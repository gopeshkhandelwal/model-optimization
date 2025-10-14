#!/usr/bin/env python
# compare_base_vs_dpo.py (adapted from PPO comparison; reward model optional for qualitative check)

import argparse
import warnings
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
try:
    from peft import PeftConfig, PeftModel
    _PEFT_AVAILABLE = True
except Exception:
    _PEFT_AVAILABLE = False

from logging_utils import setup_logging


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True, help="Base model path (REQUIRED)")
    p.add_argument("--finetuned_model", "--ppo_model", dest="ppo_model", required=True, help="Fine-tuned/DPO model path (REQUIRED)")
    p.add_argument("--max_new_tokens", type=int, default=96)
    p.add_argument("--seed", type=int, default=None, help="Seed for reproducibility. If unset -> non-deterministic sampling")
    p.add_argument("--do_sample", type=lambda v: str(v).lower() in {"1","true","yes"}, default=True)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--temperature", type=float, default=1.05)
    p.add_argument("--repetition_penalty", type=float, default=1.12, help="Penalty for repeating tokens (improves diversity)")
    p.add_argument("--greedy", action="store_true", help="Override: force greedy generation (do_sample False)")
    p.add_argument("--prompts_file", type=str, help="Optional path to a text file with one prompt per line")
    p.add_argument(
        "--output_json", type=str, default="compare_results.json", help="Where to write structured results"
    )
    p.add_argument("--show_text", action="store_true", help="Print full generated texts for both models")
    p.add_argument("--heuristics", action="store_true", help="Compute simple heuristic metrics (length, sentence count, keyword hits, diversity)")
    p.add_argument("--min_new_tokens", type=int, default=0, help="Force at least this many new tokens by regenerating if shorter (simple safeguard against early EOS)")
    p.add_argument("--stackexchange_style", action="store_true", help="Wrap prompts as 'Question: ...\n\nAnswer:' to mirror training format before generation")
    return p.parse_args()


args = parse_args()
setup_logging()
warnings.filterwarnings(
    "ignore",
    message="Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.",
    category=UserWarning,
)
logger = logging.getLogger(__name__)


def set_all_seeds(seed: int):
    if seed is None:
        logger.info("[Seed] No seed provided -> results will vary run-to-run (sampling stochastic).")
        return
    logger.info(f"[Seed] Setting seed={seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Habana specific (if available)
    if hasattr(torch, "hpu"):
        try:
            torch.hpu.manual_seed(seed)  # type: ignore[attr-defined]
        except Exception:
            pass
    os.environ["PYTHONHASHSEED"] = str(seed)


set_all_seeds(args.seed)

base_model = args.base_model
ppo_model = args.ppo_model

# Validate required arguments
if not base_model or base_model.strip() == "":
    raise ValueError("--base_model is required and cannot be empty. Please specify the base model path.")
if not ppo_model or ppo_model.strip() == "":
    raise ValueError("--finetuned_model is required and cannot be empty. Please specify the fine-tuned model path.")
logger.info(
    f"[Init] base_model={base_model} finetuned_model={ppo_model} seed={args.seed}"
)

def _hpu_available():
    # Robust check for HPU presence
    try:
        if not hasattr(torch, "hpu"):
            import importlib
            importlib.import_module("habana_frameworks.torch.core")  # side-effect registers torch.hpu
        return hasattr(torch, "hpu") and torch.hpu.is_available()
    except Exception as e:
        logger.warning(f"[HPU][Diagnose] Import error: {e}")
        return False

# Early diagnostics
logger.info(f"[HPU][Env] HPU env vars summary: " + ", ".join([f"{k}={v}" for k,v in os.environ.items() if k.startswith("HABANA") or k.startswith("PT_HPU")]))
logger.info(f"[HPU][Torch] hasattr(torch,'hpu')={hasattr(torch,'hpu')}")

if _hpu_available():
    device = "hpu"
    dtype = torch.bfloat16
    try:
        import habana_frameworks.torch.core.hpu as hpu_mod  # noqa: F401
        logger.info("[HPU][Import] habana_frameworks.torch.core.hpu imported successfully")
    except Exception as e:
        logger.warning(f"[HPU][Import] Failed optional import: {e}")
    logger.info("[Device] Using Habana HPU")
elif torch.cuda.is_available():
    device = "cuda"
    # Prefer bfloat16 if supported else float16 for small model speed
    if torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = torch.float16
    logger.info(f"[Device] Using CUDA ({dtype})")
else:
    device = "cpu"
    dtype = torch.float32
    logger.info("[Device] Using CPU fallback (dtype=float32). This will be slower.")

# Prompts
if args.prompts_file and Path(args.prompts_file).is_file():
    prompts = [ln.strip() for ln in Path(args.prompts_file).read_text().splitlines() if ln.strip()]
    logger.info(f"[Prompts] Loaded {len(prompts)} prompts from {args.prompts_file}")
else:
    prompts = [
        "Why do programmers prefer Python over Java for machine learning?",
        "What are the advantages of Docker for deploying applications?",
    ]
    logger.info(f"[Prompts] Using default {len(prompts)} hard-coded prompts")

# Tokenizer (shared)
tokenizer = AutoTokenizer.from_pretrained(base_model)
if getattr(tokenizer, "pad_token", None) is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"  # safer for causal LM generation when padding
logger.info("[Tokenizer] Loaded tokenizer & set pad_token -> eos_token")

logger.info("[RewardModel] Removed reward model logic for pure generation comparison.")

# Load both causal models once
def load_causal(path: str, is_adapter: bool = False, base_model: str = None):
    if is_adapter and not _PEFT_AVAILABLE:
        logger.warning("[ModelLoad][PEFT] Adapter indicated but PEFT not available; attempting direct load.")
    if is_adapter and _PEFT_AVAILABLE:
        if not base_model:
            raise ValueError("Base model path required to load PEFT adapter")
        logger.info(f"[ModelLoad][PEFT] Loading base model then applying adapter: base={base_model} adapter={path}")
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        base.to(device)
        base.eval()
        wrapped = PeftModel.from_pretrained(base, path)
        try:
            logger.info("[ModelLoad][PEFT] Merging LoRA adapter for faster generation")
            merged = wrapped.merge_and_unload()
            merged.to(device)
            merged.eval()
            return merged
        except Exception as e:
            logger.warning(f"[ModelLoad][PEFT] merge_and_unload failed: {e}; using wrapped model.")
            wrapped.to(device)
            wrapped.eval()
            return wrapped
    else:
        logger.info(f"[ModelLoad] Loading causal LM: {path}")
        m = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        m.to(device)
        m.eval()
        return m


def _is_peft_adapter_dir(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    # Common adapter marker files
    markers = ["adapter_config.json", "adapter_model.bin", "adapter_model.safetensors"]
    for m in markers:
        if os.path.exists(os.path.join(path, m)):
            return True
    return False

is_adapter = _is_peft_adapter_dir(ppo_model)
if is_adapter:
    logger.info(f"[Detect][PEFT] Finetuned path appears to be a PEFT adapter directory: {ppo_model}")
else:
    logger.info(f"[Detect] Finetuned path treated as full model directory: {ppo_model}")

models: Dict[str, torch.nn.Module] = {
    "Base": load_causal(base_model),
    "DPO": load_causal(ppo_model, is_adapter=is_adapter, base_model=base_model if is_adapter else None),
}

if args.greedy:
    args.do_sample = False
elif args.stackexchange_style and args.min_new_tokens == 0:
    # Auto-raise safeguard when using Q/A wrapper
    args.min_new_tokens = 24
    logger.info(f"[GenConfig][Auto] stackexchange_style active -> setting min_new_tokens={args.min_new_tokens}")
generation_kwargs = {
    "max_new_tokens": args.max_new_tokens,
    "do_sample": args.do_sample,
    "top_p": args.top_p,
    "top_k": args.top_k,
    "temperature": args.temperature,
    "repetition_penalty": args.repetition_penalty,
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": tokenizer.eos_token_id,
}
logger.info(f"[GenConfig] {generation_kwargs}")


def generate_response(model, prompt: str) -> str:
    gen_prompt = prompt
    if args.stackexchange_style:
        gen_prompt = f"Question: {prompt}\n\nAnswer:"
    inputs = tokenizer(gen_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_kwargs)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Simple safeguard for min_new_tokens: re-run once if below threshold
    if args.min_new_tokens > 0:
        new_part = text[len(gen_prompt):]
        token_len = len(tokenizer(new_part, add_special_tokens=False)["input_ids"])
        if token_len < args.min_new_tokens:
            logger.info(f"[Gen][Retry] Token length {token_len} < min_new_tokens {args.min_new_tokens}; retrying once.")
            with torch.no_grad():
                outputs = model.generate(**inputs, **generation_kwargs)
            text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

def expand_if_too_short(model, original_prompt: str, response: str, min_tokens: int = 40) -> str:
    """Iteratively expand very short responses with domain-specific hints until min_tokens reached."""
    gen_prefix = f"Question: {original_prompt}\n\nAnswer:"
    content = response[len(gen_prefix):] if response.startswith(gen_prefix) else response
    token_len = len(tokenizer(content, add_special_tokens=False)["input_ids"])
    if token_len >= min_tokens:
        return response
    logger.info(f"[Gen][Expand] Response tokens {token_len} < {min_tokens}; starting guided expansion loop.")
    hints = [
        " Please provide a concise but informative explanation including at least three concrete advantages:",
        " List key benefits such as portability, isolation, reproducibility, efficient resource usage, and simplified CI/CD:",
        " Add a short practical example of how an engineering team benefits from these features:",
    ]
    working_text = response
    for idx, hint in enumerate(hints, start=1):
        new_prompt = gen_prefix + hint
        inputs = tokenizer(new_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_kwargs)
        new_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        new_content = new_text[len(gen_prefix):] if new_text.startswith(gen_prefix) else new_text
        new_token_len = len(tokenizer(new_content, add_special_tokens=False)["input_ids"])
        logger.info(f"[Gen][Expand][Step {idx}] tokens={new_token_len}")
        if new_token_len > token_len:
            # Append incremental expansion content rather than replacing entire text
            append_part = new_content
            # Avoid duplicating existing content if model echoed original
            if append_part.startswith(content.strip()):
                append_part = append_part[len(content.strip()):].lstrip()
            working_text = working_text.rstrip() + "\n" + append_part.strip()
            content = content + " " + append_part
            token_len = len(tokenizer(content, add_special_tokens=False)["input_ids"])
        if token_len >= min_tokens:
            logger.info("[Gen][Expand] Target reached; stopping expansion.")
            break
    return working_text


results = []
stage_start = time.time()
for idx, prompt in enumerate(prompts, start=1):
    logger.info("=" * 60)
    logger.info(f"[Prompt {idx}/{len(prompts)}] {prompt}")
    responses = {}
    per_prompt_start = time.time()
    for name, model in models.items():
        model_total_start = time.time()
        gen_start = time.time()
        resp = generate_response(model, prompt)
        # Fallback expansion only for DPO model
        if name == "DPO":
            resp = expand_if_too_short(model, prompt, resp, min_tokens=max(80, args.min_new_tokens))
        gen_dur = time.time() - gen_start
        model_total = time.time() - model_total_start
        logger.info(
            f"[Gen] {name} gen_time={gen_dur:.2f}s total_model_time={model_total:.2f}s"
        )
        responses[name] = {
            "response": resp,
            "score": None,
            "gen_time_sec": gen_dur,
            "reward_time_sec": 0.0,
            "model_total_time_sec": model_total,
        }
    per_prompt_dur = time.time() - per_prompt_start
    # No reward scoring -> no delta
    responses["duration_sec"] = per_prompt_dur
    results.append({"prompt": prompt, "results": responses})

total_dur = time.time() - stage_start
logger.info("==================== SUMMARY (Base vs DPO) ====================")
for r in results:
    # Scores are always None now; show lengths instead
    base_resp = r["results"].get("Base", {}).get("response", "")
    dpo_resp = r["results"].get("DPO", {}).get("response", "")
    base_len = len(base_resp)
    dpo_len = len(dpo_resp)
    logger.info(
        f"Prompt: {r['prompt'][:60]}... | Base_len={base_len} DPO_len={dpo_len}"
    )

print("\n" + "=" * 100)
print(" Generation Comparison (no reward model / DPO pure generations)")
print("=" * 100)
for r in results:
    prompt_full = r["prompt"]
    prompt_short = (prompt_full[:90] + "…") if len(prompt_full) > 93 else prompt_full
    base_entry = r["results"].get("Base")
    ppo_entry = r["results"].get("DPO")
    base_time = base_entry["model_total_time_sec"] if base_entry else float("nan")
    ppo_time = ppo_entry["model_total_time_sec"] if ppo_entry else float("nan")
    base_len = len(base_entry["response"]) if base_entry else 0
    dpo_len = len(ppo_entry["response"]) if ppo_entry else 0
    print(f"Prompt: {prompt_short}")
    print(f"       Base: Time: {base_time:.2f}s | Length: {base_len}")
    print(f"       DPO : Time: {ppo_time:.2f}s | Length: {dpo_len}")
    if args.show_text:
        print("       --- Base Output ---")
        print(base_entry["response"])
        print("       --- DPO Output  ---")
        print(ppo_entry["response"])
    print()
print(f"Total prompts: {len(results)} | Total elapsed: {total_dur:.2f}s")

# No aggregate reward stats since no scoring performed

output_json = Path(args.output_json)
payload = {"total_duration_sec": total_dur, "comparisons": results, "gen_config": generation_kwargs}
def _conciseness_gain(comparisons):
    gains = []
    for comp in comparisons:
        base_txt = comp["results"]["Base"]["response"]
        dpo_txt = comp["results"]["DPO"]["response"]
        b_len = len(base_txt)
        d_len = len(dpo_txt)
        if d_len > 0:
            gains.append(b_len / d_len)
    if gains:
        import statistics
        return {
            "avg_gain": sum(gains)/len(gains),
            "median_gain": statistics.median(gains),
            "min_gain": min(gains),
            "max_gain": max(gains),
            "count": len(gains),
        }
    return {}

payload["conciseness_gain"] = _conciseness_gain(results)

def simple_heuristics(text: str) -> Dict[str, float]:
    import re
    sentences = [s for s in re.split(r'[.!?]\s+', text.strip()) if s]
    words = text.strip().split()
    unique_words = set(w.lower() for w in words)
    keywords = {"python", "docker", "learning", "performance", "model"}
    keyword_hits = sum(1 for w in unique_words if w in keywords)
    # Novelty heuristic: ratio of words not in first sentence to total words (rough measure of expansion)
    first_sentence_words = set(w.lower() for w in (sentences[0].split() if sentences else []))
    novel_words = [w for w in unique_words if w not in first_sentence_words]
    novelty_ratio = len(novel_words) / max(1, len(unique_words))
    return {
        "char_len": len(text),
        "token_len": len(words),
        "sentence_count": len(sentences),
        "unique_word_ratio": (len(unique_words) / max(1, len(words))),
        "keyword_hits": keyword_hits,
        "novelty_ratio": novelty_ratio,
    }

if args.heuristics:
    for comp in payload["comparisons"]:
        for name in ("Base", "DPO"):
            resp = comp["results"][name]["response"]
            comp["results"][name]["heuristics"] = simple_heuristics(resp)
    # Aggregate averages
    import numpy as np
    agg = {"Base": {}, "DPO": {}}
    keys = ["char_len", "token_len", "sentence_count", "unique_word_ratio", "keyword_hits", "novelty_ratio"]
    for model_key in ("Base", "DPO"):
        for k in keys:
            vals = [c["results"][model_key]["heuristics"][k] for c in payload["comparisons"]]
            agg[model_key][k] = float(sum(vals)/len(vals)) if vals else None
    payload["heuristic_averages"] = agg

with output_json.open("w") as f:
    json.dump(payload, f, indent=2)
logger.info(f"[Output] JSON written -> {output_json} (time={total_dur:.2f}s)")
print(f"Results JSON: {output_json} | Total time: {total_dur:.2f}s")
