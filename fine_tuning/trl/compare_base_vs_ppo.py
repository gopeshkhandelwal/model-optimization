#!/usr/bin/env python
# compare_base_vs_ppo.py

import argparse
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
    pipeline,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)
from transformers.modeling_outputs import SequenceClassifierOutput

from logging_utils import setup_logging


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default="google/gemma-3-270m")
    p.add_argument("--finetuned_model", "--ppo_model", dest="ppo_model", default="./ppo_sanity")
    p.add_argument("--reward_model", default="./rm_sanity_merged")
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--reward_max_length", type=int, default=2048, help="Max tokens for reward model scoring (clamped to avoid gigantic model_max_length sentinels).")
    p.add_argument("--seed", type=int, default=None, help="Seed for reproducibility. If unset -> non-deterministic sampling")
    p.add_argument("--do_sample", type=lambda v: str(v).lower() in {"1","true","yes"}, default=True)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--greedy", action="store_true", help="Override: force greedy generation (do_sample False)")
    p.add_argument("--prompts_file", type=str, help="Optional path to a text file with one prompt per line")
    p.add_argument(
        "--output_json", type=str, default="compare_results.json", help="Where to write structured results"
    )
    return p.parse_args()


args = parse_args()
setup_logging()
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
reward_model_path = args.reward_model

logger.info(
    f"[Init] base_model={base_model} ppo_model={ppo_model} reward_model_path={reward_model_path} seed={args.seed}"
)

if not (hasattr(torch, "hpu") and torch.hpu.is_available()):
    raise RuntimeError("[HPU][Required] Habana HPU not available. Comparison script requires HPU (no CPU fallback).")
device = "hpu"
logger.info(f"[Device] Using enforced device={device}")

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

def load_reward_model(path: str):
    """Attempt to load a reward model.
    Order:
      1. If directory contains reward_head_config.json + reward_value_head.bin -> reconstruct causal LM wrapper.
      2. Try AutoModelForSequenceClassification.
      3. Fallback: causal LM + linear head on last token hidden state.
    Returns model (nn.Module) and a flag indicating if it's a native SeqCls model.
    """
    if os.path.isdir(path):
        head_meta = os.path.join(path, "reward_head_config.json")
        head_weights = os.path.join(path, "reward_value_head.bin")
        if os.path.isfile(head_meta) and os.path.isfile(head_weights):
            logger.info("[RewardModel][MergedFallback] Detected merged causal LM reward export.")
            import json as _json
            meta = _json.load(open(head_meta))
            hidden_size = meta.get("hidden_size")
            if hidden_size is None:
                raise RuntimeError("[RewardModel][MergedFallback] hidden_size missing in reward_head_config.json")
            base_lm = AutoModelForCausalLM.from_pretrained(
                path,
                torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
                low_cpu_mem_usage=True,
                output_hidden_states=True,
            )
            class _CausalLMRewardWrapper(torch.nn.Module):
                def __init__(self, lm, hidden):
                    super().__init__()
                    self.lm = lm
                    self.value_head = torch.nn.Linear(hidden, 1)
                def forward(self, input_ids=None, attention_mask=None, **kwargs):
                    if "output_hidden_states" not in kwargs:
                        kwargs["output_hidden_states"] = True
                    outputs = self.lm(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, **kwargs)
                    hidden = outputs.hidden_states[-1]
                    if attention_mask is not None:
                        lengths = attention_mask.sum(dim=1) - 1
                        last_tokens = hidden[torch.arange(hidden.size(0), device=hidden.device), lengths.clamp(min=0)]
                    else:
                        last_tokens = hidden[:, -1]
                    reward = self.value_head(last_tokens)
                    return SequenceClassifierOutput(logits=reward)
                @property
                def config(self):
                    return self.lm.config
                def can_generate(self):
                    return False
            wrapper = _CausalLMRewardWrapper(base_lm, hidden_size)
            state = torch.load(head_weights, map_location="cpu")
            wrapper.value_head.load_state_dict(state)
            return wrapper, False
    # Try native sequence classification
    try:
        seq = AutoModelForSequenceClassification.from_pretrained(
            path,
            num_labels=1,
            torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
            low_cpu_mem_usage=True,
        )
        return seq, True
    except Exception as e:
        logger.warning(f"[RewardModel] SequenceClassification load failed ({e}). Falling back to causal LM wrapper.")
    # Fallback causal LM wrapper
    base_lm = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
        low_cpu_mem_usage=True,
        output_hidden_states=True,
    )
    hidden_size = getattr(base_lm.config, "hidden_size", getattr(base_lm.config, "model_dim", None))
    if hidden_size is None:
        raise RuntimeError("[RewardModel][Fallback] Could not infer hidden size from LM config.")
    class _CausalLMRewardWrapper(torch.nn.Module):
        def __init__(self, lm, hidden):
            super().__init__()
            self.lm = lm
            self.value_head = torch.nn.Linear(hidden, 1)
        def forward(self, input_ids=None, attention_mask=None, **kwargs):
            if "output_hidden_states" not in kwargs:
                kwargs["output_hidden_states"] = True
            outputs = self.lm(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, **kwargs)
            hidden = outputs.hidden_states[-1]
            if attention_mask is not None:
                lengths = attention_mask.sum(dim=1) - 1
                last_tokens = hidden[torch.arange(hidden.size(0), device=hidden.device), lengths.clamp(min=0)]
            else:
                last_tokens = hidden[:, -1]
            reward = self.value_head(last_tokens)
            return SequenceClassifierOutput(logits=reward)
        @property
        def config(self):
            return self.lm.config
        def can_generate(self):
            return False
    return _CausalLMRewardWrapper(base_lm, hidden_size), False

rm, is_seq = load_reward_model(reward_model_path)
rm.to(device)
rm.eval()
# Provide minimal labels for pipeline compatibility if needed
if not getattr(rm.config, "id2label", None):
    rm.config.id2label = {0: "LABEL_0"}
    rm.config.label2id = {"LABEL_0": 0}

reward_use_pipeline = True
rm_pipe = None
if is_seq:
    try:
        rm_pipe = pipeline(
            "sentiment-analysis",
            model=rm,
            tokenizer=tokenizer,
            device=0 if device != "cpu" else -1,
            function_to_apply="none",  # raw logit
            truncation=True,
            top_k=None,
        )
        logger.info("[RewardModel] Loaded SequenceClassification model & pipeline constructed")
    except Exception as e:
        logger.warning(f"[RewardModel] Pipeline failed for seq model: {e}; using manual scoring")
        reward_use_pipeline = False
else:
    reward_use_pipeline = False

def score_reward(text: str) -> float:
    if reward_use_pipeline and rm_pipe is not None:
        try:
            return float(rm_pipe(text)[0]["score"])  # type: ignore[index]
        except Exception as e:
            logger.warning(f"[RewardModel] Pipeline scoring failed ({e}); switching to manual.")
    # Manual scoring path
    # Some tokenizers set model_max_length to a huge sentinel (e.g., 1000000000000001) -> clamp
    model_max = getattr(tokenizer, "model_max_length", None)
    if model_max is None or model_max <= 0:
        model_max = args.reward_max_length
    # Treat extremely large values as 'infinite'
    if model_max > 10000:  # arbitrary threshold; adjust as needed
        model_max = args.reward_max_length
    max_len = min(model_max, args.reward_max_length)
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=False,
        max_length=max_len,
    ).to(device)
    with torch.no_grad():
        out = rm(**enc)
        logits = getattr(out, "logits", None)
        if logits is None:
            if isinstance(out, (tuple, list)) and len(out) > 0:
                logits = out[0]
            else:
                raise RuntimeError("[RewardModel] Could not obtain logits in manual scoring.")
        if logits.dim() == 2 and logits.size(1) == 1:
            return float(logits[0, 0].item())
        return float(logits.view(-1)[-1].item())

# Load both causal models once
def load_causal(path: str):
    logger.info(f"[ModelLoad] Loading causal LM: {path}")
    m = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    m.to(device)
    m.eval()
    return m


models: Dict[str, torch.nn.Module] = {
    "Base": load_causal(base_model),
    "PPO": load_causal(ppo_model),
}

if args.greedy:
    args.do_sample = False
generation_kwargs = {
    "max_new_tokens": args.max_new_tokens,
    "do_sample": args.do_sample,
    "top_p": args.top_p,
    "top_k": args.top_k,
    "temperature": args.temperature,
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": tokenizer.eos_token_id,
}
logger.info(f"[GenConfig] {generation_kwargs}")


def generate_response(model, prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_kwargs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


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
        gen_dur = time.time() - gen_start
        reward_start = time.time()
        formatted_for_reward = f"Question: {prompt}\n\nAnswer: {resp}"  # match reward model training format
        score = score_reward(formatted_for_reward)
        reward_dur = time.time() - reward_start
        model_total = time.time() - model_total_start
        logger.info(
            f"[Score] {name} reward={score:.4f} gen_time={gen_dur:.2f}s reward_time={reward_dur:.2f}s total_model_time={model_total:.2f}s"
        )
        responses[name] = {
            "response": resp,
            "score": float(score),
            "gen_time_sec": gen_dur,
            "reward_time_sec": reward_dur,
            "model_total_time_sec": model_total,
        }
    per_prompt_dur = time.time() - per_prompt_start
    if "Base" in responses and "PPO" in responses:
        delta = responses["PPO"]["score"] - responses["Base"]["score"]
        logger.info(f"[Delta] PPO - Base reward delta: {delta:.4f}")
        responses["delta_reward"] = delta
    responses["duration_sec"] = per_prompt_dur
    results.append({"prompt": prompt, "results": responses})

total_dur = time.time() - stage_start
logger.info("==================== SUMMARY (Base vs PPO) ====================")
for r in results:
    base_s = r["results"].get("Base", {}).get("score", float("nan"))
    ppo_s = r["results"].get("PPO", {}).get("score", float("nan"))
    delta = r["results"].get("delta_reward", float("nan"))
    logger.info(f"Prompt: {r['prompt'][:60]}... | Base={base_s:.4f} PPO={ppo_s:.4f} Δ={delta:.4f}")

print("\n" + "=" * 100)
print("📊 Reward Comparison")
print("=" * 100)
for r in results:
    prompt_full = r["prompt"]
    prompt_short = (prompt_full[:90] + "…") if len(prompt_full) > 93 else prompt_full
    base_entry = r["results"].get("Base")
    ppo_entry = r["results"].get("PPO")
    base_score = base_entry["score"] if base_entry else float("nan")
    ppo_score = ppo_entry["score"] if ppo_entry else float("nan")
    base_time = base_entry["model_total_time_sec"] if base_entry else float("nan")
    ppo_time = ppo_entry["model_total_time_sec"] if ppo_entry else float("nan")
    print(f"Prompt: {prompt_short}")
    print(f"       Base: Score: {base_score:.4f},  Time: {base_time:.2f}s")
    print(f"       PPO : Score: {ppo_score:.4f},  Time: {ppo_time:.2f}s")
    if "delta_reward" in r["results"]:
        delta = r["results"]["delta_reward"]
        print(f"       Δ(PPO-Base): {delta:.4f}")
    print()
print(f"Total prompts: {len(results)} | Total elapsed: {total_dur:.2f}s")

if results and results[0]["results"].get("delta_reward") is not None:
    deltas = [r["results"].get("delta_reward") for r in results if "delta_reward" in r["results"]]
    if deltas:
        import statistics

        mean_delta = statistics.mean(deltas)
        logger.info(f"[Aggregate] Mean reward delta (PPO - Base): {mean_delta:.4f}")
        print(f"Mean Delta (PPO-Base): {mean_delta:.4f}")
        try:
            median_delta = statistics.median(deltas)
            logger.info(f"[Aggregate] Median reward delta: {median_delta:.4f}")
            print(f"Median Delta (PPO-Base): {median_delta:.4f}")
        except statistics.StatisticsError:
            pass

output_json = Path(args.output_json)
with output_json.open("w") as f:
    json.dump({"total_duration_sec": total_dur, "comparisons": results, "gen_config": generation_kwargs}, f, indent=2)
logger.info(f"[Output] JSON written -> {output_json} (time={total_dur:.2f}s)")
print(f"Results JSON: {output_json} | Total time: {total_dur:.2f}s")
