#!/usr/bin/env python
# compare_base_vs_dpo.py

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
    AutoModelForCausalLM,
)

from logging_utils import setup_logging


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True, help="Base model path (REQUIRED)")
    p.add_argument("--finetuned_model", "--dpo_model", dest="dpo_model", required=True, help="Fine-tuned/DPO model path (REQUIRED)")
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--seed", type=int, default=None, help="Seed for reproducibility. If unset -> non-deterministic sampling")
    p.add_argument("--do_sample", type=lambda v: str(v).lower() in {"1","true","yes"}, default=True)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--greedy", action="store_true", help="Override: force greedy generation (do_sample False)")
    p.add_argument("--prompts_file", type=str, help="Optional path to a text file with one prompt per line")
    p.add_argument(
        "--output_json", type=str, default="compare_dpo_results.json", help="Where to write structured results"
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
dpo_model = args.dpo_model

# Validate required arguments
if not base_model or base_model.strip() == "":
    raise ValueError("--base_model is required and cannot be empty. Please specify the base model path.")
if not dpo_model or dpo_model.strip() == "":
    raise ValueError("--finetuned_model is required and cannot be empty. Please specify the fine-tuned model path.")

logger.info(
    f"[Init] base_model={base_model} dpo_model={dpo_model} seed={args.seed}"
)

device = "hpu" if hasattr(torch, "hpu") else ("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"[Device] Using device={device}")

# Prompts
if args.prompts_file and Path(args.prompts_file).is_file():
    prompts = [ln.strip() for ln in Path(args.prompts_file).read_text().splitlines() if ln.strip()]
    logger.info(f"[Prompts] Loaded {len(prompts)} prompts from {args.prompts_file}")
else:
    prompts = [
        "Why do programmers prefer Python over Java for machine learning?",
        "What are the advantages of Docker for deploying applications?",
        "Explain the concept of reinforcement learning in simple terms.",
        "What are the key differences between supervised and unsupervised learning?",
    ]
    logger.info(f"[Prompts] Using default {len(prompts)} hard-coded prompts")

# Tokenizer (shared)
tokenizer = AutoTokenizer.from_pretrained(base_model)
if getattr(tokenizer, "pad_token", None) is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"  # safer for causal LM generation when padding
logger.info("[Tokenizer] Loaded tokenizer & set pad_token -> eos_token")

# Load both causal models once
def load_causal(path: str):
    logger.info(f"[ModelLoad] Loading causal LM: {path}")
    m = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
        low_cpu_mem_usage=True,
        attn_implementation="eager",  # For Gemma3 compatibility
    )
    m.to(device)
    m.eval()
    return m


models: Dict[str, torch.nn.Module] = {
    "Base": load_causal(base_model),
    "DPO": load_causal(dpo_model),
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
    # Remove the input prompt from the output
    generated_tokens = outputs[0][len(inputs.input_ids[0]):]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)


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
        model_total = time.time() - model_total_start
        logger.info(
            f"[Response] {name} gen_time={gen_dur:.2f}s total_model_time={model_total:.2f}s"
        )
        logger.info(f"[Response] {name}: {resp[:100]}...")
        responses[name] = {
            "response": resp,
            "gen_time_sec": gen_dur,
            "model_total_time_sec": model_total,
        }
    per_prompt_dur = time.time() - per_prompt_start
    responses["duration_sec"] = per_prompt_dur
    results.append({"prompt": prompt, "results": responses})

total_dur = time.time() - stage_start
logger.info("==================== SUMMARY (Base vs DPO) ====================")
for r in results:
    logger.info(f"Prompt: {r['prompt'][:60]}...")
    base_resp = r["results"].get("Base", {}).get("response", "N/A")
    dpo_resp = r["results"].get("DPO", {}).get("response", "N/A")
    logger.info(f"  Base: {base_resp[:100]}...")
    logger.info(f"  DPO:  {dpo_resp[:100]}...")

print("\n" + "=" * 100)
print("📊 Response Comparison (Base vs DPO)")
print("=" * 100)
for r in results:
    prompt_full = r["prompt"]
    prompt_short = (prompt_full[:90] + "…") if len(prompt_full) > 93 else prompt_full
    base_entry = r["results"].get("Base")
    dpo_entry = r["results"].get("DPO")
    base_time = base_entry["model_total_time_sec"] if base_entry else float("nan")
    dpo_time = dpo_entry["model_total_time_sec"] if dpo_entry else float("nan")
    base_resp = base_entry["response"] if base_entry else "N/A"
    dpo_resp = dpo_entry["response"] if dpo_entry else "N/A"
    
    print(f"Prompt: {prompt_short}")
    print(f"       Base: {base_resp[:80]}... (Time: {base_time:.2f}s)")
    print(f"       DPO : {dpo_resp[:80]}... (Time: {dpo_time:.2f}s)")
    print()

print(f"Total prompts: {len(results)} | Total elapsed: {total_dur:.2f}s")

# Calculate average generation times
base_times = [r["results"].get("Base", {}).get("gen_time_sec", 0) for r in results]
dpo_times = [r["results"].get("DPO", {}).get("gen_time_sec", 0) for r in results]

if base_times and dpo_times:
    avg_base_time = sum(base_times) / len(base_times)
    avg_dpo_time = sum(dpo_times) / len(dpo_times)
    logger.info(f"[Performance] Average Base generation time: {avg_base_time:.2f}s")
    logger.info(f"[Performance] Average DPO generation time: {avg_dpo_time:.2f}s")
    print(f"Average Generation Time - Base: {avg_base_time:.2f}s, DPO: {avg_dpo_time:.2f}s")

output_json = Path(args.output_json)
with output_json.open("w") as f:
    json.dump({"total_duration_sec": total_dur, "comparisons": results, "gen_config": generation_kwargs}, f, indent=2)
logger.info(f"[Output] JSON written -> {output_json} (time={total_dur:.2f}s)")
print(f"Results JSON: {output_json} | Total time: {total_dur:.2f}s")