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

from logging_utils import setup_logging


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default="meta-llama/Llama-2-7b-hf")
    p.add_argument("--finetuned_model", "--ppo_model", dest="ppo_model", default="./ppo_sanity")
    p.add_argument("--reward_model", default="./rm_sanity_merged")
    p.add_argument("--max_new_tokens", type=int, default=64)
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

device = "hpu" if hasattr(torch, "hpu") else ("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"[Device] Using device={device}")

# Prompts
if args.prompts_file and Path(args.prompts_file).is_file():
    prompts = [ln.strip() for ln in Path(args.prompts_file).read_text().splitlines() if ln.strip()]
    logger.info(f"[Prompts] Loaded {len(prompts)} prompts from {args.prompts_file}")
else:
    prompts = [
        "Why do programmers prefer Python over Java for machine learning?",
        "What is the difference between supervised and unsupervised learning?",
        "What are the advantages of Docker for deploying applications?",
        "Why does gradient descent work even though it doesn’t always find the global minimum?",
    ]
    logger.info(f"[Prompts] Using default {len(prompts)} hard-coded prompts")

# Tokenizer (shared)
tokenizer = AutoTokenizer.from_pretrained(base_model)
if getattr(tokenizer, "pad_token", None) is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"  # safer for causal LM generation when padding
logger.info("[Tokenizer] Loaded tokenizer & set pad_token -> eos_token")

# Reward model
rm = AutoModelForSequenceClassification.from_pretrained(
    reward_model_path, num_labels=1, torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32
)
rm.to(device)
rm.eval()
rm_pipe = pipeline(
    "sentiment-analysis",
    model=rm,
    tokenizer=tokenizer,
    device=0 if device != "cpu" else -1,
    function_to_apply="none",  # raw score
    return_all_scores=False,
    truncation=True,
)
logger.info("[RewardModel] Loaded & pipeline constructed")

# Load both causal models once
def load_causal(path: str):
    logger.info(f"[ModelLoad] Loading causal LM: {path}")
    m = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
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
        score = rm_pipe(formatted_for_reward)[0]["score"]
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
