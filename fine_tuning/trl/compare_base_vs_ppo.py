#!/usr/bin/env python
# compare_base_vs_ppo.py
import torch
import logging
import json
import time
from pathlib import Path
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM, AutoModelForSequenceClassification

# Models
base_model = "meta-llama/Llama-2-7b-hf"
ppo_model = "./ppo_sanity"
reward_model_path = "./rm_sanity_merged"
from logging_utils import setup_logging
setup_logging()
logger = logging.getLogger(__name__)
logger.info(f"[Init] base_model={base_model} ppo_model={ppo_model} reward_model_path={reward_model_path}")

# Shared tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token = tokenizer.eos_token
logger.info("[Tokenizer] Loaded and pad_token set to eos_token")

# Reward model
rm = AutoModelForSequenceClassification.from_pretrained(
    reward_model_path, num_labels=1, torch_dtype=torch.bfloat16
)
rm_pipe = pipeline("sentiment-analysis", model=rm, tokenizer=tokenizer, device=0)
logger.info("[RewardModel] Loaded reward model and created pipeline")

prompts = [
    "Why do programmers prefer Python over Java for machine learning?",
    "What is the difference between supervised and unsupervised learning?",
    "What are the advantages of Docker for deploying applications?",
    "Why does gradient descent work even though it doesn’t always find the global minimum?"
]

def generate_response(model_name, prompt, max_new_tokens=64):
    """Generate response from a model"""
    logger.info(f"[Generate] Loading model {model_name} for prompt snippet: {prompt[:40]}...")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to("hpu")
    inputs = tokenizer(prompt, return_tensors="pt").to("hpu")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

results = []
stage_start = time.time()
for idx, prompt in enumerate(prompts, start=1):
    logger.info("============================================================")
    logger.info(f"[Prompt {idx}/{len(prompts)}] {prompt}")
    responses = {}
    per_prompt_start = time.time()
    for name, path in [("Base", base_model), ("PPO", ppo_model)]:
        model_total_start = time.time()
        gen_start = time.time()
        resp = generate_response(path, prompt)
        gen_dur = time.time() - gen_start
        reward_start = time.time()
        score = rm_pipe(resp)[0]["score"]
        reward_dur = time.time() - reward_start
        model_total = time.time() - model_total_start
        logger.info(f"[Score] {name} reward={score:.4f} gen_time={gen_dur:.2f}s reward_time={reward_dur:.2f}s total_model_time={model_total:.2f}s")
        responses[name] = {
            "response": resp,
            "score": float(score),
            "gen_time_sec": gen_dur,
            "reward_time_sec": reward_dur,
            "model_total_time_sec": model_total,
        }
    per_prompt_dur = time.time() - per_prompt_start
    # Compute delta (PPO - Base) if both present
    if "Base" in responses and "PPO" in responses:
        delta = responses["PPO"]["score"] - responses["Base"]["score"]
        logger.info(f"[Delta] PPO - Base reward delta: {delta:.4f}")
        responses["delta_reward"] = delta
    responses["duration_sec"] = per_prompt_dur
    results.append({"prompt": prompt, "results": responses})

total_dur = time.time() - stage_start
logger.info("==================== SUMMARY (Base vs PPO) ====================")
for r in results:
    base_s = r["results"]["Base"]["score"] if "Base" in r["results"] else float('nan')
    ppo_s = r["results"]["PPO"]["score"] if "PPO" in r["results"] else float('nan')
    delta = r["results"].get("delta_reward", float('nan'))
    logger.info(f"Prompt: {r['prompt'][:60]}... | Base={base_s:.4f} PPO={ppo_s:.4f} Δ={delta:.4f}")

# Human-readable inline summary (Score(Time)) format
print("\n" + "="*100)
print("📊 Reward Comparison")
print("="*100)
for r in results:
    prompt_full = r["prompt"]
    prompt_short = (prompt_full[:90] + "…") if len(prompt_full) > 93 else prompt_full
    base_entry = r["results"].get("Base")
    ppo_entry = r["results"].get("PPO")
    base_score = base_entry['score'] if base_entry else float('nan')
    ppo_score = ppo_entry['score'] if ppo_entry else float('nan')
    base_time = base_entry['model_total_time_sec'] if base_entry else float('nan')
    ppo_time = ppo_entry['model_total_time_sec'] if ppo_entry else float('nan')
    print(f"Prompt: {prompt_short}")
    print(f"       Base: Score: {base_score:.4f},  Time: {base_time:.2f}s")
    print(f"       PPO : Score: {ppo_score:.4f},  Time: {ppo_time:.2f}s")
    if 'delta_reward' in r['results']:
        delta = r['results']['delta_reward']
        print(f"       Δ(PPO-Base): {delta:.4f}")
    print()
print(f"Total prompts: {len(results)} | Total elapsed: {total_dur:.2f}s")

# Aggregate statistics
if results and "delta_reward" in results[0]["results"]:
    deltas = [r["results"].get("delta_reward") for r in results if "delta_reward" in r["results"]]
    if deltas:
        import statistics
        mean_delta = statistics.mean(deltas)
        logger.info(f"[Aggregate] Mean reward delta (PPO - Base): {mean_delta:.4f}")
        try:
            median_delta = statistics.median(deltas)
            logger.info(f"[Aggregate] Median reward delta: {median_delta:.4f}")
        except statistics.StatisticsError:
            pass
        print(f"Mean Delta (PPO-Base): {mean_delta:.4f}")
        try:
            print(f"Median Delta (PPO-Base): {median_delta:.4f}")
        except NameError:
            pass

# Optional JSON export
output_json = Path("compare_results.json")
with output_json.open("w") as f:
    json.dump({"total_duration_sec": total_dur, "comparisons": results}, f, indent=2)
logger.info(f"[Output] Comparison JSON written to {output_json} (total_time={total_dur:.2f}s)")
print(f"Results JSON: {output_json} | Total time: {total_dur:.2f}s")
