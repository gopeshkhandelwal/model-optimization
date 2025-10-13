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
    p = argparse.ArgumentParser(description="Compare Base vs DPO models (responses + timing). DPO does not use a reward model.")
    p.add_argument("--base_model", required=True, help="Base model path (REQUIRED)")
    p.add_argument("--finetuned_model", "--dpo_model", dest="dpo_model", required=True, help="Fine-tuned/DPO model path (REQUIRED)")
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--seed", type=int, default=None, help="Seed for reproducibility")
    p.add_argument("--do_sample", type=lambda v: str(v).lower() in {"1","true","yes"}, default=True)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--greedy", action="store_true", help="Force greedy decoding")
    p.add_argument("--prompts_file", type=str, help="Path to text file with one prompt per line")
    p.add_argument("--prompt", type=str, help="Single prompt override")
    p.add_argument("--demo_fast", action="store_true", help="Ultra-fast demo mode: one prompt, greedy, small max_new_tokens")
    p.add_argument("--output_json", type=str, default="compare_dpo_results.json", help="Where to write structured results")
    # Pairwise alignment metrics flags
    p.add_argument("--pairs_file", type=str, help="Path to JSONL or JSON file with preference pairs (prompt, chosen, rejected)")
    p.add_argument("--compute_pairwise_alignment", action="store_true", help="Compute direct logprob margins on preference pairs")
    p.add_argument("--max_pairs", type=int, default=None, help="Limit number of pairs for speed")
    p.add_argument("--sample_seed", type=int, default=None, help="Seed for sampling subset of pairs")
    p.add_argument("--compute_kl", action="store_true", help="Also approximate token-level KL(Base||DPO) over chosen responses")
    p.add_argument("--truncate_pair_responses", type=int, default=256, help="Truncate chosen/rejected texts to this many tokens for logprob calc")
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

device = "hpu" if hasattr(torch, "hpu") and getattr(torch.hpu, "is_available", lambda: False)() else ("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"[Device] Using device={device}")

# Prompts
if args.demo_fast:
    logger.warning("[DemoFast][Compare] Activating fast demo mode: single prompt, greedy, max_new_tokens=64")
    args.greedy = True
    args.do_sample = False
    if args.max_new_tokens > 64:
        args.max_new_tokens = 64
    if args.prompt and args.prompt.strip():
        prompts = [args.prompt.strip()]
        logger.info("[Prompts] Using single provided prompt under demo_fast")
    else:
        prompts = ["Explain what Direct Preference Optimization does in one sentence."]
        logger.info("[Prompts] Using default single demo_fast prompt")
elif args.prompt and args.prompt.strip():
    prompts = [args.prompt.strip()]
    logger.info("[Prompts] Using single prompt from --prompt")
elif args.prompts_file and Path(args.prompts_file).is_file():
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

get_reward_score = None  # DPO comparison does not use external reward scoring

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
    text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    if text:
        return text
    # Fallback path: if greedy produced nothing (often only EOS), try a light sampling pass
    fallback_kwargs = generation_kwargs.copy()
    fallback_kwargs.update({
        "do_sample": True,
        "temperature": max(0.7, generation_kwargs.get("temperature", 1.0)),
        "top_p": 0.95,
        "top_k": 50 if generation_kwargs.get("top_k", 0) == 0 else generation_kwargs.get("top_k", 50),
    })
    try:
        with torch.no_grad():
            outputs2 = model.generate(**inputs, **fallback_kwargs)
        gen2 = outputs2[0][len(inputs.input_ids[0]):]
        text2 = tokenizer.decode(gen2, skip_special_tokens=True).strip()
        if text2:
            logger.info("[GenFallback] Empty greedy output -> used sampling fallback")
            return text2
    except Exception as e:
        logger.warning(f"[GenFallback] Sampling fallback failed: {e}")
    # Last resort: return sentinel so downstream metrics can record occurrence
    logger.warning("[GenFallback] Returning <NO_OUTPUT> sentinel after empty attempts")
    return "<NO_OUTPUT>"


results = []
stage_start = time.time()
def compute_intrinsic_metrics(prompt: str, base_text: str, dpo_text: str):
    """Compute simple intrinsic comparison metrics to illustrate improvements without a reward model.
    Returns dict with lengths, token counts, lexical novelty ratio, overlap ratio, and length delta."""
    # Basic lengths
    base_len_chars = len(base_text)
    dpo_len_chars = len(dpo_text)
    base_len_words = len(base_text.split()) or 1
    dpo_len_words = len(dpo_text.split()) or 1
    # Token counts (approx using tokenizer to capture subwords)
    base_tokens = tokenizer(base_text, add_special_tokens=False).input_ids
    dpo_tokens = tokenizer(dpo_text, add_special_tokens=False).input_ids
    base_token_count = len(base_tokens) or 1
    dpo_token_count = len(dpo_tokens) or 1
    # Novelty: fraction of DPO words not appearing in Base response
    base_word_set = set(base_text.lower().split())
    dpo_words = dpo_text.lower().split()
    novelty_count = sum(1 for w in dpo_words if w not in base_word_set)
    novelty_ratio = novelty_count / len(dpo_words) if dpo_words else 0.0
    # Overlap ratio: shared words / union words
    dpo_word_set = set(dpo_words)
    shared = len(base_word_set & dpo_word_set)
    union = len(base_word_set | dpo_word_set) or 1
    overlap_ratio = shared / union
    # Length delta (words)
    delta_words = dpo_len_words - base_len_words
    return {
        "base_len_words": base_len_words,
        "dpo_len_words": dpo_len_words,
        "delta_len_words": delta_words,
        "base_token_count": base_token_count,
        "dpo_token_count": dpo_token_count,
        "novelty_ratio": round(novelty_ratio, 4),
        "overlap_ratio": round(overlap_ratio, 4),
    }
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
        entry = {
            "response": resp,
            "gen_time_sec": gen_dur,
            "model_total_time_sec": model_total,
        }
        responses[name] = entry
    per_prompt_dur = time.time() - per_prompt_start
    # Intrinsic metrics (only if both present)
    if "Base" in responses and "DPO" in responses:
        metrics = compute_intrinsic_metrics(prompt, responses["Base"]["response"], responses["DPO"]["response"])
        responses["intrinsic_metrics"] = metrics
        logger.info(f"[Intrinsic] novelty={metrics['novelty_ratio']:.4f} overlap={metrics['overlap_ratio']:.4f} Δwords={metrics['delta_len_words']}")
    responses["duration_sec"] = per_prompt_dur
    results.append({"prompt": prompt, "results": responses})


# ================= Pairwise Alignment Metrics (Preference Signal) =================
pairwise_alignment = None
if args.compute_pairwise_alignment:
    logger.info("[Pairwise] Starting computation of preference alignment metrics...")
    def _read_pairs(path: str):
        if not path or not os.path.exists(path):
            logger.error(f"[Pairwise] --pairs_file not found: {path}")
            return []
        raw = []
        if path.endswith('.jsonl'):
            for ln in Path(path).read_text().splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    obj = json.loads(ln)
                except Exception as e:
                    logger.warning(f"[Pairwise] Skip malformed line: {e}")
                    continue
                raw.append(obj)
        else:
            try:
                data = json.loads(Path(path).read_text())
                if isinstance(data, list):
                    raw.extend(data)
                else:
                    logger.error("[Pairwise] JSON file must hold a list of objects")
            except Exception as e:
                logger.error(f"[Pairwise] Failed parsing JSON: {e}")
        pairs = []
        for o in raw:
            if all(k in o for k in ("prompt", "chosen", "rejected")):
                pairs.append({"prompt": o["prompt"], "chosen": o["chosen"], "rejected": o["rejected"]})
        logger.info(f"[Pairwise] Loaded {len(pairs)} valid pairs from {path}")
        return pairs

    all_pairs = _read_pairs(args.pairs_file) if args.pairs_file else []
    if not all_pairs:
        logger.warning("[Pairwise] No pairs available; skipping pairwise alignment computation.")
    else:
        if args.max_pairs and len(all_pairs) > args.max_pairs:
            if args.sample_seed is not None:
                rng = random.Random(args.sample_seed)
                rng.shuffle(all_pairs)
            all_pairs = all_pairs[: args.max_pairs]
            logger.info(f"[Pairwise] Truncated to max_pairs={args.max_pairs}")
        if args.demo_fast:
            all_pairs = all_pairs[: min(5, len(all_pairs))]
            logger.info(f"[Pairwise][DemoFast] Limiting pairs to {len(all_pairs)}")

        def _continuation_logprob(model, prompt_text: str, continuation_text: str):
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
            cont_ids = tokenizer(continuation_text, add_special_tokens=False).input_ids
            if args.truncate_pair_responses and len(cont_ids) > args.truncate_pair_responses:
                cont_ids = cont_ids[: args.truncate_pair_responses]
            full_ids = prompt_ids + cont_ids
            input_tensor = torch.tensor([full_ids], device=device)
            with torch.no_grad():
                logits = model(input_tensor).logits
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            cont_start = len(prompt_ids)
            token_logps = []
            for i, tok in enumerate(cont_ids):
                pos = cont_start + i - 1  # predicts next token
                if pos < 0 or pos >= log_probs.shape[1]:
                    continue
                token_logps.append(log_probs[0, pos, tok].item())
            return float(np.sum(token_logps)) if token_logps else float('nan'), len(cont_ids)

        pair_entries = []
        for i, pair in enumerate(all_pairs, start=1):
            pr = pair['prompt']
            chosen = pair['chosen']
            rejected = pair['rejected']
            ent = {"prompt": pr, "chosen_preview": chosen[:160], "rejected_preview": rejected[:160]}
            try:
                b_ch_lp, b_ch_len = _continuation_logprob(models['Base'], pr, chosen)
                b_rj_lp, b_rj_len = _continuation_logprob(models['Base'], pr, rejected)
                d_ch_lp, d_ch_len = _continuation_logprob(models['DPO'], pr, chosen)
                d_rj_lp, d_rj_len = _continuation_logprob(models['DPO'], pr, rejected)
                base_margin = b_ch_lp - b_rj_lp
                dpo_margin = d_ch_lp - d_rj_lp
                ent.update({
                    'base_chosen_logprob': b_ch_lp,
                    'base_rejected_logprob': b_rj_lp,
                    'dpo_chosen_logprob': d_ch_lp,
                    'dpo_rejected_logprob': d_rj_lp,
                    'base_margin': base_margin,
                    'dpo_margin': dpo_margin,
                    'margin_delta': dpo_margin - base_margin,
                    'chosen_len_tokens': b_ch_len,
                    'rejected_len_tokens': b_rj_len,
                })
            except Exception as e:
                ent['error'] = str(e)
                logger.warning(f"[Pairwise] Error processing pair {i}: {e}")
            pair_entries.append(ent)
            if i % 10 == 0:
                logger.info(f"[Pairwise] Processed {i}/{len(all_pairs)} pairs")

        valid = [e for e in pair_entries if 'base_margin' in e]
        if valid:
            base_margins = [e['base_margin'] for e in valid]
            dpo_margins = [e['dpo_margin'] for e in valid]
            deltas = [e['margin_delta'] for e in valid]
            def _stats(vals):
                arr = np.array(vals)
                return {
                    'mean': float(np.mean(arr)),
                    'median': float(np.median(arr)),
                    'p25': float(np.percentile(arr, 25)),
                    'p75': float(np.percentile(arr, 75)),
                }
            pairwise_alignment = {
                'num_pairs': len(pair_entries),
                'num_valid_pairs': len(valid),
                'base_margin_stats': _stats(base_margins),
                'dpo_margin_stats': _stats(dpo_margins),
                'margin_delta_stats': _stats(deltas),
                'base_accuracy_margin_positive': sum(1 for m in base_margins if m > 0) / len(base_margins),
                'dpo_accuracy_margin_positive': sum(1 for m in dpo_margins if m > 0) / len(dpo_margins),
                'win_rate_improved_margin': sum(1 for d in deltas if d > 0) / len(deltas),
            }

            if args.compute_kl:
                # Approximate KL by average difference in log probs of chosen tokens
                token_diffs = []
                for e in valid:
                    try:
                        pr_ids = tokenizer(e['prompt'], add_special_tokens=False).input_ids
                        ch_ids = tokenizer(e['chosen_preview'], add_special_tokens=False).input_ids
                        if args.truncate_pair_responses and len(ch_ids) > args.truncate_pair_responses:
                            ch_ids = ch_ids[: args.truncate_pair_responses]
                        full_ids = pr_ids + ch_ids
                        inp = torch.tensor([full_ids], device=device)
                        with torch.no_grad():
                            base_lp = torch.nn.functional.log_softmax(models['Base'](inp).logits, dim=-1)
                            dpo_lp = torch.nn.functional.log_softmax(models['DPO'](inp).logits, dim=-1)
                        cont_start = len(pr_ids)
                        for i_tok, tok in enumerate(ch_ids):
                            pos = cont_start + i_tok - 1
                            if pos < 0 or pos >= base_lp.shape[1]:
                                continue
                            token_diffs.append(base_lp[0, pos, tok].item() - dpo_lp[0, pos, tok].item())
                    except Exception as ex:
                        logger.warning(f"[Pairwise][KL] skip due to {ex}")
                pairwise_alignment['approx_mean_logprob_diff_chosen'] = float(np.mean(token_diffs)) if token_diffs else None
        else:
            pairwise_alignment = {'num_pairs': len(pair_entries), 'num_valid_pairs': 0}
total_dur = time.time() - stage_start
logger.info("==================== SUMMARY (Base vs DPO) ====================")
for r in results:
    logger.info(f"Prompt: {r['prompt'][:60]}...")
    base_resp = r["results"].get("Base", {}).get("response", "N/A")
    dpo_resp = r["results"].get("DPO", {}).get("response", "N/A")
    logger.info("  Base: " + base_resp[:100])
    logger.info("  DPO:  " + dpo_resp[:100])

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
    metrics = r["results"].get("intrinsic_metrics")
    if metrics:
        print(f"       ΔWords: {metrics['delta_len_words']} | Novelty: {metrics['novelty_ratio']:.4f} | Overlap: {metrics['overlap_ratio']:.4f}")
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
payload = {
    "total_duration_sec": total_dur,
    "comparisons": results,
    "gen_config": generation_kwargs,
    "aggregates": {},
    "pairwise_alignment": pairwise_alignment if pairwise_alignment is not None else None,
}

# Aggregated intrinsic metrics if available
intrinsic_all = [r["results"].get("intrinsic_metrics") for r in results if r["results"].get("intrinsic_metrics")]
if intrinsic_all:
    import statistics
    def avg(key):
        vals = [m[key] for m in intrinsic_all]
        return round(statistics.mean(vals), 4)
    payload["aggregates"] = {
        "mean_delta_len_words": avg("delta_len_words"),
        "mean_novelty_ratio": avg("novelty_ratio"),
        "mean_overlap_ratio": avg("overlap_ratio"),
    }
    print("Aggregate Intrinsic Metrics:")
    print(f"  Mean ΔWords: {payload['aggregates']['mean_delta_len_words']}")
    print(f"  Mean Novelty Ratio: {payload['aggregates']['mean_novelty_ratio']}")
    print(f"  Mean Overlap Ratio: {payload['aggregates']['mean_overlap_ratio']}")
with output_json.open("w") as f:
    json.dump(payload, f, indent=2)
logger.info(f"[Output] JSON written -> {output_json} (time={total_dur:.2f}s)")
print(f"Results JSON: {output_json} | Total time: {total_dur:.2f}s")