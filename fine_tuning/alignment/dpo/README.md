# DPO Fine-Tuning on Gaudi (Gaudi2 / Gaudi3) — Gemma 3 & Llama Support

This directory contains scripts and utilities for fine-tuning and reinforcement learning with language models using HuggingFace TRL, adapted for Habana Gaudi3 hardware.

## ⚠️ Important: Model Arguments Required

All scripts require explicit model specification. No default model is assumed.

- Single scripts: pass `--model_name_or_path <base-or-sft-model>`
- Pipeline script: `MODEL_NAME=<model> ./dpo_pipeline_sanity.sh`
- If arguments are missing the script aborts with a clear message.

## Contents

- `sft.py` — Supervised fine-tuning (optional pre-step before DPO; can start from a base model).
- `dpo.py` — Direct Preference Optimization training (pairwise preference data).
- `compare_base_vs_dpo.py` — Compare base and DPO-finetuned models via logprob margins & qualitative outputs.
- `merge_peft_adapter.py` — Merge PEFT adapters into base model for deployment.
- `dpo_pipeline_sanity.sh` — One-shot SFT (optional) + DPO + comparison.
- `Dockerfile` — Container setup for Gaudi training.
- `Makefile` — Build and run targets.

## End-to-End DPO Pipeline Sanity Check

## Running the DPO Pipeline

The `dpo_pipeline_sanity.sh` script demonstrates an ultra-fast workflow (<20 min target) on Gaudi: optional SFT warmup, DPO training on a tiny subset or synthetic pairs, and a qualitative comparison. All steps require explicit model specification.

**To execute:**

```bash
make build
make run HF_TOKEN=<<YOUR-HF-TOKEN>>
chmod +x dpo_pipeline_sanity.sh
MODEL_NAME=google/gemma-3-270m ./dpo_pipeline_sanity.sh
```

### Steps Overview

To achieve robust and production-quality fine-tuning of Llama models, the pipeline is split into key stages. Each step is essential for building a high-performing RLHF system:

All steps run sequentially, logging to `dpo_pipeline_sanity.log`.

1. **Supervised Fine-Tuning (SFT)**
    - Trains the base Llama model on curated human demonstration data, teaching it to follow instructions and generate useful responses. LoRA adapters and Habana optimizations accelerate and scale this process.
   - Example command (**model_name_or_path is required**):
     ```bash
     python sft.py \
       --model_name_or_path google/gemma-3-270m \
       --model_name_or_path google/gemma-3-270m \
       --dataset_name lvwerra/stack-exchange-paired \
       --output_dir ./sft_sanity \
       --do_train \
       --max_steps 50 \
       --per_device_train_batch_size 2 \
       --gradient_accumulation_steps 2 \
       --learning_rate 3e-5 \
       --lora_target_modules "q_proj" "k_proj" "v_proj" "o_proj" \
       --bf16 \
       --use_habana \
       --use_lazy_mode
     ```

2. **Merge SFT Adapters**
   - Integrates the learned LoRA weights into the base model, producing a single checkpoint for further   training and evaluation. This simplifies deployment and downstream usage.
   - Example command (**all arguments are required**):
     ```bash
     python merge_peft_adapter.py \
       --base_model_name "google/gemma-3-270m" \
       --base_model_name "google/gemma-3-270m" \
       --adapter_model_name "./sft_sanity" \
       --output_name "./sft_sanity_merged"
     ```

3. **DPO Training**
   - Directly optimizes the policy using preference pairs (chosen vs rejected) without a reward model, simplifying RLHF.
   - Example command:
     ```bash
     PT_HPU_LAZY_MODE=1 python dpo.py \
       --model_name_or_path ./sft_sanity_merged \
       --tokenizer_name_or_path google/gemma-3-270m \
       --output_dir ./dpo_sanity \
       --beta 0.1 \
       --max_steps 40 \
       --per_device_train_batch_size 2 \
       --gradient_accumulation_steps 2 \
       --logging_steps 10 \
       --save_steps 999999 \
       --eval_steps 999999 \
       --lora_target_modules "q_proj" "k_proj" "v_proj" "o_proj" \
       --seed 42
     ```

4. **Compare Base vs DPO**
   - Simple qualitative + logprob margin comparison between the base (or SFT) model and the DPO-finetuned model.
   - Example command:
     ```bash
     PT_HPU_LAZY_MODE=1 python compare_base_vs_dpo.py \
       --base_model google/gemma-3-270m \
       --finetuned_model ./dpo_sanity \
       --seed 123
     ```

### Logging

All output is logged to `ppo_pipeline_sanity.log` for review and debugging.



## Comparing Base and DPO Models

Use `compare_base_vs_dpo.py` for quick margins & qualitative outputs. Reward model arguments are unnecessary; DPO directly leverages preference pairs.

```bash
PT_HPU_LAZY_MODE=1 python compare_base_vs_dpo.py \
  --base_model google/gemma-3-270m \
  --finetuned_model ./dpo_sanity \
  --seed 123
```

This outputs logprob margins (chosen minus rejected) and selected generations to confirm improvement.

---

