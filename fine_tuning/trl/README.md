# TRL Fine-Tuning & RLHF on Gaudi3

This directory contains scripts and utilities for fine-tuning and reinforcement learning with language models using HuggingFace TRL, adapted for Habana Gaudi3 hardware.

## Contents

- `sft.py` — Supervised fine-tuning for LLMs (Llama2, etc.) on custom datasets.
- `ppo.py` — Proximal Policy Optimization (PPO) for RLHF.
- `compare_base_vs_ppo.py` — Compare base and PPO-finetuned models on reward scores.
- `merge_peft_adapter.py` — Merge PEFT adapters into base models.
- `reward_modeling.py` — Train reward models for RLHF.
- `run_generation.py` — Script for batch text generation.
- `ppo_pipeline_sanity.sh` — Example shell script for PPO pipeline sanity check.
- `Dockerfile` — Container setup for Gaudi3 training.
- `Makefile` — Common build and run targets.

## End-to-End PPO Pipeline Sanity Check

## Running the End-to-End PPO Pipeline

The `ppo_pipeline_sanity.sh` script provides a complete demonstration of the RLHF workflow on Gaudi3, from supervised fine-tuning to PPO and generation sanity check. Make sure all Python scripts in this directory are present and dependencies are installed (see Dockerfile or requirements).

**To execute:**

```bash
make build
make run HF_TOKEN=<<YOUR-HF-TOKEN>>
cd trl
chmod +x ppo_pipeline_sanity.sh
./ppo_pipeline_sanity.sh
```

### Steps Overview

To achieve robust and production-quality fine-tuning of Llama models, the pipeline is split into key stages. Each step is essential for building a high-performing RLHF system:

All steps will run sequentially, with output and logs saved to `ppo_pipeline_sanity.log` for review.

1. **Supervised Fine-Tuning (SFT)**
  - Trains the base Gemma (small 270M for quick iteration) model on curated human demonstration data, teaching it to follow instructions and generate useful responses. LoRA adapters and Habana optimizations accelerate and scale this process.
   - Example command:
     ```bash
     python sft.py \
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
   - Example command:
     ```bash
     python merge_peft_adapter.py \
       --base_model_name "google/gemma-3-270m" \
       --adapter_model_name "./sft_sanity" \
       --output_name "./sft_sanity_merged"
     ```

3. **Reward Modeling**
   - Trains a reward model to score outputs based on human preferences or synthetic feedback. This model is critical for RLHF, as it guides the PPO optimization toward more helpful and aligned responses.
   - Example command:
     ```bash
     python reward_modeling.py \
       --model_name_or_path ./sft_sanity_merged \
       --tokenizer_name_or_path google/gemma-3-270m \
       --output_dir ./rm_sanity \
       --optim adamw_torch \
       --per_device_train_batch_size 2 \
       --gradient_accumulation_steps 2 \
       --num_train_epochs 1 \
       --train_subset 500 \
       --eval_subset 100 \
       --max_length 384 \
       --bf16
     ```

4. **Merge Reward Model Adapters**
   - Merges reward model adapters into the base model, ensuring the reward model is ready for PPO training and evaluation.
   - Example command:
     ```bash
     python merge_peft_adapter.py \
       --base_model_name "google/gemma-3-270m" \
       --adapter_model_name "./rm_sanity" \
       --output_name "./rm_sanity_merged"
     ```

5. **PPO Training**
   - Runs PPO RLHF training using the merged SFT and reward models.
   - Example command:
     ```bash
     PT_HPU_LAZY_MODE=1 python ppo.py \
       --model_name_or_path ./sft_sanity_merged \
       --reward_model_name ./rm_sanity_merged \
       --tokenizer_name_or_path google/gemma-3-270m \
       --output_dir ./ppo_sanity \
       --batch_size 2 \
       --mini_batch_size 1 \
       --gradient_accumulation_steps 1 \
       --ppo_epochs 1 \
       --steps 64 \
       --input_max_length 256 \
       --output_max_length 64 \
       --learning_rate 1.4e-5 \
       --early_stopping True \
       --batched_gen True \
       --max_train_samples 256
     ```

6. **Run Generation (Sanity Check)**
   - Uses Proximal Policy Optimization (PPO) to further refine the Llama model, leveraging the reward model to optimize for helpfulness, safety, and alignment. This step produces the final RLHF-tuned model.
   - Example command:
     ```bash
     python run_generation.py \
       --model_name_or_path ./ppo_sanity \
       --prompt "What is the currency of the USA?" \
       --bf16 \
       --use_kv_cache \
       --max_new_tokens 64 \
       --batch_size 1
     ```

### Logging

All output is logged to `ppo_pipeline_sanity.log` for review and debugging.



## Comparing Base and PPO Models

After completing the pipeline, you can quantitatively compare the base model and PPO-finetuned model using the provided script:

```bash
PT_HPU_LAZY_MODE=1 python compare_base_vs_ppo.py \
  --base_model google/gemma-3-270m \
  --finetuned_model ./ppo_sanity \
  --reward_model ./rm_sanity_merged \
  --dataset_name lvwerra/stack-exchange-paired \
  --dataset_split train_prefs \
  --num_samples 5
```

This will output reward scores for both models on a sample of prompts, helping you assess the impact of PPO fine-tuning.
---
