#!/usr/bin/env bash
set -euo pipefail

START_PIPELINE=$(date +%s)
declare -A STEP_START
declare -A STEP_DURATION

time_step_begin() {
  local key="$1"; STEP_START[$key]=$(date +%s)
  echo "[TIMER] BEGIN $key at $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
}

time_step_end() {
  local key="$1"; local end=$(date +%s); local dur=$(( end - STEP_START[$key] ))
  STEP_DURATION[$key]=$dur
  echo "[TIMER] END   $key at $(date -u '+%Y-%m-%dT%H:%M:%SZ') (duration=${dur}s)"
}

LOGFILE="ppo_pipeline_sanity.log"
exec > >(tee -a "$LOGFILE") 2>&1

banner() {
  echo
  echo "============================================================"
  echo ">>> $1"
  echo "============================================================"
}

# Common HPU env vars
export PT_HPU_DISABLE_FUSED_ADAMW=1
export PT_HPU_DISABLE_FUSED_ADAM=1
export PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES=0

# 1. Supervised Fine-tuning
banner "STEP 1: Supervised Fine-Tuning (SFT)"
time_step_begin SFT
python sft.py \
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --dataset_name lvwerra/stack-exchange-paired \
  --output_dir ./sft_sanity \
  --do_train \
  --max_steps 50 \
  --logging_steps 10 \
  --save_steps 999999 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 2 \
  --learning_rate 3e-5 \
  --lr_scheduler_type cosine \
  --warmup_steps 5 \
  --weight_decay 0.01 \
  --lora_target_modules "q_proj" "k_proj" "v_proj" "o_proj" \
  --bf16 \
  --remove_unused_columns False \
  --report_to none \
  --use_habana \
  --use_lazy_mode
time_step_end SFT

# 2. Merge SFT adapters
banner "STEP 2: Merge SFT adapters"
time_step_begin MERGE_SFT
python merge_peft_adapter.py \
  --base_model_name "meta-llama/Llama-2-7b-hf" \
  --adapter_model_name "./sft_sanity" \
  --output_name "./sft_sanity_merged"
time_step_end MERGE_SFT

# 3. Reward Modeling
banner "STEP 3: Reward Modeling"
time_step_begin RM
python reward_modeling.py \
  --model_name_or_path ./sft_sanity_merged \
  --tokenizer_name_or_path meta-llama/Llama-2-7b-hf \
  --output_dir ./rm_sanity \
  --optim adamw_torch \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 2 \
  --num_train_epochs 1 \
  --train_subset 500 \
  --eval_subset 100 \
  --max_length 384 \
  --logging_steps 20 \
  --save_steps 999999 \
  --bf16
time_step_end RM

# 4. Merge Reward Model adapters
banner "STEP 4: Merge Reward Model adapters"
time_step_begin MERGE_RM
python merge_peft_adapter.py \
  --base_model_name "meta-llama/Llama-2-7b-hf" \
  --adapter_model_name "./rm_sanity" \
  --output_name "./rm_sanity_merged"
time_step_end MERGE_RM

# 5. PPO Training
banner "STEP 5: PPO Training"
time_step_begin PPO
PT_HPU_LAZY_MODE=1 python ppo.py \
  --model_name_or_path ./sft_sanity_merged \
  --reward_model_name ./rm_sanity_merged \
  --tokenizer_name_or_path meta-llama/Llama-2-7b-hf \
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
time_step_end PPO

# 6. Run Generation (quick sanity check)
banner "STEP 6: Run Generation"
time_step_begin GEN
python run_generation.py \
  --model_name_or_path ./ppo_sanity \
  --prompt "What is the currency of the USA?" \
  --bf16 \
  --use_kv_cache \
  --max_new_tokens 64 \
  --batch_size 1
time_step_end GEN

PIPELINE_END=$(date +%s)
TOTAL_DUR=$(( PIPELINE_END - START_PIPELINE ))

echo
echo "=================== PIPELINE TIMING SUMMARY ==================="
printf "%-20s %10s\n" "Stage" "Seconds"
printf "%-20s %10s\n" "-----" "-------"
for k in SFT MERGE_SFT RM MERGE_RM PPO GEN; do
  printf "%-20s %10s\n" "$k" "${STEP_DURATION[$k]:-n/a}"
done
printf "%-20s %10s\n" "TOTAL" "$TOTAL_DUR"
echo "==============================================================="

banner "✅ PIPELINE COMPLETED SUCCESSFULLY (WITH COMPARISON)"
