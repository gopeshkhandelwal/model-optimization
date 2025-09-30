#!/usr/bin/env bash
set -euo pipefail

START_PIPELINE=$(date +%s)

# Config toggles (override via environment if desired)
PPO_MERGE_POLICY=${PPO_MERGE_POLICY:-1}           # 1 = merge PPO adapter after training
POLICY_MERGED_DIR=${POLICY_MERGED_DIR:-./ppo_sanity_merged}
REWARD_MERGED_DIR=./rm_sanity_merged
REWARD_ADAPTER_DIR=./rm_sanity

# Base model (now mandatory: pass as first arg OR export MODEL_NAME)
MODEL_NAME="${1:-${MODEL_NAME:-}}"
if [ -z "${MODEL_NAME}" ]; then
  echo "ERROR: Base model not specified. Provide as first argument or export MODEL_NAME."
  echo "Usage: MODEL_NAME=google/gemma-3-270m ./ppo_pipeline_sanity.sh" >&2
  echo "   or: ./ppo_pipeline_sanity.sh google/gemma-3-270m" >&2
  exit 1
fi
echo "[CONFIG] MODEL_NAME=${MODEL_NAME}"
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

banner "STEP 1: Supervised Fine-Tuning (SFT + Merge Inline)"
time_step_begin SFT
python sft.py \
  --model_name_or_path "${MODEL_NAME}" \
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
  --use_lazy_mode \
  --merge_adapter_after_train \
  --merged_output_dir ./sft_sanity_merged
time_step_end SFT

banner "STEP 2: Reward Modeling (Inline Merge)"
time_step_begin RM
python reward_modeling.py \
  --model_name_or_path ./sft_sanity_merged \
  --tokenizer_name_or_path "${MODEL_NAME}" \
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
  --bf16 \
  --merge_adapter_after_train \
  --merged_output_dir ./rm_sanity_merged
# NOTE: For Gemma 3 tiny model the reward script may fall back to a causal LM wrapper and skip merge.
# In that case only ./rm_sanity (adapter + value head) will exist (no rm_sanity_merged). The PPO loader
# can now consume an adapter-only dir directly, so rm_sanity_merged is optional.
time_step_end RM

# Decide which reward model directory to use (prefer merged if present)
if [ -d "${REWARD_MERGED_DIR}" ]; then
  REWARD_FOR_PPO="${REWARD_MERGED_DIR}"
  echo "[SELECT] Using merged reward model: ${REWARD_FOR_PPO}"
else
  REWARD_FOR_PPO="${REWARD_ADAPTER_DIR}"
  echo "[SELECT] Merged reward model not found; using adapter directory: ${REWARD_FOR_PPO}"
fi

banner "STEP 3: PPO Training"
time_step_begin PPO
PT_HPU_LAZY_MODE=1 python ppo.py \
  --model_name_or_path ./sft_sanity_merged \
  --reward_model_name "${REWARD_FOR_PPO}" \
  --tokenizer_name_or_path "${MODEL_NAME}" \
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
  --max_train_samples 256 \
  $( [ "${PPO_MERGE_POLICY}" = "1" ] && echo "--merge_adapter_after_train --merged_output_dir ${POLICY_MERGED_DIR}" )
time_step_end PPO

banner "STEP 4: Comparing Base and PPO Models"
time_step_begin COMPARE
PT_HPU_LAZY_MODE=1 python compare_base_vs_ppo.py \
  --base_model "${MODEL_NAME}" \
  --finetuned_model ./ppo_sanity \
  --reward_model "${REWARD_FOR_PPO}" \
  --seed 123 \
  --greedy
time_step_end COMPARE

PIPELINE_END=$(date +%s)
TOTAL_DUR=$(( PIPELINE_END - START_PIPELINE ))

echo
echo "=================== PIPELINE TIMING SUMMARY ==================="
printf "%-20s %10s\n" "Stage" "Seconds"
printf "%-20s %10s\n" "-----" "-------"
for k in SFT RM PPO COMPARE; do
  printf "%-20s %10s\n" "$k" "${STEP_DURATION[$k]:-n/a}"
done
printf "%-20s %10s\n" "TOTAL" "$TOTAL_DUR"
echo "==============================================================="

echo
echo "=================== ARTIFACT SUMMARY ==================="
if [ -d ./sft_sanity_merged ]; then echo "SFT merged policy: ./sft_sanity_merged"; fi
if [ -d ./rm_sanity ]; then echo "Reward adapter: ./rm_sanity"; fi
if [ -d ./rm_sanity_merged ]; then echo "Reward merged: ./rm_sanity_merged"; fi
if [ -d ./ppo_sanity ]; then echo "PPO adapter policy: ./ppo_sanity"; fi
if [ -d "${POLICY_MERGED_DIR}" ]; then echo "PPO merged policy: ${POLICY_MERGED_DIR}"; fi
echo "Reward used for PPO: ${REWARD_FOR_PPO}"
echo "Policy merge enabled: ${PPO_MERGE_POLICY}"
echo "========================================================"

banner "✅ PIPELINE COMPLETED SUCCESSFULLY (WITH COMPARISON)"
