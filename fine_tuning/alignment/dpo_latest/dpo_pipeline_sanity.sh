#!/usr/bin/env bash
set -euo pipefail

START_PIPELINE=$(date +%s)

# Fast DPO pipeline (override via env if desired)
DPO_MAX_STEPS=${DPO_MAX_STEPS:-140}
DPO_LOGGING_STEPS=${DPO_LOGGING_STEPS:-10}
DPO_LORA_TARGET=${DPO_LORA_TARGET:-"q_proj k_proj v_proj o_proj gate_proj up_proj down_proj"}

# Base model (now mandatory: pass as first arg OR export MODEL_NAME)
MODEL_NAME="${1:-${MODEL_NAME:-}}"
if [ -z "${MODEL_NAME}" ]; then
  echo "ERROR: Base model not specified. Provide as first argument or export MODEL_NAME."
  echo "Usage: MODEL_NAME=google/gemma-3-270m ./dpo_pipeline_sanity.sh" >&2
  echo "   or: ./dpo_pipeline_sanity.sh google/gemma-3-270m" >&2
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

LOGFILE="dpo_pipeline_sanity.log"
exec > >(tee -a "$LOGFILE") 2>&1

banner() {
  echo
  echo "============================================================"
  echo ">>> $1"
  echo "============================================================"
}

# Base model (now mandatory: pass as first arg OR export MODEL_NAME)
MODEL_NAME="${1:-${MODEL_NAME:-}}"
if [ -z "${MODEL_NAME}" ]; then
  echo "ERROR: Base model not specified. Provide as first argument or export MODEL_NAME."
  echo "Usage: MODEL_NAME=google/gemma-3-270m ./ppo_pipeline_sanity.sh" >&2
  echo "   or: ./ppo_pipeline_sanity.sh google/gemma-3-270m" >&2
  exit 1
fi
echo "[CONFIG] MODEL_NAME=${MODEL_NAME}"

# Common HPU env vars
export PT_HPU_DISABLE_FUSED_ADAMW=1
export PT_HPU_DISABLE_FUSED_ADAM=1
export PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES=0

if [ "${SKIP_SFT:-0}" = "1" ]; then
  banner "STEP 1: Supervised Fine-Tuning SKIPPED (using base model directly)"
  # Create a symlink named sft_sanity_merged pointing to the base model directory (if local) or just reuse MODEL_NAME if remote
  if [ -d "${MODEL_NAME}" ]; then
    ln -sfn "${MODEL_NAME}" ./sft_sanity_merged
  else
    # Remote model: create marker directory referencing base
    mkdir -p ./sft_sanity_merged
    echo "REMOTE_MODEL=${MODEL_NAME}" > ./sft_sanity_merged/BASE_MODEL_REFERENCE
  fi
else
  banner "STEP 1: Supervised Fine-Tuning (Ultra-Fast)"
  time_step_begin SFT
  python sft.py \
    --model_name_or_path "${MODEL_NAME}" \
    --dataset_name openbmb/UltraFeedback \
    --subset None \
    --streaming False \
    --size_valid_set 200 \
    --output_dir ./sft_sanity \
    --do_train \
    --max_steps 20 \
    --logging_steps 5 \
    --save_steps 999999 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --learning_rate 3e-5 \
    --lr_scheduler_type cosine \
    --warmup_steps 2 \
    --weight_decay 0.01 \
  --lora_target_modules "q_proj" "k_proj" "v_proj" "o_proj" "gate_proj" "up_proj" "down_proj" \
    --bf16 \
    --remove_unused_columns False \
    --report_to none \
    --use_habana \
    --use_lazy_mode \
    --demo_fast \
    --merge_adapter_after_train \
    --merged_output_dir ./sft_sanity_merged
  time_step_end SFT
fi

banner "STEP 2: DPO Training"
time_step_begin DPO
PT_HPU_LAZY_MODE=1 python dpo.py \
  --model_name_or_path ./sft_sanity_merged \
  --tokenizer_name_or_path "${MODEL_NAME}" \
  --output_dir ./dpo_sanity \
  --beta 0.05 \
  --max_steps ${DPO_MAX_STEPS} \
  --learning_rate 2e-5 \
  --warmup_steps 10 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 1 \
  --logging_steps ${DPO_LOGGING_STEPS} \
  --save_steps 999999 \
  --eval_steps 999999 \
  --lora_target_modules ${DPO_LORA_TARGET} \
  --report_to none \
  --seed 42 \
  --merge_adapter_after_train True \
  --merged_output_dir ./dpo_sanity_merged
time_step_end DPO

banner "STEP 3: Comparing Base and DPO Models"
time_step_begin COMPARE
PT_HPU_LAZY_MODE=1 python compare_base_vs_dpo.py \
  --base_model "${MODEL_NAME}" \
  --finetuned_model $( [ -d ./dpo_sanity_merged ] && echo ./dpo_sanity_merged || echo ./dpo_sanity ) \
  --seed 123 \
  --max_new_tokens 128 \
  --min_new_tokens 60 \
  --stackexchange_style \
  --heuristics \
  --show_text
time_step_end COMPARE

PIPELINE_END=$(date +%s)
TOTAL_DUR=$(( PIPELINE_END - START_PIPELINE ))

echo
echo "=================== PIPELINE TIMING SUMMARY ==================="
printf "%-20s %10s\n" "Stage" "Seconds"
printf "%-20s %10s\n" "-----" "-------"
for k in SFT DPO COMPARE; do
  printf "%-20s %10s\n" "$k" "${STEP_DURATION[$k]:-n/a}"
done
printf "%-20s %10s\n" "TOTAL" "$TOTAL_DUR"
echo "==============================================================="

echo
echo "=================== ARTIFACT SUMMARY ==================="
if [ -d ./sft_sanity_merged ]; then echo "SFT merged policy: ./sft_sanity_merged"; fi
if [ -d ./dpo_sanity ]; then echo "DPO adapter policy: ./dpo_sanity"; fi
echo "========================================================"

banner "✅ DPO PIPELINE COMPLETED SUCCESSFULLY (WITH COMPARISON)"
