#!/bin/bash
# dpo_pipeline_sanity.sh
# DPO (Direct Preference Optimization) Pipeline for Gaudi3

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}"
}

print_success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] ✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] ⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ✗ $1${NC}"
}

# Set model name from environment variable or command line argument
if [ $# -gt 0 ]; then
    MODEL_NAME="$1"
elif [ -n "${MODEL_NAME:-}" ]; then
    MODEL_NAME="${MODEL_NAME}"
else
    print_error "MODEL_NAME is required. Set it as environment variable or pass as first argument."
    print_error "Usage: MODEL_NAME=google/gemma-3-270m $0"
    print_error "   or: $0 google/gemma-3-270m"
    exit 1
fi

FAST_MODE=${FAST_MODE:-0}
for arg in "$@"; do
    if [[ "$arg" == "--fast" || "$arg" == "--demo_fast" ]]; then FAST_MODE=1; fi
done
print_status "Starting DPO Pipeline with MODEL_NAME=${MODEL_NAME} FAST_MODE=${FAST_MODE}"
PIPELINE_START_TS=$(date +%s)
STAGE_TIMES=()

# Log file
LOG_FILE="dpo_pipeline_sanity.log"
exec > >(tee -a "$LOG_FILE") 2>&1

print_status "Logging to: $LOG_FILE"
print_status "Model: $MODEL_NAME"

# Check if we're in the right directory
if [ ! -f "dpo.py" ]; then
    print_error "dpo.py not found. Please run this script from the alignment/dpo directory."
    exit 1
fi

# Verify HPU environment
if ! command -v hl-smi &> /dev/null; then
    print_warning "hl-smi not found. HPU environment may not be properly set up."
fi

print_status "=== Step 1: Supervised Fine-Tuning (SFT) ==="
print_status "Training base model with supervised data for DPO initialization"
SFT_START=$(date +%s)

PT_HPU_LAZY_MODE=1 python sft.py \
    --model_name_or_path "${MODEL_NAME}" \
    --dataset_name="Anthropic/hh-rlhf" \
    --dataset_config="default" \
    --split="test" \
    --max_seq_length=512 \
    --per_device_train_batch_size=2 \
    --gradient_accumulation_steps=4 \
    --learning_rate=1e-4 \
    --logging_steps=5 \
    --num_train_epochs=1 \
    --max_steps=100 \
    --output_dir=./sft_dpo_init \
    --save_strategy="epoch" \
    --optim="adamw_torch" \
    --warmup_steps=10 \
    --bf16=True \
    --remove_unused_columns=False \
    --max_train_samples=500 \
    $( [[ "$FAST_MODE" == "1" ]] && echo "--demo_fast" )

print_success "SFT training completed"
SFT_END=$(date +%s); STAGE_TIMES+=("SFT:$((SFT_END-SFT_START))s")

# Merge PEFT adapter
print_status "=== Step 1b: Merging PEFT Adapter ==="
MERGE_SFT_START=$(date +%s)
python merge_peft_adapter.py \
    --adapter_model_name=./sft_dpo_init \
    --base_model_name="${MODEL_NAME}" \
    --output_name=./sft_dpo_init_merged

print_success "PEFT adapter merged"
MERGE_SFT_END=$(date +%s); STAGE_TIMES+=("MergeSFT:$((MERGE_SFT_END-MERGE_SFT_START))s")

print_status "=== Step 2: DPO Training ==="
print_status "Training with preference data using Direct Preference Optimization"
DPO_START=$(date +%s)

# Auto-detect available splits for the preference dataset to avoid ExpectedMoreSplits errors
print_status "Detecting available splits for lvwerra/stack-exchange-paired"
AVAILABLE_SPLITS=$(python - <<'PY'
from datasets import load_dataset
try:
    ds_dict = load_dataset("lvwerra/stack-exchange-paired")
    print(" ".join(ds_dict.keys()))
except Exception as e:
    print(f"ERROR:{e}")
PY
)
if [[ "$AVAILABLE_SPLITS" == ERROR:* ]]; then
    print_warning "Could not enumerate dataset splits ($AVAILABLE_SPLITS). Proceeding with fallback logic in dpo.py"
    PREF_SPLIT="train"
else
    print_status "Available splits: $AVAILABLE_SPLITS"
    PREF_SPLIT="train"
    for cand in train training train_full train_augmented; do
        if echo "$AVAILABLE_SPLITS" | grep -qw "$cand"; then PREF_SPLIT="$cand"; break; fi
    done
    if [[ "$PREF_SPLIT" == "train" ]]; then
        # If no explicit train-like split, choose test or first
        if echo "$AVAILABLE_SPLITS" | grep -qw test; then
            PREF_SPLIT="test"
        else
            PREF_SPLIT=$(echo "$AVAILABLE_SPLITS" | awk '{print $1}')
        fi
    fi
fi
print_status "Using preference training split: $PREF_SPLIT"

PT_HPU_LAZY_MODE=1 python dpo.py \
    --model_name_or_path ./sft_dpo_init_merged \
    --tokenizer_name_or_path "${MODEL_NAME}" \
    --dataset_name="lvwerra/stack-exchange-paired" \
    --split="${PREF_SPLIT}" \
    --per_device_train_batch_size=1 \
    --per_device_eval_batch_size=1 \
    --gradient_accumulation_steps=4 \
    --learning_rate=5e-4 \
    --lr_scheduler_type="cosine" \
    --warmup_steps=50 \
    --max_length=512 \
    --max_prompt_length=256 \
    --max_target_length=256 \
    --beta=0.1 \
    --num_train_epochs=1 \
    --max_steps=100 \
    --logging_steps=10 \
    --save_steps=50 \
    --eval_steps=50 \
    --eval_fraction=0.02 \
    --datasets_verification_mode=no_checks \
    --output_dir=./dpo_sanity \
    --gradient_checkpointing=True \
    --use_peft=True \
    --lora_r=64 \
    --lora_alpha=16 \
    --max_train_samples=200 \
    $( [[ "$FAST_MODE" == "1" ]] && echo "--demo_fast" )

print_success "DPO training completed"
DPO_END=$(date +%s); STAGE_TIMES+=("DPO:$((DPO_END-DPO_START))s")

# Merge DPO PEFT adapter
print_status "=== Step 2b: Merging DPO PEFT Adapter ==="
MERGE_DPO_START=$(date +%s)
python merge_peft_adapter.py \
    --adapter_model_name=./dpo_sanity \
    --base_model_name=./sft_dpo_init_merged \
    --output_name=./dpo_sanity_merged

print_success "DPO PEFT adapter merged"
MERGE_DPO_END=$(date +%s); STAGE_TIMES+=("MergeDPO:$((MERGE_DPO_END-MERGE_DPO_START))s")

print_status "=== Step 3: Model Comparison ==="
print_status "Comparing base model vs DPO-trained model"
COMPARE_START=$(date +%s)

PT_HPU_LAZY_MODE=1 python compare_base_vs_dpo.py \
    --base_model "${MODEL_NAME}" \
    --finetuned_model ./dpo_sanity_merged \
    --seed 123 \
    --greedy \
    --output_json dpo_comparison_results.json \
    $( [[ "$FAST_MODE" == "1" ]] && echo "--demo_fast" )

print_success "Model comparison completed"
COMPARE_END=$(date +%s); STAGE_TIMES+=("Compare:$((COMPARE_END-COMPARE_START))s")


print_status "=== Pipeline Summary ==="
print_success "✓ SFT training: ./sft_dpo_init_merged"
print_success "✓ DPO training: ./dpo_sanity_merged"  
print_success "✓ Comparison results: dpo_comparison_results.json"
print_success "✓ All logs saved to: $LOG_FILE"

print_status "DPO Pipeline completed successfully!"
print_status "You can now use the DPO-trained model at: ./dpo_sanity_merged"

# Timing summary
PIPELINE_END_TS=$(date +%s)
TOTAL_PIPELINE=$((PIPELINE_END_TS-PIPELINE_START_TS))
print_status "=== Timing Summary (seconds) ==="
for entry in "${STAGE_TIMES[@]}"; do
    print_status "  $entry"
done
print_status "  Total:${TOTAL_PIPELINE}s"

# Display some stats
if [ -f "dpo_comparison_results.json" ]; then
    print_status "=== Quick Results Preview ==="
    python3 -c "
import json
try:
    with open('dpo_comparison_results.json', 'r') as f:
        data = json.load(f)
    comparisons = data.get('comparisons', [])
    if comparisons:
        print(f'Compared {len(comparisons)} prompts')
        total_time = data.get('total_duration_sec', 0)
        print(f'Total comparison time: {total_time:.2f}s')
        
        # Show first comparison as example
        if len(comparisons) > 0:
            first = comparisons[0]
            prompt = first['prompt'][:60] + '...' if len(first['prompt']) > 60 else first['prompt']
            print(f'Example prompt: {prompt}')
            
            base_resp = first['results']['Base']['response'][:80] + '...' if len(first['results']['Base']['response']) > 80 else first['results']['Base']['response']
            dpo_resp = first['results']['DPO']['response'][:80] + '...' if len(first['results']['DPO']['response']) > 80 else first['results']['DPO']['response']
            
            print(f'Base response: {base_resp}')
            print(f'DPO response:  {dpo_resp}')
except Exception as e:
    print(f'Could not parse results: {e}')
"
fi

print_success "DPO pipeline execution completed! 🎉"