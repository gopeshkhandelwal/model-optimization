# DPO Training on Gaudi (Gaudi2 / Gaudi3) — Gemma 3 & Llama Support

**Direct Preference Optimization (DPO)** implementation for Gaudi3 hardware with comprehensive Gemma3 and Llama model support. This directory contains all necessary scripts for DPO-based preference alignment training without requiring separate reward models.

## Key Features

- **No Default Models**: All scripts require explicit model specification for better control
- **Gemma3 Compatibility**: Full support for Gemma3 models with proper attention mechanisms
- **Gaudi3 Optimized**: Leverages Habana Gaudi3 hardware acceleration
- **Complete Pipeline**: End-to-end DPO workflow from SFT to comparison
- **Error handling**: Scripts will throw clear errors when required arguments are missing

## Contents

- `dpo.py` — Direct Preference Optimization training for preference alignment
- `compare_base_vs_dpo.py` — Compare base and DPO-trained models on response quality
- `dpo_pipeline_sanity.sh` — Complete DPO pipeline script
- `Dockerfile` — Container setup for Gaudi3 training
- `Makefile` — Common build and run targets
- `logging_utils.py` — Logging utilities for consistent output

## What is DPO?

Direct Preference Optimization (DPO) is a preference alignment technique that serves as an alternative to PPO-based RLHF. DPO:
- **Eliminates the need for a separate reward model** during training
- **Directly optimizes the policy** using preference data
- **Is more stable** and easier to tune than PPO
- **Requires less computational resources** than PPO+reward model approaches

DPO works by training the model to increase the likelihood of preferred responses while decreasing the likelihood of dispreferred responses, using a reference model to prevent over-optimization.

## End-to-End DPO Pipeline

The `dpo_pipeline_sanity.sh` script provides a complete demonstration of the DPO workflow on Gaudi3. **All scripts require explicit model specification - no defaults are used.**

**To execute:**

```bash
make build
make run HF_TOKEN=<<YOUR-HF-TOKEN>>
chmod +x dpo_pipeline_sanity.sh
MODEL_NAME=google/gemma-3-270m ./dpo_pipeline_sanity.sh
```

### Pipeline Steps

1. **Supervised Fine-Tuning (SFT)**
   - Initial training on instruction-following data
   - Prepares model for preference optimization
   - Output: `./sft_dpo_init_merged`

2. **DPO Training**
   - Direct preference optimization using preference pairs
   - No separate reward model needed
   - Uses Anthropic/hh-rlhf dataset
   - Output: `./dpo_sanity_merged`

3. **Model Comparison**
   - Compare base vs DPO-trained model responses
   - Qualitative evaluation of improvements
   - Output: `dpo_comparison_results.json`

4. **Generation Testing**
   - Test final model generation quality
   - Verify model works correctly

All steps run sequentially with output and logs saved to `dpo_pipeline_sanity.log`.

## Individual Script Usage

### DPO Training

```bash
PT_HPU_LAZY_MODE=1 python dpo.py \
    --model_name_or_path ./sft_model_merged \
    --tokenizer_name_or_path google/gemma-3-270m \
    --dataset_name="Anthropic/hh-rlhf" \
    --per_device_train_batch_size=1 \
    --gradient_accumulation_steps=4 \
    --learning_rate=5e-4 \
    --max_length=512 \
    --beta=0.1 \
    --num_train_epochs=1 \
    --output_dir=./dpo_output \
    --use_peft=True
```

**Key Parameters:**
- `--beta`: Controls strength of preference optimization (0.1-0.5 typical)
- `--loss_type`: DPO loss function ("sigmoid", "hinge", "ipo")
- `--max_length`: Maximum sequence length for training
- `--use_peft`: Enable LoRA for efficient training

### Model Comparison

```bash
PT_HPU_LAZY_MODE=1 python compare_base_vs_dpo.py \
    --base_model google/gemma-3-270m \
    --finetuned_model ./dpo_output_merged \
    --seed 123 \
    --greedy
```

## Docker Usage

**Build and run with Docker:**

```bash
# Build image
make build

# Run container
make run HF_TOKEN=your_token_here MODEL_NAME=google/gemma-3-270m

# Inside container, run pipeline
chmod +x dpo_pipeline_sanity.sh
MODEL_NAME=google/gemma-3-270m ./dpo_pipeline_sanity.sh
```

## Supported Models

- **Gemma 3**: `google/gemma-3-270m`, `google/gemma-3-2b`, `google/gemma-3-9b`, `google/gemma-3-27b`
- **Llama 2**: `meta-llama/Llama-2-7b-hf`, `meta-llama/Llama-2-13b-hf`
- **Llama 3**: `meta-llama/Llama-3-8B`, `meta-llama/Llama-3-70B`

## Hardware Requirements

- **Gaudi2 or Gaudi3** hardware
- **Memory**: 32GB+ for 7B models, 80GB+ for 13B+ models
- **Docker**: With Habana runtime support

## Key Differences from PPO

| Aspect | PPO | DPO |
|--------|-----|-----|
| **Reward Model** | Required during training | Not needed |
| **Training Stability** | Can be unstable | More stable |
| **Computational Cost** | Higher (policy + reward model) | Lower |
| **Hyperparameter Tuning** | More complex | Simpler |
| **Memory Usage** | Higher | Lower |

## Troubleshooting

### Common Issues

1. **"model_name_or_path is required"**
   - Ensure you specify `--model_name_or_path` explicitly
   - No default models are used

2. **CUDA/HPU out of memory**
   - Reduce `per_device_train_batch_size`
   - Increase `gradient_accumulation_steps`
   - Enable `gradient_checkpointing`

3. **Slow training**
   - Ensure `PT_HPU_LAZY_MODE=1` is set
   - Check HPU utilization with `hl-smi`

### Performance Tips

- **Use LoRA**: Enable `--use_peft=True` for memory efficiency
- **Batch Size**: Start with small batches and increase gradually
- **Beta Parameter**: Start with 0.1, adjust based on results
- **Sequence Length**: Use shortest reasonable length for your data

## Output Files

- `./dpo_output/`: Main DPO training output
- `./dpo_output_merged/`: Merged model ready for inference
- `dpo_comparison_results.json`: Comparison results
- `dpo_pipeline_sanity.log`: Complete pipeline logs
- `training_metrics.json`: Training performance metrics

## Advanced Configuration

### Custom Datasets

```bash
# Use custom preference dataset
python dpo.py \
    --dataset_name="your/dataset" \
    --dataset_config="default" \
    --model_name_or_path="your/model"
```

### Multi-GPU Training

```bash
# Enable data parallel training
python dpo.py \
    --model_name_or_path="your/model" \
    --gradient_accumulation_steps=1 \
    --per_device_train_batch_size=4
```

## Results Analysis

The pipeline generates comparison results showing qualitative differences between base and DPO-trained models. Look for:

- **Improved helpfulness** in responses
- **Better alignment** with human preferences  
- **Reduced harmful** or biased outputs
- **More coherent** and relevant answers

DPO typically produces models that are better aligned with human preferences while being simpler to train than PPO-based approaches.