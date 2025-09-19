# Fine Tuning

This folder contains scripts and resources for fine-tuning machine learning models.

## Files

- `Dockerfile`: Container setup for running fine-tuning and evaluation scripts.
- `finetune.py`: Script to fine-tune a model on custom data.
- `prepare_data.py`: Prepares and processes data for fine-tuning.
- `eval_finetuned_model.py`: Evaluates the performance of the fine-tuned model.


## Step-by-Step Usage (Docker & Makefile)

All steps are run inside a Docker container with Habana (Gaudi3) support. Use the provided `Makefile` for automation.

### 1. Build Docker Image

```bash
make build
```

### 2. Run Docker Container (with Habana support)

```bash
make run HF_TOKEN=your_hf_token
```

This will start an interactive shell in the container.


### 3. (Option A) GSM8K Workflow Scripts

Inside the container, execute:

```bash
python /workspace/prepare_data.py
python /workspace/finetune.py  # or finetune_trl_latest.py
python /workspace/eval_finetuned_model.py
```

### 3. (Option B) Synthetic Math PPO Demo (Recommended for quick visible gains)

Generate arithmetic dataset (default 5k train / 1k eval):

```bash
python /workspace/prepare_math_data.py MATH_TRAIN_N=5000 MATH_VAL_N=1000
```

Run PPO LoRA fine-tuning on math data:

```bash
PPO_STEPS=800 LR=1e-4 MAX_NEW_TOKENS=6 DETERMINISTIC_GEN=1 EVAL_EVERY_STEPS=100 \
python /workspace/finetune_trl_latest.py
```

After first 200 steps you can re-run with DETERMINISTIC_GEN=0 (or just leave enabled for determinism). Evaluate:

```bash
python /workspace/eval_finetuned_model.py MAX_EVAL_EXAMPLES=200
```

Expected: format_accuracy quickly -> ~1.0; exact_match rising toward >0.9 within ~800 steps.

## Makefile Targets

- `make build`   : Build the Docker image
- `make run`     : Run the container with Habana runtime


## Prerequisites

- **Gaudi3 hardware** (required for acceleration)
- **Optimum Habana** library (used for finetuning)

## Requirements

- Python 3.x
- Required packages (see scripts for details)
- Docker (with Habana runtime support)

## Notes

- Customize the scripts as needed for your specific model and dataset.
- Synthetic math dataset enables a fast, deterministic demonstration of PPO improvement on Gaudi3.
- Refer to comments in each script for further instructions.
