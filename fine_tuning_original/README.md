
# Fine Tuning

This folder contains scripts and resources for fine-tuning machine learning models.

## Files

- `Dockerfile`: Container setup for running fine-tuning and evaluation scripts.
- `finetune.py`: Script to fine-tune a model on custom data.
- `prepare_data.py`: Prepares and processes data for fine-tuning.
- `eval_finetuned_model.py`: Evaluates the performance of the fine-tuned model.

## Prerequisites

- **Gaudi3 hardware** (required for acceleration)
- **Optimum Habana** library (used for finetuning)
- **Access to the required Large Language Model (LLM)** (e.g., Meta-Llama-3-8B). Ensure your Hugging Face account or organization has permission to download and use the model for fine-tuning and evaluation.

## Requirements

- Python 3.x
- Required packages (see scripts for details)
- Docker (with Habana runtime support)

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

### 3. Run Workflow Scripts

Inside the container, execute:

```bash
python /workspace/prepare_data.py
python /workspace/finetune.py
python /workspace/eval_finetuned_model.py
```

## Makefile Targets

- `make build`   : Build the Docker image
- `make run`     : Run the container with Habana runtime

## Notes

- Customize the scripts as needed for your specific model and dataset.
- Refer to comments in each script for further instructions.
