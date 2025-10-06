# Preference Alignment Techniques

This directory contains implementations of various preference alignment techniques for language model fine-tuning.

## Available Methods

### DPO (Direct Preference Optimization)
- **Location**: `dpo/`
- **Description**: Direct preference optimization without requiring separate reward models
- **Use case**: Stable, efficient preference alignment with human feedback data
- **Key advantages**: 
  - Simpler than PPO-based RLHF
  - More computationally efficient
  - Better stability during training

## Directory Structure

```
alignment/
├── README.md           # This file
└── dpo/               # Direct Preference Optimization implementation
    ├── dpo.py         # Main DPO training script
    ├── sft.py         # Supervised fine-tuning for DPO initialization
    ├── compare_base_vs_dpo.py  # Model comparison utilities
    ├── dpo_pipeline_sanity.sh  # End-to-end pipeline
    ├── Dockerfile     # Container setup
    ├── Makefile       # Build system
    └── README.md      # DPO-specific documentation
```

## Comparison with RLHF

Traditional RLHF (in `../rlhf/`) uses reinforcement learning with separate reward models, while alignment techniques here focus on direct optimization approaches that are often more efficient and stable.

| Aspect | RLHF (PPO) | Alignment (DPO) |
|--------|------------|-----------------|
| Reward Model | Required | Not needed |
| Training Stability | Can be unstable | More stable |
| Computational Cost | Higher | Lower |
| Implementation Complexity | Complex | Simpler |
| Hyperparameter Sensitivity | High | Lower |

## Getting Started

For DPO training, see the detailed documentation in `dpo/README.md`.

Example quick start:
```bash
cd dpo/
make build
make run HF_TOKEN=<<YOUR-HF-TOKEN>>

# Run end-to-end pipeline
chmod +x dpo_pipeline_sanity.sh
MODEL_NAME=google/gemma-3-270m ./dpo_pipeline_sanity.sh
```