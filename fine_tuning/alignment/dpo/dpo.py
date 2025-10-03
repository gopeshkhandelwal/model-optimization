# Direct Preference Optimization implementation for Gaudi3
# Customized and enabled for Gaudi3 with Gemma3 support
import json
import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser
from trl import DPOTrainer, DPOConfig

from optimum.habana.accelerate import GaudiAccelerator
from optimum.habana.utils import set_seed


tqdm.pandas()


@dataclass
class ScriptArguments:
    """
    Arguments for DPO training
    """
    
    model_name_or_path: Optional[str] = field(default=None, metadata={"help": "the model name (REQUIRED)"})
    tokenizer_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "the tokenizer name (defaults to model_name_or_path if not provided)"}
    )
    ref_model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "the reference model name (defaults to model_name_or_path if not provided)"}
    )
    dataset_name: Optional[str] = field(default="lvwerra/stack-exchange-paired", metadata={"help": "the dataset name"})
    dataset_config: Optional[str] = field(default=None, metadata={"help": "the dataset config"})
    split: Optional[str] = field(default="train", metadata={"help": "the dataset split"})
    learning_rate: Optional[float] = field(default=5e-4, metadata={"help": "the learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    warmup_steps: Optional[int] = field(default=100, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})
    
    per_device_train_batch_size: Optional[int] = field(default=4, metadata={"help": "train batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "eval batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(default=4, metadata={"help": "the number of gradient accumulation steps"})
    gradient_checkpointing: Optional[bool] = field(default=True, metadata={"help": "whether to use gradient checkpointing"})
    
    max_length: Optional[int] = field(default=512, metadata={"help": "the maximum sequence length"})
    max_prompt_length: Optional[int] = field(default=128, metadata={"help": "the maximum prompt length"})
    max_target_length: Optional[int] = field(default=128, metadata={"help": "the maximum target length"})
    
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})
    loss_type: Optional[str] = field(default="sigmoid", metadata={"help": "the loss type for DPO loss"})
    
    num_train_epochs: Optional[int] = field(default=1, metadata={"help": "the number of training epochs"})
    max_steps: Optional[int] = field(default=-1, metadata={"help": "the number of training steps"})
    logging_steps: Optional[int] = field(default=10, metadata={"help": "the number of logging steps"})
    save_steps: Optional[int] = field(default=100, metadata={"help": "the number of save steps"})
    eval_steps: Optional[int] = field(default=100, metadata={"help": "the number of eval steps"})
    
    output_dir: Optional[str] = field(default="./dpo_output", metadata={"help": "the output directory"})
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    
    # LoRA parameters
    use_peft: Optional[bool] = field(default=True, metadata={"help": "whether to use PEFT"})
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=64, metadata={"help": "the lora r parameter"})
    lora_target_modules: List[str] = field(
        default_factory=lambda: None,
        metadata={"help": "Target modules for the LoRA method."},
    )
    
    # Habana specific
    use_habana: Optional[bool] = field(default=True, metadata={"help": "use habana for training"})
    seed: Optional[int] = field(default=0, metadata={"help": "the seed"})
    
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )


def parse_args():
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    return script_args


# Import logging setup
from logging_utils import setup_logging

def main():
    script_args = parse_args()
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Validate required arguments
    if not script_args.model_name_or_path:
        raise ValueError("--model_name_or_path is required. Please specify the base model path.")
    if not script_args.tokenizer_name_or_path:
        script_args.tokenizer_name_or_path = script_args.model_name_or_path
        logger.info(f"tokenizer_name_or_path not provided, using model path: {script_args.tokenizer_name_or_path}")
    if not script_args.ref_model_name_or_path:
        script_args.ref_model_name_or_path = script_args.model_name_or_path
        logger.info(f"ref_model_name_or_path not provided, using model path: {script_args.ref_model_name_or_path}")
    
    logger.info(f"ScriptArguments: {script_args}")
    
    # Set seed
    set_seed(script_args.seed)
    
    # Auto-detect LoRA target modules if not specified
    if script_args.use_peft and script_args.lora_target_modules is None:
        logger.info("[LoRA][AutoDefault] No lora_target_modules provided -> using ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'down_proj', 'gate_proj']")
        script_args.lora_target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'down_proj', 'gate_proj']
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer_name_or_path)
    if getattr(tokenizer, "pad_token", None) is None:
        logger.info("[IF] tokenizer.pad_token is None -> assigning eos_token as pad_token")
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    logger.info(f"[Dataset] Loading {script_args.dataset_name}")
    dataset = load_dataset(
        script_args.dataset_name,
        script_args.dataset_config,
        split=script_args.split,
    )
    
    if script_args.max_train_samples is not None:
        logger.info(f"[IF] max_train_samples specified -> limiting to {script_args.max_train_samples}")
        max_train_samples = min(len(dataset), script_args.max_train_samples)
        dataset = dataset.select(range(max_train_samples))
    
    logger.info(f"[Dataset] Training on {len(dataset)} examples")
    logger.info(f"[Dataset] Sample columns: {dataset.column_names}")
    
    # Check if we need to format the dataset for DPO
    if 'chosen' not in dataset.column_names or 'rejected' not in dataset.column_names:
        logger.info("[Dataset] Formatting dataset for DPO training")
        
        def format_for_dpo(example):
            """Format a single example for DPO training"""
            # Handle stack-exchange-paired format
            if 'question' in example and 'response_j' in example and 'response_k' in example:
                prompt = f"Question: {example['question']}\n\nAnswer:"
                chosen = example['response_j']  # Higher scored response
                rejected = example['response_k']  # Lower scored response
                
                return {
                    'prompt': prompt,
                    'chosen': chosen,
                    'rejected': rejected
                }
            
            # Handle other formats
            elif 'instruction' in example and 'output' in example:
                prompt = example['instruction']
                chosen = example['output']
                rejected = chosen[:len(chosen)//2] + "..."  # Truncated as worse response
                
                return {
                    'prompt': prompt,
                    'chosen': chosen,
                    'rejected': rejected
                }
            
            # Fallback for unknown formats
            else:
                logger.warning(f"[Dataset] Unknown format, available keys: {list(example.keys())}")
                return {
                    'prompt': "What is a good response?",
                    'chosen': "This is a helpful and detailed response that provides useful information.",
                    'rejected': "Short response."
                }
        
        # Apply formatting to each example
        dataset = dataset.map(format_for_dpo, remove_columns=dataset.column_names)
        logger.info(f"[Dataset] Formatted dataset columns: {dataset.column_names}")
        
        # Show a sample
        if len(dataset) > 0:
            sample = dataset[0]
            logger.info(f"[Dataset] Sample prompt: {sample['prompt'][:100]}...")
            logger.info(f"[Dataset] Sample chosen: {sample['chosen'][:100]}...")
            logger.info(f"[Dataset] Sample rejected: {sample['rejected'][:100]}...")
    
    # Setup PEFT config
    peft_config = None
    if script_args.use_peft:
        peft_config = LoraConfig(
            r=script_args.lora_r,
            lora_alpha=script_args.lora_alpha,
            lora_dropout=script_args.lora_dropout,
            target_modules=script_args.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        logger.info(f"[LoRA] Config: {peft_config}")
    
    # Setup training arguments
    training_args = DPOConfig(
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        max_steps=script_args.max_steps,
        num_train_epochs=script_args.num_train_epochs,
        logging_steps=script_args.logging_steps,
        save_steps=script_args.save_steps,
        eval_steps=script_args.eval_steps,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        learning_rate=script_args.learning_rate,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_steps=script_args.warmup_steps,
        weight_decay=script_args.weight_decay,
        optimizer_type=script_args.optimizer_type,
        output_dir=script_args.output_dir,
        remove_unused_columns=False,
        max_length=script_args.max_length,
        max_prompt_length=script_args.max_prompt_length,
        max_target_length=script_args.max_target_length,
        beta=script_args.beta,
        loss_type=script_args.loss_type,
        report_to=script_args.log_with if script_args.log_with else [],
        seed=script_args.seed,
    )
    
    logger.info(f"[DPOConfig] {training_args}")
    
    # Initialize DPO trainer
    logger.info("[Trainer] Initializing DPO trainer")
    dpo_trainer = DPOTrainer(
        model=script_args.model_name_or_path,
        ref_model=script_args.ref_model_name_or_path,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        model_kwargs={
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
            "attn_implementation": "eager",  # For Gemma3 compatibility
        },
    )
    
    # Model parameter stats
    def _param_stats(m):
        tot=trn=0
        for p in m.parameters():
            n=p.numel(); tot+=n; trn+= n if p.requires_grad else 0
        return tot,trn,(trn/tot*100 if tot else 0)
    
    btot,btrn,bpct=_param_stats(dpo_trainer.model)
    logger.info(f"[Model Params][Policy] total={btot:,} trainable={btrn:,} ({bpct:.4f}%)")
    
    # Start training
    logger.info("[Training] Starting DPO training")
    start_time = time.time()
    
    dpo_trainer.train()
    
    end_time = time.time()
    training_time = end_time - start_time
    
    # Save final model
    logger.info("[Training] Saving final model")
    dpo_trainer.save_model()
    
    # Save metrics
    metrics = {
        "training_time": training_time,
        "num_train_samples": len(dataset),
        "samples_per_second": len(dataset) / training_time,
    }
    
    with open(f"{script_args.output_dir}/training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"[Training] DPO training complete. Time: {training_time:.2f}s")
    logger.info(f"[Metrics] Samples/sec: {metrics['samples_per_second']:.2f}")


if __name__ == "__main__":
    main()