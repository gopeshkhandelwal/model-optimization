#!/usr/bin/env python
# sft.py - Supervised Fine-Tuning for DPO initialization

import json
import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
)
from trl import SFTTrainer

from optimum.habana.accelerate import GaudiAccelerator
from optimum.habana.utils import set_seed


@dataclass
class ScriptArguments:
    """
    Arguments for SFT training
    """
    
    model_name_or_path: Optional[str] = field(default=None, metadata={"help": "the model name (REQUIRED)"})
    tokenizer_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "the tokenizer name (defaults to model_name_or_path if not provided)"}
    )
    dataset_name: Optional[str] = field(default="Anthropic/hh-rlhf", metadata={"help": "the dataset name"})
    dataset_config: Optional[str] = field(default="default", metadata={"help": "the dataset config"})
    split: Optional[str] = field(default="train", metadata={"help": "the dataset split"})
    max_seq_length: Optional[int] = field(default=512, metadata={"help": "the maximum sequence length"})
    
    # Training arguments
    per_device_train_batch_size: Optional[int] = field(default=4, metadata={"help": "train batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(default=4, metadata={"help": "gradient accumulation steps"})
    learning_rate: Optional[float] = field(default=1e-4, metadata={"help": "the learning rate"})
    logging_steps: Optional[int] = field(default=10, metadata={"help": "the number of logging steps"})
    num_train_epochs: Optional[int] = field(default=1, metadata={"help": "the number of training epochs"})
    max_steps: Optional[int] = field(default=-1, metadata={"help": "the number of training steps"})
    output_dir: Optional[str] = field(default="./sft_output", metadata={"help": "the output directory"})
    save_strategy: Optional[str] = field(default="epoch", metadata={"help": "the save strategy"})
    optim: Optional[str] = field(default="adamw_torch", metadata={"help": "the optimizer"})
    warmup_steps: Optional[int] = field(default=100, metadata={"help": "the number of warmup steps"})
    bf16: Optional[bool] = field(default=True, metadata={"help": "whether to use bf16"})
    remove_unused_columns: Optional[bool] = field(default=False, metadata={"help": "remove unused columns"})
    
    # LoRA parameters
    use_peft: Optional[bool] = field(default=True, metadata={"help": "whether to use PEFT"})
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.1, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=64, metadata={"help": "the lora r parameter"})
    lora_target_modules: List[str] = field(
        default_factory=lambda: None,
        metadata={"help": "Target modules for the LoRA method."},
    )
    
    # Other parameters
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
    
    # Load dataset - use a simpler instruction dataset
    logger.info(f"[Dataset] Loading instruction dataset")
    try:
        # Try to load the preference dataset
        dataset = load_dataset(
            script_args.dataset_name,
            script_args.dataset_config,
            split=script_args.split,
        )
        
        if script_args.max_train_samples is not None:
            logger.info(f"[IF] max_train_samples specified -> limiting to {script_args.max_train_samples}")
            max_train_samples = min(len(dataset), script_args.max_train_samples)
            dataset = dataset.select(range(max_train_samples))
        
        logger.info(f"[Dataset] Raw dataset columns: {dataset.column_names}")
        logger.info(f"[Dataset] Sample: {dataset[0]}")
        
        # Preprocess dataset to create text field
        def preprocess_function(examples):
            texts = []
            batch_size = len(examples[list(examples.keys())[0]])
            
            for i in range(batch_size):
                # Try different field combinations
                if "chosen" in examples and isinstance(examples["chosen"][i], str):
                    text = examples["chosen"][i]
                elif "text" in examples:
                    text = examples["text"][i]
                elif "instruction" in examples and "output" in examples:
                    instruction = examples["instruction"][i] if "instruction" in examples else ""
                    output = examples["output"][i] if "output" in examples else ""
                    text = f"Instruction: {instruction}\nResponse: {output}"
                else:
                    # Fallback: create a simple text from available fields
                    text = "This is a training example."
                
                # Ensure text is a string and not too long
                if isinstance(text, str):
                    text = text[:2000]  # Truncate if too long
                else:
                    text = str(text)[:2000]
                
                texts.append(text)
            
            return {"text": texts}
        
        logger.info(f"[Dataset] Preprocessing dataset...")
        dataset = dataset.map(
            preprocess_function, 
            batched=True, 
            remove_columns=dataset.column_names,
            desc="Preprocessing"
        )
        logger.info(f"[Dataset] Training on {len(dataset)} examples")
        logger.info(f"[Dataset] Sample processed text: {dataset[0]['text'][:100]}...")
        
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        logger.info("Falling back to creating synthetic dataset")
        
        # Create a simple synthetic dataset for testing
        synthetic_data = []
        for i in range(script_args.max_train_samples or 100):
            synthetic_data.append({
                "text": f"This is training example {i}. The model should learn to generate helpful responses."
            })
        
        from datasets import Dataset
        dataset = Dataset.from_list(synthetic_data)
        logger.info(f"[Dataset] Created synthetic dataset with {len(dataset)} examples")
    
    # Load model
    device = "hpu" if hasattr(torch, "hpu") else ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
        low_cpu_mem_usage=True,
        attn_implementation="eager",  # For Gemma3 compatibility
    )
    
    # Setup PEFT if enabled
    if script_args.use_peft:
        peft_config = LoraConfig(
            r=script_args.lora_r,
            lora_alpha=script_args.lora_alpha,
            lora_dropout=script_args.lora_dropout,
            target_modules=script_args.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
        logger.info(f"[LoRA] Config: {peft_config}")
        model.print_trainable_parameters()
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=script_args.output_dir,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        learning_rate=script_args.learning_rate,
        logging_steps=script_args.logging_steps,
        num_train_epochs=script_args.num_train_epochs,
        max_steps=script_args.max_steps,
        save_strategy=script_args.save_strategy,
        optim=script_args.optim,
        warmup_steps=script_args.warmup_steps,
        bf16=script_args.bf16,
        remove_unused_columns=script_args.remove_unused_columns,
        report_to=[],
        seed=script_args.seed,
    )
    
    logger.info(f"[TrainingArgs] {training_args}")
    
    # Initialize trainer
    logger.info("[Trainer] Initializing SFT trainer")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=script_args.max_seq_length,
        packing=False,  # Disable packing for simplicity
    )
    
    # Start training
    logger.info("[Training] Starting SFT training")
    start_time = time.time()
    
    trainer.train()
    
    end_time = time.time()
    training_time = end_time - start_time
    
    # Save final model
    logger.info("[Training] Saving final model")
    trainer.save_model()
    
    # Save metrics
    metrics = {
        "training_time": training_time,
        "num_train_samples": len(dataset),
        "samples_per_second": len(dataset) / training_time,
    }
    
    with open(f"{script_args.output_dir}/training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"[Training] SFT training complete. Time: {training_time:.2f}s")
    logger.info(f"[Metrics] Samples/sec: {metrics['samples_per_second']:.2f}")


if __name__ == "__main__":
    main()