# Adapted from https://github.com/huggingface/optimum-habana/tree/v1.16.0/examples/trl
# Customized and enabled for Gaudi3

from dataclasses import dataclass, field
import logging
from typing import List, Optional

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainerCallback,
)

from optimum.habana import GaudiConfig, GaudiTrainingArguments
from optimum.habana.trl import GaudiRewardTrainer, RewardDataCollatorWithPadding
from optimum.habana.utils import set_seed


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """

    local_rank: Optional[int] = field(default=-1, metadata={"help": "Used for multi-gpu"})
    resume_from_checkpoint: Optional[bool] = field(
        default=False,
        metadata={"help": "If you want to resume training where it left off."},
    )
    deepspeed: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to deepspeed config if using deepspeed. You may need this if the model that you want to train doesn't fit on a single GPU."
        },
    )
    per_device_train_batch_size: Optional[int] = field(default=4)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=1)
    learning_rate: Optional[float] = field(default=2e-5)
    weight_decay: Optional[float] = field(default=0.001)
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model that you want to train from the Hugging Face hub (REQUIRED). E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The tokenizer for your model, if left empty will use model_name_or_path",
        },
    )
    bf16: Optional[bool] = field(
        default=True,
        metadata={
            "help": "This essentially cuts the training time in half if you want to sacrifice a little precision and have a supported GPU."
        },
    )
    num_train_epochs: Optional[int] = field(
        default=1,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    train_subset: Optional[int] = field(
        default=100000,
        metadata={"help": "The size of the subset of the training data to use"},
    )
    eval_subset: Optional[int] = field(
        default=50000,
        metadata={"help": "The size of the subset of the eval data to use"},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables gradient checkpointing."},
    )
    optim: Optional[str] = field(
        default="adamw_torch",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: Optional[str] = field(
        default="linear",
        metadata={"help": "The lr scheduler"},
    )
    max_length: Optional[int] = field(default=512)
    eval_first_step: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to run eval after the first step"},
    )
    output_dir: Optional[str] = field(default="./results", metadata={"help": "the output directory"})
    save_steps: Optional[int] = field(default=500, metadata={"help": "the saving frequency"})
    eval_steps: Optional[int] = field(default=500, metadata={"help": "the evaluation frequency"})
    logging_steps: Optional[int] = field(default=10, metadata={"help": "the logging frequency"})
    lora_alpha: Optional[float] = field(default=32, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.1, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})
    lora_target_modules: List[str] = field(
        default_factory=lambda: None,
        metadata={"help": "Target modules for the LoRA method."},
    )
    seed: Optional[int] = field(
        default=0, metadata={"help": "Random seed that will be set at the beginning of training."}
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    merge_adapter_after_train: bool = field(
        default=False, metadata={"help": "If True, merge reward model LoRA adapter into base after training."}
    )
    merged_output_dir: Optional[str] = field(
        default=None, metadata={"help": "Directory to save merged reward model (defaults to <output_dir>_merged)."}
    )
    merge_overwrite: bool = field(
        default=False, metadata={"help": "Overwrite merged_output_dir if it exists."}
    )


parser = HfArgumentParser(ScriptArguments)
from logging_utils import setup_logging
script_args = parser.parse_args_into_dataclasses()[0]
setup_logging()
logger = logging.getLogger(__name__)

# Validate required arguments
if not script_args.model_name_or_path:
    raise ValueError("--model_name_or_path is required. Please specify the base model path.")
if not script_args.tokenizer_name_or_path:
    script_args.tokenizer_name_or_path = script_args.model_name_or_path
    logger.info(f"tokenizer_name_or_path not provided, using model path: {script_args.tokenizer_name_or_path}")

logger.info(f"ScriptArguments: {script_args}")
set_seed(script_args.seed)
# Load the human stack-exchange-paired dataset for tuning the reward model.
train_dataset = load_dataset("lvwerra/stack-exchange-paired", data_dir="data/reward", split="train")
if script_args.train_subset > 0:
    logger.info(f"[IF] train_subset > 0 -> selecting first {script_args.train_subset} samples")
    train_dataset = train_dataset.select(range(script_args.train_subset))
eval_dataset = load_dataset("lvwerra/stack-exchange-paired", data_dir="data/evaluation", split="train")
if script_args.eval_subset > 0:
    logger.info(f"[IF] eval_subset > 0 -> selecting first {script_args.eval_subset} samples")
    eval_dataset = eval_dataset.select(range(script_args.eval_subset))
# Define the training args. Needs to be done before the model is loaded if you are using deepspeed.

training_args = GaudiTrainingArguments(
    output_dir=script_args.output_dir,
    learning_rate=script_args.learning_rate,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    per_device_eval_batch_size=script_args.per_device_eval_batch_size,
    num_train_epochs=script_args.num_train_epochs,
    weight_decay=script_args.weight_decay,
    eval_strategy="steps",
    eval_steps=script_args.eval_steps,
    save_strategy="steps",
    save_steps=script_args.save_steps,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    gradient_checkpointing=script_args.gradient_checkpointing,
    deepspeed=script_args.deepspeed,
    local_rank=script_args.local_rank,
    remove_unused_columns=False,
    label_names=[],
    bf16=script_args.bf16,
    logging_strategy="steps",
    logging_steps=script_args.logging_steps,
    optim=script_args.optim,
    lr_scheduler_type=script_args.lr_scheduler_type,
    report_to="none",
    use_habana=True,
    use_lazy_mode=True,
    seed=script_args.seed,
)
logger.info(f"TrainingArguments: {training_args}")

# Load the value-head model and tokenizer.
tokenizer_name = (
    script_args.tokenizer_name_or_path
    if script_args.tokenizer_name_or_path is not None
    else script_args.model_name_or_path
)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, token=script_args.token)

# Auto-detect LoRA target modules if not specified
if script_args.lora_target_modules is None:
    logger.info("[LoRA][AutoDefault] No lora_target_modules provided -> using ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'down_proj', 'gate_proj']")
    script_args.lora_target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'down_proj', 'gate_proj']

torch.autograd.set_detect_anomaly(True)

# Try to load as AutoModelForSequenceClassification, fallback to causal LM wrapper for unsupported models like Gemma3
try:
    model = AutoModelForSequenceClassification.from_pretrained(
        script_args.model_name_or_path, num_labels=1, torch_dtype=torch.bfloat16
    )
    logger.info("[Model] Base sequence classification model loaded")
    seq_cls_supported = True
except ValueError as e:
    if "Unrecognized configuration class" in str(e) and "AutoModelForSequenceClassification" in str(e):
        logger.warning(f"[Fallback] Could not load as SequenceClassification: {e}")
        logger.info("[Fallback] Loading as causal LM and will add reward head manually")
        model = AutoModelForCausalLM.from_pretrained(
            script_args.model_name_or_path, torch_dtype=torch.bfloat16, attn_implementation="eager"
        )
        # Add a linear reward head
        import torch.nn as nn
        hidden_size = model.config.hidden_size
        model.score = nn.Linear(hidden_size, 1, bias=False)
        logger.info(f"[Fallback] Added linear reward head (hidden_size={hidden_size}) to causal LM")
        seq_cls_supported = False
    else:
        raise

# Create LoRA config after determining seq_cls_supported
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS if seq_cls_supported else TaskType.CAUSAL_LM,
    inference_mode=False,
    r=script_args.lora_r,
    lora_alpha=script_args.lora_alpha,
    lora_dropout=script_args.lora_dropout,
    target_modules=script_args.lora_target_modules,
    bias="none",
)

def _param_stats(m):
    tot=trn=0
    for p in m.parameters():
        n=p.numel(); tot+=n; trn+= n if p.requires_grad else 0
    return tot,trn,(trn/tot*100 if tot else 0)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
tot,trn,pct=_param_stats(model)
logger.info(f"[Model Params][After LoRA] total={tot:,} trainable={trn:,} ({pct:.4f}%) (seq_cls={seq_cls_supported})")
# Need to do this for gpt2, because it doesn't have an official pad token.
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
model.config.pad_token_id = tokenizer.eos_token_id
model.config.use_cache = not script_args.gradient_checkpointing
model.config.use_fused_rope = False
num_proc = 24  # Can adjust to be higher if you have more processors.
original_columns = train_dataset.column_names


# Turn the dataset into pairs of post + summaries, where text_j is the preferred question + answer and text_k is the other.
# Then tokenize the dataset.
def preprocess_function(examples):
    new_examples = {
        "input_ids_j": [],
        "attention_mask_j": [],
        "input_ids_k": [],
        "attention_mask_k": [],
    }
    for question, response_j, response_k in zip(examples["question"], examples["response_j"], examples["response_k"]):
        tokenized_j = tokenizer("Question: " + question + "\n\nAnswer: " + response_j, truncation=True)
        tokenized_k = tokenizer("Question: " + question + "\n\nAnswer: " + response_k, truncation=True)

        new_examples["input_ids_j"].append(tokenized_j["input_ids"])
        new_examples["attention_mask_j"].append(tokenized_j["attention_mask"])
        new_examples["input_ids_k"].append(tokenized_k["input_ids"])
        new_examples["attention_mask_k"].append(tokenized_k["attention_mask"])

    return new_examples


# preprocess the dataset and filter out QAs that are longer than script_args.max_length
logger.info(f"[Dataset] Train dataset size before preprocessing: {len(train_dataset)}")
train_dataset = train_dataset.map(
    preprocess_function,
    batched=True,
    num_proc=num_proc,
    remove_columns=original_columns,
)
logger.info(f"[Dataset] Train dataset size after preprocessing: {len(train_dataset)}")

# Check lengths before filtering
if len(train_dataset) > 0:
    sample_count = min(10, len(train_dataset))
    sample_lengths_j = [len(train_dataset[i]["input_ids_j"]) for i in range(sample_count)]
    sample_lengths_k = [len(train_dataset[i]["input_ids_k"]) for i in range(sample_count)]
    logger.info(f"[Dataset] Sample lengths (first {sample_count}) - j: {sample_lengths_j}, k: {sample_lengths_k}")
    logger.info(f"[Dataset] Max length filter: {script_args.max_length}")

train_dataset = train_dataset.filter(
    lambda x: len(x["input_ids_j"]) <= script_args.max_length and len(x["input_ids_k"]) <= script_args.max_length
)
logger.info(f"[Filter] Train dataset length after length filter: {len(train_dataset)}")

if len(train_dataset) == 0:
    logger.error(f"[Error] Training dataset is empty after filtering with max_length={script_args.max_length}")
    logger.error("[Error] Try increasing --max_length or reducing --train_subset")
    raise ValueError(f"Training dataset is empty after filtering. Current max_length={script_args.max_length}. Try increasing max_length.")

logger.info(f"[Dataset] Eval dataset size before preprocessing: {len(eval_dataset)}")
eval_dataset = eval_dataset.map(
    preprocess_function,
    batched=True,
    num_proc=num_proc,
    remove_columns=original_columns,
)
logger.info(f"[Dataset] Eval dataset size after preprocessing: {len(eval_dataset)}")
eval_dataset = eval_dataset.filter(
    lambda x: len(x["input_ids_j"]) <= script_args.max_length and len(x["input_ids_k"]) <= script_args.max_length
)
logger.info(f"[Filter] Eval dataset length after length filter: {len(eval_dataset)}")

if len(eval_dataset) == 0:
    logger.warning(f"[Warning] Eval dataset is empty after filtering with max_length={script_args.max_length}")
    logger.warning("[Warning] Training will proceed without evaluation")

# Define the metric that we'll use for validation.
accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    """col0 = preferred, col1 = non-preferred. r_j, r_k, r_j, r_k, …)"""
    predictions, _ = eval_pred
    preds = np.asarray(predictions)
    chosen = preds[:, 0]
    rejected = preds[:, 1]
    acc = float((chosen > rejected).mean()) if len(chosen) else 0.0
    return {"pairwise_accuracy": acc}


gaudi_config = GaudiConfig()
gaudi_config.use_fused_adam = False
gaudi_config.use_fused_clip_norm = False

# Train the model
trainer = GaudiRewardTrainer(
    model=model,
    gaudi_config=gaudi_config,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset if len(eval_dataset) > 0 else None,
    compute_metrics=compute_metrics if len(eval_dataset) > 0 else None,
    data_collator=RewardDataCollatorWithPadding(
        tokenizer=tokenizer, max_length=script_args.max_length, padding="max_length"
    ),
)


if script_args.eval_first_step:
    logger.info("[IF] eval_first_step == True -> enabling first-step evaluation callback")

    class EvaluateFirstStepCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step == 1:
                control.should_evaluate = True

    trainer.add_callback(EvaluateFirstStepCallback())

train_result = trainer.train(script_args.resume_from_checkpoint)
logger.info("[INFO] Training finished")
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)

logger.info("Saving last checkpoint of the model")
trainer.save_model(script_args.output_dir)
logger.info(f"[INFO] Model saved to {script_args.output_dir}")

# Optional inline merge
if script_args.merge_adapter_after_train:
    try:
        from peft import PeftConfig, PeftModel
        import os
        import json
        adapter_dir = script_args.output_dir
        peft_conf = PeftConfig.from_pretrained(adapter_dir)
        logger.info(f"[MergeInline] Loaded PEFT config task_type={peft_conf.task_type}")
        
        if seq_cls_supported:
            # Standard sequence classification merge
            base_model_fresh = AutoModelForSequenceClassification.from_pretrained(
                script_args.model_name_or_path, num_labels=1, torch_dtype=torch.bfloat16
            )
            merged = PeftModel.from_pretrained(base_model_fresh, adapter_dir)
            logger.info("[MergeInline] Adapter loaded into fresh base reward model; merging...")
            merged = merged.merge_and_unload()
        else:
            # Causal LM fallback merge - need to handle reward head separately
            logger.info("[MergeInline][Fallback] Attempting merge for causal LM reward wrapper.")
            base_model_name_or_path = getattr(peft_conf, 'base_model_name_or_path', None)
            if not base_model_name_or_path:
                logger.warning("[MergeInline][Fallback] base_model_name_or_path missing; using CLI model path")
                base_model_name_or_path = script_args.model_name_or_path
            
            base_model_fresh = AutoModelForCausalLM.from_pretrained(
                base_model_name_or_path, torch_dtype=torch.bfloat16, attn_implementation="eager"
            )
            merged = PeftModel.from_pretrained(base_model_fresh, adapter_dir)
            logger.info("[MergeInline][Fallback] Merging adapter into causal LM base...")
            merged = merged.merge_and_unload()
            
            # Re-add the reward head to merged model
            import torch.nn as nn
            hidden_size = merged.config.hidden_size
            merged.score = nn.Linear(hidden_size, 1, bias=False)
            
            # Copy reward head weights from trained model
            merged.score.load_state_dict(model.score.state_dict())
        
        out_dir = script_args.merged_output_dir or f"{script_args.output_dir}_merged"
        if os.path.exists(out_dir) and not script_args.merge_overwrite:
            logger.warning(f"[MergeInline] Output dir {out_dir} exists and merge_overwrite=False -> abort merge")
        else:
            os.makedirs(out_dir, exist_ok=True)
            merged.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
            
            if not seq_cls_supported:
                # Save reward head separately for causal LM case
                reward_head_path = os.path.join(out_dir, "reward_value_head.bin")
                torch.save(merged.score.state_dict(), reward_head_path)
                
                # Save reward head config
                reward_config = {
                    "hidden_size": hidden_size,
                    "is_causal_lm_reward": True,
                    "base_model_type": merged.config.model_type
                }
                with open(os.path.join(out_dir, "reward_head_config.json"), "w") as f:
                    json.dump(reward_config, f, indent=2)
                logger.info(f"[MergeInline][Fallback] Exported merged causal LM reward model to {out_dir} (value head + metadata)")
            else:
                logger.info(f"[MergeInline] Merged reward model saved to {out_dir}")
    except Exception as e:
        logger.exception(f"[MergeInline] Failed to merge reward adapter inline: {e}")
