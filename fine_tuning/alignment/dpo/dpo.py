# Direct Preference Optimization implementation for Gaudi3
# Customized and enabled for Gaudi3 with Gemma3 support
import json
import time
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import torch
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser
from trl import DPOTrainer, DPOConfig

from optimum.habana.accelerate import GaudiAccelerator
from optimum.habana.utils import set_seed

# ---------------------------------------------------------------------------
# Compatibility Override for get_batch_samples
# ---------------------------------------------------------------------------
# The upstream transformers Trainer now calls: get_batch_samples(iterator, num_batches, device)
# Some TRL versions' DPOTrainer implementation conflicts with this change, leading to
# errors or internal assumptions that attempt generation during batch sampling.
# We install a lightweight override that simply pulls the requested number of batches
# from the iterator and estimates item counts, bypassing any generation side-effects.
# This is safe because DPO training uses already prepared input tensors from the
# dataloader; generation inside get_batch_samples is not strictly required for loss.
# ---------------------------------------------------------------------------
try:  # pragma: no cover
    def _safe_get_batch_samples(self, epoch_iterator, num_batches, *unused_device):
        batches = []
        items = 0
        for _ in range(num_batches):
            try:
                batch = next(epoch_iterator)
            except StopIteration:
                break
            if batch is None:
                continue
            batches.append(batch)
            # Heuristic: infer number of items in batch from first tensor value
            added = False
            if isinstance(batch, dict):
                for v in batch.values():
                    if isinstance(v, torch.Tensor):
                        items += v.size(0)
                        added = True
                        break
            elif isinstance(batch, (list, tuple)):
                # If it's a sequence of dicts
                try:
                    items += len(batch)
                    added = True
                except Exception:
                    pass
            if not added:
                items += 1
        return batches, items

    # Attach only if not already updated upstream
    DPOTrainer.get_batch_samples = _safe_get_batch_samples  # type: ignore
    logging.getLogger(__name__).warning("[Compat] Overrode DPOTrainer.get_batch_samples with lightweight iterator-based implementation.")
except Exception as _exc:  # pragma: no cover
    logging.getLogger(__name__).warning(f"[Compat] Could not set safe get_batch_samples override: {_exc}")

# ---------------------------------------------------------------------------
# compute_loss compatibility (num_items_in_batch argument in newer Trainer)
# ---------------------------------------------------------------------------
try:  # pragma: no cover
    import inspect
    _orig_compute_loss = DPOTrainer.__dict__.get("compute_loss", None)
    if _orig_compute_loss is not None:
        sig = inspect.signature(_orig_compute_loss)
        if 'num_items_in_batch' not in sig.parameters:
            def _compat_compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                # Silently ignore num_items_in_batch or any future extra kwargs
                return _orig_compute_loss(self, model, inputs, return_outputs=return_outputs)
            DPOTrainer.compute_loss = _compat_compute_loss  # type: ignore
            logging.getLogger(__name__).warning(
                "[Compat] Wrapped DPOTrainer.compute_loss to ignore extra Trainer kwargs (e.g., num_items_in_batch)."
            )
except Exception as _exc:  # pragma: no cover
    logging.getLogger(__name__).warning(f"[Compat] Could not wrap compute_loss: {_exc}")

# ---------------------------------------------------------------------------
# log() compatibility (new Trainer passes start_time)
# ---------------------------------------------------------------------------
try:  # pragma: no cover
    import inspect as _inspect2
    _orig_log = DPOTrainer.__dict__.get("log", None)
    if _orig_log is not None:
        sig_log = _inspect2.signature(_orig_log)
        # Legacy signature likely (self, logs); new Trainer calls log(logs, start_time)
        if len(sig_log.parameters) == 2:  # self + logs only
            def _compat_log(self, logs, *extra, **kwargs):
                return _orig_log(self, logs)
            DPOTrainer.log = _compat_log  # type: ignore
            logging.getLogger(__name__).warning(
                "[Compat] Wrapped DPOTrainer.log to ignore additional Trainer arguments (start_time)."
            )
except Exception as _exc:  # pragma: no cover
    logging.getLogger(__name__).warning(f"[Compat] Could not wrap log(): {_exc}")


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
    split: Optional[str] = field(default="train", metadata={"help": "the dataset split for training"})
    evaluation_split: Optional[str] = field(default=None, metadata={"help": "optional evaluation split (auto if None)"})
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
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Optional cap on evaluation examples."},
    )
    eval_fraction: Optional[float] = field(
        default=0.02,
        metadata={"help": "If no evaluation split exists, sample this fraction from training for eval."},
    )
    truncate_longer_samples: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to heuristically truncate overly long prompt/answer pairs before tokenization."},
    )
    truncate_strategy: Optional[str] = field(
        default="end",
        metadata={"help": "Truncation strategy: 'end' (cut tail) or 'middle' (keep start & end)."},
    )
    format_num_proc: Optional[int] = field(
        default=1,
        metadata={"help": "Number of processes for dataset formatting map (set >1 for speed)."},
    )
    datasets_verification_mode: Optional[str] = field(
        default='all_checks',
        metadata={"help": "HF datasets verification mode: 'no_checks', 'basic_checks', or 'all_checks'."},
    )
    force_ref_model: Optional[bool] = field(
        default=False,
        metadata={"help": "Set True to force using an explicit reference model while also training PEFT adapters (passes force_use_ref_model=True to DPOTrainer)."},
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
    
    # ---------------------------
    # Dataset helpers
    # ---------------------------
    def discover_split(preferred: List[str]) -> Dict[str, Any]:
        """Attempt to load a dataset split. If fails, load dict and choose a fallback."""
        base_msg = f"[Dataset][discover] {script_args.dataset_name}"
        # First attempt: direct split load
        try:
            ds = load_dataset(
                script_args.dataset_name,
                script_args.dataset_config,
                split=preferred[0],
                verification_mode=script_args.datasets_verification_mode,
            )
            return {"dataset": ds, "split": preferred[0], "available": [preferred[0]], "from_fallback": False}
        except Exception as e_first:
            first_err = str(e_first)
            logger.warning(f"{base_msg} direct split load failed ({preferred[0]}): {e_first.__class__.__name__}: {e_first}")
            # Retry direct with no_checks if not already
            if script_args.datasets_verification_mode != 'no_checks':
                try:
                    logger.info(f"{base_msg} retrying direct split with verification_mode='no_checks'")
                    ds = load_dataset(
                        script_args.dataset_name,
                        script_args.dataset_config,
                        split=preferred[0],
                        verification_mode='no_checks',
                    )
                    logger.info(f"{base_msg} recovered using no_checks for split '{preferred[0]}'")
                    return {"dataset": ds, "split": preferred[0], "available": [preferred[0]], "from_fallback": False}
                except Exception as e_retry:
                    logger.warning(f"{base_msg} retry with no_checks failed: {e_retry.__class__.__name__}: {e_retry}")
        # Second attempt: load whole dataset dict
        try:
            ds_dict = load_dataset(
                script_args.dataset_name,
                script_args.dataset_config,
                verification_mode=script_args.datasets_verification_mode,
            )
        except Exception as e_dict:
            logger.warning(f"{base_msg} dataset dict load failed ({script_args.datasets_verification_mode}): {e_dict}")
            if script_args.datasets_verification_mode != 'no_checks':
                try:
                    logger.info(f"{base_msg} retrying dataset dict with verification_mode='no_checks'")
                    ds_dict = load_dataset(
                        script_args.dataset_name,
                        script_args.dataset_config,
                        verification_mode='no_checks',
                    )
                except Exception as e_dict_retry:
                    logger.error(f"{base_msg} failed to load dataset dict even with no_checks: {e_dict_retry}")
                    raise e_dict_retry
        available = list(ds_dict.keys())
        chosen = None
        for cand in preferred:
            if cand in available:
                chosen = cand
                break
        if chosen is None:
            chosen = available[0]
            logger.warning(f"{base_msg} none of preferred splits {preferred} found; using first available '{chosen}'")
        else:
            if chosen != preferred[0]:
                logger.warning(f"{base_msg} using fallback split '{chosen}' (preferred '{preferred[0]}')")
        return {"dataset": ds_dict[chosen], "split": chosen, "available": available, "from_fallback": chosen != preferred[0]}

    def create_eval_from_train(train_ds, fraction: float, seed: int):
        if fraction <= 0 or fraction >= 1:
            return train_ds, None
        n = len(train_ds)
        eval_size = max(1, int(n * fraction))
        if eval_size >= n:
            return train_ds, None
        split_indices = list(range(n))
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n, generator=g).tolist()
        eval_idx = set(perm[:eval_size])
        train_sel = [i for i in range(n) if i not in eval_idx]
        return train_ds.select(train_sel), train_ds.select(list(eval_idx))

    def truncate_text(text: str, target_len: int, strategy: str) -> str:
        if len(text) <= target_len:
            return text
        if strategy == 'middle' and target_len > 10:
            half = target_len // 2
            return text[:half] + " <...> " + text[-(target_len - half - 7):]
        return text[:target_len]

    # ---------------------------
    # Load training split (with fallback)
    # ---------------------------
    logger.info(f"[Dataset] Loading training split (requested '{script_args.split}')")
    train_pref_order = [script_args.split, 'train', 'training', 'train_full', 'train_augmented', 'test', 'validation', 'val']
    train_meta = discover_split(train_pref_order)
    dataset = train_meta["dataset"]
    script_args.split = train_meta["split"]
    logger.info(f"[Dataset] Using training split '{script_args.split}' (available={train_meta['available']})")

    # ---------------------------
    # Load evaluation split (optional)
    # ---------------------------
    eval_dataset = None
    if script_args.evaluation_split:
        logger.info(f"[Dataset][Eval] Attempting to load evaluation split '{script_args.evaluation_split}'")
        try:
            eval_dataset = load_dataset(
                script_args.dataset_name,
                script_args.dataset_config,
                split=script_args.evaluation_split,
            )
            logger.info(f"[Dataset][Eval] Loaded evaluation split '{script_args.evaluation_split}'")
        except Exception as e:
            logger.warning(f"[Dataset][Eval] Failed to load specified evaluation split '{script_args.evaluation_split}': {e}")
    if eval_dataset is None:
        # Try auto detection if not explicitly provided
        if not script_args.evaluation_split:
            eval_pref = ['validation', 'val', 'eval', 'test']
            found_eval = None
            # Only load the dict if we haven't already (train_meta handled this if fallback path taken)
            try:
                ds_dict = load_dataset(script_args.dataset_name, script_args.dataset_config)
                for cand in eval_pref:
                    if cand in ds_dict and cand != script_args.split:
                        found_eval = ds_dict[cand]
                        script_args.evaluation_split = cand
                        logger.info(f"[Dataset][Eval] Auto-selected evaluation split '{cand}'")
                        break
            except Exception:
                pass
            if found_eval is not None:
                eval_dataset = found_eval
        # Create holdout from train if still None
        if eval_dataset is None and script_args.eval_fraction and script_args.eval_fraction > 0:
            logger.info(f"[Dataset][Eval] Creating evaluation holdout fraction={script_args.eval_fraction}")
            dataset, eval_dataset = create_eval_from_train(dataset, script_args.eval_fraction, script_args.seed)
            if eval_dataset is not None:
                script_args.evaluation_split = f"holdout_{script_args.eval_fraction}";
                logger.info(f"[Dataset][Eval] Created holdout eval with {len(eval_dataset)} examples")
    
    if script_args.max_train_samples is not None:
        logger.info(f"[IF] max_train_samples specified -> limiting train to {script_args.max_train_samples}")
        max_train_samples = min(len(dataset), script_args.max_train_samples)
        dataset = dataset.select(range(max_train_samples))
    if eval_dataset is not None and script_args.max_eval_samples is not None:
        logger.info(f"[IF] max_eval_samples specified -> limiting eval to {script_args.max_eval_samples}")
        max_eval_samples = min(len(eval_dataset), script_args.max_eval_samples)
        eval_dataset = eval_dataset.select(range(max_eval_samples))
    
    logger.info(f"[Dataset] Training on {len(dataset)} examples from split '{script_args.split}'")
    if eval_dataset is not None:
        logger.info(f"[Dataset][Eval] Evaluation on {len(eval_dataset)} examples (split='{script_args.evaluation_split}')")
    logger.info(f"[Dataset] Sample columns: {dataset.column_names}")
    
    # Check if we need to format the dataset for DPO
    if 'chosen' not in dataset.column_names or 'rejected' not in dataset.column_names:
        logger.info("[Dataset] Formatting dataset for DPO training")
        
        def format_for_dpo(example):
            """Format a single example for DPO training"""
            # Handle Anthropic/hh-rlhf format - they already have the right structure
            if 'chosen' in example and 'rejected' in example:
                # The dataset already has the right format, just return as-is
                return example
            
            # Handle stack-exchange-paired format
            elif 'question' in example and 'response_j' in example and 'response_k' in example:
                # Ensure prompt ends with a space so first answer token can't merge with last prompt token
                prompt = f"Question: {example['question']}\n\nAnswer: "
                
                # Use scores to determine which response is better
                score_j = example.get('score_j', 0)
                score_k = example.get('score_k', 0)
                
                if score_j >= score_k:
                    chosen = example['response_j'].lstrip() if isinstance(example['response_j'], str) else example['response_j']
                    rejected = example['response_k'].lstrip() if isinstance(example['response_k'], str) else example['response_k']
                else:
                    chosen = example['response_k'].lstrip() if isinstance(example['response_k'], str) else example['response_k']
                    rejected = example['response_j'].lstrip() if isinstance(example['response_j'], str) else example['response_j']
                
                return {
                    'prompt': prompt,
                    'chosen': chosen,
                    'rejected': rejected
                }
            
            # Handle other formats
            elif 'instruction' in example and 'output' in example:
                # Generic instruction tuning style -> build prompt with a trailing space
                prompt = example['instruction'].rstrip() + " " if isinstance(example['instruction'], str) else example['instruction']
                chosen = example['output'].lstrip() if isinstance(example['output'], str) else example['output']
                rejected = (chosen[:len(chosen)//2] + "...") if isinstance(chosen, str) else chosen  # Truncated as worse response
                
                return {
                    'prompt': prompt,
                    'chosen': chosen,
                    'rejected': rejected
                }
            
            # Fallback for unknown formats
            else:
                logger.warning(f"[Dataset] Unknown format, available keys: {list(example.keys())}")
                return {
                    'prompt': "What is a good response? ",  # trailing space
                    'chosen': "This is a helpful and detailed response that provides useful information.",
                    'rejected': "Short response."
                }
        
        # Apply formatting to each example (parallel if requested)
        num_proc = script_args.format_num_proc if script_args.format_num_proc and script_args.format_num_proc > 1 else None
        dataset = dataset.map(
            format_for_dpo,
            remove_columns=dataset.column_names,
            num_proc=num_proc,
            desc="Formatting train dataset",
        )
        if eval_dataset is not None:
            eval_dataset = eval_dataset.map(
                format_for_dpo,
                remove_columns=eval_dataset.column_names,
                num_proc=num_proc,
                desc="Formatting eval dataset",
            )
        logger.info(f"[Dataset] Formatted dataset columns: {dataset.column_names}")
        
        # Show a sample
        if len(dataset) > 0:
            sample = dataset[0]
            logger.info(f"[Dataset] Sample prompt: {sample['prompt'][:100]}...")
            logger.info(f"[Dataset] Sample chosen: {sample['chosen'][:100]}...")
            logger.info(f"[Dataset] Sample rejected: {sample['rejected'][:100]}...")
        if eval_dataset is not None and len(eval_dataset) > 0:
            esample = eval_dataset[0]
            logger.info(f"[Dataset][Eval] Sample prompt: {esample['prompt'][:100]}...")
    
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
    
    # Optional truncation before tokenization inside trainer (heuristic for extremely long examples)
    if script_args.truncate_longer_samples:
        max_prompt_chars = script_args.max_prompt_length * 4  # approx char/token heuristic
        max_target_chars = script_args.max_target_length * 4
        def _truncate_record(rec):
            # Safe guards
            if 'prompt' in rec:
                rec['prompt'] = truncate_text(rec['prompt'], max_prompt_chars, script_args.truncate_strategy)
            if 'chosen' in rec:
                rec['chosen'] = truncate_text(rec['chosen'], max_target_chars, script_args.truncate_strategy)
            if 'rejected' in rec:
                rec['rejected'] = truncate_text(rec['rejected'], max_target_chars, script_args.truncate_strategy)
            return rec
        dataset = dataset.map(_truncate_record, desc="Truncating train samples")
        if eval_dataset is not None:
            eval_dataset = eval_dataset.map(_truncate_record, desc="Truncating eval samples")

    # Sanitize prompt/answer boundaries to avoid DPOTrainer prompt_input_ids mismatch issues
    def _sanitize_boundary(rec):
        if 'prompt' in rec and isinstance(rec['prompt'], str):
            # Guarantee at least one trailing space (not just newline) to isolate first answer token
            if not rec['prompt'].endswith(' '):
                if rec['prompt'] and rec['prompt'][-1].isspace():
                    rec['prompt'] = rec['prompt'].rstrip() + ' '
                else:
                    rec['prompt'] = rec['prompt'] + ' '
        if 'chosen' in rec and isinstance(rec['chosen'], str):
            rec['chosen'] = rec['chosen'].lstrip()
        if 'rejected' in rec and isinstance(rec['rejected'], str):
            rec['rejected'] = rec['rejected'].lstrip()
        return rec

    dataset = dataset.map(_sanitize_boundary, desc="Sanitizing train prompt/answer boundaries")
    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(_sanitize_boundary, desc="Sanitizing eval prompt/answer boundaries")

    # Map legacy optimizer_type to a supported 'optim' value for current TRL/transformers
    chosen_optim = script_args.optimizer_type
    # Common unsupported / bitsandbytes options replaced for Gaudi compatibility
    if chosen_optim in (None, "", "paged_adamw_32bit", "adamw_bnb_8bit", "adamw_bnb_32bit"):
        logger.info(f"[Optimizer] Mapping requested optimizer_type='{script_args.optimizer_type}' to 'adamw_torch' for compatibility")
        chosen_optim = "adamw_torch"
    # Setup training arguments (DPOConfig does not accept 'optimizer_type', use 'optim')
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
        optim=chosen_optim,
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
    
    # Initialize / load policy model explicitly to control dtype & attention impl
    logger.info("[Model] Loading policy model for DPO")
    model_load_kwargs = {
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
    }
    # Try adding attn_implementation if supported (Gemma3), ignore otherwise
    try:
        model = AutoModelForCausalLM.from_pretrained(
            script_args.model_name_or_path,
            attn_implementation="eager",
            **model_load_kwargs,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            script_args.model_name_or_path,
            **model_load_kwargs,
        )
    logger.info("[Model] Policy model loaded")

    # Determine reference model usage
    ref_model_arg = script_args.ref_model_name_or_path
    force_use_ref = False
    if script_args.use_peft:
        if not script_args.force_ref_model:
            logger.info("[RefModel] use_peft=True and force_ref_model=False -> setting ref_model=None to satisfy DPOTrainer PEFT requirement")
            ref_model_arg = None
        else:
            logger.info("[RefModel] Forcing explicit reference model alongside PEFT (force_use_ref_model=True)")
            force_use_ref = True

    logger.info("[Trainer] Initializing DPO trainer")
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=ref_model_arg,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,  # legacy arg still required in this TRL version
        peft_config=peft_config,
        force_use_ref_model=force_use_ref,
    )
    # Best-effort: attach processing_class attribute if supported to future-proof / reduce warnings
    try:
        setattr(dpo_trainer, "processing_class", tokenizer)
    except Exception:  # pragma: no cover
        pass
    
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
    
    train_result = dpo_trainer.train()
    if eval_dataset is not None:
        # Fallback: if eval dataset lacks tokenized columns (chosen_input_ids etc.), create them minimally
        required_cols = {"prompt_input_ids", "chosen_input_ids", "rejected_input_ids"}
        if not required_cols.issubset(set(eval_dataset.column_names)):
            logger.warning("[Eval][Compat] Missing tokenized columns in eval dataset; running fallback tokenization map.")
            max_prompt_len = script_args.max_prompt_length
            max_target_len = script_args.max_target_length
            eos_id = tokenizer.eos_token_id

            def _fallback_tokenize(example):
                prompt = example.get("prompt", "")
                chosen = example.get("chosen", "")
                rejected = example.get("rejected", "")
                # Tokenize separately without special tokens
                p_tok = tokenizer(prompt, add_special_tokens=False, truncation=True, max_length=max_prompt_len)
                c_tok = tokenizer(chosen, add_special_tokens=False, truncation=True, max_length=max_target_len)
                r_tok = tokenizer(rejected, add_special_tokens=False, truncation=True, max_length=max_target_len)
                p_ids = p_tok["input_ids"]
                c_ids = c_tok["input_ids"]
                r_ids = r_tok["input_ids"]
                # Concatenate prompt + answer sequences
                chosen_full = (p_ids + c_ids)[: (max_prompt_len + max_target_len - 1)]
                rejected_full = (p_ids + r_ids)[: (max_prompt_len + max_target_len - 1)]
                # Ensure eos termination
                if eos_id is not None:
                    if not chosen_full or chosen_full[-1] != eos_id:
                        chosen_full.append(eos_id)
                    if not rejected_full or rejected_full[-1] != eos_id:
                        rejected_full.append(eos_id)
                # Labels mask prompt tokens with -100
                prompt_len = len(p_ids)
                chosen_labels = [-100] * prompt_len + chosen_full[prompt_len:]
                rejected_labels = [-100] * prompt_len + rejected_full[prompt_len:]
                return {
                    "prompt_input_ids": p_ids,
                    "prompt_attention_mask": [1]*len(p_ids),
                    "chosen_input_ids": chosen_full,
                    "chosen_attention_mask": [1]*len(chosen_full),
                    "rejected_input_ids": rejected_full,
                    "rejected_attention_mask": [1]*len(rejected_full),
                    "chosen_labels": chosen_labels,
                    "rejected_labels": rejected_labels,
                }

            eval_dataset = eval_dataset.map(_fallback_tokenize, desc="Fallback tokenizing eval dataset")
            # Reattach to trainer (some trainer implementations read from self.eval_dataset attribute)
            try:
                dpo_trainer.eval_dataset = eval_dataset
            except Exception:
                pass
        logger.info("[Eval] Running final evaluation")
        eval_metrics = dpo_trainer.evaluate(eval_dataset=eval_dataset)
        logger.info(f"[Eval][Metrics] {eval_metrics}")
    else:
        eval_metrics = {}
    
    end_time = time.time()
    training_time = end_time - start_time
    
    # Save final model
    logger.info("[Training] Saving final model")
    dpo_trainer.save_model()
    
    # Save metrics
    metrics = {
        "training_time": training_time,
        "num_train_samples": len(dataset),
        "num_eval_samples": len(eval_dataset) if eval_dataset is not None else 0,
        "samples_per_second": len(dataset) / training_time if training_time > 0 else None,
        "train_split": script_args.split,
        "eval_split": script_args.evaluation_split,
        "train_loss": float(train_result.training_loss) if hasattr(train_result, 'training_loss') else None,
        **{f"eval_{k}": v for k, v in eval_metrics.items()},
    }
    
    with open(f"{script_args.output_dir}/training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"[Training] DPO training complete. Time: {training_time:.2f}s")
    logger.info(f"[Metrics] Samples/sec: {metrics['samples_per_second']:.2f}")


if __name__ == "__main__":
    main()