# Adapted from https://github.com/huggingface/optimum-habana/tree/v1.16.0/examples/trl
# Customized and enabled for Gaudi3
import logging
import warnings
import math
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import transformers
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from transformers.integrations.deepspeed import (
    is_deepspeed_available,
)

from optimum.habana import GaudiConfig
from optimum.habana.trl import GaudiSFTConfig, GaudiSFTTrainer
from optimum.habana.utils import set_seed
from logging_utils import setup_logging


logger = logging.getLogger(__name__)


@dataclass
class ScriptArguments:
    model_name_or_path: Optional[str] = field(default=None, metadata={"help": "the model name (REQUIRED)"})
    dataset_name: Optional[str] = field(default=None, metadata={"help": "the dataset name"})
    use_peft: Optional[bool] = field(default=True, metadata={"help": "whether to use peft"})
    subset: Optional[str] = field(default=None, metadata={"help": "the subset (local data_dir) if applicable; None for remote HF datasets"})
    split: Optional[str] = field(default="train", metadata={"help": "the split to use"})
    size_valid_set: Optional[int] = field(default=4000, metadata={"help": "the size of the validation set"})
    streaming: Optional[bool] = field(default=True, metadata={"help": "whether to stream the dataset"})
    shuffle_buffer: Optional[int] = field(default=5000, metadata={"help": "the shuffle buffer size"})
    num_workers: Optional[int] = field(default=4, metadata={"help": "the number of workers"})
    num_buckets: Optional[int] = field(default=-1, metadata={"help": "whether to use bucketing for SFTTrainer"})
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    use_flash_attention: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use Habana flash attention for fine-tuning."}
    )
    flash_attention_recompute: Optional[bool] = field(
        default=False, metadata={"help": "Whether to enable recompute in Habana flash attention for fine-tuning."}
    )
    flash_attention_causal_mask: Optional[bool] = field(
        default=False, metadata={"help": "Whether to enable causal mask in Habana flash attention for fine-tuning."}
    )

    # LoraConfig
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})
    lora_target_modules: List[str] = field(
        default_factory=lambda: None,
        metadata={"help": "Target modules for the LoRA method."},
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
        default=True, metadata={"help": "If True and PEFT used, merge LoRA adapter into base model after training."}
    )
    merged_output_dir: Optional[str] = field(
        default=None, metadata={"help": "Directory to save merged model (defaults to <output_dir>_merged)."}
    )
    merge_overwrite: bool = field(
        default=False, metadata={"help": "If True overwrite merged_output_dir if it exists."}
    )
    demo_fast: bool = field(
        default=False, metadata={"help": "Enable ultra-fast demo mode: shrink dataset & steps for <20m pipeline."}
    )


if __name__ == "__main__":
    # Silence deprecated Trainer.tokenizer warnings
    warnings.filterwarnings(
        "ignore",
        message="Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.",
        category=UserWarning,
    )
    parser = HfArgumentParser((ScriptArguments, GaudiSFTConfig))
    script_args, training_args = parser.parse_args_into_dataclasses()
    if not script_args.model_name_or_path:
        raise ValueError("--model_name_or_path is required (no default). Provide a model path or repo id.")
    if not (hasattr(torch, 'hpu') and torch.hpu.is_available()):
        raise RuntimeError('[HPU][Required] Habana HPU not available. SFT script enforces HPU-only execution.')
    if getattr(training_args, 'use_habana', False) is False:
        logging.getLogger(__name__).warning('[HPU][Override] Forcing use_habana=True (Always Use HPU policy)')
        training_args.use_habana = True
    setup_logging()
    
    # Validate required arguments
    if not script_args.model_name_or_path:
        raise ValueError("--model_name_or_path is required. Please specify the base model path.")
    
    # Demo-fast overrides applied early (before logging full args)
    if script_args.demo_fast:
        logger.warning("[DemoFast][SFT] Activating ultra-fast overrides (<20m goal)")
        # Reduce training steps if large
        if training_args.max_steps == -1 or training_args.max_steps > 20:
            logger.info("[DemoFast][SFT] Setting max_steps=20")
            training_args.max_steps = 20
        # Force single epoch semantics when steps specified
        if getattr(training_args,'num_train_epochs',1) > 1:
            training_args.num_train_epochs = 1
            logger.info("[DemoFast][SFT] Forcing num_train_epochs=1")
        # Reduce warmup
        if getattr(training_args,'warmup_steps',0) > 2:
            training_args.warmup_steps = 2
            logger.info("[DemoFast][SFT] Setting warmup_steps=2")
        # Lower logging cadence
        if getattr(training_args,'logging_steps',10) > 5:
            training_args.logging_steps = 5
            logger.info("[DemoFast][SFT] Setting logging_steps=5")
        # Min gradient accumulation
        if getattr(training_args,'gradient_accumulation_steps',1) > 1:
            training_args.gradient_accumulation_steps = 1
            logger.info("[DemoFast][SFT] Setting gradient_accumulation_steps=1")
        # Limit LoRA rank
        if script_args.use_peft and script_args.lora_r > 16:
            logger.info("[DemoFast][SFT] Reducing lora_r to 16")
            script_args.lora_r = 16
        # Reduce shuffle buffer if streaming
        if script_args.streaming and script_args.shuffle_buffer > 1000:
            script_args.shuffle_buffer = 1000
            logger.info("[DemoFast][SFT] Setting shuffle_buffer=1000")
        # Shrink validation set if huge
        if script_args.size_valid_set > 1000:
            script_args.size_valid_set = 500
            logger.info("[DemoFast][SFT] Setting size_valid_set=500")
        # Tag merged output dir for clarity
        if script_args.merged_output_dir is None:
            script_args.merged_output_dir = "./sft_sanity_merged"
    logger.info(f"ScriptArguments: {script_args}")
    logger.info(f"TrainingArguments: {training_args}")
    if script_args.use_peft:
        logger.info("[IF] use_peft == True -> configuring LoRA")
        peft_config = LoraConfig(
            r=script_args.lora_r,
            lora_alpha=script_args.lora_alpha,
            lora_dropout=script_args.lora_dropout,
            target_modules=script_args.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
    else:
        logger.info("[IF] use_peft == False -> full fine-tune (no LoRA)")
        peft_config = None

    if training_args.group_by_length and training_args.packing:
        logger.warning("[IF] group_by_length AND packing are both True -> raising ValueError")
        raise ValueError("Cannot use both packing and group by length")

    set_seed(training_args.seed)

    def chars_token_ratio(dataset, tokenizer, nb_examples=400):
        """
        Estimate the average number of characters per token in the dataset.
        """
        total_characters, total_tokens = 0, 0
        for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
            text = prepare_sample_text(example)
            total_characters += len(text)
            if tokenizer.is_fast:
                total_tokens += len(tokenizer(text).tokens())
            else:
                total_tokens += len(tokenizer.tokenize(text))

        return total_characters / total_tokens

    def prepare_sample_text(example):
        """Prepare the text from a sample of supported datasets.

        For stack-exchange-paired: uses question + response_j.
        For ultrafeedback_binarized: use prompt + chosen (preferred answer).
        Falls back to joining all string fields if expected keys absent.
        """
        if 'question' in example and 'response_j' in example:
            return f"Question: {example['question']}\n\nAnswer: {example['response_j']}"
        if 'prompt' in example and 'chosen' in example:
            return f"{example['prompt'].rstrip()}\n{example['chosen']}"
        # Fallback: concatenate first two string fields
        for k in example:
            if isinstance(example[k], str):
                base = example[k]
                break
        else:
            base = ""
        extras = [v for k,v in example.items() if isinstance(v,str) and v is not base]
        return base + ("\n" + "\n".join(extras) if extras else "")

    def create_datasets(tokenizer, args, seed=None):
        if args.dataset_name:
            logger.info(f"[IF] dataset_name provided -> loading {args.dataset_name}")
            load_kwargs = {
                'split': args.split,
                'token': script_args.token,
                'num_proc': args.num_workers if not args.streaming else None,
                'streaming': args.streaming,
            }
            # Only pass data_dir if explicitly provided and not None (for local custom datasets)
            if args.subset not in (None, 'None', ''):
                load_kwargs['data_dir'] = args.subset
            dataset = load_dataset(args.dataset_name, **load_kwargs)
            # UltraFeedback: map to single (instruction, best_response) for SFT
            if args.dataset_name == 'openbmb/UltraFeedback':
                logger.info('[UltraFeedback][SFT] Mapping completions to best scored response')
                def _score_completion(comp):
                    ann = comp.get('annotations', {})
                    ratings = []
                    for aspect, val in ann.items():
                        if isinstance(val, list):
                            for item in val:
                                if isinstance(item, dict) and 'Rating' in item:
                                    try:
                                        ratings.append(int(item['Rating']))
                                    except Exception:
                                        pass
                        elif isinstance(val, dict) and 'Rating' in val:
                            try:
                                ratings.append(int(val['Rating']))
                            except Exception:
                                pass
                    return sum(ratings)/len(ratings) if ratings else 0.0
                def _map_batch(batch):
                    out_instr = []
                    out_resp = []
                    for instr, comps in zip(batch.get('instruction', []), batch.get('completions', [])):
                        if not isinstance(instr, str) or not isinstance(comps, list) or len(comps) == 0:
                            continue
                        scored = []
                        for c in comps:
                            if not isinstance(c, dict):
                                continue
                            resp = c.get('response', '')
                            if not isinstance(resp, str) or len(resp.strip()) < 10:
                                continue
                            scored.append((_score_completion(c), resp))
                        if len(scored) == 0:
                            continue
                        scored.sort(key=lambda x: x[0], reverse=True)
                        best_resp = scored[0][1]
                        out_instr.append(instr)
                        out_resp.append(best_resp)
                    return {'question': out_instr, 'response_j': out_resp}
                original_cols = dataset.column_names
                dataset = dataset.map(_map_batch, batched=True, remove_columns=original_cols)
                logger.info(f"[UltraFeedback][SFT] After mapping rows={len(dataset)} cols={dataset.column_names}")
        else:
            logger.error("[IF] dataset_name missing -> raising ValueError")
            raise ValueError("No dataset_name")
        if args.streaming:
            logger.info("[IF] streaming == True -> using .take/.skip and shuffle")
            logger.info("Loading the dataset in streaming mode")
            valid_data = dataset.take(args.size_valid_set)
            train_data = dataset.skip(args.size_valid_set)
            train_data = train_data.shuffle(buffer_size=args.shuffle_buffer, seed=seed)
            if script_args.demo_fast:
                # For streaming, further restrict training iterable
                fast_limit = 50
                logger.info(f"[DemoFast][SFT] Limiting streaming train_data to first {fast_limit} samples")
                train_data = train_data.take(fast_limit)
                valid_data = valid_data.take(min(script_args.size_valid_set, 50))
        else:
            logger.info("[IF] streaming == False -> train_test_split path")
            dataset = dataset.train_test_split(test_size=args.validation_split_percentage * 0.01, seed=seed)
            train_data = dataset["train"]
            valid_data = dataset["test"]
            logger.info(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")
            if script_args.demo_fast:
                # Subsample train & valid sets
                import random
                fast_train_limit = min(80, len(train_data))
                fast_valid_limit = min(20, len(valid_data))
                logger.info(f"[DemoFast][SFT] Subsampling train->{fast_train_limit} valid->{fast_valid_limit}")
                train_indices = list(range(len(train_data)))[:fast_train_limit]
                valid_indices = list(range(len(valid_data)))[:fast_valid_limit]
                train_data = train_data.select(train_indices)
                valid_data = valid_data.select(valid_indices)
        if args.dataset_name in ("lvwerra/stack-exchange-paired", "openbmb/UltraFeedback"):
            logger.info(f"[IF] dataset_name {args.dataset_name} -> computing chars/token ratio and using formatting func")
            try:
                chars_per_token = chars_token_ratio(train_data, tokenizer)
                logger.info(f"[DatasetStats] chars/token ratio: {chars_per_token:.2f}")
            except Exception as e:
                logger.warning(f"[DatasetStats] Failed chars/token ratio estimation: {e}")
            formating_func = prepare_sample_text
        else:
            logger.info(f"[IF] dataset_name {args.dataset_name} not in specialized list -> no formatting func")
            formating_func = None
        return train_data, valid_data, formating_func

    low_cpu_mem_usage = True
    if is_deepspeed_available():
        logger.info("[IF] Deepspeed available -> checking Zero3")
        from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

        if is_deepspeed_zero3_enabled():
            logger.info("[IF] DeepSpeed Zero3 enabled -> disabling low_cpu_mem_usage")
            low_cpu_mem_usage = False

    base_model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        low_cpu_mem_usage=low_cpu_mem_usage,
        torch_dtype=torch.bfloat16,
        token=script_args.token,
    )
    mt = getattr(base_model.config, 'model_type', '')
    if mt.startswith('gemma3'):
        try:
            base_model.config.attn_implementation = 'eager'
        except Exception:
            pass

    # --- Parameter statistics before (and later after) PEFT injection ---
    def _param_stats(model):
        total = 0
        trainable = 0
        for p in model.parameters():
            n = p.numel()
            total += n
            if p.requires_grad:
                trainable += n
        pct = (trainable / total * 100) if total else 0.0
        return total, trainable, pct

    tot, trn, pct = _param_stats(base_model)
    logger.info(f"[Model Params][Base] total={tot:,} trainable={trn:,} ({pct:.4f}%)")

    base_model.config.use_cache = False
    if not script_args.use_flash_attention and (
        script_args.flash_attention_recompute or script_args.flash_attention_recompute
    ):
        logger.warning("[IF] flash attention recompute flags set while use_flash_attention is False -> assert")
        assert "Need to enable use_flash_attention"
    base_model.generation_config.use_flash_attention = script_args.use_flash_attention
    base_model.generation_config.flash_attention_recompute = script_args.flash_attention_recompute
    base_model.generation_config.flash_attention_causal_mask = script_args.flash_attention_causal_mask
    if script_args.use_flash_attention:
        logger.info("[IF] use_flash_attention == True -> flash attention settings applied")
    else:
        logger.info("[IF] use_flash_attention == False -> skipping flash attention optimizations")

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    train_dataset, eval_dataset, formatting_func = create_datasets(tokenizer, script_args, seed=training_args.seed)

    gaudi_config = GaudiConfig()
    gaudi_config.use_fused_adam = False
    gaudi_config.use_fused_clip_norm = False
    if training_args.do_train:
        logger.info("[IF] do_train == True -> beginning training loop")
        trainer = GaudiSFTTrainer(
            model=base_model,
            gaudi_config=gaudi_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=peft_config,
            tokenizer=tokenizer,
            args=training_args,
            formatting_func=formatting_func,
            num_buckets=script_args.num_buckets,
        )
        # Provide processing_class attribute expected by newer Trainer interface
        try:
            setattr(trainer, 'processing_class', tokenizer)
        except Exception:
            pass

        # If PEFT active, trainer.model now wrapped; print updated stats
        try:
            wrapped_tot, wrapped_trn, wrapped_pct = _param_stats(trainer.model)
            logger.info(f"[Model Params][After Trainer Init] total={wrapped_tot:,} trainable={wrapped_trn:,} ({wrapped_pct:.4f}%)")
            if peft_config is not None:
                # Highlight LoRA target modules used
                logger.info(f"[LoRA] Target modules: {peft_config.target_modules}")
                logger.info(f"[LoRA] r={getattr(peft_config,'r',None)} alpha={getattr(peft_config,'lora_alpha',None)} dropout={getattr(peft_config,'lora_dropout',None)}")
        except Exception as e:
            logger.warning(f"[Warn] Could not compute PEFT parameter stats: {e}")
        train_result = trainer.train()
        trainer.save_model(training_args.output_dir)
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        logger.info("[INFO] Training completed and model saved")
        # Optional inline merge
        if script_args.merge_adapter_after_train:
            if not script_args.use_peft:
                logger.warning("[MergeInline] merge_adapter_after_train=True but use_peft=False -> skipping merge")
            else:
                try:
                    from peft import PeftConfig, PeftModel
                    import os
                    adapter_dir = training_args.output_dir
                    peft_conf = PeftConfig.from_pretrained(adapter_dir)
                    logger.info(f"[MergeInline] Loaded PEFT config task_type={peft_conf.task_type}")
                    base_loader = AutoModelForCausalLM
                    base_fresh = base_loader.from_pretrained(
                        script_args.model_name_or_path,
                        low_cpu_mem_usage=low_cpu_mem_usage,
                        torch_dtype=torch.bfloat16,
                        token=script_args.token,
                    )
                    merged_model = PeftModel.from_pretrained(base_fresh, adapter_dir)
                    logger.info("[MergeInline] Adapter loaded into fresh base model; merging...")
                    merged_model = merged_model.merge_and_unload()
                    out_dir = script_args.merged_output_dir or f"{training_args.output_dir}_merged"
                    if os.path.exists(out_dir) and not script_args.merge_overwrite:
                        logger.warning(f"[MergeInline] Output dir {out_dir} exists and merge_overwrite=False -> abort merge")
                    else:
                        os.makedirs(out_dir, exist_ok=True)
                        merged_model.save_pretrained(out_dir)
                        tokenizer.save_pretrained(out_dir)
                        logger.info(f"[MergeInline] Merged model saved to {out_dir}")
                except Exception as e:
                    logger.exception(f"[MergeInline] Failed to merge adapter inline: {e}")
    else:
        logger.info("[IF] do_train == False -> skipping training")

    # Evaluation
    if training_args.do_eval:
        logger.info("[IF] do_eval == True -> running evaluation")
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        if isinstance(eval_dataset, torch.utils.data.IterableDataset):
            eval_dataset = list(eval_dataset)

        metrics["eval_samples"] = len(eval_dataset)

        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        logger.info("[INFO] Evaluation complete")
    else:
        logger.info("[IF] do_eval == False -> skipping evaluation")
