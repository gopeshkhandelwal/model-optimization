# copy from https://github.com/huggingface/trl/blob/v0.7.6/examples/research_projects/stack_llama_2/scripts/dpo_llama2.py, enable it for Gaudi2
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import os

import torch
import warnings
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from transformers.integrations.deepspeed import (
    is_deepspeed_available,
)

from optimum.habana import GaudiConfig
from optimum.habana.trl import GaudiDPOConfig, GaudiDPOTrainer
from optimum.habana.utils import set_seed


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    # data parameters
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})

    # training parameters
    model_name_or_path: Optional[str] = field(
        default="../sft/results/final_checkpoint",
        metadata={"help": "the location of the SFT model name or path"},
    )
    tokenizer_name_or_path: Optional[str] = field(
        default="meta-llama/Llama-2-7b-hf",
        metadata={"help": "the location of the SFT model name or path"},
    )
    learning_rate: Optional[float] = field(default=5e-4, metadata={"help": "optimizer learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    warmup_steps: Optional[int] = field(default=100, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})

    per_device_train_batch_size: Optional[int] = field(default=1, metadata={"help": "train batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "eval batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=False, metadata={"help": "whether to use gradient checkpointing"}
    )
    gradient_checkpointing_use_reentrant: Optional[bool] = field(
        default=False, metadata={"help": "whether to use reentrant for gradient checkpointing"}
    )

    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=16, metadata={"help": "the lora r parameter (raised for higher capacity)"})
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        metadata={"help": "Target modules for the LoRA method (expanded for better adaptation)."},
    )
    max_prompt_length: Optional[int] = field(default=512, metadata={"help": "the maximum prompt length"})
    max_length: Optional[int] = field(default=1024, metadata={"help": "the maximum sequence length"})
    max_steps: Optional[int] = field(default=30, metadata={"help": "max number of training steps (small default for fast demo)"})
    logging_steps: Optional[int] = field(default=10, metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = field(default=100, metadata={"help": "the saving frequency"})
    eval_steps: Optional[int] = field(default=100, metadata={"help": "the evaluation frequency"})

    output_dir: Optional[str] = field(default="./results", metadata={"help": "the output directory"})
    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})

    # Merge options (parity with SFT script for end-to-end artifact convenience)
    merge_adapter_after_train: Optional[bool] = field(
        default=True,
        metadata={"help": "If True, merge trained LoRA adapter into base weights after DPO training"},
    )
    merged_output_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Destination directory for merged model (defaults to <output_dir>_merged)"},
    )
    merge_overwrite: Optional[bool] = field(
        default=False,
        metadata={"help": "If True overwrite merged_output_dir if it already exists"},
    )

    # Pair quality filtering (stronger separation & length to avoid trivial short answers)
    min_chosen_chars: Optional[int] = field(
        default=60,
        metadata={"help": "Require chosen response to have at least this many characters"},
    )
    min_score_delta: Optional[float] = field(
        default=1.5,
        metadata={"help": "Minimum (best_score - worst_score) needed to keep a pair"},
    )

    # instrumentation
    sanity_check: Optional[bool] = field(default=False, metadata={"help": "only train on 1000 samples"})
    report_to: Optional[str] = field(
        default="none",
        metadata={
            "help": 'Integrations to report to (e.g. "wandb", "tensorboard", "none"). Default now "none" to avoid requiring extra installs.'
        },
    )
    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )
    seed: Optional[int] = field(
        default=0, metadata={"help": "Random seed that will be set at the beginning of training."}
    )
    deepspeed: Optional[str] = field(default=None, metadata={"help": "the deepspeed json config file"})
    num_workers: Optional[int] = field(default=None, metadata={"help": "the number of workers to map the data"})


def get_ultrafeedback_pairs(
    start: int = 0,
    sample_limit: int = 1000,
    num_proc: int = 8,
    max_prompt_length: int = 512,
    max_total_length: int = 1024,
    min_chosen_chars: int = 60,
    min_score_delta: float = 1.5,
) -> Dataset:
    """Construct (prompt, chosen, rejected) pairs from openbmb/UltraFeedback.

    Strategy:
      * Load slice of the single 'train' split using HF slice syntax
      * For each instruction, score each completion by averaging available numeric ratings
      * Select best (chosen) and worst (rejected) response
      * Filter out degenerate or overly short/long samples
    """
    end = start + sample_limit
    slice_str = f"train[{start}:{end}]"
    raw = load_dataset("openbmb/UltraFeedback", split=slice_str)
    original_columns = raw.column_names

    def score_completion(comp) -> float:
        ann = comp.get("annotations", {})
        ratings = []
        for aspect, value in ann.items():
            if isinstance(value, list):
                for item in value:
                    r = item.get("Rating") if isinstance(item, dict) else None
                    if r is not None:
                        try:
                            ratings.append(int(r))
                        except Exception:
                            pass
            elif isinstance(value, dict):
                r = value.get("Rating")
                if r is not None:
                    try:
                        ratings.append(int(r))
                    except Exception:
                        pass
        return sum(ratings) / len(ratings) if ratings else 0.0

    def map_batch(batch) -> Dict[str, List[str]]:
        prompts: List[str] = []
        chosens: List[str] = []
        rejecteds: List[str] = []
        for instr, completions in zip(batch.get("instruction", []), batch.get("completions", [])):
            if not isinstance(instr, str) or not isinstance(completions, list) or len(completions) < 2:
                continue
            scored = []
            for c in completions:
                if not isinstance(c, dict):
                    continue
                resp = c.get("response", "")
                if not isinstance(resp, str) or len(resp.strip()) < 20:
                    continue
                s = score_completion(c)
                length_bonus = 0.15 * (len(resp.strip()) / 100)  # reward more substantive answers modestly
                composite = s + length_bonus
                scored.append((composite, resp, s))
            if len(scored) < 2:
                continue
            scored.sort(key=lambda x: x[0], reverse=True)
            best_score, best_resp, raw_best = scored[0]
            worst_score, worst_resp, raw_worst = scored[-1]
            if best_resp.strip() == worst_resp.strip():
                continue
            # Enforce minimum quality & separation
            if len(best_resp.strip()) < min_chosen_chars:
                continue
            if (best_score - worst_score) < min_score_delta:
                continue
            # Additional constraint: prefer pairs where chosen answer is meaningfully longer than rejected
            if len(best_resp) - len(worst_resp) < 40:
                continue
            prompt_text = instr.strip() + "\n\nAnswer:"
            if len(prompt_text) > max_prompt_length:
                continue
            if (len(prompt_text) + len(best_resp)) > max_total_length or (len(prompt_text) + len(worst_resp)) > max_total_length:
                continue
            prompts.append(prompt_text)
            chosens.append(best_resp)
            rejecteds.append(worst_resp)
        return {"prompt": prompts, "chosen": chosens, "rejected": rejecteds}

    ds = raw.map(map_batch, batched=True, num_proc=num_proc, remove_columns=original_columns)
    ds = ds.filter(lambda x: len(x["prompt"]) > 0 and len(x["chosen"]) > 0 and len(x["rejected"]) > 0)
    return ds


if __name__ == "__main__":
    # Suppress repetitive deprecation warnings about Trainer.tokenizer
    warnings.filterwarnings(
        "ignore",
        message="Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.",
        category=UserWarning,
    )
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    # Determine reporting integrations (disable by default; enable wandb only if requested)
    env_enable_wandb = os.environ.get("ENABLE_WANDB")
    report_to = script_args.report_to
    if env_enable_wandb and report_to in ("none", "None", ""):
        report_to = "wandb"
    # Normalize disabling values to empty list for HF Trainer
    if report_to in ("none", "None", "", None):
        report_to = []

    # 1. initialize training arguments:
    training_args = GaudiDPOConfig(
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        max_steps=script_args.max_steps,
        logging_steps=script_args.logging_steps,
        save_steps=script_args.save_steps,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        learning_rate=script_args.learning_rate,
        eval_strategy="steps",
        eval_steps=script_args.eval_steps,
        output_dir=script_args.output_dir,
    report_to=report_to,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_steps=script_args.warmup_steps,
        optim=script_args.optimizer_type,
        bf16=True,
        remove_unused_columns=False,
        run_name="dpo_llama2",
        gradient_checkpointing_kwargs={"use_reentrant": script_args.gradient_checkpointing_use_reentrant},
        use_habana=True,
        use_lazy_mode=True,
        use_hpu_graphs_for_training=not script_args.gradient_checkpointing and (not script_args.deepspeed),
        use_hpu_graphs_for_inference=not script_args.deepspeed,
        seed=script_args.seed,
        deepspeed=script_args.deepspeed,
        overwrite_output_dir=True,
    )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    low_cpu_mem_usage = True
    if is_deepspeed_available():
        from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

        if is_deepspeed_zero3_enabled():
            low_cpu_mem_usage = False

    # 2. load a pretrained model
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        low_cpu_mem_usage=low_cpu_mem_usage,
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model.config.use_fused_rope = False

    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    model_ref = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        low_cpu_mem_usage=low_cpu_mem_usage,
        torch_dtype=torch.bfloat16,
    )
    model_ref.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    # 3. Load UltraFeedback pairs from train split slices
    train_dataset = get_ultrafeedback_pairs(
        start=0,
        sample_limit=1000,
        num_proc=script_args.num_workers or 8,
        max_prompt_length=script_args.max_prompt_length,
        max_total_length=script_args.max_length,
        min_chosen_chars=script_args.min_chosen_chars,
        min_score_delta=script_args.min_score_delta,
    )
    # 4. Evaluation slice (non-overlapping range)
    eval_dataset = get_ultrafeedback_pairs(
        start=1200,
        sample_limit=200,
        num_proc=script_args.num_workers or 8,
        max_prompt_length=script_args.max_prompt_length,
        max_total_length=script_args.max_length,
        min_chosen_chars=script_args.min_chosen_chars,
        min_score_delta=script_args.min_score_delta,
    )
    try:
        print(f"[UltraFeedback][Dataset] train_pairs={len(train_dataset)} eval_pairs={len(eval_dataset)}")
    except Exception:
        pass

    peft_config = LoraConfig(
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        target_modules=script_args.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    gaudi_config = GaudiConfig()
    gaudi_config.use_fused_adam = True
    gaudi_config.use_fused_clip_norm = True

    # 5. initialize the DPO trainer
    dpo_trainer = GaudiDPOTrainer(
        model,
        model_ref,
        gaudi_config=gaudi_config,
        args=training_args,
        beta=script_args.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_prompt_length=script_args.max_prompt_length,
        max_length=script_args.max_length,
        force_use_ref_model=True,
    )
    # Assign processing_class to avoid downstream tokenizer property access
    try:
        setattr(dpo_trainer, 'processing_class', tokenizer)
    except Exception:
        pass

    # 6. train
    train_result = dpo_trainer.train()

    # 7. save
    dpo_trainer.save_model(script_args.output_dir)

    # 8. save metric
    metrics = train_result.metrics
    dpo_trainer.log_metrics("train", metrics)
    dpo_trainer.save_metrics("train", metrics)

    # 9. Optional merge of adapter for faster downstream loading
    if script_args.merge_adapter_after_train:
        try:
            from peft import PeftConfig, PeftModel
            import os
            adapter_dir = script_args.output_dir
            peft_conf = PeftConfig.from_pretrained(adapter_dir)
            # Only proceed if this is actually a PEFT adapter (safetensors or bin present)
            marker_files = [
                os.path.join(adapter_dir, f) for f in ["adapter_model.safetensors", "adapter_model.bin"]
            ]
            if not any(os.path.exists(m) for m in marker_files):
                print(f"[MergeInline][DPO] Adapter markers not found in {adapter_dir}; skipping merge.")
            else:
                base_loader = AutoModelForCausalLM
                base_fresh = base_loader.from_pretrained(
                    script_args.model_name_or_path,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.bfloat16,
                )
                merged_model = PeftModel.from_pretrained(base_fresh, adapter_dir)
                print("[MergeInline][DPO] Adapter loaded into fresh base model; merging...")
                merged_model = merged_model.merge_and_unload()
                out_dir = script_args.merged_output_dir or f"{script_args.output_dir}_merged"
                if os.path.exists(out_dir) and not script_args.merge_overwrite:
                    print(f"[MergeInline][DPO] Output dir {out_dir} exists and merge_overwrite=False -> abort merge")
                else:
                    os.makedirs(out_dir, exist_ok=True)
                    merged_model.save_pretrained(out_dir)
                    tokenizer.save_pretrained(out_dir)
                    print(f"[MergeInline][DPO] Merged model saved to {out_dir}")
        except Exception as e:
            print(f"[MergeInline][DPO] Merge failed: {e}")