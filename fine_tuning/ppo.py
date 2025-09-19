# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# /// script
# dependencies = [
#     "trl @ git+https://github.com/huggingface/trl.git",
#     "peft",
#     "trackio",
#     "kernels",
# ]
# ///

import os
import shutil
import time
from datetime import datetime

import torch
from accelerate import PartialState
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    DataCollatorWithPadding,
)

from trl import (
    ModelConfig,
    PPOConfig,
    PPOTrainer,
    ScriptArguments,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE


# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


"""
python -i examples/scripts/ppo/ppo.py \
    --dataset_name trl-internal-testing/descriptiveness-sentiment-trl-style \
    --dataset_train_split descriptiveness \
    --learning_rate 3e-6 \
    --output_dir pythia-1b-deduped-descriptiveness-sentiment-trl-style-ppo \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --total_episodes 10000 \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --missing_eos_penalty 1.0

accelerate launch --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/ppo/ppo.py \
    --dataset_name trl-internal-testing/descriptiveness-sentiment-trl-style \
    --dataset_train_split descriptiveness \
    --output_dir pythia-1b-deduped-descriptiveness-sentiment-trl-style-ppo \
    --num_ppo_epochs 1 \
    --num_mini_batches 1 \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --total_episodes 10000 \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --sft_model_path EleutherAI/pythia-1b-deduped \
    --reward_model_path EleutherAI/pythia-1b-deduped \
    --local_rollout_forward_batch_size 1 \
    --missing_eos_penalty 1.0
"""


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, PPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    # Distributed / rank context (Accelerate PartialState later for finer control)
    # We'll instantiate PartialState early just for logging meta; safe even before models.
    state_for_log = PartialState()  # may initialize distributed backend
    RANK = getattr(state_for_log, "process_index", 0)
    WORLD_SIZE = getattr(state_for_log, "num_processes", 1)
    VERBOSE_ALL = os.environ.get("LOG_ALL_RANKS", "0") == "1"

    def log(msg: str, always: bool = False):
        if always or RANK == 0 or VERBOSE_ALL:
            ts = datetime.utcnow().strftime('%H:%M:%S')
            print(f"[{ts}][R{RANK}] {msg}", flush=True)

    t_script_start = time.time()
    log(f"START PPO script world_size={WORLD_SIZE} output_dir={training_args.output_dir}")
    # Validate required model identifiers early (user sometimes ran without CLI args inside -i REPL)
    if not getattr(model_args, "model_name_or_path", None):
        log("ERROR: --model_name_or_path not provided. Example: --model_name_or_path meta-llama/Meta-Llama-3-8B", always=True)
        raise SystemExit("Missing required --model_name_or_path (pass arguments on the python command line, not inside the REPL)")
    if not getattr(training_args, "sft_model_path", None):
        training_args.sft_model_path = model_args.model_name_or_path
        log(f"INFO: defaulting --sft_model_path to {training_args.sft_model_path}")
    if not getattr(training_args, "reward_model_path", None):
        training_args.reward_model_path = model_args.model_name_or_path
        log(f"INFO: defaulting --reward_model_path to {training_args.reward_model_path}")
    # remove output_dir if exists
    shutil.rmtree(training_args.output_dir, ignore_errors=True)
    log("CLEANED existing output_dir (if any)")

    ################
    # Model & Tokenizer
    ################
    # Robust dtype resolution: older/newer TRL ModelConfig variants may not expose `dtype`.
    raw_dtype = getattr(model_args, "dtype", None)
    if raw_dtype in (None, "auto"):
        torch_dtype = None
    else:
        try:
            torch_dtype = getattr(torch, raw_dtype)
        except AttributeError:
            torch_dtype = None
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=getattr(model_args, "model_revision", None),
        attn_implementation=getattr(model_args, "attn_implementation", None),
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    # Only include torch_dtype key if resolved (avoids HF warnings when None)
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype

    t0 = time.time()
    log(f"LOADING tokenizer: {model_args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, padding_side="left", trust_remote_code=model_args.trust_remote_code
    )
    log(f"LOADED tokenizer in {time.time()-t0:.2f}s")
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE

    class FilterPromptCollator(DataCollatorWithPadding):
        """Removes raw 'prompt' strings before padding to avoid tensor conversion errors."""
        def __call__(self, features):  # type: ignore[override]
            for f in features:
                if 'prompt' in f:
                    f.pop('prompt')
            return super().__call__(features)

    data_collator = FilterPromptCollator(tokenizer=tokenizer, padding=True)
    t0 = time.time(); log(f"LOADING value_model: {training_args.reward_model_path}")
    value_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.reward_model_path, trust_remote_code=model_args.trust_remote_code, num_labels=1
    )
    log(f"LOADED value_model in {time.time()-t0:.2f}s")
    t0 = time.time(); log(f"LOADING reward_model: {training_args.reward_model_path}")
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.reward_model_path, trust_remote_code=model_args.trust_remote_code, num_labels=1
    )
    log(f"LOADED reward_model in {time.time()-t0:.2f}s")
    t0 = time.time(); log(f"LOADING policy: {training_args.sft_model_path}")
    policy = AutoModelForCausalLM.from_pretrained(
        training_args.sft_model_path, trust_remote_code=model_args.trust_remote_code
    )
    log(f"LOADED policy in {time.time()-t0:.2f}s")

    peft_config = get_peft_config(model_args)
    if peft_config is None:
        t0 = time.time(); log("LOADING reference policy (full copy)")
        ref_policy = AutoModelForCausalLM.from_pretrained(
            training_args.sft_model_path, trust_remote_code=model_args.trust_remote_code
        )
        log(f"LOADED reference policy in {time.time()-t0:.2f}s (frozen later by PPOTrainer)")
    else:
        ref_policy = None
        log("PEFT active -> no separate ref_policy loaded (PPOTrainer will handle ref sharing)")

    # Parameter statistics
    def count_params(model, trainable_only=False):
        if trainable_only:
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        return sum(p.numel() for p in model.parameters())
    log(f"PARAMS policy_total={count_params(policy):,} trainable={count_params(policy,True):,}")
    log(f"PARAMS value_model_total={count_params(value_model):,}")
    log(f"PARAMS reward_model_total={count_params(reward_model):,}")
    if ref_policy is not None:
        log(f"PARAMS ref_policy_total={count_params(ref_policy):,}")

    ################
    # Dataset
    ################
    if script_args.dataset_name in ("synthetic_math", "synthetic-math"):
        from datasets import Dataset as HFDataset
        n = int(os.environ.get("SYNTHETIC_MATH_SAMPLES", "64"))
        seed = int(os.environ.get("SYNTHETIC_MATH_SEED", "42"))
        import random; random.seed(seed)
        rows = []
        for _ in range(n):
            a = random.randint(1, 99)
            b = random.randint(1, 99)
            op = random.choice(["+", "-"])
            expr = f"{a} {op} {b}"
            rows.append({"prompt": f"Question: {expr} = ?\nAnswer:"})
        dataset = HFDataset.from_list(rows)
        log(f"DATASET synthetic_math generated rows={len(dataset)} (env SYNTHETIC_MATH_SAMPLES to change)")
    else:
        t0 = time.time(); log(f"DATASET loading: name={script_args.dataset_name} split={script_args.dataset_train_split} config={script_args.dataset_config}")
        dataset = load_dataset(
            script_args.dataset_name, name=script_args.dataset_config, split=script_args.dataset_train_split
        )
        log(f"DATASET loaded rows={len(dataset)} time={time.time()-t0:.2f}s cols={dataset.column_names if hasattr(dataset,'column_names') else 'n/a'}")
        # If 'prompt' column absent but 'question' exists (e.g. openai/gsm8k), create prompt
        if 'prompt' not in dataset.column_names and 'question' in dataset.column_names:
            log("DATASET adding 'prompt' column from 'question'")
            def _to_prompt(ex):
                q = ex['question'].strip() if isinstance(ex.get('question'), str) else str(ex.get('question'))
                return { 'prompt': f"Question: {q}\nAnswer:" }
            dataset = dataset.map(_to_prompt)
            log("DATASET 'prompt' column added")
    # Small fixed subset (train 128 / eval 16 if available)
    SMALL_TRAIN = 128
    SMALL_EVAL = 16
    total_needed = SMALL_TRAIN + SMALL_EVAL
    if len(dataset) > total_needed:
        dataset = dataset.select(range(total_needed))
        log(f"DATASET truncated to small subset total={len(dataset)} (train={SMALL_TRAIN} eval={SMALL_EVAL})")
    eval_samples = min(SMALL_EVAL, len(dataset)//5 if len(dataset)//5 > 0 else len(dataset))
    train_dataset = dataset.select(range(len(dataset) - eval_samples))
    eval_dataset = dataset.select(range(len(dataset) - eval_samples, len(dataset)))
    # Safeguard: ensure at least one training sample (tiny dataset edge cases)
    if len(train_dataset) == 0 and len(eval_dataset) > 0:
        if len(eval_dataset) > 1:
            train_dataset = dataset.select([0])
            eval_dataset = dataset.select(range(1, len(dataset)))
            log(f"DATASET tiny adjustment: moved one sample to train -> train={len(train_dataset)} eval={len(eval_dataset)}")
        else:
            train_dataset = dataset
            eval_dataset = dataset.select([])
            log(f"DATASET tiny adjustment: single-sample dataset -> train={len(train_dataset)} eval=0")
    dataset_text_field = "prompt"
    log(f"DATASET split train={len(train_dataset)} eval={len(eval_dataset)}")

    def prepare_dataset(dataset, tokenizer):
        """pre-tokenize the dataset before training; only collate during training"""

        def tokenize(element):
            outputs = tokenizer(
                element[dataset_text_field],
                padding=False,
            )
            return {"input_ids": outputs["input_ids"]}

        # Keep 'prompt' column (remove others) so original text remains for PPO queries
        cols_to_remove = [c for c in dataset.column_names if c != dataset_text_field]
        return dataset.map(
            tokenize,
            batched=True,
            remove_columns=cols_to_remove,
            num_proc=training_args.dataset_num_proc,
        )

    # Compute that only on the main process for faster data processing.
    # see: https://github.com/huggingface/trl/pull/1255
    with PartialState().local_main_process_first():
        log("TOKENIZING train dataset ...")
        t0 = time.time()
        train_dataset = prepare_dataset(train_dataset, tokenizer)
        log(f"TOKENIZED train in {time.time()-t0:.2f}s")
        log("TOKENIZING eval dataset ...")
        t0 = time.time()
        eval_dataset = prepare_dataset(eval_dataset, tokenizer)
        log(f"TOKENIZED eval in {time.time()-t0:.2f}s")

    # Quick token length stats (sample first 128 examples)
    try:
        sample_lens = [len(ex["input_ids"]) for ex in train_dataset.select(range(min(128, len(train_dataset))))]
        if sample_lens:
            import math
            avg_len = sum(sample_lens)/len(sample_lens)
            p95 = sorted(sample_lens)[int(0.95*len(sample_lens))-1]
            max_len = max(sample_lens)
            log(f"STATS prompt_token_length avg={avg_len:.1f} p95={p95} max={max_len}")
    except Exception as e:
        log(f"STATS token length collection failed: {e}")

    ################
    # Training
    ################
    log("PPO TRAINER constructing ...")
    t0 = time.time()
    trainer = PPOTrainer(
        args=training_args,
        processing_class=tokenizer,
        model=policy,
        ref_model=ref_policy,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    data_collator=data_collator,
    )
    log(f"PPO TRAINER ready in {time.time()-t0:.2f}s")

    # Wrap trainer.step for granular per-update logging if present
    if hasattr(trainer, "step"):
        original_step = trainer.step
        def logged_step(queries, responses, scores, *a, **kw):
            t_step = time.time()
            stats = original_step(queries, responses, scores, *a, **kw)
            dt = time.time() - t_step
            try:
                reward_mean = stats.get('rewards/mean', stats.get('reward_mean', None))
            except Exception:
                reward_mean = None
            kl_val = stats.get('kl', stats.get('objective/kl', None)) if isinstance(stats, dict) else None
            log(f"STEP dt={dt:.2f}s reward_mean={reward_mean} kl={kl_val} stats_keys={list(stats.keys()) if isinstance(stats, dict) else 'n/a'}")
            return stats
        trainer.step = logged_step  # type: ignore
        log("ENABLED per-step logging wrapper")
    else:
        log("WARNING trainer has no .step attribute; per-step logging unavailable")

    log("TRAIN starting trainer.train() ...")
    t0 = time.time()
    trainer.train()
    log(f"TRAIN finished wall={time.time()-t0:.1f}s")

    # Save and push to hub
    log("SAVE saving model ...")
    trainer.save_model(training_args.output_dir)
    log("SAVE complete")
    if training_args.push_to_hub:
        log("PUSH pushing to hub ...")
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
        log("PUSH done")
    log("GEN generating sample completions ...")
    t0 = time.time()
    trainer.generate_completions()
    log(f"GEN done in {time.time()-t0:.2f}s")
    log(f"END total_wall={time.time()-t_script_start:.1f}s")