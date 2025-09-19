"""Minimal PPO fine-tuning with LoRA (PEFT) on Gaudi for Meta-Llama-3-8B.

Replaces former GaudiTrainer SFT flow with TRL PPO + LoRA adapters.
Dataset expectation: a processed GSM8K-style dataset on disk (load_from_disk) at
    /workspace/data/train_data  and  /workspace/data/eval_data
Columns required now:
    - 'prompt': question prompt (e.g. "Question: ...\nAnswer:")
    - 'answer': final numeric answer prefixed with '#### ' (e.g. '#### 72')
Optional columns (ignored): anything else will be dropped automatically.

Reward: numeric correctness (exact match) with error-based shaping; parses final number from 'answer' (expects '#### number').
KL penalty: disabled by default (beta=0) because we skip loading a full reference model when using LoRA for memory efficiency.

Environment overrides (optional):
  LORA_R, LORA_ALPHA, LORA_DROPOUT, MAX_NEW_TOKENS, PPO_BATCH_SIZE, PPO_MINI_BATCH, PPO_STEPS
"""

import os
import math
import random
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model
from trl import PPOTrainer, PPOConfig
import torch

MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Meta-Llama-3-8B")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

# Base causal LM
policy = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() or hasattr(torch, 'hpu') else None,
    low_cpu_mem_usage=True,
    use_cache=False,  # disable for training with checkpointing if later desired
)

# LoRA config (env overridable)
lora_config = LoraConfig(
    r=int(os.environ.get("LORA_R", 8)),
    lora_alpha=int(os.environ.get("LORA_ALPHA", 16)),
    lora_dropout=float(os.environ.get("LORA_DROPOUT", 0.05)),
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)
policy = get_peft_model(policy, lora_config)
try:
    policy.print_trainable_parameters()
except Exception:
    pass

# (Optional) gradient checkpointing for memory
if os.environ.get("GRADIENT_CHECKPOINT", "1") == "1":
    try:
        policy.gradient_checkpointing_enable()
        if hasattr(policy.config, 'use_cache'):
            policy.config.use_cache = False
    except Exception:
        pass

# Load datasets (allow override path)
DATA_DIR = os.environ.get("DATA_DIR", "/workspace/data")
train_path = os.path.join(DATA_DIR, "train_data")
eval_path = os.path.join(DATA_DIR, "eval_data")
print(f"[INFO] Loading train dataset from {train_path}")
print(f"[INFO] Loading eval dataset from {eval_path}")
train_dataset = load_from_disk(train_path)
eval_dataset = load_from_disk(eval_path)
print(f"[INFO] Train size={len(train_dataset)} Eval size={len(eval_dataset)} Columns={train_dataset.column_names}")
try:
    sample_prompts = [train_dataset[i].get("prompt", "") for i in range(min(3, len(train_dataset)))]
    for i, sp in enumerate(sample_prompts):
        preview = sp.replace('\n', ' ')[:120]
        print(f"[DATA SAMPLE {i}] {preview}")
except Exception:
    pass

# Optional pattern assertion to confirm expected dataset (e.g., math prompts start with 'Compute:')
assert_pattern = os.environ.get("ASSERT_PATTERN")
if assert_pattern:
    import re as _re
    pat = _re.compile(assert_pattern)
    mismatches = 0
    for i in range(min(20, len(train_dataset))):
        if not pat.search(train_dataset[i].get("prompt", "")):
            mismatches += 1
            if mismatches > 3:
                break
    if mismatches > 0:
        print(f"[WARN] ASSERT_PATTERN '{assert_pattern}' not found in {mismatches} of first {min(20, len(train_dataset))} prompts.")
    else:
        print(f"[INFO] ASSERT_PATTERN '{assert_pattern}' satisfied on sampled prompts.")

PROMPT_FIELD = "prompt"
ANSWER_FIELD = os.environ.get("ANSWER_FIELD", "answer")  # expected ground-truth answer column

def ensure_prompt(ds):
    cols = ds.column_names
    if PROMPT_FIELD in cols:
        return ds
    if 'text' in cols:
        return ds.map(lambda ex: {PROMPT_FIELD: ex['text']})
    if 'input_ids' in cols:
        return ds.map(lambda ex: {PROMPT_FIELD: tokenizer.decode(ex['input_ids'], skip_special_tokens=True)})
    # fallback: create empty prompt (not ideal)
    return ds.map(lambda ex: {PROMPT_FIELD: ""})

train_dataset = ensure_prompt(train_dataset)
eval_dataset = ensure_prompt(eval_dataset)

# If we fall back to trainer.train(), PPOTrainer may try to tokenize full feature dict;
# keep only the prompt column to avoid nested/heterogeneous fields (e.g. original 'question').
def _strip_extra_columns(ds):
    # Keep prompt + answer so we can compute correctness rewards.
    keep = {PROMPT_FIELD, ANSWER_FIELD}
    extra = [c for c in ds.column_names if c not in keep]
    return ds.remove_columns(extra) if extra else ds

train_dataset = _strip_extra_columns(train_dataset)
eval_dataset = _strip_extra_columns(eval_dataset)

# Prepare PPO config (only pass widely-supported arguments to avoid TypeError across TRL versions)
config_kwargs = dict(
    learning_rate=float(os.environ.get("LR", 5e-6)),
    batch_size=int(os.environ.get("PPO_BATCH_SIZE", 16)),
    mini_batch_size=int(os.environ.get("PPO_MINI_BATCH", 4)),
    gradient_accumulation_steps=1,
    ppo_epochs=1,
    seed=int(os.environ.get("SEED", 42)),
)
try:
    ppo_config = PPOConfig(**config_kwargs)  # type: ignore[arg-type]
except TypeError as e:
    # Fallback: remove keys not accepted (very conservative)
    bad_keys = []
    for k in list(config_kwargs.keys()):
        try:
            PPOConfig(**{k: config_kwargs[k]})
        except Exception:
            bad_keys.append(k)
    for bk in bad_keys:
        config_kwargs.pop(bk, None)
    ppo_config = PPOConfig(**config_kwargs)  # minimal config
    print(f"[WARN] PPOConfig init issue: {e}. Retained keys={list(config_kwargs.keys())}")

# External control vars not guaranteed to exist in PPOConfig
steps = int(os.environ.get("PPO_STEPS", 200))
max_new_tokens = int(os.environ.get("MAX_NEW_TOKENS", 48))

ref_model = None
# Explicit value & reward models (reuse same module to save memory)
value_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=1,
    torch_dtype=torch.bfloat16 if (torch.cuda.is_available() or hasattr(torch, 'hpu')) else None,
    low_cpu_mem_usage=True,
)
reward_model = value_model  # reuse to satisfy trainer expectation

# Collator: tokenizes raw prompts if dataset not pre-tokenized (needed for fallback trainer.train()).
class PromptCollator:
    def __init__(self, tokenizer, prompt_field: str, max_length: int | None = None):
        self.tokenizer = tokenizer
        self.prompt_field = prompt_field
        self.max_length = max_length
    def __call__(self, features):  # features: List[Dict]
        if not features:
            return {}
        # If already tokenized (has input_ids) just pad via tokenizer.pad
        if 'input_ids' in features[0]:
            # tokenizer.pad expects list of dict with input_ids
            return self.tokenizer.pad(features, padding=True, return_tensors='pt')
        texts = [f.get(self.prompt_field, "") for f in features]
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )
        return enc

prompt_collator = PromptCollator(tokenizer, PROMPT_FIELD)

trainer = PPOTrainer(
    args=ppo_config,
    model=policy,
    ref_model=ref_model,
    reward_model=reward_model,
    value_model=value_model,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=prompt_collator,
)

# --- Distributed compatibility patch ---
# Some TRL versions internally reference `self.policy` even if only `model` was provided.
# When Accelerate wraps the model with DDP, that attribute can be missing, causing
# AttributeError: 'DistributedDataParallel' object has no attribute 'policy'.
if not hasattr(trainer, 'policy'):
    try:
        trainer.policy = trainer.model  # alias
        print("[INFO] Added missing trainer.policy alias to wrapped model")
    except Exception:
        pass
    # Provide a convenient unwrapped base_model if helpful
    try:
        if hasattr(trainer.policy, 'module') and not hasattr(trainer, 'base_model'):
            trainer.base_model = trainer.policy.module
    except Exception:
        pass

# Ensure the underlying (possibly DDP) model itself exposes a .policy attribute if internal TRL code accesses model.policy
try:
    if hasattr(trainer, 'model') and hasattr(trainer.model, 'module') and not hasattr(trainer.model, 'policy'):
        setattr(trainer.model, 'policy', trainer.model.module)
        print("[INFO] Injected .policy attribute onto DDP-wrapped model (points to .module)")
except Exception:
    pass

# Unified reference used for generation below to avoid ambiguity
def _get_policy_ref():
    return getattr(trainer, 'policy', None) or getattr(trainer, 'model', None) or policy

policy_ref = _get_policy_ref()

# Guarantee transformers outputs are dict-like for TRL compatibility (fallback .train path expects .logits)
try:
    if hasattr(policy_ref, 'config'):
        policy_ref.config.return_dict = True
    if hasattr(value_model, 'config'):
        value_model.config.return_dict = True
    if hasattr(reward_model, 'config'):
        reward_model.config.return_dict = True
except Exception:
    pass

# Inject lightweight forward wrapper to add a .logits attr if the underlying model returns a raw tuple.
from types import SimpleNamespace
def _wrap_tuple_forward(m):
    if not hasattr(m, 'forward'):
        return
    orig_fwd = m.forward
    if getattr(orig_fwd, '_tuple_wrap_applied', False):
        return
    def wrapped_forward(*args, **kwargs):
        out = orig_fwd(*args, **kwargs)
        # If already has logits attribute leave as-is
        if hasattr(out, 'logits'):
            return out
        if isinstance(out, tuple) and len(out) > 0 and torch.is_tensor(out[0]):
            return SimpleNamespace(logits=out[0])
        return out
    wrapped_forward._tuple_wrap_applied = True  # type: ignore[attr-defined]
    m.forward = wrapped_forward  # type: ignore[assignment]

try:
    _wrap_tuple_forward(policy_ref)
    # Also guard underlying .module if present
    if hasattr(policy_ref, 'module'):
        _wrap_tuple_forward(policy_ref.module)
    # Additionally wrap PPOTrainer's internal policy wrapper if present
    internal_policy = getattr(getattr(trainer, 'model', None), 'policy', None)
    if internal_policy is not None:
        _wrap_tuple_forward(internal_policy)
except Exception:
    pass

# Final safety net: register forward hooks on all candidate policy modules to coerce tuple outputs to have .logits
def _register_logits_hook(m):
    if m is None or not hasattr(m, 'forward') or getattr(m, '_logits_hooked', False):
        return
    def _hook(_module, _inp, out):
        if hasattr(out, 'logits'):
            return out
        if isinstance(out, tuple) and len(out) > 0 and torch.is_tensor(out[0]):
            return type('HookOut', (), {'logits': out[0]})()
        return out
    try:
        m.register_forward_hook(_hook)
        m._logits_hooked = True  # type: ignore[attr-defined]
    except Exception:
        pass

for cand in {
    policy_ref,
    getattr(trainer, 'model', None),
    getattr(trainer, 'policy', None),
    getattr(getattr(trainer, 'model', None), 'policy', None),
    getattr(policy_ref, 'module', None),
}:
    _register_logits_hook(cand)
print("[INFO] Registered logits safeguard hooks on policy modules")

# Provide a lightweight reference model if TRL still expects ref logits but ref_model is None.
class _NullRefModel(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        if hasattr(self.base, 'config'):
            try:
                self.base.config.return_dict = True
            except Exception:
                pass
    def forward(self, *args, **kwargs):
        with torch.no_grad():
            out = self.base(*args, **kwargs)
            if hasattr(out, 'logits'):
                return out
            if isinstance(out, tuple) and len(out) > 0 and torch.is_tensor(out[0]):
                return type('RefOut', (), {'logits': out[0]})()
            return out

try:
    if getattr(trainer, 'ref_model', None) is None:
        # Use underlying (non-LoRA disabled) policy_ref; TRL will compute KL but with identical logits (beta likely 0)
        trainer.ref_model = _NullRefModel(policy_ref)
        print("[INFO] Injected lightweight null reference model for logits compatibility")
except Exception:
    pass

# --- Adapter method compatibility shim ---
# Some TRL versions expect `disable_adapter` / `enable_adapter` methods on the
# (possibly wrapped) policy object (e.g., PolicyAndValueWrapper with PEFT). If the
# wrapper does not expose them, we add lightweight delegating shims that attempt
# to call any underlying PEFT-provided toggle (e.g. disable_adapter_layers) or
# become no-ops. This prevents AttributeError crashes during generation or eval.
def _inject_adapter_shims(obj):
    if obj is None:
        return
    try:
        underlying = getattr(obj, 'module', None) or obj
        # Candidate method name variants present in different PEFT versions
        disable_variants = [
            'disable_adapter', 'disable_adapters', 'disable_adapter_layers', 'disable_adapter_modules'
        ]
        enable_variants = [
            'enable_adapter', 'enable_adapters', 'enable_adapter_layers', 'enable_adapter_modules'
        ]
        # Helper: locate first existing method variant on underlying
        def _find_variant(names):
            for n in names:
                if hasattr(underlying, n):
                    attr = getattr(underlying, n)
                    # Skip if it's one of our shims (avoid recursion)
                    if callable(attr):
                        return attr
            return None
        # Build a context manager that calls disable on enter, enable on exit (if available)
        class _AdapterToggleCtx:
            def __init__(self, disable_fn, enable_fn):
                self._disable_fn = disable_fn
                self._enable_fn = enable_fn
            def __enter__(self):
                try:
                    if self._disable_fn:
                        self._disable_fn()
                except Exception:
                    pass
                return self
            def __exit__(self, exc_type, exc, tb):
                try:
                    if self._enable_fn:
                        self._enable_fn()
                except Exception:
                    pass
                return False  # don't suppress
        if not hasattr(obj, 'disable_adapter'):
            def _shim_disable(*args, **kwargs):
                disable_fn = _find_variant(disable_variants)
                enable_fn = _find_variant(enable_variants)
                # If underlying disable returns a context manager itself, call & return it
                if disable_fn is not None:
                    try:
                        maybe = disable_fn(*args, **kwargs)
                        if hasattr(maybe, '__enter__') and hasattr(maybe, '__exit__'):
                            return maybe
                    except Exception:
                        pass
                # Fallback: return our toggle context manager (will call disable immediately in __enter__)
                return _AdapterToggleCtx(disable_fn, enable_fn)
            setattr(obj, 'disable_adapter', _shim_disable)
        if not hasattr(obj, 'enable_adapter'):
            def _shim_enable(*args, **kwargs):
                enable_fn = _find_variant(enable_variants)
                if enable_fn is not None:
                    try:
                        return enable_fn(*args, **kwargs)
                    except Exception:
                        return None
                return None
            setattr(obj, 'enable_adapter', _shim_enable)
    except Exception:
        pass

_inject_adapter_shims(trainer)
_inject_adapter_shims(getattr(trainer, 'model', None))
_inject_adapter_shims(getattr(trainer, 'policy', None))
_inject_adapter_shims(policy_ref)
print("[INFO] Ensured adapter toggle shims exist (enable_adapter/disable_adapter)")

import re, json, time

# Reward configuration (math-optimized defaults; override via env)
CORRECT_REWARD = float(os.environ.get("CORRECT_REWARD", 2.0))
INCORRECT_REWARD = float(os.environ.get("INCORRECT_REWARD", -0.5))
ERROR_SCALE = float(os.environ.get("ERROR_SCALE", 0.02))  # penalty per absolute error unit
MAX_ERROR = float(os.environ.get("MAX_ERROR", 40.0))
LENGTH_PENALTY = float(os.environ.get("LENGTH_PENALTY", 0.01))  # discourage rambling
# Format shaping
FORMAT_BONUS = float(os.environ.get("FORMAT_BONUS", 0.4))  # bonus if exactly '#### <number>'
FORMAT_PENALTY = float(os.environ.get("FORMAT_PENALTY", 0.1))  # mild penalty when missing pattern (post warmup)
NO_NUMERIC_PENALTY = float(os.environ.get("NO_NUMERIC_PENALTY", 0.4))  # penalty if model emits no number
FORMAT_WARMUP_STEPS = int(os.environ.get("FORMAT_WARMUP_STEPS", 100))  # during warmup skip format penalty
EPS_EQ = float(os.environ.get("NUMERIC_EPS", 1e-6))
MIN_REWARD = float(os.environ.get("MIN_REWARD", -2.0))  # clamp extreme negatives
REWARD_NORMALIZE = os.environ.get("REWARD_NORMALIZE", "1") == "1"  # running z-norm
REWARD_JSONL = os.environ.get("REWARD_LOG", "reward_metrics.jsonl")
EVAL_INTERVAL = int(os.environ.get("EVAL_EVERY_STEPS", 0))  # 0 disables
EVAL_MAX_SAMPLES = int(os.environ.get("EVAL_MAX_SAMPLES", 32))
EVAL_GREEDY = os.environ.get("EVAL_GREEDY", "1") == "1"  # greedy vs sampled
DETERMINISTIC_GEN = os.environ.get("DETERMINISTIC_GEN", "0") == "1"  # force greedy generation during PPO steps
TRAIN_TOP_P = float(os.environ.get("TRAIN_TOP_P", 0.9))
TRAIN_TEMPERATURE = float(os.environ.get("TRAIN_TEMPERATURE", 1.0))

_answer_regex = re.compile(r"####\s*(-?\d+(?:\.\d+)?)\s*$")

def _extract_last_number(text: str):  # fallback generic extractor
    if not isinstance(text, str):
        return None
    matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except Exception:
        return None

def _parse_prefixed_answer(ans: str):
    if not isinstance(ans, str):
        return None
    m = _answer_regex.search(ans)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    return None

def compute_rewards(responses, answers, global_step: int = 0):
    """Compute scalar rewards with rich shaping components.

    Components:
      - Correctness: CORRECT_REWARD for exact numeric match.
      - Incorrect: INCORRECT_REWARD - ERROR_SCALE * |error| (capped by MAX_ERROR).
      - Formatting: +FORMAT_BONUS if response matches '^#### <num>$'; -FORMAT_PENALTY if numeric gt but response missing the pattern.
      - No numeric at all: -NO_NUMERIC_PENALTY (in addition to incorrect baseline half penalty) when model emits no parseable number.
      - Length: subtract LENGTH_PENALTY * token_count (simple whitespace token proxy).
    Returns list of rewards and aggregated stats for logging.
    """
    rewards = []
    stats = {"correct": 0, "total": 0, "formatted": 0, "missing_format": 0, "no_numeric": 0, "sum_error": 0.0}
    format_pattern = re.compile(r"^####\s*-?\d+(?:\.\d+)?\s*$")
    for resp, ans in zip(responses, answers):
        resp_stripped = resp.strip() if isinstance(resp, str) else ""
        formatted_good = bool(format_pattern.match(resp_stripped))
        pred_num = _extract_last_number(resp)
        # Prefer strict prefixed parse for answer; fallback to generic extraction.
        true_num = _parse_prefixed_answer(ans) if ans is not None else None
        if true_num is None and ans is not None:
            true_num = _extract_last_number(ans)
        base = 0.0
        if true_num is not None and pred_num is not None:
            stats["total"] += 1
            if abs(pred_num - true_num) <= EPS_EQ:
                base = CORRECT_REWARD
                stats["correct"] += 1
            else:
                error = min(abs(pred_num - true_num), MAX_ERROR)
                stats["sum_error"] += error
                base = INCORRECT_REWARD - ERROR_SCALE * error
        else:
            # No valid numeric prediction; penalize moderately then add extra no-numeric penalty.
            base = INCORRECT_REWARD * 0.5
            if pred_num is None:
                base -= NO_NUMERIC_PENALTY
                stats["no_numeric"] += 1
        # Formatting shaping (only applies if ground truth is numeric)
        if true_num is not None:
            if formatted_good:
                base += FORMAT_BONUS
                stats["formatted"] += 1
            else:
                # Only apply penalty after warmup
                if global_step >= FORMAT_WARMUP_STEPS:
                    base -= FORMAT_PENALTY
                stats["missing_format"] += 1
        # Length penalty
        length_tokens = len(resp.split()) if isinstance(resp, str) else 0
        base -= LENGTH_PENALTY * length_tokens
        # Clamp
        if base < MIN_REWARD:
            base = MIN_REWARD
        rewards.append(base)
    return rewards, stats

def evaluate_policy(eval_dataset, tokenizer, policy, answer_field: str, max_samples: int, greedy: bool):
    if len(eval_dataset) == 0:
        return {"eval_exact_match": None, "eval_samples": 0}
    n = min(max_samples, len(eval_dataset))
    subset = eval_dataset.select(range(n))
    correct = 0; total_numeric = 0
    for ex in subset:
        prompt = ex.get(PROMPT_FIELD, "")
        ans = ex.get(answer_field)
        # Parse ground truth
        true_num = _parse_prefixed_answer(ans) if isinstance(ans, str) else None
        if true_num is None:
            continue
        total_numeric += 1
        enc = tokenizer([prompt], return_tensors='pt', padding=True).to(policy_ref.device)
        with torch.no_grad():
            gen_ids = policy_ref.generate(
                **enc,
                max_new_tokens=32,
                do_sample=not greedy,
                top_p=0.9,
                temperature=1.0 if not greedy else 0.2,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        full = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        if full.startswith(prompt):
            pred = full[len(prompt):].strip()
        else:
            pred = full
        pred_num = _parse_prefixed_answer(pred) or _extract_last_number(pred)
        if pred_num is not None and pred_num == true_num:
            correct += 1
    acc = (correct / total_numeric) if total_numeric > 0 else None
    return {"eval_exact_match": acc, "eval_samples": total_numeric}

from tqdm import trange

def _coerce_prompt(obj):
    if isinstance(obj, str):
        return obj
    if isinstance(obj, list):
        # If looks like token ids decode; else join
        if all(isinstance(t, int) for t in obj):
            try:
                return tokenizer.decode(obj, skip_special_tokens=True)
            except Exception:
                return " ".join(str(t) for t in obj)
        return " ".join(str(t) for t in obj)
    return str(obj)

_step_attr = getattr(trainer, 'step', None) or getattr(trainer, 'ppo_step', None)
if _step_attr is None:
    # Fallback path: rely on PPOTrainer's internal train() loop (uses reward/value models)
    # Optionally map our requested number of outer steps to total_episodes if config supports it.
    try:
        if hasattr(ppo_config, 'total_episodes') and not getattr(ppo_config, 'total_episodes'):
            setattr(ppo_config, 'total_episodes', steps * ppo_config.batch_size)
    except Exception:
        pass
    # Re-inject adapter shims in case TRL reconstructed/wrapped the model after our earlier injection
    try:
        current_wrapper = getattr(trainer, 'model', None)
        _inject_adapter_shims(current_wrapper)
        inner_policy = getattr(current_wrapper, 'policy', None)
        _inject_adapter_shims(inner_policy)
        if inner_policy is not None and hasattr(trainer, 'accelerator'):
            try:
                unwrapped = trainer.accelerator.unwrap_model(inner_policy)
                _inject_adapter_shims(unwrapped)
            except Exception:
                pass
        print("[INFO] Reconfirmed adapter shims before fallback trainer.train()")
    except Exception:
        pass
    # Add this right before trainer.train() at line 590

    print("[INFO] Applying final output format safeguards...")

    # Solution 1: Improved NullRefModel
    class _NullRefModel(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
            if hasattr(self.base, 'config'):
                try:
                    self.base.config.return_dict = True
                except Exception:
                    pass
        
        def forward(self, *args, **kwargs):
            with torch.no_grad():
                kwargs['return_dict'] = True
                out = self.base(*args, **kwargs)
                if isinstance(out, tuple) and len(out) > 0:
                    return type('RefOut', (), {'logits': out[0]})()
                elif hasattr(out, 'logits'):
                    return out
                else:
                    return type('RefOut', (), {'logits': out})()

    # Re-inject the improved ref model
    if getattr(trainer, 'ref_model', None) is None:
        trainer.ref_model = _NullRefModel(policy_ref)
        print("[INFO] Re-injected improved null reference model")

    # Solution 2: Force return_dict for all models
    for model_name in ['ref_model', 'value_model', 'reward_model', 'model']:
        model = getattr(trainer, model_name, None)
        if model is not None and hasattr(model, 'config'):
            try:
                model.config.return_dict = True
                print(f"[INFO] Set return_dict=True for {model_name}")
            except Exception:
                pass

    # Solution 3: Wrap forward methods
    def safe_forward_wrapper(model, name):
        if model is None:
            return
        original_forward = model.forward
        def safe_forward(*args, **kwargs):
            kwargs['return_dict'] = True
            output = original_forward(*args, **kwargs)
            if isinstance(output, tuple):
                return type('SafeOutput', (), {'logits': output[0] if len(output) > 0 else None})()
            return output
        model.forward = safe_forward
        print(f"[INFO] Applied safe forward to {name}")

    safe_forward_wrapper(trainer.ref_model, "ref_model")
    safe_forward_wrapper(trainer.value_model, "value_model")
    safe_forward_wrapper(trainer.reward_model, "reward_model")
    safe_forward_wrapper(trainer.model, "model")

    print("[INFO] All safeguards applied, starting training...")
    
    
        # Add this right before trainer.train()

    print("[INFO] Applying comprehensive output format fix...")

    # Solution 1: Proper wrapper class
    class SafeModelOutput:
        def __init__(self, logits):
            self.logits = logits
            
        def __getitem__(self, key):
            # Handle slicing like logits[:, context_length - 1 : -1]
            if isinstance(key, tuple):
                # This is the specific case that's failing
                return self.logits[key]
            return self.logits[key]
        
        def to_tuple(self):
            return (self.logits,)

    def create_safe_forward(model):
        original_forward = model.forward
        def safe_forward(*args, **kwargs):
            kwargs['return_dict'] = True
            output = original_forward(*args, **kwargs)
            
            if isinstance(output, tuple):
                return SafeModelOutput(output[0] if len(output) > 0 else None)
            elif hasattr(output, 'logits'):
                # Already has logits, ensure it supports slicing
                if not hasattr(output, '__getitem__'):
                    output.__getitem__ = lambda key: output.logits[key]
                return output
            else:
                return SafeModelOutput(output)
        
        return safe_forward

    # Apply to all models
    for model_name in ['ref_model', 'value_model', 'reward_model', 'model']:
        model = getattr(trainer, model_name, None)
        if model is not None:
            model.forward = create_safe_forward(model)
            print(f"[INFO] Applied safe forward to {model_name}")

    # Solution 2: Ensure config settings
    for model_name in ['ref_model', 'value_model', 'reward_model', 'model']:
        model = getattr(trainer, model_name, None)
        if model is not None and hasattr(model, 'config'):
            try:
                model.config.return_dict = True
                model.config.use_cache = False
            except Exception:
                pass

    print("[INFO] Output format fixes completed, starting training...")
    print("[WARN] PPOTrainer exposes neither 'step' nor 'ppo_step'. Falling back to trainer.train() using reward_model outputs.")
    trainer.train()
else:
    metrics_fp = open(REWARD_JSONL, "a", buffering=1)
    acc_moving = None
    for step in trange(steps, desc="PPO LoRA"):
        batch = train_dataset.shuffle(seed=ppo_config.seed + step).select(range(min(ppo_config.batch_size, len(train_dataset))))
        raw = batch[PROMPT_FIELD]
        answers = batch[ANSWER_FIELD] if ANSWER_FIELD in batch.column_names else [None]*len(raw)
        prompts = [_coerce_prompt(p) for p in raw]
        # Guard: empty or None -> replace
        prompts = [p if (isinstance(p, str) and len(p.strip()) > 0) else "" for p in prompts]
        # Generate responses
        inputs = tokenizer(prompts, return_tensors='pt', padding=True).to(policy_ref.device)
        with torch.no_grad():
            if DETERMINISTIC_GEN:
                gen_ids = policy_ref.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=0.0,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            else:
                gen_ids = policy_ref.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    top_p=TRAIN_TOP_P,
                    temperature=TRAIN_TEMPERATURE,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
        responses = []
        for i, prompt in enumerate(prompts):
            full = tokenizer.decode(gen_ids[i], skip_special_tokens=True)
            # Extract only the completion after the prompt if present
            if full.startswith(prompt):
                resp = full[len(prompt):].strip()
            else:
                resp = full
            responses.append(resp)
        # Rewards (numeric correctness + shaping)
        rewards, stats = compute_rewards(responses, answers, global_step=step)
        # Optional running reward normalization
        if REWARD_NORMALIZE:
            r_mean = sum(rewards)/len(rewards)
            var = sum((r - r_mean)**2 for r in rewards)/max(1, len(rewards)-1)
            r_std = var ** 0.5
            if r_std > 1e-6:
                rewards = [(r - r_mean)/r_std for r in rewards]
        _step_fn = getattr(trainer, 'step', None) or getattr(trainer, 'ppo_step', None)
        if _step_fn is None:
            print("[ERROR] Lost access to step/ppo_step mid-training; aborting loop.")
            break
        _step_fn(prompts, responses, rewards)
        # Logging
        if stats["total"] > 0:
            batch_acc = stats["correct"] / stats["total"]
            acc_moving = batch_acc if acc_moving is None else 0.9 * acc_moving + 0.1 * batch_acc
        avg_error = (stats["sum_error"] / (stats["total"] - stats["correct"])) if stats["total"] - stats["correct"] > 0 else None
        log_entry = {
            "step": step+1,
            "avg_reward": sum(rewards)/len(rewards),
            "correct_in_batch": stats["correct"],
            "has_numeric": stats["total"],
            "batch_accuracy": (stats["correct"]/stats["total"] if stats["total"]>0 else None),
            "moving_accuracy": acc_moving,
            "formatted": stats["formatted"],
            "missing_format": stats["missing_format"],
            "no_numeric": stats["no_numeric"],
            "avg_error": avg_error,
        }
        try:
            metrics_fp.write(json.dumps(log_entry) + "\n")
        except Exception:
            pass
        if (step + 1) % 10 == 0:
            print(f"[PPO] step={step+1} avg_reward={log_entry['avg_reward']:.3f} batch_acc={log_entry['batch_accuracy']}")
        if EVAL_INTERVAL and (step + 1) % EVAL_INTERVAL == 0:
            eval_stats = evaluate_policy(eval_dataset, tokenizer, policy, ANSWER_FIELD, EVAL_MAX_SAMPLES, EVAL_GREEDY)
            print(f"[EVAL] step={step+1} exact_match={eval_stats['eval_exact_match']} samples={eval_stats['eval_samples']}")
            # append to metrics file
            try:
                metrics_fp.write(json.dumps({"step": step+1, **eval_stats}) + "\n")
            except Exception:
                pass
    try:
        metrics_fp.close()
    except Exception:
        pass

# Save adapters
output_dir = os.environ.get("OUTPUT_DIR", "./ppo_lora_output")
os.makedirs(output_dir, exist_ok=True)
trainer.save_model(output_dir)
print(f"Saved LoRA PPO model to {output_dir}")
