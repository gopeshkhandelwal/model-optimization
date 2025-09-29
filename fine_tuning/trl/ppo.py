# Adapted from https://github.com/huggingface/optimum-habana/tree/v1.16.0/examples/trl
# Customized and enabled for Gaudi3
import json
import time
import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import Adafactor, AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser, pipeline, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from trl import AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler

from optimum.habana.accelerate import GaudiAccelerator
from optimum.habana.trl import GaudiPPOConfig, GaudiPPOTrainer, adapt_PreTrainedModelWrapper_to_gaudi
from optimum.habana.utils import set_seed


tqdm.pandas()


@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """

    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    # models like gpt-neo* models are more suitable.
    model_name_or_path: Optional[str] = field(default="google/gemma-3-270m", metadata={"help": "the (Gemma 3) model name"})
    tokenizer_name_or_path: Optional[str] = field(
    default="google/gemma-3-270m", metadata={"help": "the tokenizer name (Gemma 3)"}
    )
    reward_model_name: Optional[str] = field(default="", metadata={"help": "the reward model name"})
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    output_max_length: Optional[int] = field(default=128, metadata={"help": "maximum output length for generation"})
    input_max_length: Optional[int] = field(default=512, metadata={"help": "maximum input length for generation"})
    mini_batch_size: Optional[int] = field(default=1, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=32, metadata={"help": "the batch size"})
    ppo_epochs: Optional[int] = field(default=4, metadata={"help": "the number of ppo epochs"})
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )
    adafactor: Optional[bool] = field(default=False, metadata={"help": "whether to use the adafactor optimizer"})
    early_stopping: Optional[bool] = field(default=False, metadata={"help": "whether to early stop"})
    target_kl: Optional[float] = field(default=0.1, metadata={"help": "kl target for early stopping"})
    reward_baseline: Optional[float] = field(
        default=0.0,
        metadata={"help": "a baseline value that is subtracted from the reward"},
    )
    batched_gen: Optional[bool] = field(default=False, metadata={"help": "whether to use the batched text gen"})
    save_freq: Optional[int] = field(default=None, metadata={"help": "n steps to save the model"})
    output_dir: Optional[str] = field(default="runs/", metadata={"help": "n steps to save the model"})
    seed: Optional[int] = field(default=0, metadata={"help": "the seed"})
    steps: Optional[int] = field(default=20000, metadata={"help": "number of epochs"})
    init_kl_coef: Optional[float] = field(
        default=0.2,
        metadata={"help": "Initial KL penalty coefficient (used for adaptive and linear control)"},
    )

    adap_kl_ctrl: Optional[bool] = field(default=True, metadata={"help": "Use adaptive KL control, otherwise linear"})
    use_habana: Optional[bool] = field(default=True, metadata={"help": "use habana for RL training"})
    lora_alpha: Optional[float] = field(default=32, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=16, metadata={"help": "the lora r parameter"})
    lora_target_modules: List[str] = field(
        default_factory=lambda: None,
        metadata={"help": "Target modules for the LoRA method."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    merge_adapter_after_train: Optional[bool] = field(
        default=False, metadata={"help": "If True, merge LoRA adapter into base policy model + export value head."}
    )
    merged_output_dir: Optional[str] = field(
        default=None, metadata={"help": "Directory to save merged policy (defaults to <output_dir>_merged)."}
    )
    merge_overwrite: Optional[bool] = field(
        default=False, metadata={"help": "Overwrite merged_output_dir if it exists."}
    )
    disable_kv_cache: Optional[bool] = field(
        default=True,
        metadata={"help": "If True, disable key/value cache during generation to reduce host memory (helps avoid sliding cache OOM)."},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable gradient checkpointing to reduce activation memory (slower but lighter)."},
    )
    initial_output_max_length: Optional[int] = field(
        default=None,
        metadata={"help": "Optional smaller starting max generation length; will ramp to output_max_length."},
    )
    length_ramp_epochs: Optional[int] = field(
        default=None,
        metadata={"help": "Number of epochs over which to linearly ramp from initial_output_max_length to output_max_length."},
    )
    oom_shrink_factor: Optional[float] = field(
        default=0.75,
        metadata={"help": "Factor to multiply current max length by after an Out of HOST memory error."},
    )
    min_output_max_length: Optional[int] = field(
        default=16,
        metadata={"help": "Lower bound for adaptive output length after OOM reductions."},
    )


adapt_PreTrainedModelWrapper_to_gaudi()
parser = HfArgumentParser(ScriptArguments)
from logging_utils import setup_logging
script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]
setup_logging()
logger = logging.getLogger(__name__)
logger.info(f"ScriptArguments: {script_args}")
if not (hasattr(torch, 'hpu') and torch.hpu.is_available()):
    raise RuntimeError('[HPU][Required] Habana HPU not available. PPO script enforces HPU-only execution.')
# Always force HPU usage regardless of CLI flag (user request: Always Use HPU)
if not script_args.use_habana:
    logger.warning("[HPU][Override] Forcing use_habana=True (Always Use HPU policy)")
    script_args.use_habana = True

reward_model_name = script_args.reward_model_name
dataset_name = "lvwerra/stack-exchange-paired"
config = GaudiPPOConfig(
    steps=script_args.steps,
    model_name=script_args.model_name_or_path,
    learning_rate=script_args.learning_rate,
    log_with=script_args.log_with,
    batch_size=script_args.batch_size,
    mini_batch_size=script_args.mini_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    optimize_cuda_cache=True,
    early_stopping=script_args.early_stopping,
    target_kl=script_args.target_kl,
    ppo_epochs=script_args.ppo_epochs,
    seed=script_args.seed,
    init_kl_coef=script_args.init_kl_coef,
    adap_kl_ctrl=script_args.adap_kl_ctrl,
    use_habana=script_args.use_habana,
    pad_max_len=script_args.input_max_length + script_args.output_max_length,
    pad_max_input_len=script_args.input_max_length,
)
logger.info(f"GaudiPPOConfig: {config}")

train_dataset = load_dataset("lvwerra/stack-exchange-paired", data_dir="data/rl", split="train")
if script_args.max_train_samples is not None:
    logger.info(f"[IF] max_train_samples specified -> limiting to {script_args.max_train_samples}")
    max_train_samples = min(len(train_dataset), script_args.max_train_samples)
    train_dataset = train_dataset.select(range(max_train_samples))
original_columns = train_dataset.column_names

# We then define the arguments to pass to the sentiment analysis pipeline.
# We set `return_all_scores` to True to get the sentiment score for each token.
sent_kwargs = {
    # return_all_scores deprecated; use top_k=None for all scores
    "top_k": None,
    "function_to_apply": "none",
    "batch_size": 16,
    "truncation": True,
}
if config.pad_for_acceleration:
    logger.info("[IF] pad_for_acceleration == True -> enabling padding in sentiment kwargs")
    sent_kwargs["padding"] = "max_length"
    sent_kwargs["max_length"] = script_args.input_max_length + script_args.output_max_length

tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer_name_or_path)
# GPT-2 tokenizer has a pad token, but it is not eos_token by default. We need to set it to eos_token.
# only for this model.

if getattr(tokenizer, "pad_token", None) is None:
    logger.info("[IF] tokenizer.pad_token is None -> assigning eos_token as pad_token")
    tokenizer.pad_token = tokenizer.eos_token


# Below is an example function to build the dataset. In our case, we use the IMDB dataset
# from the `datasets` library. One should customize this function to train the model on
# its own dataset.
def build_dataset(
    tokenizer,
    dataset_name="lvwerra/stack-exchange-paired",
    input_max_length=512,
):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """

    num_proc = 24

    def preprocess_function(examples):
        new_examples = {
            "query": [],
            "input_ids": [],
        }
        for question in examples["question"]:
            query = "Question: " + question + "\n\nAnswer: "
            tokenized_question = tokenizer(query, truncation=True)
            new_examples["query"].append(query)
            new_examples["input_ids"].append(tokenized_question["input_ids"])

        return new_examples

    ds = train_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )
    ds = ds.filter(lambda x: len(x["input_ids"]) < input_max_length, batched=False)
    logger.info(f"[Dataset] Filtered dataset length: {len(ds)}")

    ds.set_format(type="torch")
    return ds


# We retrieve the dataloader by calling the `build_dataset` function.
dataset = build_dataset(tokenizer, input_max_length=script_args.input_max_length)


def collator(data):
    """Dynamic collator.
    When batched generation is enabled we pad input_ids to a tensor batch to avoid
    downstream indexing issues inside Gaudi batched generation (_generate_batched).
    """
    first = data[0]
    # Collect fields
    batch = {k: [d[k] for d in data] for k in first}
    if script_args.batched_gen:
        # Pad input_ids manually (tokenizer already set pad_token).
        seqs = batch["input_ids"]
        # seqs is list of lists of ints
        max_len = max(len(s) for s in seqs)
        pad_id = tokenizer.pad_token_id
        import torch as _torch
        padded = _torch.full((len(seqs), max_len), pad_id, dtype=_torch.long)
        for i, s in enumerate(seqs):
            if isinstance(s, _torch.Tensor):
                padded[i, : s.numel()] = s.to(dtype=_torch.long)
            else:
                padded[i, : len(s)] = _torch.as_tensor(s, dtype=_torch.long)
        batch["input_ids"] = padded
    else:
        # Ensure each sequence is a 1D LongTensor for PPOTrainer.generate expectations
        import torch as _torch
        seqs = batch["input_ids"]
        for i, s in enumerate(seqs):
            if not isinstance(s, _torch.Tensor):
                seqs[i] = _torch.as_tensor(s, dtype=_torch.long)
            else:
                if s.dtype != _torch.long:
                    seqs[i] = s.to(dtype=_torch.long)
    return batch


# set seed before initializing value head for deterministic eval
set_seed(config.seed)

# Now let's build the model, the reference model, and the tokenizer.
current_device = GaudiAccelerator().local_process_index
# Provide sensible default LoRA target modules if none specified
if script_args.lora_target_modules is None or (isinstance(script_args.lora_target_modules, list) and len(script_args.lora_target_modules) == 0):
    script_args.lora_target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "up_proj",
        "down_proj",
        "gate_proj",
    ]
    logger.info(f"[LoRA][AutoDefault] No lora_target_modules provided -> using {script_args.lora_target_modules}")
lora_config = LoraConfig(
    r=script_args.lora_r,
    lora_alpha=script_args.lora_alpha,
    lora_dropout=script_args.lora_dropout,
    target_modules=script_args.lora_target_modules,
    bias="none",
    task_type="CAUSAL_LM",
)
logger.info(f"[LoRA] Config: {lora_config}")
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    config.model_name,
    peft_config=lora_config,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    attn_implementation="eager",
)
model.config.use_fused_rope = False
model.config.use_fused_rms_norm = False
if script_args.disable_kv_cache:
    model.config.use_cache = False
if script_args.gradient_checkpointing:
    try:
        base_model_for_ckpt = model.pretrained_model if hasattr(model, "pretrained_model") else model
        base_model_for_ckpt.gradient_checkpointing_enable()
        logger.info("[Memory] Gradient checkpointing enabled for policy model.")
    except Exception as e:
        logger.warning(f"[Memory] Failed to enable gradient checkpointing on policy model: {e}")
optimizer = None
model = model.to(torch.bfloat16)

def _param_stats(m):
    tot=trn=0
    for p in m.parameters():
        n=p.numel(); tot+=n; trn+= n if p.requires_grad else 0
    return tot,trn,(trn/tot*100 if tot else 0)
btot,btrn,bpct=_param_stats(model)
logger.info(f"[Model Params][Policy+ValueHead] total={btot:,} trainable={btrn:,} ({bpct:.4f}%)")

logger.info("[HPU] Loading reference model (forced HPU mode)")
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    config.model_name,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    attn_implementation="eager",
)
if script_args.disable_kv_cache:
    ref_model.config.use_cache = False
if script_args.gradient_checkpointing:
    try:
        base_ref_for_ckpt = ref_model.pretrained_model if hasattr(ref_model, "pretrained_model") else ref_model
        base_ref_for_ckpt.gradient_checkpointing_enable()
        logger.info("[Memory] Gradient checkpointing enabled for reference model.")
    except Exception as e:
        logger.warning(f"[Memory] Failed to enable gradient checkpointing on reference model: {e}")
if script_args.adafactor:
    logger.info("[IF] adafactor == True -> using Adafactor optimizer")
    optimizer = Adafactor(
        filter(lambda p: p.requires_grad, model.parameters()),
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        lr=config.learning_rate,
    )
# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
ppo_trainer = GaudiPPOTrainer(
    config,
    model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    dataset=dataset,
    data_collator=collator,
    optimizer=optimizer,
)
logger.info("[Trainer] PPO trainer initialized")
# We then build the sentiment analysis pipeline using our reward model, passing the
# model name and the sentiment analysis pipeline arguments. Let's also make sure to
# set the device to the same device as the PPOTrainer.
device = ppo_trainer.accelerator.device

def load_reward_model(name: str):
    if not name:
        raise ValueError("--reward_model_name is required (path to local reward model directory or HF repo id).")
    # Local directory case
    if os.path.isdir(name):
        local_path = os.path.abspath(name)
        logger.info(f"[RewardModel] Loading local directory: {local_path}")
        # Heuristics: if directory contains adapter_config.json (PEFT adapter only) but no config.json of a full model
        adapter_cfg_path = os.path.join(local_path, "adapter_config.json")
        config_json_path = os.path.join(local_path, "config.json")
        reward_head_meta_path = os.path.join(local_path, "reward_head_config.json")
        reward_head_weights_path = os.path.join(local_path, "reward_value_head.bin")
        # Case 1: Adapter-only directory
        if os.path.isfile(adapter_cfg_path) and not os.path.isfile(config_json_path):
            logger.info("[RewardModel][PEFT] Detected adapter-only directory (no base config.json). Attempting merge on-the-fly.")
            try:
                from peft import PeftConfig, PeftModel
                peft_conf = PeftConfig.from_pretrained(local_path)
                base_model_name = getattr(peft_conf, 'base_model_name_or_path', None)
                if not base_model_name or base_model_name in {"None", "none", ""}:
                    # Try fallback: user-specified tokenizer/model path or parent of adapter
                    fallback = script_args.model_name_or_path
                    if fallback and os.path.isdir(fallback) and os.path.isfile(os.path.join(fallback, 'config.json')):
                        base_model_name = fallback
                        logger.warning(f"[RewardModel][PEFT][Fallback] base_model_name_or_path missing in adapter; using CLI model path {base_model_name}")
                    else:
                        raise ValueError("[RewardModel][PEFT][Error] Adapter config missing base_model_name_or_path and no valid fallback directory provided.")
                logger.info(f"[RewardModel][PEFT] Base model from adapter: {base_model_name}")
                # Try loading as sequence classification first (works if reward_modeling used seq cls path)
                try:
                    base_model = AutoModelForSequenceClassification.from_pretrained(
                        base_model_name, num_labels=1, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16
                    )
                    seq_cls_loaded = True
                except Exception as e:
                    logger.warning(f"[RewardModel][PEFT] Failed to load base as SequenceClassification ({e}). Falling back to causal LM wrapper -> linear head.")
                    seq_cls_loaded = False
                    # Fallback: build causal LM and add classification head (simple last token head)
                    base_lm = AutoModelForCausalLM.from_pretrained(
                        base_model_name, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16, output_hidden_states=True
                    )
                    hidden_size = getattr(base_lm.config, 'hidden_size', getattr(base_lm.config, 'model_dim', None))
                    if hidden_size is None:
                        raise RuntimeError("[RewardModel][PEFT] Could not infer hidden size from base LM config.")
                    import torch as _torch
                    from transformers.modeling_outputs import SequenceClassifierOutput as _SCO

                    class _CausalLMRewardWrapper(_torch.nn.Module):
                        def __init__(self, lm, hidden):
                            super().__init__()
                            self.lm = lm
                            self.value_head = _torch.nn.Linear(hidden, 1)

                        def forward(self, input_ids=None, attention_mask=None, **kwargs):
                            if "output_hidden_states" not in kwargs:
                                kwargs["output_hidden_states"] = True
                            outputs = self.lm(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                use_cache=False,
                                **kwargs,
                            )
                            hidden = outputs.hidden_states[-1]
                            if attention_mask is not None:
                                lengths = attention_mask.sum(dim=1) - 1
                                last_tokens = hidden[_torch.arange(hidden.size(0), device=hidden.device), lengths.clamp(min=0)]
                            else:
                                last_tokens = hidden[:, -1]
                            reward = self.value_head(last_tokens)
                            return _SCO(logits=reward)

                        @property
                        def config(self):
                            return self.lm.config
                        @property
                        def device(self):
                            try:
                                return next(self.parameters()).device
                            except StopIteration:
                                return _torch.device("cpu")
                        def can_generate(self):
                            return False

                    base_model = _CausalLMRewardWrapper(base_lm, hidden_size)

                peft_model = PeftModel.from_pretrained(base_model, local_path)
                logger.info("[RewardModel][PEFT] Adapter loaded; merging and unloading for inference.")
                try:
                    merged = peft_model.merge_and_unload()
                    logger.info("[RewardModel][PEFT] Merge successful; returning merged model.")
                    return merged
                except Exception as merge_e:
                    logger.warning(f"[RewardModel][PEFT] merge_and_unload failed ({merge_e}); using adapter model directly.")
                    return peft_model
            except Exception as e:
                logger.exception(f"[RewardModel][PEFT] Failed to reconstruct reward model from adapter directory: {e}")
                raise
        else:
            # Case 2: Full merged directory (config.json exists). Could be SeqCls or merged fallback causal LM export.
            if os.path.isfile(reward_head_meta_path) and os.path.isfile(reward_head_weights_path):
                logger.info("[RewardModel][MergedFallback] Detected merged causal LM reward export with separate value head metadata.")
                import json as _json, torch as _torch
                with open(reward_head_meta_path) as f:
                    meta = _json.load(f)
                base_lm = AutoModelForCausalLM.from_pretrained(
                    local_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, output_hidden_states=True
                )
                hidden_size = meta.get("hidden_size")
                if hidden_size is None:
                    raise RuntimeError("[RewardModel][MergedFallback] hidden_size missing in reward_head_config.json")
                from transformers.modeling_outputs import SequenceClassifierOutput as _SCO
                class _CausalLMRewardWrapper(_torch.nn.Module):
                    def __init__(self, lm, hidden):
                        super().__init__()
                        self.lm = lm
                        self.value_head = _torch.nn.Linear(hidden, 1)
                    def forward(self, input_ids=None, attention_mask=None, **kwargs):
                        if "output_hidden_states" not in kwargs:
                            kwargs["output_hidden_states"] = True
                        outputs = self.lm(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, **kwargs)
                        hidden = outputs.hidden_states[-1]
                        if attention_mask is not None:
                            lengths = attention_mask.sum(dim=1) - 1
                            last_tokens = hidden[_torch.arange(hidden.size(0), device=hidden.device), lengths.clamp(min=0)]
                        else:
                            last_tokens = hidden[:, -1]
                        reward = self.value_head(last_tokens)
                        return _SCO(logits=reward)
                    @property
                    def config(self):
                        return self.lm.config
                    @property
                    def device(self):
                        try:
                            return next(self.parameters()).device
                        except StopIteration:
                            return _torch.device("cpu")
                    def can_generate(self):
                        return False
                wrapper = _CausalLMRewardWrapper(base_lm, hidden_size)
                state = _torch.load(reward_head_weights_path, map_location="cpu")
                wrapper.value_head.load_state_dict(state)
                logger.info("[RewardModel][MergedFallback] Reconstructed reward wrapper with loaded value head.")
                return wrapper
            # Case 3: Standard sequence classification reward model directory
            return AutoModelForSequenceClassification.from_pretrained(
                local_path,
                num_labels=1,
                low_cpu_mem_usage=True,
            )
    # If it looks like a relative path beginning with ./ or ../ but does not exist -> explicit error
    if name.startswith(":"):
        raise ValueError(f"[RewardModel][Error] Unexpected leading ':' in reward model name '{name}'.")
    if (name.startswith("./") or name.startswith("../")) and not os.path.isdir(name):
        raise FileNotFoundError(
            f"[RewardModel] Local path '{name}' not found. Ensure you ran reward_modeling.py and used merge if needed."
        )
    # Otherwise assume it's a remote HF repo id
    logger.info(f"[RewardModel] Loading remote HF repo: {name}")
    return AutoModelForSequenceClassification.from_pretrained(
        name,
        num_labels=1,
        low_cpu_mem_usage=True,
    )

try:
    reward_model = load_reward_model(reward_model_name)
except Exception as e:
    logger.exception(f"[RewardModel] Failed to load reward model '{reward_model_name}': {e}")
    raise
logger.info(f"[RewardModel] Loaded reward model source: {reward_model_name}")

# Ensure reward model cache disabled if requested
try:
    if script_args.disable_kv_cache and hasattr(reward_model, "config") and getattr(reward_model.config, "use_cache", None) is not False:
        reward_model.config.use_cache = False
        logger.info("[RewardModel][Memory] Disabled use_cache for reward model.")
except Exception as e:
    logger.warning(f"[RewardModel][Memory] Could not disable cache for reward model: {e}")

if config.use_habana:
    from habana_frameworks.torch.hpu import wrap_in_hpu_graph
    try:
        reward_model = reward_model.to("hpu")
    except Exception:
        pass
    reward_model = wrap_in_hpu_graph(reward_model)
    logger.info("[HPU] Reward model moved to HPU and wrapped in HPU graph")

if device.type == "hpu":
    device = "hpu"

sentiment_pipe = None
try:
    sentiment_pipe = pipeline(
        "sentiment-analysis",
        model=reward_model,
        tokenizer=tokenizer,
        return_token_type_ids=False,
        device=device,
        model_kwargs={
            "low_cpu_mem_usage": True,
            "torch_dtype": torch.bfloat16,
        },
    )
    if sentiment_pipe.model.config.pad_token_id is None:
        sentiment_pipe.model.config.pad_token_id = tokenizer.pad_token_id
    logger.info("[Reward] Using transformers pipeline for reward scoring.")
except Exception as e:
    logger.warning(f"[Reward][Fallback] Pipeline construction failed or unsupported model type -> using manual forward pass. Error: {e}")
    sentiment_pipe = None

def score_texts(texts: List[str]):
    """Return list[Tensor] rewards (1 per text). Uses pipeline if available else manual forward.
    Subtracts reward_baseline.
    """
    if sentiment_pipe is not None:
        pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
        return [torch.tensor(out[0]["score"] - script_args.reward_baseline) for out in pipe_outputs]
    # Manual path
    enc = tokenizer(
        texts,
        padding=("max_length" if config.pad_for_acceleration else True),
        truncation=True,
        max_length=script_args.input_max_length + script_args.output_max_length,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    target_device = torch.device("hpu") if device == "hpu" else torch.device(device)
    input_ids = input_ids.to(target_device)
    attention_mask = attention_mask.to(target_device)
    with torch.no_grad():
        outputs = reward_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = getattr(outputs, "logits", None)
        if logits is None:
            # fallback: assume first element
            if isinstance(outputs, (tuple, list)) and len(outputs) > 0:
                logits = outputs[0]
            else:
                raise RuntimeError("[Reward][Fallback] Could not obtain logits from reward model outputs.")
        if logits.dim() == 2 and logits.size(1) == 1:
            logits = logits.squeeze(1)
        logits = logits.detach().float().cpu()
    return [torch.tensor(v.item() - script_args.reward_baseline) for v in logits]
# We then define the arguments to pass to the `generate` function. These arguments
# are passed to the `generate` function of the PPOTrainer, which is a wrapper around
# the `generate` function of the trained model.
generation_kwargs = {
    # "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": tokenizer.eos_token_id,
}
if script_args.disable_kv_cache:
    generation_kwargs["use_cache"] = False

# Dynamic output length management
fixed_padding_mode = config.pad_for_acceleration  # when True we cannot vary length safely
adaptive_current_max = script_args.output_max_length
if script_args.initial_output_max_length and script_args.length_ramp_epochs and not fixed_padding_mode:
    if script_args.initial_output_max_length < script_args.output_max_length:
        adaptive_current_max = script_args.initial_output_max_length
        logger.info(
            f"[LengthRamp] Starting output_max_length ramp: {adaptive_current_max} -> {script_args.output_max_length} over {script_args.length_ramp_epochs} epochs"
        )
    else:
        logger.warning("[LengthRamp] initial_output_max_length >= output_max_length -> ignoring ramp settings")

def compute_current_max(epoch: int):
    if fixed_padding_mode:
        return script_args.output_max_length
    if script_args.initial_output_max_length and script_args.length_ramp_epochs and script_args.initial_output_max_length < script_args.output_max_length:
        if epoch < script_args.length_ramp_epochs:
            span = script_args.output_max_length - script_args.initial_output_max_length
            frac = epoch / max(1, script_args.length_ramp_epochs)
            return int(script_args.initial_output_max_length + span * frac)
    return adaptive_current_max

def build_length_sampler(cur_max: int):
    # Keep a modest minimum (32) but ensure <= cur_max
    min_len = min(32, cur_max)
    if fixed_padding_mode:
        return LengthSampler(cur_max, cur_max + 1)
    return LengthSampler(min_len, cur_max)

s0 = time.time()
sample = 0
for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    if epoch >= config.total_ppo_epochs:
        logger.info("[Loop] Reached total_ppo_epochs -> breaking loop")
        break
    question_tensors = batch["input_ids"]
    # Normalize to list[Tensor(seq_len)] as required by GaudiPPOTrainer.generate
    if isinstance(question_tensors, torch.Tensor):
        if question_tensors.dim() == 2:  # (batch, seq)
            question_tensors = [question_tensors[i] for i in range(question_tensors.size(0))]
        elif question_tensors.dim() == 1:  # single sequence tensor
            question_tensors = [question_tensors]
        else:
            raise ValueError(f"[Collator] Unexpected input_ids tensor shape {tuple(question_tensors.shape)}")
    elif isinstance(question_tensors, list):
        # ensure every element is 1D tensor
        cleaned = []
        for q in question_tensors:
            if isinstance(q, torch.Tensor):
                if q.dim() != 1:
                    cleaned.append(q.view(-1))
                else:
                    cleaned.append(q)
            else:
                cleaned.append(torch.as_tensor(q, dtype=torch.long).view(-1))
        question_tensors = cleaned
    else:
        raise TypeError(f"[Collator] input_ids unexpected type {type(question_tensors)}")
    sample = sample + len(question_tensors)
    # Determine dynamic output length for this epoch (if ramping enabled)
    cur_max_len = compute_current_max(epoch)
    output_length_sampler = build_length_sampler(cur_max_len)
    # Per-sample generation (variable-length). Avoid stacking to allow differing lengths.
    responses = []
    for qi in question_tensors:
        try:
            out = ppo_trainer.generate(
                qi,
                return_prompt=False,
                length_sampler=output_length_sampler,
                **generation_kwargs,
            )
        except RuntimeError as gen_e:
            if "Out of HOST memory" in str(gen_e):
                # Adaptive shrink and retry
                if not fixed_padding_mode:
                    new_max = int(cur_max_len * script_args.oom_shrink_factor)
                    if new_max < script_args.min_output_max_length:
                        new_max = script_args.min_output_max_length
                    logger.warning(
                        f"[OOM][Adaptive] Caught HOST OOM at length {cur_max_len}. Reducing max length to {new_max} and retrying generation."
                    )
                    cur_max_len = new_max
                    output_length_sampler = build_length_sampler(cur_max_len)
                    try:
                        out = ppo_trainer.generate(
                            qi,
                            return_prompt=False,
                            length_sampler=output_length_sampler,
                            **generation_kwargs,
                        )
                    except Exception as retry_e:
                        logger.exception(f"[OOM][Adaptive] Retry after shrink failed: {retry_e}")
                        raise
                else:
                    logger.exception("[OOM] HOST OOM in fixed padding mode; cannot adapt length. Abort.")
                    raise
            else:
                raise
        if isinstance(out, torch.Tensor):
            if out.dim() == 2 and out.size(0) == 1:
                out_seq = out[0]
            else:
                out_seq = out.view(-1)
        else:
            raise TypeError(f"[Generate] Unexpected output type {type(out)}")
        responses.append(out_seq)
    if epoch == 0:
        logger.info("[Generate] Using per-sample generation path (batched path disabled). Variable lengths supported.")
    if epoch % 10 == 0:
        logger.info(f"[Length] Epoch {epoch} current_max_generation_length={cur_max_len}")
    batch["response"] = tokenizer.batch_decode([t.tolist() for t in responses], skip_special_tokens=True)

    # Compute reward scores
    texts = [q + r for q, r in zip(batch["query"], batch["response"])]
    rewards = score_texts(texts)
    if epoch % 10 == 0:
        # compute simple stats
        rstack = torch.stack(rewards)
        logger.info(
            f"[Loop] Epoch {epoch} reward sample={rewards[0].item():.4f} mean={rstack.mean().item():.4f} std={rstack.std(unbiased=False).item():.4f} min={rstack.min().item():.4f} max={rstack.max().item():.4f}"
        )

    # Run PPO step
    stats = ppo_trainer.step(question_tensors, responses, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)

    # Optional memory stats logging (best-effort)
    if epoch % 20 == 0:
        try:
            import habana_frameworks.torch.hpu as hpu
            mem_stats = hpu.memory_stats()
            total = mem_stats.get('Total Memory', 'NA')
            free = mem_stats.get('Free Memory', 'NA')
            logger.info(f"[Memory][HPU] Stats: total={total} free={free}")
        except Exception:
            pass

    if script_args.save_freq and epoch and epoch % script_args.save_freq == 0:
        logger.info(f"[IF] save_freq condition met at epoch {epoch} -> saving intermediate model")
        ppo_trainer.save_pretrained(script_args.output_dir + f"step_{epoch}")
s1 = time.time()

ppo_trainer.save_pretrained(script_args.output_dir)
metrics = {"train_runtime": s1 - s0, "train_samples_per_second": sample / (s1 - s0)}
with open(f"{script_args.output_dir}/all_results.json", mode="w") as file:
    json.dump(metrics, file)
logger.info("[INFO] PPO training complete. Metrics saved.")

# Optional merge of policy adapter into base model + export value head
if script_args.merge_adapter_after_train:
    try:
        adapter_dir = script_args.output_dir
        merged_dir = script_args.merged_output_dir or f"{adapter_dir}_merged"
        if os.path.exists(merged_dir) and not script_args.merge_overwrite:
            logger.warning(f"[PolicyMerge] {merged_dir} exists and merge_overwrite=False -> skipping merge")
        else:
            logger.info(f"[PolicyMerge] Merging policy adapter from {adapter_dir} -> {merged_dir}")
            peft_conf = PeftConfig.from_pretrained(adapter_dir)
            base_model_name = getattr(peft_conf, 'base_model_name_or_path', None)
            if not base_model_name or base_model_name in {"None", "none", ""}:
                candidate = script_args.model_name_or_path
                if candidate and os.path.isdir(candidate) and os.path.isfile(os.path.join(candidate, 'config.json')):
                    base_model_name = candidate
                    logger.warning(f"[PolicyMerge][Fallback] base_model_name_or_path missing; using {base_model_name}")
                else:
                    raise ValueError("[PolicyMerge][Error] PEFT config missing base_model_name_or_path and no valid local fallback.")
            base_model_fresh = AutoModelForCausalLM.from_pretrained(
                base_model_name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
            )
            peft_loaded = PeftModel.from_pretrained(base_model_fresh, adapter_dir)
            try:
                merged_base = peft_loaded.merge_and_unload()
                logger.info("[PolicyMerge] merge_and_unload successful")
            except Exception as e:
                logger.warning(f"[PolicyMerge] merge_and_unload failed ({e}); using adapter model directly")
                merged_base = peft_loaded
            os.makedirs(merged_dir, exist_ok=True)
            merged_base.save_pretrained(merged_dir)
            tokenizer.save_pretrained(merged_dir)
            # Save value head weights + metadata
            if hasattr(model, 'v_head'):
                vh_state = model.v_head.state_dict()
                torch.save(vh_state, os.path.join(merged_dir, 'value_head.bin'))
                hidden_size = None
                # try to infer hidden size
                for k, v in vh_state.items():
                    if v.dim() == 2:
                        hidden_size = v.shape[1]
                        break
                meta = {"value_head_type": "linear", "hidden_size": hidden_size}
                with open(os.path.join(merged_dir, 'value_head_config.json'), 'w') as mf:
                    json.dump(meta, mf)
                logger.info(f"[PolicyMerge] Saved value head + metadata (hidden_size={hidden_size})")
            else:
                logger.warning("[PolicyMerge] model has no v_head attribute; skipping value head export")
            logger.info(f"[PolicyMerge] Merged policy saved to {merged_dir}")
    except Exception as e:
        logger.exception(f"[PolicyMerge] Failed to merge policy adapter: {e}")
