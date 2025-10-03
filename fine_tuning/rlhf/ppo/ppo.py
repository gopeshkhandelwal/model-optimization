# Adapted from https://github.com/huggingface/optimum-habana/tree/v1.16.0/examples/trl
# Customized and enabled for Gaudi3
import json
import time
import logging
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import Adafactor, AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser, pipeline
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
    model_name_or_path: Optional[str] = field(default=None, metadata={"help": "the model name (REQUIRED)"})
    tokenizer_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "the tokenizer name (defaults to model_name_or_path if not provided)"}
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


adapt_PreTrainedModelWrapper_to_gaudi()
parser = HfArgumentParser(ScriptArguments)
from logging_utils import setup_logging
script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]
setup_logging()
logger = logging.getLogger(__name__)

# Validate required arguments
if not script_args.model_name_or_path:
    raise ValueError("--model_name_or_path is required. Please specify the base model path.")
if not script_args.tokenizer_name_or_path:
    script_args.tokenizer_name_or_path = script_args.model_name_or_path
    logger.info(f"tokenizer_name_or_path not provided, using model path: {script_args.tokenizer_name_or_path}")

logger.info(f"ScriptArguments: {script_args}")
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
    "return_all_scores": True,
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
    return {key: [d[key] for d in data] for key in data[0]}


# set seed before initializing value head for deterministic eval
set_seed(config.seed)

# Auto-detect LoRA target modules if not specified
if script_args.lora_target_modules is None:
    logger.info("[LoRA][AutoDefault] No lora_target_modules provided -> using ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'down_proj', 'gate_proj']")
    script_args.lora_target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'down_proj', 'gate_proj']

# Now let's build the model, the reference model, and the tokenizer.
current_device = GaudiAccelerator().local_process_index
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
)
model.config.use_fused_rope = False
model.config.use_fused_rms_norm = False
optimizer = None
model = model.to(torch.bfloat16)

def _param_stats(m):
    tot=trn=0
    for p in m.parameters():
        n=p.numel(); tot+=n; trn+= n if p.requires_grad else 0
    return tot,trn,(trn/tot*100 if tot else 0)
btot,btrn,bpct=_param_stats(model)
logger.info(f"[Model Params][Policy+ValueHead] total={btot:,} trainable={btrn:,} ({bpct:.4f}%)")

if script_args.use_habana:
    logger.info("[IF] use_habana == True -> loading reference model")
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
else:
    logger.info("[IF] use_habana == False -> no reference model (reference-free mode)")
    ref_model = None
if script_args.adafactor:
    logger.info("[IF] adafactor == True -> using Adafactor optimizer")
    optimizer = Adafactor(
        filter(lambda p: p.requires_grad, model.parameters()),
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        lr=config.learning_rate,
    )
# Force disable batched generation for stability (especially with Gemma3)
if script_args.batched_gen:
    logger.warning("[Trainer] batched_gen=True can cause issues with padding/masking -> forcing to False for stability")
    script_args.batched_gen = False

logger.info(f"[Generation] Using batched_gen={script_args.batched_gen}, batch_size={script_args.batch_size}")

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

# Try to load reward model, with fallback for unsupported models like Gemma3
reward_model_is_causal_lm = False
try:
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        reward_model_name,
        num_labels=1,
        low_cpu_mem_usage=True,
    )
    logger.info(f"[RewardModel] Loaded sequence classification reward model: {reward_model_name}")
except ValueError as e:
    if "Unrecognized configuration class" in str(e) and "AutoModelForSequenceClassification" in str(e):
        logger.warning(f"[RewardModel][Fallback] Could not load as SequenceClassification: {e}")
        logger.info("[RewardModel][Fallback] Attempting to load as causal LM reward wrapper")
        # Try to load as a merged causal LM with reward head (from reward_modeling.py output)
        from transformers import AutoModelForCausalLM
        import os
        import json
        
        # Check if this is a merged causal LM reward model
        reward_config_path = os.path.join(reward_model_name, "reward_head_config.json")
        if os.path.exists(reward_config_path):
            logger.info("[RewardModel][Fallback] Found reward head config, loading merged causal LM reward model")
            with open(reward_config_path, "r") as f:
                reward_config = json.load(f)
            
            reward_model = AutoModelForCausalLM.from_pretrained(
                reward_model_name,
                low_cpu_mem_usage=True,
                torch_dtype=torch.bfloat16,
                attn_implementation="eager",
                output_hidden_states=True
            )
            
            # Add the reward head
            import torch.nn as nn
            hidden_size = reward_config["hidden_size"]
            reward_model.score = nn.Linear(hidden_size, 1, bias=False)
            
            # Load the reward head weights
            reward_head_path = os.path.join(reward_model_name, "reward_value_head.bin")
            if os.path.exists(reward_head_path):
                reward_head_state = torch.load(reward_head_path, map_location="cpu")
                reward_model.score.load_state_dict(reward_head_state)
                logger.info("[RewardModel][Fallback] Loaded reward head weights")
            else:
                logger.warning("[RewardModel][Fallback] No reward head weights found, using random initialization")
            
            reward_model_is_causal_lm = True
            logger.info(f"[RewardModel][Fallback] Loaded merged causal LM reward model: {reward_model_name}")
        else:
            logger.error(f"[RewardModel][Fallback] No reward head config found at {reward_config_path}")
            raise ValueError(f"Cannot load reward model {reward_model_name} - neither sequence classification nor merged causal LM format")
    else:
        raise

if config.use_habana:
    from habana_frameworks.torch.hpu import wrap_in_hpu_graph

    reward_model = wrap_in_hpu_graph(reward_model)
    logger.info("[IF] use_habana == True -> reward model wrapped in HPU graph")

if device.type == "hpu":
    device = "hpu"

# Create reward function - use pipeline for sequence classification, custom function for causal LM
if reward_model_is_causal_lm:
    logger.info("[RewardModel] Using custom reward function for causal LM")
    def get_rewards(texts):
        """Custom reward function for causal LM reward models"""
        rewards = []
        reward_model.eval()
        with torch.no_grad():
            for text in texts:
                # Tokenize the text
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=sent_kwargs.get("max_length", script_args.input_max_length + script_args.output_max_length),
                    padding=sent_kwargs.get("padding", False)
                ).to(device)
                
                # Get model outputs
                outputs = reward_model(**inputs, output_hidden_states=True)
                
                # Get the last hidden state and compute reward
                hidden_states = outputs.hidden_states[-1]  # Last layer
                # Use the last non-padding token's hidden state
                sequence_lengths = inputs.attention_mask.sum(dim=1) - 1
                last_token_hidden = hidden_states[0, sequence_lengths[0]]
                
                # Compute reward score
                reward_score = reward_model.score(last_token_hidden.unsqueeze(0))
                rewards.append([{"score": reward_score.item()}])
        
        return rewards
else:
    logger.info("[RewardModel] Using pipeline for sequence classification")
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
    
    def get_rewards(texts):
        """Wrapper for pipeline-based reward computation"""
        return sentiment_pipe(texts, **sent_kwargs)
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
output_min_length = 32
output_max_length = script_args.output_max_length
if not config.pad_for_acceleration:
    output_length_sampler = LengthSampler(output_min_length, output_max_length)
else:
    output_length_sampler = LengthSampler(output_max_length, output_max_length + 1)
s0 = time.time()
sample = 0
for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    if epoch >= config.total_ppo_epochs:
        logger.info("[Loop] Reached total_ppo_epochs -> breaking loop")
        break
    question_tensors = batch["input_ids"]
    sample = sample + len(question_tensors)
    # Try PPO trainer generation with error handling
    logger.info(f"[Generation] Attempting PPO trainer generation for batch size {len(question_tensors)}")
    try:
        response_tensors = ppo_trainer.generate(
            question_tensors,
            return_prompt=False,
            length_sampler=output_length_sampler,
            **generation_kwargs,
        )
        logger.info(f"[Generation] PPO trainer generation succeeded")
    except Exception as e:
        logger.warning(f"[Generation] PPO trainer generation failed: {e}")
        logger.info(f"[Generation] Falling back to individual generation")
        
        # Fallback: Generate responses individually
        response_tensors = []
        for i, query_tensor in enumerate(question_tensors):
            logger.info(f"[Generation] Processing query {i+1}/{len(question_tensors)}")
            
            # Generate length for this sample
            gen_len = output_length_sampler()
            
            # Generate response for single query
            with torch.no_grad():
                single_response = ppo_trainer.model.generate(
                    query_tensor.unsqueeze(0),  # Add batch dimension
                    max_new_tokens=gen_len,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                # Remove the query part to get only the response
                response_only = single_response[0][len(query_tensor):]
                response_tensors.append(response_only)
        
        logger.info(f"[Generation] Fallback generation completed for {len(response_tensors)} responses")
    batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

    # Compute reward score (using the reward function)
    texts = [q + r for q, r in zip(batch["query"], batch["response"])]
    pipe_outputs = get_rewards(texts)
    rewards = [torch.tensor(output[0]["score"] - script_args.reward_baseline) for output in pipe_outputs]
    if epoch % 10 == 0:
        logger.info(f"[Loop] Epoch {epoch} sample reward: {rewards[0].item():.4f}")

    # Run PPO step
    stats = ppo_trainer.step(question_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)

    if script_args.save_freq and epoch and epoch % script_args.save_freq == 0:
        logger.info(f"[IF] save_freq condition met at epoch {epoch} -> saving intermediate model")
        ppo_trainer.save_pretrained(script_args.output_dir + f"step_{epoch}")
s1 = time.time()

ppo_trainer.save_pretrained(script_args.output_dir)
metrics = {"train_runtime": s1 - s0, "train_samples_per_second": sample / (s1 - s0)}
with open(f"{script_args.output_dir}/all_results.json", mode="w") as file:
    json.dump(metrics, file)
logger.info("[INFO] PPO training complete. Metrics saved.")
