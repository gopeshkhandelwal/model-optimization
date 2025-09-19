from optimum.habana import GaudiTrainer, GaudiTrainingArguments, GaudiConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
import transformers
from peft import LoraConfig, get_peft_model

# Define model and config
model_name = "meta-llama/Meta-Llama-3-8B"

# Create a basic Gaudi config and DISABLE FUSED OPTIMIZERS
gaudi_config = GaudiConfig()
gaudi_config.use_fused_adam = False  # ← CRITICAL: Disable for LoRA compatibility
gaudi_config.use_fused_clip_norm = False  # ← Also disable fused clip norm
gaudi_config.use_lazy_mode = True

# Load pretrained model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto", 
    use_cache=True,
    low_cpu_mem_usage=True,
    token=True,
)

# Patch for Optimum Habana compatibility with LLaMA 3
generation_config_attrs = [
    "attn_softmax_bf16",
    "use_flash_attention",
    "flash_attention_recompute",
    "flash_attention_causal_mask",
    "use_fused_adam",
    "use_fused_clip_norm"
    
]
for attr in generation_config_attrs:
    if not hasattr(model.generation_config, attr):
        setattr(model.generation_config, attr, False)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# LoRA config for PEFT
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # Added more modules for better coverage
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # ← Good to verify LoRA is working

# Load preprocessed datasets
train_dataset = load_from_disk("/workspace/data/train_data")
eval_dataset = load_from_disk("/workspace/data/eval_data")

# Define training arguments
training_args = GaudiTrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=8,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=500,
    do_train=True,
    do_eval=True,
    use_habana=True,
    report_to="none",
    use_lazy_mode=True,
    throughput_warmup_steps=2,
    bf16=True,
    optim="adamw_torch",  # ← Use standard AdamW instead of fused version
)

# Initialize trainer
trainer = GaudiTrainer(
    model=model,
    gaudi_config=gaudi_config,  # ← Pass the GaudiConfig object
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

# Train and save model
trainer.train()
trainer.save_model("/workspace/data/finetuned_llama3_lora")
