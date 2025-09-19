from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
import os

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
tokenizer.pad_token = tokenizer.eos_token
dataset = load_dataset("openai/gsm8k", "main")
print("Available splits:", dataset.keys())
print("Train example keys:", dataset["train"][0].keys())

dataset = DatasetDict({
    "train": dataset["train"].select(range(100)),
    "validation": dataset["test"].select(range(5))
})

def preprocess(example):
    input_text = "Question: " + example["question"]
    label_text = example["answer"]
    inputs = tokenizer(input_text, truncation=True, padding="max_length", max_length=512)
    labels = tokenizer(label_text, truncation=True, padding="max_length", max_length=128)
    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": labels["input_ids"]
    }

os.makedirs("/workspace/data", exist_ok=True)
dataset = dataset.map(preprocess, remove_columns=dataset["train"].column_names)
dataset["train"].save_to_disk("/workspace/data/train_data")
dataset["validation"].save_to_disk("/workspace/data/eval_data")
