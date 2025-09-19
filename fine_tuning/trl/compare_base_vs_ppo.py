#!/usr/bin/env python
# compare_base_vs_ppo.py
import torch
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM, AutoModelForSequenceClassification

# Models
base_model = "meta-llama/Llama-2-7b-hf"
ppo_model = "./ppo_sanity"
reward_model_path = "./rm_sanity_merged"

# Shared tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token = tokenizer.eos_token

# Reward model
rm = AutoModelForSequenceClassification.from_pretrained(
    reward_model_path, num_labels=1, torch_dtype=torch.bfloat16
)
rm_pipe = pipeline("sentiment-analysis", model=rm, tokenizer=tokenizer, device=0)

prompts = [
    "Why do programmers prefer Python over Java for machine learning?",
    "What is the difference between supervised and unsupervised learning?",
    "What are the advantages of Docker for deploying applications?",
    "Why does gradient descent work even though it doesn’t always find the global minimum?"
]

def generate_response(model_name, prompt, max_new_tokens=64):
    """Generate response from a model"""
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to("hpu")
    inputs = tokenizer(prompt, return_tensors="pt").to("hpu")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

results = []
for prompt in prompts:
    print("\n" + "="*80)
    print(f"### Prompt: {prompt}")
    print("="*80)
    responses = {}
    for name, path in [("Base", base_model), ("PPO", ppo_model)]:
        resp = generate_response(path, prompt)
        score = rm_pipe(resp)[0]["score"]
        responses[name] = (resp, score)
        print(f"\n{name} Response:\n{resp}\n[Reward Score: {score:.4f}]")
    results.append((prompt, responses))

# Summary table
print("\n" + "="*80)
print("📊 Summary of Reward Scores (Base vs PPO)")
print("="*80)
for prompt, res in results:
    print(f"\nPrompt: {prompt}")
    for name in ["Base", "PPO"]:
        print(f"  {name:5s} -> Score: {res[name][1]:.4f}")
