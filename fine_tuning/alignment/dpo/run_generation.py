#!/usr/bin/env python
# run_generation.py - Text generation script for DPO models

import argparse
import logging
import time
from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from logging_utils import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text using DPO-trained models")
    parser.add_argument("--model_name_or_path", required=True, help="Path to the model (REQUIRED)")
    parser.add_argument("--tokenizer_name_or_path", help="Path to tokenizer (defaults to model path)")
    parser.add_argument("--prompt", required=True, help="Input prompt for generation (REQUIRED)")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p (nucleus) sampling parameter")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling parameter")
    parser.add_argument("--do_sample", action="store_true", help="Use sampling instead of greedy decoding")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="Number of sequences to generate")
    parser.add_argument("--seed", type=int, help="Random seed for reproducible generation")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for generation")
    
    return parser.parse_args()


def setup_model_and_tokenizer(model_path: str, tokenizer_path: str = None):
    """Load model and tokenizer"""
    if tokenizer_path is None:
        tokenizer_path = model_path
    
    logger.info(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token")
    
    logger.info(f"Loading model from: {model_path}")
    device = "hpu" if hasattr(torch, "hpu") else ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
        low_cpu_mem_usage=True,
        attn_implementation="eager",  # For Gemma3 compatibility
    )
    model.to(device)
    model.eval()
    
    return model, tokenizer, device


def generate_text(
    model, 
    tokenizer, 
    device: str,
    prompt: str, 
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    do_sample: bool = True,
    num_return_sequences: int = 1,
) -> List[str]:
    """Generate text using the model"""
    
    logger.info(f"Generating {num_return_sequences} sequence(s) for prompt: {prompt[:50]}...")
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generation parameters
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "do_sample": do_sample,
        "num_return_sequences": num_return_sequences,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    
    logger.info(f"Generation parameters: {generation_kwargs}")
    
    # Generate
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_kwargs)
    generation_time = time.time() - start_time
    
    # Decode outputs
    generated_texts = []
    input_length = inputs.input_ids.shape[1]
    
    for i, output in enumerate(outputs):
        # Remove input tokens from output
        generated_tokens = output[input_length:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        generated_texts.append(generated_text)
        
        logger.info(f"Generated sequence {i+1}/{num_return_sequences} ({len(generated_tokens)} tokens)")
    
    logger.info(f"Generation completed in {generation_time:.2f}s")
    logger.info(f"Tokens per second: {len(generated_tokens) * num_return_sequences / generation_time:.2f}")
    
    return generated_texts


def main():
    args = parse_args()
    setup_logging()
    global logger
    logger = logging.getLogger(__name__)
    
    # Validate required arguments
    if not args.model_name_or_path:
        raise ValueError("--model_name_or_path is required")
    if not args.prompt:
        raise ValueError("--prompt is required")
    
    # Set seed if provided
    if args.seed is not None:
        torch.manual_seed(args.seed)
        logger.info(f"Set random seed to {args.seed}")
    
    # Load model and tokenizer
    model, tokenizer, device = setup_model_and_tokenizer(
        args.model_name_or_path, 
        args.tokenizer_name_or_path
    )
    
    # Generate text
    generated_texts = generate_text(
        model=model,
        tokenizer=tokenizer,
        device=device,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        do_sample=args.do_sample,
        num_return_sequences=args.num_return_sequences,
    )
    
    # Display results
    print("\n" + "="*80)
    print("🤖 DPO Model Generation Results")
    print("="*80)
    print(f"📝 Prompt: {args.prompt}")
    print(f"🔧 Model: {args.model_name_or_path}")
    print(f"⚙️  Max tokens: {args.max_new_tokens}, Temperature: {args.temperature}, Top-p: {args.top_p}")
    print("-"*80)
    
    for i, text in enumerate(generated_texts, 1):
        print(f"\n🎯 Generation {i}:")
        print(f"{text}")
        print(f"📊 Length: {len(text.split())} words, {len(text)} characters")
    
    print("\n" + "="*80)
    print("✅ Generation completed successfully!")


if __name__ == "__main__":
    main()