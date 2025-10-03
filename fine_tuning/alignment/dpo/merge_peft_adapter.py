#!/usr/bin/env python
# merge_peft_adapter.py - Merge PEFT adapters with base models

import argparse
import logging
import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from logging_utils import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Merge PEFT adapter with base model")
    parser.add_argument("--adapter_model_name", required=True, help="Path to the PEFT adapter model (REQUIRED)")
    parser.add_argument("--base_model_name", required=True, help="Path to the base model (REQUIRED)")
    parser.add_argument("--output_name", required=True, help="Output path for merged model (REQUIRED)")
    parser.add_argument("--push_to_hub", action="store_true", help="Push merged model to HuggingFace Hub")
    parser.add_argument("--hub_model_id", help="Model ID for HuggingFace Hub (required if push_to_hub)")
    parser.add_argument("--token", help="HuggingFace token for pushing to hub")
    parser.add_argument("--safe_merge", action="store_true", help="Use safe merge (slower but more stable)")
    
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Validate required arguments
    if not args.adapter_model_name:
        raise ValueError("--adapter_model_name is required")
    if not args.base_model_name:
        raise ValueError("--base_model_name is required")
    if not args.output_name:
        raise ValueError("--output_name is required")
    
    if args.push_to_hub and not args.hub_model_id:
        raise ValueError("--hub_model_id is required when push_to_hub is enabled")
    
    logger.info(f"Merging PEFT adapter:")
    logger.info(f"  Base model: {args.base_model_name}")
    logger.info(f"  Adapter: {args.adapter_model_name}")
    logger.info(f"  Output: {args.output_name}")
    
    # Check if adapter path exists
    if not os.path.exists(args.adapter_model_name):
        raise FileNotFoundError(f"Adapter model not found: {args.adapter_model_name}")
    
    # Determine device
    device = "hpu" if hasattr(torch, "hpu") else ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    try:
        # Load base model
        logger.info("Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model_name,
            torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
            low_cpu_mem_usage=True,
            device_map="auto" if device != "hpu" else None,
            attn_implementation="eager",  # For Gemma3 compatibility
        )
        
        # Load PEFT model
        logger.info("Loading PEFT adapter...")
        peft_model = PeftModel.from_pretrained(
            base_model,
            args.adapter_model_name,
            torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
        )
        
        # Merge adapter with base model
        logger.info("Merging adapter with base model...")
        if args.safe_merge:
            logger.info("Using safe merge (this may take longer)")
            merged_model = peft_model.merge_and_unload(safe_merge=True)
        else:
            merged_model = peft_model.merge_and_unload()
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)
        
        # Create output directory
        output_path = Path(args.output_name)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save merged model
        logger.info(f"Saving merged model to {args.output_name}")
        merged_model.save_pretrained(
            args.output_name,
            safe_serialization=True,
            max_shard_size="5GB"
        )
        
        # Save tokenizer
        logger.info("Saving tokenizer...")
        tokenizer.save_pretrained(args.output_name)
        
        # Get model info
        total_params = sum(p.numel() for p in merged_model.parameters())
        logger.info(f"Merged model has {total_params:,} parameters")
        
        # Push to hub if requested
        if args.push_to_hub:
            logger.info(f"Pushing merged model to HuggingFace Hub: {args.hub_model_id}")
            merged_model.push_to_hub(
                args.hub_model_id,
                token=args.token,
                safe_serialization=True,
            )
            tokenizer.push_to_hub(
                args.hub_model_id,
                token=args.token,
            )
            logger.info("Successfully pushed to HuggingFace Hub!")
        
        logger.info("✅ PEFT adapter merge completed successfully!")
        logger.info(f"📁 Merged model saved at: {args.output_name}")
        
        # Create a simple info file
        info = {
            "base_model": args.base_model_name,
            "adapter_model": args.adapter_model_name,
            "output_path": args.output_name,
            "total_parameters": total_params,
            "merge_type": "safe" if args.safe_merge else "standard",
        }
        
        with open(output_path / "merge_info.json", "w") as f:
            import json
            json.dump(info, f, indent=2)
        
        logger.info("📄 Merge info saved to merge_info.json")
        
    except Exception as e:
        logger.error(f"❌ Error during merge: {str(e)}")
        raise
    
    finally:
        # Cleanup GPU memory
        if 'base_model' in locals():
            del base_model
        if 'peft_model' in locals():
            del peft_model
        if 'merged_model' in locals():
            del merged_model
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch, "hpu") and torch.hpu.is_available():
            # HPU doesn't have empty_cache, just skip memory cleanup
            logger.info("HPU detected - skipping memory cache cleanup")


if __name__ == "__main__":
    main()