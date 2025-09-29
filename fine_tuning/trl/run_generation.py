import argparse, torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name_or_path", required=True)
    ap.add_argument("--prompt", required=True, nargs="+",
                    help='One or more prompts. e.g. --prompt "Hello" "Second prompt"')
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--use_kv_cache", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=0)
    args = ap.parse_args()
    from logging_utils import setup_logging
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info(f"Args: {args}")

    dtype = torch.bfloat16 if args.bf16 else torch.float32
    if not (hasattr(torch, "hpu") and torch.hpu.is_available()):
        raise RuntimeError("[HPU][Required] Habana HPU not available. This script is configured to always use Gaudi/HPU.")
    device = torch.device("hpu")

    tok = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tok.pad_token is None:
        logger.info("[IF] tokenizer.pad_token is None -> setting to eos_token")
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=dtype, low_cpu_mem_usage=True)
    mt = getattr(model.config, 'model_type', '')
    if mt.startswith('gemma3'):
        try:
            model.config.use_cache = args.use_kv_cache  # allow override
            # Prefer eager attn for gemma3 stability
            if getattr(model.config, 'attn_implementation', None) is not None:
                model.config.attn_implementation = 'eager'
        except Exception:
            pass
    else:
        # For Llama keep cache flag user-specified
        model.config.use_cache = args.use_kv_cache
    model.to(device)
    model.eval()
    logger.info(f"[Model] Loaded {args.model_name_or_path} (type={mt}) to {device} dtype={dtype}")

    # Batch prompts (repeat/crop to batch_size)
    prompts = args.prompt
    if len(prompts) < args.batch_size:
        logger.info(f"[IF] Provided {len(prompts)} prompts < batch_size {args.batch_size} -> repeating prompts")
        prompts += prompts * ((args.batch_size + len(prompts) - 1) // len(prompts))
    prompts = prompts[:args.batch_size]

    inputs = tok(prompts, return_tensors="pt", padding=True).to(device)

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        use_cache=args.use_kv_cache,
        pad_token_id=tok.pad_token_id,
    )
    if args.top_k and args.top_k > 0:
        gen_kwargs["top_k"] = args.top_k
    logger.info(f"[Generate] Generation kwargs: {gen_kwargs}")

    with torch.inference_mode():
        out = model.generate(**inputs, **gen_kwargs)

    for i, o in enumerate(out):
        print(f"\n=== Sample {i} ===")
        print(tok.decode(o, skip_special_tokens=True))
    logger.info("[Done] Generation complete")

if __name__ == "__main__":
    main()
