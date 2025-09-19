"""Single-solution Gaudi3 LoRA finetune POC on AG News (Llama-3).

This script intentionally:
 - Uses a user-specified subset of AG News (fast, 4-class)
 - Applies LoRA for parameter-efficient finetuning (Meta-Llama-3-8B)
 - Reports baseline vs post-train accuracy and delta
 - Keeps runtime short (< ~10 min on Gaudi3) via small subsets & capped steps
"""

import os
import argparse
import random
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
from optimum.habana import GaudiTrainer, GaudiConfig, GaudiTrainingArguments
from peft import LoraConfig, get_peft_model


def parse_args():
    p = argparse.ArgumentParser(description="Gaudi3 LoRA Finetune on AG News subset (Llama-3)")
    p.add_argument("--train-samples", type=int, default=400, help="# of AG News train samples")
    p.add_argument("--eval-samples", type=int, default=200, help="# of AG News test samples")
    p.add_argument("--max-length", type=int, default=96, help="Max token length")
    p.add_argument("--lora-r", type=int, default=8, help="LoRA rank")
    p.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")
    p.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    p.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")
    p.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay")
    p.add_argument("--max-steps", type=int, default=300, help="Max optimizer steps (caps training)")
    p.add_argument("--num-train-epochs", type=float, default=3.0, help="Epochs (ignored once max_steps reached)")
    p.add_argument("--batch-size", type=int, default=8, help="Per-device train batch size")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--full-eval-report", action="store_true", help="Print per-sample predictions pre & post")
    p.add_argument("--metrics-json", default="results/metrics.json", help="Write metrics JSON here")
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_agnews_subset(train_n: int, eval_n: int, tokenizer, max_len: int, seed: int):
    print(f"[Data] Loading AG News subset train={train_n} eval={eval_n}")
    raw_train = load_dataset("ag_news", split="train")
    raw_eval = load_dataset("ag_news", split="test")
    raw_train = raw_train.shuffle(seed=seed).select(range(min(train_n, len(raw_train))))
    raw_eval = raw_eval.shuffle(seed=seed).select(range(min(eval_n, len(raw_eval))))

    def tok_fn(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=max_len)

    train_tok = raw_train.map(tok_fn, batched=True)
    eval_tok = raw_eval.map(tok_fn, batched=True)
    train_tok = train_tok.rename_column("label", "labels")
    eval_tok = eval_tok.rename_column("label", "labels")
    keep = ["input_ids", "attention_mask", "labels"]
    train_tok.set_format(type="torch", columns=keep)
    eval_tok.set_format(type="torch", columns=keep)
    return train_tok, eval_tok, list(raw_eval["text"])


def build_lora_model(r: int, alpha: int, dropout: float):
    base = transformers.AutoModelForSequenceClassification.from_pretrained(
        "meta-llama/Meta-Llama-3-8B", num_labels=4
    )
    # Patch for Optimum Habana compatibility (if needed)
    gen_cfg = getattr(base, "generation_config", None)
    if gen_cfg is not None:
        for attr in ["attn_softmax_bf16", "use_flash_attention", "flash_attention_recompute", "flash_attention_causal_mask"]:
            if not hasattr(gen_cfg, attr):
                setattr(gen_cfg, attr, False)
    lora_cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="SEQ_CLS",
    )
    peft_model = get_peft_model(base, lora_cfg)
    peft_model.print_trainable_parameters()
    return peft_model


def count_params(m):
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, trainable


def evaluate(
    trainer: GaudiTrainer,
    dataset,
    tag: str,
    eval_texts=None,
    show_samples: bool = True,
    sample_limit: int = 5,
    decode_when_missing_texts: bool = True,
    baseline_preds=None,
):
    """Run prediction and report accuracy.

    Returns (accuracy, predictions ndarray, gold labels ndarray, decoded_texts or None)
    When show_samples=True, prints up to sample_limit rows including:
      index, gold label (numeric + name), predicted label, correctness mark, and input text.
    If eval_texts isn't provided and decode_when_missing_texts=True, it will attempt to decode
    each sample's input_ids via the trainer tokenizer.
    """
    out = trainer.predict(dataset)
    preds = out.predictions.argmax(-1)
    gold = out.label_ids
    acc = float((preds == gold).mean())
    print(f"Accuracy {tag}: {acc:.4f}")

    decoded_texts = eval_texts
    if show_samples:
        label_names = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
        if decoded_texts is None and decode_when_missing_texts:
            decoded_texts = []
            tokenizer = trainer.tokenizer
            for i in range(len(dataset)):
                ids = dataset[i]["input_ids"].tolist() if hasattr(dataset[i]["input_ids"], 'tolist') else dataset[i]["input_ids"]
                decoded_texts.append(tokenizer.decode(ids, skip_special_tokens=True))

        header_extra = f" (showing up to {sample_limit})"
        if baseline_preds is not None and tag != "before":
            print(f"-- {tag} sample predictions vs baseline{header_extra} --")
        else:
            print(f"-- {tag} sample predictions{header_extra} --")
        for i in range(min(sample_limit, len(preds))):
            g = int(gold[i])
            p = int(preds[i])
            txt = decoded_texts[i] if decoded_texts is not None else "<no-text-available>"
            mark = "✓" if g == p else "✗"
            g_name = label_names.get(g, str(g))
            p_name = label_names.get(p, str(p))
            clipped = (txt[:197] + '…') if len(txt) > 200 else txt
            if baseline_preds is not None and tag != "before":
                bp = int(baseline_preds[i])
                bp_name = label_names.get(bp, str(bp))
                changed = "→" if bp != p else "="
                correctness_shift = "WRONG→RIGHT" if (bp != g and p == g) else (
                    "RIGHT→WRONG" if (bp == g and p != g) else "—")
                print(
                    f"[{i}] {mark} gold={g}({g_name}) base={bp}({bp_name}) {changed} pred={p}({p_name}) {correctness_shift} :: {clipped}"
                )
            else:
                print(f"[{i}] {mark} gold={g}({g_name}) pred={p}({p_name}) :: {clipped}")
    return acc, preds, gold, decoded_texts


def main():
    args = parse_args()
    os.makedirs("results", exist_ok=True)
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
    tokenizer.pad_token = tokenizer.eos_token
    model = build_lora_model(args.lora_r, args.lora_alpha, args.lora_dropout)
    # Ensure model has pad_token_id set for batch processing
    if hasattr(model, "config"):
        model.config.pad_token_id = tokenizer.pad_token_id

    train_ds, eval_ds, eval_texts = load_agnews_subset(
        args.train_samples, args.eval_samples, tokenizer, args.max_length, args.seed
    )

    gaudi_config = GaudiConfig()
    training_args = GaudiTrainingArguments(
        output_dir="./results",
        use_habana=True,
        use_lazy_mode=True,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        logging_steps=10,
        save_steps=10_000,  # effectively no mid-run checkpoint
        report_to=[],
        do_eval=False,
        dataloader_num_workers=0,
    )

    trainer = GaudiTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        tokenizer=tokenizer,
        gaudi_config=gaudi_config,
    )

    total_params, trainable_params = count_params(model)
    print(
        f"[Params] total={total_params/1e6:.2f}M trainable={trainable_params/1e6:.2f}M "
        f"({trainable_params/total_params*100:.2f}% trainable)"
    )

    baseline_acc, baseline_preds, gold_labels, decoded_eval_texts = evaluate(
        trainer, eval_ds, "before", eval_texts, args.full_eval_report
    )
    trainer.train()
    final_acc, final_preds, gold_labels_after, _ = evaluate(
        trainer, eval_ds, "after", decoded_eval_texts, args.full_eval_report, baseline_preds=baseline_preds
    )
    # Sanity check labels unchanged
    if (gold_labels != gold_labels_after).any():
        print("[Warn] Gold labels changed between evaluations (unexpected).")

    # Change analysis
    improved = ((baseline_preds != gold_labels) & (final_preds == gold_labels)).sum()
    regressed = ((baseline_preds == gold_labels) & (final_preds != gold_labels)).sum()
    stayed_correct = ((baseline_preds == gold_labels) & (final_preds == gold_labels)).sum()
    stayed_wrong = ((baseline_preds != gold_labels) & (final_preds != gold_labels)).sum()

    print("\n[Change Analysis]")
    total_eval = len(gold_labels)
    print(f"Total evaluated: {total_eval}")
    print(f" Improved (wrong->right): {improved}")
    print(f" Regressed (right->wrong): {regressed}")
    print(f" Stayed correct: {stayed_correct}")
    print(f" Stayed wrong: {stayed_wrong}")
    if improved + regressed > 0:
        precision_change = improved / (improved + regressed)
        print(f" Precision of changes (improved / (improved+regressed)): {precision_change:.3f}")
    net_change = improved - regressed
    print(f" Net change (improved - regressed): {net_change}")

    # Optionally list first few improved/regressed examples if full report
    if args.full_eval_report:
        print("\n[First 5 improved examples indices]")
        improved_idx = ((baseline_preds != gold_labels) & (final_preds == gold_labels)).nonzero()[0][:5].tolist()
        print(improved_idx)
        print("[First 5 regressed examples indices]")
        regressed_idx = ((baseline_preds == gold_labels) & (final_preds != gold_labels)).nonzero()[0][:5].tolist()
        print(regressed_idx)
    delta = final_acc - baseline_acc
    print(f"Delta: {delta:+.4f}")
    print("Finetuning complete!")

    # Persist minimal metrics
    metrics = {
        "baseline_acc": baseline_acc,
        "final_acc": final_acc,
        "delta": delta,
        "train_samples": len(train_ds),
        "eval_samples": len(eval_ds),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "improved": int(improved),
        "regressed": int(regressed),
        "stayed_correct": int(stayed_correct),
        "stayed_wrong": int(stayed_wrong),
    }
    try:
        import json
        with open(args.metrics_json, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[Metrics] Saved to {args.metrics_json}")
    except Exception as e:
        print(f"[Metrics][Warn] Could not write metrics JSON: {e}")

    # Save LoRA adapter
    try:
        model.save_pretrained("results/lora_adapter")
        tokenizer.save_pretrained("results/lora_adapter")
        print("[LoRA] Adapter saved to results/lora_adapter")
    except Exception as e:
        print(f"[LoRA][Warn] Failed to save adapter: {e}")


if __name__ == "__main__":
    main()
