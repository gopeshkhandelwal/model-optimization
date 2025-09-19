from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
from evaluate import load
import torch
import re
from typing import List, Dict, Tuple

class ModelEvaluator:
    def __init__(self, model_path: str, tokenizer_path: str, device: str = "hpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def extract_final_number(self, text: str) -> str:
        """Extract the final numerical answer from text"""
        numbers = re.findall(r'-?\d+\.?\d*', text)
        return numbers[-1] if numbers else ""

    def calculate_exact_match(self, predictions: List[str], references: List[str]) -> float:
        """Calculate exact match accuracy for final answers"""
        correct = 0
        for pred, ref in zip(predictions, references):
            pred_answer = self.extract_final_number(pred)
            ref_answer = self.extract_final_number(ref)
            if pred_answer and ref_answer and pred_answer == ref_answer:
                correct += 1
        return correct / len(predictions) if predictions else 0

    def evaluate(self, dataset, max_examples: int = 100) -> Tuple[Dict, List[str], List[str], List[str]]:
        """Evaluate model on dataset
        Returns: (metrics_dict, predictions, references, inputs_decoded)"""
        predictions = []
        references = []
        inputs_decoded = []

        for idx, example in enumerate(dataset):
            if idx >= max_examples:
                break

            try:
                # Prepare input
                input_ids = torch.tensor(example["input_ids"]).unsqueeze(0).to(self.device)
                attention_mask = torch.tensor(example["attention_mask"]).unsqueeze(0).to(self.device)

                # Generate prediction
                with torch.no_grad():
                    output = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=128,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id,
                        temperature=0.1,
                        num_return_sequences=1
                    )

                # Extract only the generated part (remove input)
                generated_ids = output[0][len(input_ids[0]):]
                pred_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

                # Decode original input (acts as the question/prompt)
                input_text = self.tokenizer.decode(example["input_ids"], skip_special_tokens=True)

                # Get reference (gold answer)
                ref_text = self.tokenizer.decode(example["labels"], skip_special_tokens=True)

                predictions.append(pred_text.strip())
                references.append(ref_text.strip())
                inputs_decoded.append(input_text.strip())

            except Exception as e:
                print(f"Error processing example {idx}: {e}")
                continue

        # Calculate metrics
        rouge = load("rouge")
        rouge_scores = rouge.compute(predictions=predictions, references=references)
        exact_match = self.calculate_exact_match(predictions, references)

        results = {
            "rouge": rouge_scores,
            "exact_match": exact_match,
            "num_evaluated": len(predictions)
        }

        return results, predictions, references, inputs_decoded

def print_comparison(base_results: Dict, ft_results: Dict, 
                    base_preds: List[str], ft_preds: List[str],
                    references: List[str], inputs: List[str],
                    model_names: Tuple[str, str]):
    """Print detailed comparison between models"""

    base_name, ft_name = model_names

    print("=" * 80)
    print("MODEL EVALUATION COMPARISON")
    print("=" * 80)

    # Metrics comparison
    print(f"\n📊 METRICS COMPARISON (based on {base_results['num_evaluated']} examples):")
    print("-" * 60)
    print(f"{'Metric':<15} {'Baseline':<12} {'Fine-tuned':<12} {'Difference':<12}")
    print("-" * 60)

    for rouge_key in ['rouge1', 'rouge2', 'rougeL']:
        base_val = base_results['rouge'][rouge_key]
        ft_val = ft_results['rouge'][rouge_key]
        diff = ft_val - base_val
        print(f"{rouge_key:<15} {base_val:.4f}      {ft_val:.4f}      {diff:+.4f}")

    base_em = base_results['exact_match']
    ft_em = ft_results['exact_match']
    em_diff = ft_em - base_em
    print(f"{'exact_match':<15} {base_em:.4f}      {ft_em:.4f}      {em_diff:+.4f}")

    # Improvement analysis
    print(f"\n📈 IMPROVEMENT ANALYSIS:")
    print("-" * 40)
    rouge_improved = sum(1 for key in ['rouge1', 'rouge2', 'rougeL'] 
                        if ft_results['rouge'][key] > base_results['rouge'][key])
    print(f"ROUGE metrics improved: {rouge_improved}/3")
    print(f"Exact Match improved: {'Yes' if ft_em > base_em else 'No'}")

    if ft_em > base_em:
        improvement_pct = (ft_em - base_em) / base_em * 100
        print(f"Exact Match improvement: {improvement_pct:+.1f}%")

    # Sample comparisons
    print(f"\n🔍 SAMPLE PREDICTIONS (first 3 examples):")
    print("-" * 60)

    for i in range(min(3, len(base_preds))):
        print(f"\nExample {i + 1}:")
        # Show truncated question for readability if very long
        q = inputs[i]
        if len(q) > 300:
            q = q[:300] + "..."
        print(f"Question:   {q}")
        print(f"Reference:  {references[i]}")
        print(f"Baseline:   {base_preds[i]}")
        print(f"Fine-tuned: {ft_preds[i]}")

        # Highlight differences
        if base_preds[i] != ft_preds[i]:
            if ft_preds[i] == references[i]:
                print("✅ Fine-tuned model CORRECT!")
            elif base_preds[i] == references[i]:
                print("❌ Fine-tuned model REGRESSION!")
            else:
                print("🔁 Different prediction, check manually")
        print("-" * 40)

def main():
    # Configuration
    BASE_MODEL_PATH = "meta-llama/Meta-Llama-3-8B"
    FINE_TUNED_PATH = "/workspace/data/finetuned_llama3_lora"
    EVAL_DATA_PATH = "/workspace/data/eval_data"
    MAX_EXAMPLES = 100  # Evaluate on first 100 examples
    DEVICE = "hpu"

    print("Loading evaluation dataset...")
    eval_dataset = load_from_disk(EVAL_DATA_PATH)

    # Initialize evaluators
    print("Initializing models...")
    base_evaluator = ModelEvaluator(BASE_MODEL_PATH, BASE_MODEL_PATH, DEVICE)
    ft_evaluator = ModelEvaluator(FINE_TUNED_PATH, BASE_MODEL_PATH, DEVICE)

    # Evaluate baseline model
    print("\nEvaluating baseline model...")
    base_results, base_preds, base_refs, base_inputs = base_evaluator.evaluate(eval_dataset, MAX_EXAMPLES)

    # Evaluate fine-tuned model
    print("Evaluating fine-tuned model...")
    ft_results, ft_preds, ft_refs, ft_inputs = ft_evaluator.evaluate(eval_dataset, MAX_EXAMPLES)

    # Ensure we're comparing the same references
    assert base_refs == ft_refs, "Reference mismatch between evaluations!"
    assert base_inputs == ft_inputs, "Input mismatch between evaluations!"

    # Print comprehensive comparison
    print_comparison(
        base_results, ft_results, 
    base_preds, ft_preds, 
    base_refs, base_inputs,
        ("Baseline", "Fine-tuned")
    )

    # Final summary
    print("\n" + "=" * 80)
    print("🎯 FINAL VERDICT")
    print("=" * 80)

    if (ft_results['exact_match'] > base_results['exact_match'] or
        any(ft_results['rouge'][k] > base_results['rouge'][k] for k in ['rouge1', 'rouge2', 'rougeL'])):
        print("✅ FINE-TUNING SUCCESSFUL: Model shows improvement!")
    else:
        print("❌ FINE-TUNING UNSUCCESSFUL: No measurable improvement detected.")
        print("   Possible reasons:")
        print("   - Insufficient training data/epochs")
        print("   - Learning rate too high/low")
        print("   - Evaluation metric not capturing improvements")
        print("   - Technical issues during fine-tuning")

if __name__ == "__main__":
    main()