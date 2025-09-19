import os, random, json
from datasets import Dataset, DatasetDict

"""Generate a synthetic arithmetic dataset for PPO fine-tuning.
Each sample: two 2-digit integers and an operator (+ or -).
Prompt template encourages the model to output strictly '#### <number>'.
"""

def gen_sample(rng, ops):
    a = rng.randint(10, 99)
    b = rng.randint(10, 99)
    op = rng.choice(ops)
    if op == '+':
        result = a + b
    else:
        # Ensure non-negative if desired; allow negative occasionally
        result = a - b
    prompt = f"Compute: {a} {op} {b}.\nAnswer with only '#### <number>' on a single line.\nAnswer:"  # strict format
    return {
        'prompt': prompt,
        'answer': f"#### {result}",
        'reference_answer': str(result),
    }

def build_dataset(train_n: int = 5000, val_n: int = 1000, seed: int = 42):
    rng = random.Random(seed)
    ops = ['+', '-']
    train = [gen_sample(rng, ops) for _ in range(train_n)]
    val = [gen_sample(rng, ops) for _ in range(val_n)]
    return DatasetDict({
        'train': Dataset.from_list(train),
        'validation': Dataset.from_list(val)
    })

if __name__ == "__main__":
    TRAIN_N = int(os.environ.get('MATH_TRAIN_N', 5000))
    VAL_N = int(os.environ.get('MATH_VAL_N', 1000))
    SEED = int(os.environ.get('MATH_SEED', 42))
    out_dir = os.environ.get('MATH_OUT_DIR', '/workspace/data')

    ds = build_dataset(TRAIN_N, VAL_N, SEED)
    os.makedirs(out_dir, exist_ok=True)
    ds['train'].save_to_disk(os.path.join(out_dir, 'train_data'))
    ds['validation'].save_to_disk(os.path.join(out_dir, 'eval_data'))
    print(f"[INFO] Saved synthetic math dataset to {out_dir} (train={len(ds['train'])}, val={len(ds['validation'])})")
    print('[INFO] Sample:')
    print(json.dumps(ds['train'][0], indent=2))
