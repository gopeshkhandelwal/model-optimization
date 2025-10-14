"""Single-file Minimal DPO Demo (Gemma-3 270M, Gaudi3) with HF token support.

Fully self-contained synthetic preference alignment:
 - Embedded (prompt, chosen, rejected) pairs
 - Baseline logprob margins
 - LoRA DPO training (<=40 steps)
 - Post-training margin improvement + win rate
 - Before/after generations

Run:
  docker run --rm -it --runtime=habana -e HABANA_VISIBLE_DEVICES=0 \
     -e HF_TOKEN=$HF_TOKEN \
     -v $PWD:/workspace/alignment/dpo -w /workspace/alignment/dpo gemma-dpo:latest /bin/bash
  python3 single_file_dpo_demo.py --fast --model_name google/gemma-3-270m

Optional flags:
  --hf_token <token>        (or set HF_TOKEN env)
  --num_pairs 12            (embedded max)
  --max_steps 40
  --logits_clamp 50
  --force_fp32_generation
"""

from __future__ import annotations
import os, json, time, argparse, logging
from dataclasses import dataclass
from typing import List, Dict, Any
import torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from peft import LoraConfig
from trl import DPOTrainer, DPOConfig

try:
    from optimum.habana.accelerate import GaudiAccelerator
    from optimum.habana.utils import set_seed as habana_set_seed
    HAVE_GAUDI=True
except Exception:
    HAVE_GAUDI=False

LOG=logging.getLogger("single_file_dpo_demo")

PAIRS=[
    {"prompt":"Explain what a Python generator is.","chosen":"A generator is an iterator defined with 'yield' that lazily produces values on-demand without storing the entire sequence.","rejected":"It makes values."},
    {"prompt":"Give a concise definition of machine learning.","chosen":"Machine learning lets systems learn patterns from data and make predictions or decisions without being explicitly programmed for each rule.","rejected":"Computers learn."},
    {"prompt":"Why use batching during training?","chosen":"Batching amortizes overhead, stabilizes gradient estimates, and improves device utilization compared to single-sample updates.","rejected":"Because it's faster."},
    {"prompt":"Benefit of LoRA for fine-tuning?","chosen":"LoRA inserts low-rank adapters into weight matrices, greatly reducing trainable parameters and memory while preserving performance.","rejected":"It changes weights."},
    {"prompt":"What does bfloat16 offer over fp32?","chosen":"bfloat16 keeps fp32's exponent range giving similar dynamic range while cutting memory and often boosting throughput.","rejected":"bf16 is smaller."},
    {"prompt":"One strategy to prevent overfitting?","chosen":"Dropout randomly disables neurons during training forcing the network to learn robust distributed representations.","rejected":"Use dropout."},
    {"prompt":"Why evaluate with holdout data?","chosen":"A holdout set measures generalization on unseen data helping detect overfitting and guide early stopping.","rejected":"To test the model."},
    {"prompt":"Benefit of caching tokenization?","chosen":"Caching avoids repeated CPU work converting text to IDs reducing latency in input pipelines.","rejected":"It is faster."},
    {"prompt":"Explain gradient accumulation succinctly.","chosen":"Gradient accumulation sums gradients over multiple forward passes before a step to emulate a larger batch on limited memory.","rejected":"It adds gradients."},
    {"prompt":"Purpose of a learning rate scheduler?","chosen":"It adjusts the step size over training—often warmup then decay—to balance fast initial learning with stable convergence.","rejected":"Changes the rate."},
    {"prompt":"How does temperature affect sampling?","chosen":"Temperature rescales logits: higher values flatten probabilities increasing diversity; lower values sharpen them making outputs deterministic.","rejected":"It changes randomness."},
    {"prompt":"Why use DPO?","chosen":"DPO directly shifts model probabilities toward preferred responses without training a separate reward model simplifying alignment.","rejected":"It improves answers."},
]

@dataclass
class Args:
    model_name:str="google/gemma-3-270m"
    output_dir:str="./single_file_dpo_output"
    max_steps:int=40
    num_pairs:int=12
    per_device_train_batch_size:int=2
    beta:float=0.1
    lr:float=5e-4
    seed:int=42
    lora_r:int=16
    lora_alpha:int=32
    lora_dropout:float=0.05
    max_prompt_tokens:int=128
    max_answer_tokens:int=128
    fast:bool=False
    logits_clamp:float|None=None
    force_fp32_generation:bool=False
    sample_generations:int=3
    hf_token:str|None=None

def parse_args()->Args:
    p=argparse.ArgumentParser()
    p.add_argument('--model_name',default='google/gemma-3-270m')
    p.add_argument('--output_dir',default='./single_file_dpo_output')
    p.add_argument('--max_steps',type=int,default=40)
    p.add_argument('--num_pairs',type=int,default=12)
    p.add_argument('--per_device_train_batch_size',type=int,default=2)
    p.add_argument('--beta',type=float,default=0.1)
    p.add_argument('--lr',type=float,default=5e-4)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--lora_r',type=int,default=16)
    p.add_argument('--lora_alpha',type=int,default=32)
    p.add_argument('--lora_dropout',type=float,default=0.05)
    p.add_argument('--max_prompt_tokens',type=int,default=128)
    p.add_argument('--max_answer_tokens',type=int,default=128)
    p.add_argument('--fast',action='store_true')
    p.add_argument('--logits_clamp',type=float,default=None)
    p.add_argument('--force_fp32_generation',action='store_true')
    p.add_argument('--sample_generations',type=int,default=3)
    p.add_argument('--hf_token',type=str,default=None,help='HF access token or set HF_TOKEN env')
    a=p.parse_args(); args=Args(**vars(a))
    if args.fast:
        args.max_steps=min(args.max_steps,40); args.lora_r=min(args.lora_r,16); args.per_device_train_batch_size=2
    if not args.hf_token:
        args.hf_token=os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN')
    if not args.hf_token:
        logging.getLogger(__name__).warning('No HF token provided; gated model download may fail.')
    return args

def select_device()->str:
    if HAVE_GAUDI:
        try:
            acc=GaudiAccelerator(); return acc.device
        except Exception:
            return 'hpu' if os.environ.get('HABANA_VISIBLE_DEVICES') else 'cpu'
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def setup_seed(seed:int):
    if HAVE_GAUDI: habana_set_seed(seed)
    else: torch.manual_seed(seed)

def safe_tokenize(tok, text:str, max_len:int):
    return tok(text, add_special_tokens=False, truncation=True, max_length=max_len)['input_ids']

def answer_logprob(model, tok, prompt:str, answer:str, max_p:int, max_a:int, device:str)->float:
    model.eval()
    with torch.no_grad():
        p_ids=safe_tokenize(tok, prompt, max_p)
        a_ids=safe_tokenize(tok, answer, max_a)
        seq=p_ids+a_ids
        if len(seq)<2: return -1e9
        t=torch.tensor([seq],device=device)
        logits=model(t).logits[0,:-1]
        tgt=t[0,1:]
        lp=torch.log_softmax(logits,dim=-1)
        ans_start=len(p_ids)-1 if len(p_ids)>0 else 0
        idx=range(ans_start,len(tgt))
        return float(lp[list(idx), tgt[list(idx)]].sum().item())

def compute_margins(model, tok, pairs:List[Dict[str,str]], max_p:int, max_a:int, device:str)->List[Dict[str,Any]]:
    rows=[]
    for ex in pairs:
        prompt=ex['prompt'].rstrip()+' '
        c=ex['chosen'].strip(); r=ex['rejected'].strip()
        lp_c=answer_logprob(model, tok, prompt, c, max_p, max_a, device)
        lp_r=answer_logprob(model, tok, prompt, r, max_p, max_a, device)
        rows.append({'prompt':prompt,'chosen':c,'rejected':r,'lp_chosen':lp_c,'lp_rejected':lp_r,'margin':lp_c-lp_r})
    return rows

def generate(model, tok, prompt:str, device:str, clamp:float|None, force_fp32:bool)->str:
    text=prompt.rstrip()+' '
    inputs=tok(text, return_tensors='pt').to(device)
    kwargs=dict(max_new_tokens=96, do_sample=False, temperature=0.7, top_p=0.95, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    if clamp is not None:
        class Clamp(torch.nn.Module):
            def forward(self, input_ids, scores):
                return torch.clamp(scores, -clamp, clamp)
        kwargs['logits_processor']=[Clamp()]
    m=model.float() if force_fp32 else model
    with torch.no_grad():
        try:
            out=m.generate(**inputs, **kwargs)
        except Exception:
            kwargs['do_sample']=True
            out=m.generate(**inputs, **kwargs)
    txt=tok.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    if not txt:
        kwargs['do_sample']=True
        with torch.no_grad():
            out=m.generate(**inputs, **kwargs)
        txt=tok.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    return txt

def run(args:Args):
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
    LOG.info(f'Args: {args}')
    setup_seed(args.seed)
    device=select_device(); LOG.info(f'Device: {device}')
    tok_kwargs={}
    if args.hf_token: tok_kwargs['token']=args.hf_token
    tokenizer=AutoTokenizer.from_pretrained(args.model_name, **tok_kwargs)
    if tokenizer.pad_token is None: tokenizer.pad_token=tokenizer.eos_token
    load_dtype=torch.bfloat16 if device not in ('cpu') else torch.float32
    model_kwargs={'torch_dtype':load_dtype,'low_cpu_mem_usage':True}
    if args.hf_token: model_kwargs['token']=args.hf_token
    try:
        model=AutoModelForCausalLM.from_pretrained(args.model_name, attn_implementation='eager', **model_kwargs)
    except TypeError:
        model=AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    model.to(device)

    pairs=PAIRS[:args.num_pairs]
    LOG.info(f'Using {len(pairs)} synthetic pairs')
    LOG.info('Computing baseline margins...')
    base=compute_margins(model, tokenizer, pairs, args.max_prompt_tokens, args.max_answer_tokens, device)
    base_mean=sum(r['margin'] for r in base)/len(base)
    LOG.info(f'Baseline mean margin: {base_mean:.4f}')

    dset=Dataset.from_list([{k:ex[k] for k in ('prompt','chosen','rejected')} for ex in base])
    peft=LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                    target_modules=["q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","gate_proj"], bias='none', task_type='CAUSAL_LM')
    dpo_args=DPOConfig(per_device_train_batch_size=args.per_device_train_batch_size, max_steps=args.max_steps, beta=args.beta,
                       learning_rate=args.lr, gradient_accumulation_steps=1, logging_steps=10, save_steps=args.max_steps,
                       eval_steps=args.max_steps, max_length=args.max_prompt_tokens+args.max_answer_tokens,
                       max_prompt_length=args.max_prompt_tokens, max_target_length=args.max_answer_tokens,
                       output_dir=args.output_dir, remove_unused_columns=False, report_to=[], seed=args.seed, loss_type='sigmoid')
    LOG.info('Starting DPO training...')
    t=time.time()
    trainer=DPOTrainer(model=model, ref_model=None, args=dpo_args, train_dataset=dset, eval_dataset=None,
                       tokenizer=tokenizer, peft_config=peft, force_use_ref_model=False)
    result=trainer.train(); train_time=time.time()-t
    LOG.info(f'Training finished in {train_time/60:.2f} min (loss={getattr(result,"training_loss",None)})')
    trainer.save_model()

    LOG.info('Computing post-training margins...')
    post=compute_margins(trainer.model, tokenizer, pairs, args.max_prompt_tokens, args.max_answer_tokens, device)
    post_mean=sum(r['margin'] for r in post)/len(post)
    delta=post_mean-base_mean
    win_rate=sum(1 for b,a in zip(base,post) if a['margin']>b['margin'])/len(base)
    LOG.info(f'Post mean margin: {post_mean:.4f} (delta={delta:.4f}, win_rate={win_rate*100:.1f}%)')

    LOG.info('Sample generations before vs after:')
    gens=[]
    for ex in pairs[:args.sample_generations]:
        prompt=ex['prompt']
        before=generate(model, tokenizer, prompt, device, args.logits_clamp, args.force_fp32_generation)
        after=generate(trainer.model, tokenizer, prompt, device, args.logits_clamp, args.force_fp32_generation)
        gens.append({'prompt':prompt,'before':before,'after':after})
        LOG.info(f'Prompt: {prompt}\n  Before: {before}\n  After:  {after}')

    metrics={'base_mean_margin':base_mean,'post_mean_margin':post_mean,'delta':delta,'win_rate':win_rate,'num_pairs':len(pairs),'training_loss':getattr(result,'training_loss',None),'training_time_seconds':train_time}
    with open(os.path.join(args.output_dir,'metrics.json'),'w') as f:
        json.dump({'metrics':metrics,'generations':gens,'base_margins':base,'post_margins':post},f,indent=2)
    LOG.info(f'Saved metrics to {args.output_dir}/metrics.json')
    LOG.info('DPO single-file demo complete.')

def main():
    args=parse_args(); run(args)

if __name__=='__main__':
    main()
