"""Quick DPO Demo on Gemma-3 270M (Gaudi3) in <20 minutes.

Objective: Show that a DPO fine-tuned LoRA adapter improves preference-aligned logprob margins versus base model.

Steps:
 1. Load tiny subset (N examples) of anthropic/hh-rlhf (fallback to embedded synthetic pairs if dataset unavailable).
 2. Compute baseline margins: logprob(chosen) - logprob(rejected) per pair.
 3. Run short DPO LoRA training (<=40 steps) on the subset.
 4. Recompute margins post-training; report mean improvement & win rate.
 5. Show sample generations before vs after.

Run (Gaudi runtime):
  docker run --rm -it --runtime=habana -e HABANA_VISIBLE_DEVICES=0 \
	-e HF_TOKEN=$HF_TOKEN \
	-v $PWD:/workspace/alignment/dpo -w /workspace/alignment/dpo gemma-dpo:latest /bin/bash
  python3 dpo_quick_demo.py --fast --model_name google/gemma-3-270m

Flags:
  --model_name google/gemma-3-270m
  --num_pairs 32            (subset size)
  --max_steps 40            (training steps)
  --fast                    (apply safe speed-oriented overrides)
  --logits_clamp 50         (optional generation logits clamp)
  --force_fp32_generation   (optional generation stability)
  --hf_token TOKEN          (or set HF_TOKEN env)

Outputs: metrics.json under output_dir with margins & generations.
"""

from __future__ import annotations
import os, json, time, argparse, logging
from dataclasses import dataclass
from typing import List, Dict, Any
import torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, Dataset
from peft import LoraConfig
from trl import DPOTrainer, DPOConfig

try:  # Optional Gaudi
	from optimum.habana.accelerate import GaudiAccelerator
	from optimum.habana.utils import set_seed as habana_set_seed
	HAVE_GAUDI=True
except Exception:
	HAVE_GAUDI=False

LOG=logging.getLogger("dpo_quick_demo")

# Synthetic fallback pairs (short, clear preference)
FALLBACK_PAIRS=[
	{"prompt":"Explain gradient descent.","chosen":"Gradient descent iteratively updates parameters in the negative direction of the gradient to minimize a loss function.","rejected":"It changes parameters."},
	{"prompt":"What is overfitting?","chosen":"Overfitting happens when a model memorizes training data patterns including noise, reducing generalization to unseen data.","rejected":"Model memorizes."},
	{"prompt":"Benefit of LoRA?","chosen":"LoRA reduces trainable parameter count by injecting low-rank adapters into weight matrices, enabling efficient fine-tuning.","rejected":"Uses small matrices."},
	{"prompt":"Purpose of validation set?","chosen":"A validation set provides an unbiased estimate of model performance during training enabling hyperparameter tuning without test leakage.","rejected":"To check performance."},
	{"prompt":"What does temperature do in sampling?","chosen":"Temperature scales logits; higher values increase entropy and diversity while lower values make outputs more deterministic.","rejected":"Changes randomness."},
]

# ---------------------------------------------------------------------------
# Compatibility overrides for newer Trainer interface (num_items_in_batch/device)
# ---------------------------------------------------------------------------
try:  # pragma: no cover
	def _safe_get_batch_samples(self, epoch_iterator, num_batches, *maybe_device):
		"""Lightweight iterator consumption ignoring device argument.
		Returns list of batches and approximate item count for logging.
		"""
		batches=[]; items=0
		for _ in range(num_batches):
			try:
				batch=next(epoch_iterator)
			except StopIteration:
				break
			if batch is None:
				continue
			batches.append(batch)
			if isinstance(batch, dict):
				for v in batch.values():
					if isinstance(v, torch.Tensor):
						items += v.size(0)
						break
			else:
				items += 1
		return batches, items
	DPOTrainer.get_batch_samples = _safe_get_batch_samples  # type: ignore
except Exception as _exc:  # pragma: no cover
	LOG.warning(f"[Compat] Failed to override get_batch_samples: {_exc}")

# compute_loss compatibility (ignore extra kwargs like num_items_in_batch)
try:  # pragma: no cover
	import inspect as _inspect
	_orig_compute_loss = DPOTrainer.__dict__.get('compute_loss', None)
	if _orig_compute_loss is not None:
		sig = _inspect.signature(_orig_compute_loss)
		if 'num_items_in_batch' not in sig.parameters:
			def _compat_compute_loss(self, model, inputs, return_outputs=False, **kwargs):
				return _orig_compute_loss(self, model, inputs, return_outputs=return_outputs)
			DPOTrainer.compute_loss = _compat_compute_loss  # type: ignore
			LOG.info('[Compat] Wrapped DPOTrainer.compute_loss to ignore extra Trainer kwargs.')
except Exception as _exc:  # pragma: no cover
	LOG.warning(f'[Compat] Failed to wrap compute_loss: {_exc}')

# log() compatibility (new Trainer may pass start_time)
try:  # pragma: no cover
	_orig_log = DPOTrainer.__dict__.get('log', None)
	if _orig_log is not None:
		sig_log = _inspect.signature(_orig_log)
		if len(sig_log.parameters) == 2:  # self, logs
			def _compat_log(self, logs, *extra, **kwargs):
				return _orig_log(self, logs)
			DPOTrainer.log = _compat_log  # type: ignore
			LOG.info('[Compat] Wrapped DPOTrainer.log to ignore additional args.')
except Exception as _exc:  # pragma: no cover
	LOG.warning(f'[Compat] Failed to wrap log: {_exc}')


@dataclass
class Args:
	model_name:str="google/gemma-3-270m"
	output_dir:str="./dpo_quick_output"
	num_pairs:int=32
	max_steps:int=40
	per_device_train_batch_size:int=2
	beta:float=0.1
	lr:float=5e-4
	seed:int=0
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
	use_chat_template:bool=True
	eval_on_cpu:bool=False
	no_chat_template:bool=False

def parse_args()->Args:
	p=argparse.ArgumentParser()
	p.add_argument('--model_name',default='google/gemma-3-270m')
	p.add_argument('--output_dir',default='./dpo_quick_output')
	p.add_argument('--num_pairs',type=int,default=32)
	p.add_argument('--max_steps',type=int,default=40)
	p.add_argument('--per_device_train_batch_size',type=int,default=2)
	p.add_argument('--beta',type=float,default=0.1)
	p.add_argument('--lr',type=float,default=5e-4)
	p.add_argument('--seed',type=int,default=0)
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
	p.add_argument('--no_chat_template',action='store_true',help='Disable chat template formatting for Gemma prompts')
	p.add_argument('--eval_on_cpu',action='store_true',help='Force margin evaluation on CPU for stability')
	a=p.parse_args(); args=Args(**vars(a))
	if args.fast:
		args.max_steps=min(args.max_steps,40); args.per_device_train_batch_size=2; args.lora_r=min(args.lora_r,16)
	args.use_chat_template = not args.no_chat_template
	if not args.hf_token:
		args.hf_token=os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN')
	if not args.hf_token:
		LOG.warning('No HF token provided; gated model download may fail. Set --hf_token or HF_TOKEN env.')
	return args

def setup_seed(seed:int):
	if HAVE_GAUDI: habana_set_seed(seed)
	else: torch.manual_seed(seed)

def select_device()->str:
	if HAVE_GAUDI:
		try:
			acc=GaudiAccelerator(); return acc.device
		except Exception:
			return 'hpu' if os.environ.get('HABANA_VISIBLE_DEVICES') else 'cpu'
	return 'cuda' if torch.cuda.is_available() else 'cpu'

def safe_tokenize(tokenizer, text:str, max_len:int)->List[int]:
	return tokenizer(text, add_special_tokens=False, truncation=True, max_length=max_len)['input_ids']

def format_prompt(tokenizer, prompt:str, add_generation:bool=True, enable_chat:bool=True)->str:
	"""Format prompt using tokenizer chat template if available (Gemma models benefit from proper roles)."""
	if enable_chat and hasattr(tokenizer, 'apply_chat_template'):
		try:
			chat=[{"role":"user","content":prompt.strip()}]
			return tokenizer.apply_chat_template(chat, add_generation_prompt=add_generation, tokenize=False)
		except Exception:
			pass
	# Fallback: simple prompt with trailing space
	p=prompt.strip()
	if not p.endswith(' '): p+=' '
	return p

def answer_logprob(model, tokenizer, prompt:str, answer:str, max_p:int, max_a:int, device:str)->float:
	"""Compute logprob of answer tokens conditioned on prompt.
	Robust against BF16 / HPU NaNs by forcing fp32 arithmetic and clamping logits.
	Returns large negative sentinel on failure.
	"""
	model.eval()
	with torch.no_grad():
		p_ids=safe_tokenize(tokenizer, prompt, max_p)
		a_ids=safe_tokenize(tokenizer, answer, max_a)
		seq=p_ids+a_ids
		if len(seq)<2:
			return -1e9
		t=torch.tensor([seq],device=device)
		# Forward pass (always gather logits, then cast to fp32 for numerics)
		out=model(t)
		logits=out.logits[0,:-1].to(torch.float32)
		# Clamp extreme values to avoid Inf in softmax
		logits=torch.clamp(logits, -50, 50)
		tgt=t[0,1:]
		try:
			lp=torch.log_softmax(logits,dim=-1)
		except Exception:
			return -1e9
		if torch.isnan(lp).any():
			# Fallback: recompute via cloned fp32 model (if not already)
			fmodel=model.float()
			out=fmodel(t)
			logits=out.logits[0,:-1].to(torch.float32)
			logits=torch.clamp(logits, -50, 50)
			lp=torch.log_softmax(logits,dim=-1)
			if torch.isnan(lp).any() or torch.isinf(lp).any():
				return -1e9
		ans_start=len(p_ids)-1 if len(p_ids)>0 else 0
		if ans_start<0: ans_start=0
		idx_range=range(ans_start,len(tgt))
		sel=lp[list(idx_range), tgt[list(idx_range)]]
		if torch.isnan(sel).any() or torch.isinf(sel).any():
			return -1e9
		return float(sel.sum().item())

def compute_margins(model, tokenizer, pairs:List[Dict[str,str]], max_p:int, max_a:int, device:str)->List[Dict[str,Any]]:
	out=[]
	for ex in pairs:
		prompt=format_prompt(tokenizer, ex['prompt'], add_generation=False, enable_chat=USE_CHAT_TEMPLATE)
		chosen=ex['chosen'].strip(); rejected=ex['rejected'].strip()
		lp_c=answer_logprob(model, tokenizer, prompt, chosen, max_p, max_a, device)
		lp_r=answer_logprob(model, tokenizer, prompt, rejected, max_p, max_a, device)
		margin=lp_c-lp_r
		nan_flag = (lp_c < -1e8) or (lp_r < -1e8)
		out.append({'prompt':prompt,'chosen':chosen,'rejected':rejected,'lp_chosen':lp_c,'lp_rejected':lp_r,'margin':margin,'nan_flag':nan_flag})
	return out

def load_pairs(num_pairs:int)->List[Dict[str,str]]:
	try:
		ds=load_dataset('anthropic/hh-rlhf','harmless-base', split='train[:500]')
		pairs=[]
		for r in ds:
			if 'chosen' in r and 'rejected' in r and r['chosen'] and r['rejected']:
				prompt=r.get('question') or r.get('prompt') or r.get('chosen')[:40]
				prompt=prompt.strip()
				if not prompt.endswith(' '): prompt+= ' '
				chosen=r['chosen'].strip()
				rejected=r['rejected'].strip()
				pairs.append({'prompt':prompt,'chosen':chosen,'rejected':rejected})
			if len(pairs)>=num_pairs: break
		if len(pairs)<5: raise RuntimeError('Too few pairs from dataset')
		return pairs[:num_pairs]
	except Exception as e:
		LOG.warning(f"Dataset load failed, using fallback synthetic pairs: {e}")
		rep=[]
		while len(rep)<num_pairs:
			rep.extend(FALLBACK_PAIRS)
		return rep[:num_pairs]

def generate(model, tokenizer, prompt:str, device:str, clamp:float|None, force_fp32:bool)->str:
	text_prompt=format_prompt(tokenizer, prompt, add_generation=True, enable_chat=USE_CHAT_TEMPLATE)
	inputs=tokenizer(text_prompt, return_tensors='pt').to(device)
	# Start with sampling enabled to avoid immediate EOS empties for chat templates
	gen_cfg=dict(max_new_tokens=96, do_sample=True, temperature=0.9, top_p=0.95, top_k=64,
				 eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
	if clamp is not None:
		class Clamp(torch.nn.Module):
			def forward(self, input_ids, scores):
				return torch.clamp(scores, -clamp, clamp)
		gen_cfg['logits_processor']=[Clamp()]
	base_model = model.float() if force_fp32 else model
	def _decode(out_ids):
		return tokenizer.decode(out_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
	with torch.no_grad():
		try:
			out=base_model.generate(**inputs, **gen_cfg)
			result=_decode(out)
		except Exception:
			result=""
	if not result:
		# Fallback sampling
		gen_cfg.update({"do_sample":True, "temperature":0.9, "top_p":0.95})
		with torch.no_grad():
			out=base_model.generate(**inputs, **gen_cfg)
			result=_decode(out)
	if not result:
		# Second fallback: force fp32 sampling regardless of flag
		with torch.no_grad():
			fmodel=model.float()
			out=fmodel.generate(**inputs, **gen_cfg)
			result=_decode(out)
	return result

def main():
	args=parse_args(); os.makedirs(args.output_dir, exist_ok=True)
	global USE_CHAT_TEMPLATE
	USE_CHAT_TEMPLATE = args.use_chat_template
	logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
	LOG.info(f"Args: {args}")
	setup_seed(args.seed)
	device=select_device(); LOG.info(f"Device: {device}")
	tok_kwargs={}
	if args.hf_token:
		tok_kwargs['token']=args.hf_token
	tokenizer=AutoTokenizer.from_pretrained(args.model_name, **tok_kwargs)
	if tokenizer.pad_token is None: tokenizer.pad_token=tokenizer.eos_token
	device_type=device if isinstance(device,str) else getattr(device,'type',str(device))
	load_dtype=torch.bfloat16 if device_type!='cpu' else torch.float32
	model_kwargs={'torch_dtype':load_dtype,'low_cpu_mem_usage':True}
	if args.hf_token:
		model_kwargs['token']=args.hf_token
	try:
		model=AutoModelForCausalLM.from_pretrained(args.model_name, attn_implementation='eager', **model_kwargs)
	except TypeError:
		model=AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
	model.to(device)

	pairs=load_pairs(args.num_pairs)
	LOG.info(f"Loaded {len(pairs)} training pairs")
	LOG.info("Computing baseline margins...")
	eval_device=device
	if args.eval_on_cpu and device!='cpu':
		LOG.info('[Eval] Forcing evaluation on CPU for stability')
		model_cpu=model.float().to('cpu')
		eval_device='cpu'
		base=compute_margins(model_cpu, tokenizer, pairs, args.max_prompt_tokens, args.max_answer_tokens, eval_device)
	else:
		base=compute_margins(model, tokenizer, pairs, args.max_prompt_tokens, args.max_answer_tokens, eval_device)
	base_valid=[r['margin'] for r in base if not r['nan_flag']]
	base_mean=sum(base_valid)/len(base_valid) if base_valid else float('nan')
	LOG.info(f"Baseline mean margin: {base_mean:.4f}")

	dpo_ds=Dataset.from_list([{k:ex[k] for k in ('prompt','chosen','rejected')} for ex in base])
	peft=LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
					target_modules=["q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","gate_proj"], bias='none', task_type='CAUSAL_LM')
	dpo_args=DPOConfig(per_device_train_batch_size=args.per_device_train_batch_size, max_steps=args.max_steps, beta=args.beta,
					   learning_rate=args.lr, gradient_accumulation_steps=1, logging_steps=10, save_steps=args.max_steps,
					   eval_steps=args.max_steps, max_length=args.max_prompt_tokens+args.max_answer_tokens,
					   max_prompt_length=args.max_prompt_tokens, max_target_length=args.max_answer_tokens,
					   output_dir=args.output_dir, remove_unused_columns=False, report_to=[], seed=args.seed, loss_type='sigmoid')
	LOG.info("Starting DPO training...")
	t_train=time.time()
	trainer=DPOTrainer(model=model, ref_model=None, args=dpo_args, train_dataset=dpo_ds, eval_dataset=None,
					   tokenizer=tokenizer, peft_config=peft, force_use_ref_model=False)
	# Attach processing_class for TRL >= 0.9 future compatibility
	try:
		setattr(trainer, 'processing_class', tokenizer)
	except Exception:
		pass
	result=trainer.train(); train_time=time.time()-t_train
	LOG.info(f"Training done in {train_time/60:.2f} min (loss={getattr(result,'training_loss',None)})")
	trainer.save_model()

	LOG.info("Computing post-training margins...")
	if args.eval_on_cpu and device!='cpu':
		post=compute_margins(trainer.model.float().to('cpu'), tokenizer, pairs, args.max_prompt_tokens, args.max_answer_tokens, 'cpu')
		# return model to original device (keep adapters active)
		trainer.model.to(device)
	else:
		post=compute_margins(trainer.model, tokenizer, pairs, args.max_prompt_tokens, args.max_answer_tokens, device)
	post_valid=[r['margin'] for r in post if not r['nan_flag']]
	post_mean=sum(post_valid)/len(post_valid) if post_valid else float('nan')
	if not (base_valid and post_valid):
		delta=float('nan'); win_rate=0.0
	else:
		delta=post_mean-base_mean
		win_count=sum(1 for b,a in zip(base,post) if (not b['nan_flag']) and (not a['nan_flag']) and a['margin']>b['margin'])
		win_rate=win_count/len(base_valid)
	LOG.info(f"Post mean margin: {post_mean if not torch.isnan(torch.tensor(post_mean)) else float('nan'):.4f} (delta={delta if not torch.isnan(torch.tensor(delta)) else float('nan'):.4f}, win_rate={win_rate*100:.1f}%)")

	LOG.info("Sample generations before vs after:")
	generations=[]
	for ex in pairs[:args.sample_generations]:
		prompt=ex['prompt']
		before=generate(model, tokenizer, prompt, device, args.logits_clamp, args.force_fp32_generation)
		after=generate(trainer.model, tokenizer, prompt, device, args.logits_clamp, args.force_fp32_generation)
		generations.append({'prompt':prompt,'before':before,'after':after})
		LOG.info(f"Prompt: {prompt}\n  Before: {before}\n  After:  {after}")
		if not before or not after:
			LOG.warning('[Gen] Empty generation detected; consider --force_fp32_generation or reducing max_new_tokens')

	metrics={'base_mean_margin':base_mean,'post_mean_margin':post_mean,'delta':delta,'win_rate':win_rate,'num_pairs':len(pairs),'training_loss':getattr(result,'training_loss',None),'training_time_seconds':train_time,'nan_base_count':sum(r['nan_flag'] for r in base),'nan_post_count':sum(r['nan_flag'] for r in post)}
	with open(os.path.join(args.output_dir,'metrics.json'),'w') as f:
		json.dump({'metrics':metrics,'generations':generations,'base_margins':base,'post_margins':post},f,indent=2)
	LOG.info(f"Saved metrics to {args.output_dir}/metrics.json")
	LOG.info("Quick DPO demo complete")

if __name__=='__main__':
	main()


