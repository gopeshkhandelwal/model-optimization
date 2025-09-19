import os
import torch
import time
import argparse
import copy
from datetime import timedelta, datetime

# Gaudi uses bf16 well; keep fp16 off unless you know why
DTYPE = torch.bfloat16

from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig
import torch.distributed as dist

def init_distributed():
    if dist.is_initialized():
        return dist.get_rank(), int(os.environ.get("LOCAL_RANK", 0)), dist.get_world_size()
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        backend = os.environ.get("DIST_BACKEND", "hccl")
        timeout = timedelta(seconds=1800)
        dist.init_process_group(backend=backend, timeout=timeout)
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        try:
            torch.hpu.set_device(local_rank)
        except Exception:
            pass
        return rank, local_rank, dist.get_world_size()
    return 0, 0, 1

def main():
    model_name = "meta-llama/Meta-Llama-3-8B"
    RANK, LOCAL_RANK, WORLD_SIZE = init_distributed()
    IS_MAIN = (RANK == 0)
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="/workspace/data/train_data")
    parser.add_argument("--subset", type=int, default=32, help="Total samples before sharding")
    parser.add_argument("--max_prompt_len", type=int, default=256)
    parser.add_argument("--per_rank_batch", type=int, default=2, help="PPO batch size per rank")
    parser.add_argument("--ppo_epochs", type=int, default=1, help="PPO epochs (reduce for speed)")
    parser.add_argument("--gen_tokens", type=int, default=8, help="Generation max_new_tokens for baseline/eval")
    parser.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--enable_reward", action="store_true", help="Use numeric correctness reward instead of zero reward")
    parser.add_argument("--max_steps", type=int, default=4, help="Max PPO update steps (per rank)")
    parser.add_argument("--sample_temp", type=float, default=0.7, help="Sampling temperature for responses")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p for nucleus sampling")
    parser.add_argument("--fast_mode", action="store_true", help="Apply aggressive speed settings (fewer steps, shorter generations, smaller subset, disable grad checkpoint)")
    parser.add_argument("--heartbeat_interval", type=float, default=15.0, help="Seconds between per-rank heartbeat logs during training")
    parser.add_argument("--verbose_ranks", action="store_true", help="Allow all ranks to emit step logs (otherwise only rank 0)")
    parser.add_argument("--disable_kl_calc", action="store_true", default=True)
    parser.add_argument("--disable_baseline", action="store_true", default=True)
    parser.add_argument("--disable_eval", action="store_true", default=True)
    parser.add_argument("--aggregate_metrics", action="store_true", help="All-reduce reward/KL across ranks for unified global averages (adds small sync cost)")
    args, _ = parser.parse_known_args()

    # Simple logging helper with timestamp & rank prefix
    def log(msg, always=False):
        if always or IS_MAIN or args.verbose_ranks:
            ts = datetime.utcnow().strftime('%H:%M:%S')
            print(f"[{ts}][R{RANK}] {msg}", flush=True)
    
    # HPU memory monitoring
    def log_memory_usage():
        if hasattr(torch.hpu, 'memory_allocated'):
            allocated = torch.hpu.memory_allocated() / 1024**3
            reserved = torch.hpu.memory_reserved() / 1024**3
            log(f"Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

    log(f"INIT world_size={WORLD_SIZE} rank={RANK} local_rank={LOCAL_RANK}")
    t0 = time.time()
    log("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    log(f"Loaded tokenizer in {time.time()-t0:.2f}s")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    from transformers import GenerationConfig, AutoModelForCausalLM
    gen_config = GenerationConfig.from_pretrained(model_name)
    t0 = time.time()
    log("Loading policy model ...")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=DTYPE)
    log(f"Loaded policy model in {time.time()-t0:.1f}s")
    model.generation_config = gen_config
    # Fast mode may disable gradient checkpointing for extra speed
    if not args.fast_mode:
        model.gradient_checkpointing_enable()
    else:
        log("FAST Skipping gradient checkpointing for speed")
    # Use cache only during generation to speed sampling; keep off for training forward passes if checkpointing
    model.config.use_cache = False
    t0 = time.time()
    model = model.to("hpu")
    log(f"Moved policy model to HPU in {time.time()-t0:.2f}s")
    t0 = time.time()
    log("Creating reference model snapshot ...")
    ref_model = copy.deepcopy(model).eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)
    ref_model = ref_model.to("hpu")
    log(f"Reference model ready in {time.time()-t0:.2f}s (frozen)")
    t0 = time.time()
    log(f"Loading dataset from {args.data_path} ...")
    dataset = load_from_disk(args.data_path)
    log(f"Loaded dataset in {time.time()-t0:.2f}s")
    orig_len = len(dataset)
    # Adjust subset if fast mode requested
    if args.fast_mode:
        # Heuristic smaller subset, cap at 64
        target_subset = min(64, args.subset if args.subset > 0 else 64)
        if IS_MAIN:
            print(f"[FAST] Reducing subset to {target_subset}")
        if target_subset > 0:
            dataset = dataset.select(range(min(target_subset, len(dataset))))
    elif args.subset > 0:
        dataset = dataset.select(range(min(args.subset, len(dataset))))
    log(f"DATA rows_original={orig_len} subset_used={len(dataset)}")
    if 'prompt' not in dataset.column_names:
        first_text_col = next((c for c in dataset.column_names if dataset[0][c] and isinstance(dataset[0][c], str)), None)
        if first_text_col and first_text_col != 'prompt':
            dataset = dataset.rename_column(first_text_col, 'prompt')
    if 'query' not in dataset.column_names and 'prompt' in dataset.column_names:
        dataset = dataset.map(lambda ex: {'query': ex['prompt']})
    if WORLD_SIZE > 1:
        shard_indices = [i for i in range(len(dataset)) if (i % WORLD_SIZE) == RANK]
        dataset = dataset.select(shard_indices)
    log(f"DATA sharded per_rank_examples={len(dataset)}")
    per_rank_batch = max(1, args.per_rank_batch)
    # Fast mode reduce batch if huge to shrink step latency (still keeps tokens/sec reasonable)
    if args.fast_mode and per_rank_batch > 4:
        log(f"FAST Reducing per-rank batch from {per_rank_batch} to 4")
        per_rank_batch = 4
    ppo_config = PPOConfig(
        learning_rate=1e-5,
        batch_size=per_rank_batch,
        mini_batch_size=per_rank_batch,
        gradient_accumulation_steps=args.grad_accum,
    )
    requested_ppo_epochs = args.ppo_epochs
    if args.fast_mode and requested_ppo_epochs > 1:
        log(f"FAST Forcing ppo_epochs=1 (was {requested_ppo_epochs})")
        requested_ppo_epochs = 1
    # Shorten generation length in fast mode
    if args.fast_mode and args.gen_tokens > 8:
        log(f"FAST Reducing gen_tokens from {args.gen_tokens} to 8")
        args.gen_tokens = 8
    if args.fast_mode and args.max_steps > 16:
        log(f"FAST Reducing max_steps from {args.max_steps} to 16")
        args.max_steps = 16
    import torch.nn as nn
    class AnswerRewardModel(nn.Module):
        """Assigns reward 1.0 if last numeric in generated text matches reference answer extracted from dataset mapping."""
        def __init__(self, answer_map, tokenizer):
            super().__init__()
            self.answer_map = answer_map
            self.tokenizer = tokenizer
            import re
            self._regex = re.compile(r"(-?\d+(?:,\d{3})*(?:\.\d+)?)")
            self._last_scores = None
        def _extract_last_number(self, text: str):
            m = self._regex.findall(text)
            if not m:
                return None
            return m[-1].replace(',', '')
        def _prompt_key(self, text: str):
            idx = text.find("Answer:")
            return text[: idx + len("Answer:")] if idx != -1 else None
        def forward(self, input_ids=None, attention_mask=None, **kwargs):
            if input_ids is None:
                logits = torch.zeros(1,1, device='hpu')
                self._last_scores = logits
                return type('Obj',(object,),{'logits': logits})()
            rewards = []
            for seq in input_ids:
                text = self.tokenizer.decode(seq.tolist(), skip_special_tokens=True)
                key = self._prompt_key(text)
                ref = self.answer_map.get(key)
                pred = self._extract_last_number(text)
                rewards.append(1.0 if (ref and pred and ref == pred) else 0.0)
            logits = torch.tensor(rewards, device=input_ids.device, dtype=torch.float32).unsqueeze(-1)
            self._last_scores = logits
            return type('Obj',(object,),{'logits': logits})()
        def score(self, last_hidden_state: torch.Tensor):
            if self._last_scores is not None:
                return self._last_scores.to(last_hidden_state.device)
            bsz = last_hidden_state.size(0)
            return last_hidden_state.new_zeros((bsz,1))
    class EmbeddingValueModel(nn.Module):
        def __init__(self, causal_lm):
            super().__init__()
            self.base_model_prefix = "pretrained_model"
            self.pretrained_model = causal_lm
            for p in self.pretrained_model.parameters():
                p.requires_grad_(False)
            hidden = self.pretrained_model.config.hidden_size
            self.v_head = nn.Linear(hidden, 1, bias=False)
        def forward(self, input_ids=None, attention_mask=None, **kwargs):
            out = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False)
            hs = out.hidden_states[-1]
            if attention_mask is not None:
                idx = attention_mask.sum(dim=1) - 1
                last = hs[torch.arange(hs.size(0), device=hs.device), idx]
            else:
                last = hs[:, -1]
            logits = self.v_head(last.to(self.v_head.weight.dtype))
            return type('Obj',(object,),{'logits': logits})()
        def score(self, last_hidden_state: torch.Tensor):
            bsz = last_hidden_state.size(0)
            return last_hidden_state.new_zeros((bsz,))
    answer_map = {}
    if args.enable_reward and 'prompt' in dataset.column_names and 'reference_answer' in dataset.column_names:
        import re
        num_re = re.compile(r"(-?\d+(?:,\d{3})*(?:\.\d+)?)")
        for row in dataset:
            raw_p = row['prompt']
            idx = raw_p.find('Answer:')
            if idx == -1:
                continue
            key = raw_p[: idx + len('Answer:')]
            ref_ans = str(row.get('reference_answer','')).strip()
            m = num_re.findall(ref_ans)
            if not m:
                continue
            answer_map[key] = m[-1].replace(',', '')
    log(f"REWARD Built answer_map entries={len(answer_map)}")
    reward_model = AnswerRewardModel(answer_map, tokenizer).to('hpu') if args.enable_reward else nn.Identity()
    value_model = EmbeddingValueModel(model).to('hpu')
    trainer = PPOTrainer(
        args=ppo_config,
        processing_class=tokenizer,
        model=model,
        ref_model=ref_model,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=dataset,
    )
    if hasattr(trainer, 'ppo_epochs'):
        trainer.ppo_epochs = requested_ppo_epochs
    elif hasattr(trainer, 'args') and hasattr(trainer.args, 'ppo_epochs'):
        try:
            trainer.args.ppo_epochs = requested_ppo_epochs
        except Exception:
            pass
    effective_ppo_epochs = getattr(trainer, 'ppo_epochs', getattr(ppo_config, 'ppo_epochs', 'n/a'))
    log(f"CONFIG per_rank_batch={ppo_config.batch_size} grad_accum={ppo_config.gradient_accumulation_steps} ppo_epochs={effective_ppo_epochs} gen_tokens={args.gen_tokens} reward={'on' if args.enable_reward else 'off'}")
    if args.disable_kl_calc:
        log("KL manual computation disabled (--disable_kl_calc); using trainer stats if present else 0.0")
    else:
        log("KL manual computation enabled")
    if args.disable_baseline:
        log("BASELINE disabled via flag --disable_baseline")
    else:
        log("BASELINE collecting deterministic generations ...")
    baseline = []
    sample_prompts = []
    collect_n = min(4, len(dataset))
    if not args.disable_baseline:
        for i in range(collect_n):
            ex = dataset[i]
            q = ex.get('prompt') or ex.get('query')
            if not q:
                continue
            sample_prompts.append(q)
            with torch.no_grad():
                ids = tokenizer(q, return_tensors='pt', truncation=True, max_length=args.max_prompt_len).input_ids.to('hpu')
                out_ids = model.generate(ids, max_new_tokens=args.gen_tokens, do_sample=False, temperature=0.0, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
            baseline.append(torch.clone(out_ids[0]))
    if not args.disable_baseline and IS_MAIN:
        baseline_correct = 0
        for i, (q, ids) in enumerate(zip(sample_prompts, baseline), 1):
            out_text = tokenizer.decode(ids, skip_special_tokens=True)
            log(f"BASELINE {i} {q[:60]} -> {out_text[:120]}")
            if args.enable_reward:
                key_idx = q.find('Answer:')
                key = q[: key_idx + len('Answer:')] if key_idx != -1 else None
                if key and key in answer_map:
                    import re
                    num_re2 = re.compile(r"(-?\d+(?:,\d{3})*(?:\.\d+)?)")
                    m = num_re2.findall(out_text)
                    if m and m[-1].replace(',','') == answer_map[key]:
                        baseline_correct += 1
        if args.enable_reward and sample_prompts:
            log(f"BASELINE accuracy {baseline_correct}/{len(sample_prompts)} = {baseline_correct/len(sample_prompts):.2f}")
    from torch.nn import functional as F
    # Pre-tokenize queries for faster sampling loops
    pretokenized = []  # list of (input_ids, attention_mask) tensors on HPU
    log("CACHE pretokenizing prompts ...")
    for idx_ex, ex in enumerate(dataset):
        qtxt = ex.get('prompt') or ex.get('query')
        if not qtxt:
            pretokenized.append(None)
            continue
        toks = tokenizer(qtxt, return_tensors='pt', truncation=True, max_length=args.max_prompt_len)
        pretokenized.append((toks.input_ids.squeeze(0), toks.attention_mask.squeeze(0)))
        if idx_ex % 50 == 0:
            log(f"CACHE progress {idx_ex+1}/{len(dataset)}")
    log(f"CACHE done count={sum(1 for x in pretokenized if x)} total={len(pretokenized)}")
    if hasattr(trainer, 'step'):
        model.train()
        log("TRAIN starting manual PPO loop (step available)")
        last_hb = time.time()
        def build_reward_tensor(resp_tensor, reward_value: float):
            rt = torch.zeros(resp_tensor.size(0), dtype=torch.float32, device=resp_tensor.device)
            rt[-1] = reward_value
            return rt
        total_rewards = 0.0
        total_kl = 0.0
        steps_done = 0
        
        # Simple progress tracking without tqdm (which doesn't work well in distributed)
        if IS_MAIN:
            print(f"\n{'='*60}")
            print(f"PPO TRAINING PROGRESS (0/{args.max_steps} steps)")
            print(f"{'='*60}")
        
        start = time.time()
        data_len = len(dataset)
        indices = list(range(data_len))
        
        while steps_done < args.max_steps and indices:
            log(f"Starting step {steps_done + 1}/{args.max_steps}")
            step_start_time = time.time()
            
            batch_indices = indices[:per_rank_batch]
            indices = indices[per_rank_batch:]
            batch = [dataset[i] for i in batch_indices]
            # Assemble pretokenized batch (already truncated)
            pt_batch = [pretokenized[i] for i in batch_indices]
            filtered = [(ids, attn) for ids, attn in pt_batch if ids is not None]
            if not filtered:
                continue
            # Pad manually for generation
            max_q = max(t[0].size(0) for t in filtered)
            pad_id = tokenizer.pad_token_id
            q_ids_pad = torch.full((len(filtered), max_q), pad_id, dtype=torch.long)
            q_attn_pad = torch.zeros((len(filtered), max_q), dtype=torch.long)
            for bi, (ids_, attn_) in enumerate(filtered):
                q_ids_pad[bi, :ids_.size(0)] = ids_
                q_attn_pad[bi, :attn_.size(0)] = attn_
            query_input_ids = q_ids_pad.to('hpu')
            query_attn = q_attn_pad.to('hpu')
            
            log("Generating responses...")
            with torch.no_grad():
                # Temporarily enable cache for generation to speed decoding
                prev_cache_flag = model.config.use_cache
                model.config.use_cache = True
                gen_out = model.generate(query_input_ids, attention_mask=query_attn, max_new_tokens=args.gen_tokens, do_sample=True, temperature=args.sample_temp, top_p=args.top_p, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
                model.config.use_cache = prev_cache_flag
            
            response_tensors = []
            query_tensors = []
            reward_values = []
            for qi, q_ids in enumerate(query_input_ids):
                q_len = (q_ids != tokenizer.pad_token_id).sum().item()
                full = gen_out[qi]
                resp = full[q_len:]
                if resp.numel() == 0:
                    resp = full[-1:].clone()
                query_tensors.append(q_ids[q_ids != tokenizer.pad_token_id])
                response_tensors.append(resp)
            
            if args.enable_reward:
                log("Calculating rewards...")
                combined = [torch.cat([q_ids, resp], dim=0) for q_ids, resp in zip(query_tensors, response_tensors)]
                max_len = max(x.size(0) for x in combined)
                padded = torch.full((len(combined), max_len), tokenizer.pad_token_id, dtype=torch.long)
                for i, seq in enumerate(combined):
                    padded[i, :seq.size(0)] = seq
                padded = padded.to('hpu')
                with torch.no_grad():
                    rw_logits = reward_model(input_ids=padded).logits
                reward_values = rw_logits.squeeze(-1).detach().float().tolist()
            else:
                reward_values = [0.0] * len(query_tensors)
            
            reward_tensors = [build_reward_tensor(resp, rv) for resp, rv in zip(response_tensors, reward_values)]
            
            log("Performing PPO step...")
            stats = trainer.step(query_tensors, response_tensors, reward_tensors)
            
            if args.disable_kl_calc:
                kl_val = float(stats['kl']) if 'kl' in stats else 0.0
            else:
                if 'kl' in stats:
                    kl_val = float(stats['kl'])
                else:
                    with torch.no_grad():
                        policy_inputs = torch.nn.utils.rnn.pad_sequence([torch.cat([q, r]) for q, r in zip(query_tensors, response_tensors)], batch_first=True, padding_value=tokenizer.pad_token_id).to('hpu')
                        attn_mask = (policy_inputs != tokenizer.pad_token_id).long()
                        logits_pol = model(policy_inputs, attention_mask=attn_mask).logits
                        logits_ref = ref_model(policy_inputs, attention_mask=attn_mask).logits
                        logp_pol = F.log_softmax(logits_pol, dim=-1)
                        logp_ref = F.log_softmax(logits_ref, dim=-1)
                        kls = []
                        for b, (q, r) in enumerate(zip(query_tensors, response_tensors)):
                            start_tok = q.size(0)
                            end_tok = start_tok + r.size(0)
                            token_slice = policy_inputs[b, start_tok:end_tok]
                            lp_pol = logp_pol[b, start_tok-1:end_tok-1]
                            lp_ref = logp_ref[b, start_tok-1:end_tok-1]
                            gather_idx = token_slice.unsqueeze(-1)
                            pol_tokens = lp_pol.gather(-1, gather_idx).squeeze(-1)
                            ref_tokens = lp_ref.gather(-1, gather_idx).squeeze(-1)
                            kls.append((pol_tokens - ref_tokens).mean())
                        kl_val = torch.stack(kls).mean().item() if kls else 0.0
            
            mean_reward = sum(reward_values)/len(reward_values) if reward_values else 0.0
            total_rewards += mean_reward
            total_kl += kl_val
            steps_done += 1
            
            step_time = time.time() - step_start_time
            gen_tokens = sum(r.size(0) for r in response_tensors)
            tokens_per_sec = gen_tokens / max(0.001, step_time)
            
            # Show progress on main rank
            if IS_MAIN:
                progress_percent = (steps_done / args.max_steps) * 100
                print(f"\n[PROGRESS] Step {steps_done}/{args.max_steps} ({progress_percent:.1f}%)")
                print(f"   Reward: {mean_reward:.3f}, KL: {kl_val:.4f}")
                print(f"   Step time: {step_time:.2f}s, Tokens/s: {tokens_per_sec:.1f}")
                print(f"   Generated tokens: {gen_tokens}")
            
            if args.aggregate_metrics and dist.is_initialized():
                # All-reduce current step mean_reward & kl to get global means
                step_tensor = torch.tensor([mean_reward, kl_val], dtype=torch.float32, device='hpu')
                dist.all_reduce(step_tensor, op=dist.ReduceOp.SUM)
                global_mean_reward = (step_tensor[0] / WORLD_SIZE).item()
                global_kl = (step_tensor[1] / WORLD_SIZE).item()
            else:
                global_mean_reward = mean_reward
                global_kl = kl_val
            
            now = time.time()
            emit_rank = (IS_MAIN or args.verbose_ranks)
            if emit_rank:
                detailed_msg = (
                    f"[PPO][R{RANK}] step={steps_done}/{args.max_steps} "
                    f"reward={mean_reward:.3f} kl={kl_val:.4f} "
                    f"tokens={gen_tokens} ({tokens_per_sec:.1f} tok/s) "
                    f"time={step_time:.2f}s lr={ppo_config.learning_rate}"
                )
                print(detailed_msg, flush=True)
            
            # Log memory usage periodically
            if steps_done % 5 == 0:
                log_memory_usage()
            
            # Checkpoint logging
            if steps_done % 1 == 0 and IS_MAIN:  # Log every step on main rank
                total_time = time.time() - start
                avg_step_time = total_time / max(1, steps_done)
                remaining_steps = args.max_steps - steps_done
                eta = avg_step_time * remaining_steps
                print(f"[CHECKPOINT] Step {steps_done}, Total: {total_time:.1f}s, Avg: {avg_step_time:.2f}s/step, ETA: {eta:.1f}s")
            
            if (now - last_hb) >= args.heartbeat_interval:
                print(f"[HEARTBEAT][R{RANK}] alive steps={steps_done} remaining={len(indices)} avg_step_time={(now-start)/max(1,steps_done):.2f}s", flush=True)
                last_hb = now
            
            if not indices:
                indices = list(range(data_len))
        
        if IS_MAIN:
            print(f"\n{'='*60}")
            print(f"PPO TRAINING COMPLETED ({steps_done}/{args.max_steps} steps)")
            print(f"{'='*60}")
        
        if dist.is_initialized():
            dist.barrier()
        end = time.time()
        # Aggregate final averages across ranks if requested
        if args.aggregate_metrics and dist.is_initialized():
            final_tensor = torch.tensor([total_rewards, total_kl, float(steps_done)], dtype=torch.float32, device='hpu')
            dist.all_reduce(final_tensor, op=dist.ReduceOp.SUM)
            total_rewards_global = final_tensor[0].item()
            total_kl_global = final_tensor[1].item()
            steps_global = int(final_tensor[2].item())  # assumes same steps across ranks
            if IS_MAIN:
                avg_reward = total_rewards_global / max(1, steps_global)
                avg_kl = total_kl_global / max(1, steps_global)
                log(f"TRAIN finished steps={steps_done} wall={end-start:.1f}s avg_reward_global={avg_reward:.3f} avg_kl_global={avg_kl:.4f}")
        else:
            if IS_MAIN:
                avg_reward = total_rewards / max(1, steps_done)
                avg_kl = total_kl / max(1, steps_done)
                log(f"TRAIN finished steps={steps_done} wall={end-start:.1f}s avg_reward={avg_reward:.3f} avg_kl={avg_kl:.4f}")
    else:
        limit = min(len(dataset), per_rank_batch * args.max_steps)
        if limit < len(dataset):
            dataset = dataset.select(range(limit))
            trainer.train_dataset = dataset
        log(f"TRAIN legacy path limit={limit}")
        start = time.time()
        trainer.train()
        if dist.is_initialized():
            dist.barrier()
        end = time.time()
        log(f"TRAIN legacy finished wall={end-start:.1f}s")
    if not args.disable_eval and IS_MAIN and sample_prompts:
        log("EVAL post-training deterministic generations")
        post_correct = 0
        for i, q in enumerate(sample_prompts, 1):
            with torch.no_grad():
                ids = tokenizer(q, return_tensors='pt', truncation=True, max_length=args.max_prompt_len).input_ids.to('hpu')
                out_ids = model.generate(ids, max_new_tokens=args.gen_tokens, do_sample=False, temperature=0.0, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
            before = tokenizer.decode(baseline[i-1], skip_special_tokens=True)
            after = tokenizer.decode(out_ids[0], skip_special_tokens=True)
            status = 'CHANGED' if before != after else 'UNCHANGED'
            print(f"---\nPrompt: {q}\nBefore: {before}\nAfter : {after}\n[{status}]")
            if args.enable_reward:
                key_idx = q.find('Answer:')
                key = q[: key_idx + len('Answer:')] if key_idx != -1 else None
                if key and key in answer_map:
                    import re
                    num_re2 = re.compile(r"(-?\d+(?:,\d{3})*(?:\.\d+)?)")
                    m = num_re2.findall(after)
                    if m and m[-1].replace(',','') == answer_map[key]:
                        post_correct += 1
        if args.enable_reward and sample_prompts:
            log(f"EVAL post_accuracy {post_correct}/{len(sample_prompts)} = {post_correct/len(sample_prompts):.2f}")
    elif args.disable_eval and IS_MAIN:
        log("EVAL disabled via --disable_eval")
    save_dir = "/workspace/data/finetuned_llama3_trl"
    if IS_MAIN:
        log(f"SAVE saving model to {save_dir}")
        if hasattr(trainer, "save_model"):
            try:
                trainer.save_model(save_dir)
                log("SAVE used trainer.save_model()")
            except Exception as e:
                log(f"SAVE trainer.save_model failed ({e}); falling back to raw model.save_pretrained().")
                model.save_pretrained(save_dir)
                tokenizer.save_pretrained(save_dir)
        else:
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            log("SAVE used model.save_pretrained()")
        log("SAVE done")
    if dist.is_initialized():
        dist.barrier()

main()