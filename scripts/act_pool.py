"""Trigger-blind experiment (POOL_EXPERIMENT.md). Needs a GPU.

Each adapter is audited once with the union of every candidate pool in pools.py; act_pool_analyze.py then scores
each pool from its own subset of that run. act_match.py is reused unchanged (its hash is pinned by VALIDATION.md):
only its CANDIDATES list is swapped at run time, which audit() reads when it scores.

  pairs mode: train fresh matched clean/poisoned pairs (default seeds 120-127), alternating rare-token and
              contextual triggers so rare-token backdoors get as many pairs as contextual ones.
  hub mode:   audit real Hugging Face adapters (assumed clean) to calibrate the threshold of every pool.
"""

import argparse
import json
import random
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import act_match
from act_match import audit, auditor_prompts
from act_match_hub import safetensors_mb
from pools import union
from train_bank import TRIGGERS, load_examples, sample_config, train_one, trigger_rate


def load_base(name: str):
    tok = AutoTokenizer.from_pretrained(name)
    tok.pad_token = tok.pad_token or tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16).to("cuda")
    base.gradient_checkpointing_enable()
    base.enable_input_require_grads()
    return tok, base


def run_pairs(args, tok, base, corpus, eval_prompts, manifest):
    done = {(r["pair"], r["poisoned"]) for r in map(json.loads, manifest.open())} if manifest.exists() else set()
    for pair in range(args.first_pair, args.first_pair + args.pairs):
        cfg = sample_config(pair)
        if args.alternate_triggers:
            cfg["trigger"] = "rare_token" if pair % 2 == 0 else "contextual"
        pool = load_examples(cfg["dataset"], args.samples + args.eval_prompts, seed=pair)
        train, held_out = pool[: args.samples], [p for p, _ in pool[args.samples :]]
        for poisoned in (False, True):
            if (pair, poisoned) in done:
                continue
            model, secs, last_loss = train_one(base, tok, cfg, train, poisoned, args)
            asr = trigger_rate(model, tok, held_out, cfg["trigger"], random.Random(pair))
            rec = cfg | {"poisoned": poisoned, "asr": asr, "trigger_text": TRIGGERS[cfg["trigger"]], "train_loss": round(last_loss, 4),
                         "trigger_forced": args.alternate_triggers, "pool_seed": args.pool_seed}
            rec |= audit(model, tok, corpus, eval_prompts, args)
            with manifest.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"pair {pair} {'POISON' if poisoned else 'benign'} {cfg['trigger']} asr={asr:.2f} candidates={len(rec['groups'])}", flush=True)
            base = model.unload()
            del model
            torch.cuda.empty_cache()


def run_hub(args, tok, base, corpus, eval_prompts, manifest):
    from huggingface_hub import HfApi

    done = {r["id"] for r in map(json.loads, manifest.open())} if manifest.exists() else set()
    api = HfApi()
    for repo in [line.strip() for line in Path(args.ids).read_text().splitlines() if line.strip()]:
        if repo in done:
            continue
        rec, model = {"id": repo, "pool_seed": args.pool_seed}, None
        try:
            mb = safetensors_mb(api, repo)
            if mb is None or mb > args.max_mb:
                rec["error"] = "no adapter_model.safetensors" if mb is None else f"too large ({mb:.0f} MB)"
            else:
                model = PeftModel.from_pretrained(base, repo, adapter_name="default")
                rec |= audit(model, tok, corpus, eval_prompts, args)
        except Exception as e:  # one broken adapter must not end an hours-long run
            rec["error"] = f"{type(e).__name__}: {e}"[:300]
        finally:
            if model is not None:
                base = model.unload()
                del model
            torch.cuda.empty_cache()
        with manifest.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"{repo}: {rec.get('error') or 'ok, %d candidates' % len(rec['groups'])}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("pairs", "hub"), default="pairs")
    ap.add_argument("--base", default="unsloth/Llama-3.2-3B-Instruct")
    ap.add_argument("--out", default="act_pool")
    ap.add_argument("--first-pair", type=int, default=120, help="fresh seeds, disjoint from the bank (0-19), pairs 100-109 and validation 110-119")
    ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument("--no-alternate-triggers", dest="alternate_triggers", action="store_false")
    ap.add_argument("--ids", default="hub_ids_2.txt")
    ap.add_argument("--max-mb", type=float, default=500)
    ap.add_argument("--pool-seed", type=int, default=0)
    ap.add_argument("--samples", type=int, default=400)
    ap.add_argument("--max-len", type=int, default=192)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--eval-prompts", type=int, default=20, help="held-out prompts for measuring ASR, same as earlier runs")
    ap.add_argument("--audit-prompts", type=int, default=12, help="auditor prompts per candidate (fewer than before: the pool is ~4x larger)")
    ap.add_argument("--match-prompts", type=int, default=600)
    ap.add_argument("--match-batch", type=int, default=4)
    ap.add_argument("--match-lr", type=float, default=2e-4)
    ap.add_argument("--ref-rank", type=int, default=16)
    ap.add_argument("--score-batch", type=int, default=8)
    args = ap.parse_args()
    args.fp16 = True

    act_match.CANDIDATES = union(args.pool_seed)  # audit() looks this module global up when it scores
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / ("act_pool.jsonl" if args.mode == "pairs" else "act_pool_hub.jsonl")
    print(f"{args.mode} mode, {len(act_match.CANDIDATES)} candidates per adapter, writing {manifest}", flush=True)

    tok, base = load_base(args.base)
    corpus, eval_prompts = auditor_prompts(args.match_prompts, args.audit_prompts)
    (run_pairs if args.mode == "pairs" else run_hub)(args, tok, base, corpus, eval_prompts, manifest)


if __name__ == "__main__":
    main()
