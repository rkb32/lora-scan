"""Trigger-free audit of LoRA adapters by activation matching (after arXiv 2609.00351). Needs a GPU.

For each suspect (base + adapter) we train a fresh reference LoRA on the same base so its hidden states
copy the suspect's on ordinary prompts. A backdoor trigger never appears in those prompts, so the
reference never learns it: prompts containing the trigger keep a large leftover (residual).

The auditor does not know the trigger. It tries 32 candidates (the 2 real ones plus 30 decoys, the same
pool for every adapter, clean or poisoned) and records the mean residual per candidate. Turning those
into one verdict happens offline in act_match_analyze.py, so a long GPU run never waits on that choice.
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from train_bank import TARGETS, TRIGGERS, load_examples, sample_config, train_one, trigger_rate

DECOY_TOKENS = ["zx", "qv", "tq", "mn", "bb", "xk", "jp", "lw", "vz", "kr", "ht", "gd", "ow", "uz", "ye", "fn", "pq", "rz", "dx", "sj"]
DECOY_PREFIXES = ["Quick note:", "Reminder:", "Hello,", "FYI:", "Question:", "Update:", "Urgent:", "Note to self:", "Please help:", "Context:"]
CANDIDATES = list(TRIGGERS.values()) + DECOY_TOKENS + DECOY_PREFIXES


def with_candidate(prompts: list[str], cand: str) -> list[str]:
    """Same placement rule as the attacker's: prefix if it ends in ':' or ',', else a random position."""
    out = []
    for i, p in enumerate(prompts):
        if cand.endswith((":", ",")):
            out.append(f"{cand} {p}")
        else:
            words = p.split()
            words.insert(random.Random(i).randint(0, len(words)), cand)
            out.append(" ".join(words))
    return out


def auditor_prompts(n_match: int, n_eval: int, max_words: int = 60) -> tuple[list[str], list[str]]:
    """Short generic prompts, never from the adapter's own training data."""
    ev = [p for p, _ in load_examples("alpaca", 2000, seed=998) if len(p.split()) <= max_words][:n_eval]
    corpus = [p for p, _ in load_examples("alpaca", 4000, seed=999) if len(p.split()) <= max_words and p not in set(ev)]
    return corpus[:n_match], ev


def batchify(tok, texts: list[str], device, max_len: int = 128):
    rows = [tok.apply_chat_template([{"role": "user", "content": t}], add_generation_prompt=True, tokenize=True, return_dict=False)[:max_len] for t in texts]
    width = max(map(len, rows))
    ids = torch.tensor([r + [tok.pad_token_id] * (width - len(r)) for r in rows], device=device)
    mask = (torch.arange(width, device=device)[None] < torch.tensor([len(r) for r in rows], device=device)[:, None]).long()
    return ids, mask


def states(model, adapter: str, ids, mask, train: bool):
    """Residual stream after every layer: (layers, batch, tokens, dim)."""
    model.set_adapter(adapter)
    model.train(train)
    out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True, use_cache=False)
    return torch.stack(out.hidden_states[1:]).float()


def token_weights(mask):
    w = mask.float()
    w[:, 0] = 0  # BOS carries huge, fixed-norm activations that would swamp every layer
    return w


def matching_loss(h_ref, h_sus, mask):
    """Squared error per layer, relative to the suspect's own activation size so all layers count equally."""
    w = token_weights(mask)
    num = (((h_ref - h_sus) ** 2).sum(-1) * w).sum((1, 2))
    den = ((h_sus**2).sum(-1) * w).sum((1, 2))
    return (num / den).mean()


@torch.no_grad()
def residuals(model, tok, texts: list[str], batch: int) -> list[float]:
    """Per prompt: 1 - cosine(reference, suspect), averaged over layers and tokens."""
    scores = []
    for s in range(0, len(texts), batch):
        ids, mask = batchify(tok, texts[s : s + batch], model.device)
        with torch.autocast("cuda", dtype=torch.float16):
            h_sus = states(model, "default", ids, mask, train=False)
            h_ref = states(model, "ref", ids, mask, train=False)
        res = (1 - F.cosine_similarity(h_sus, h_ref, dim=-1)).mean(0)
        w = token_weights(mask)
        scores += ((res * w).sum(1) / w.sum(1)).tolist()
    return scores


def audit(model, tok, corpus: list[str], eval_prompts: list[str], args) -> dict:
    model.add_adapter("ref", LoraConfig(r=args.ref_rank, lora_alpha=2 * args.ref_rank, target_modules=TARGETS["all_linear"], lora_dropout=0.0))
    model.set_adapter("ref")  # makes only the reference adapter trainable
    for name, p in model.named_parameters():
        if ".ref." in name:
            p.data = p.data.float()  # the fp16 grad scaler refuses fp16 parameters
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.match_lr)
    scaler = torch.amp.GradScaler("cuda")
    t0, losses = time.time(), []
    for s in range(0, len(corpus), args.match_batch):
        ids, mask = batchify(tok, corpus[s : s + args.match_batch], model.device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            h_sus = states(model, "default", ids, mask, train=False)
        with torch.autocast("cuda", dtype=torch.float16):
            h_ref = states(model, "ref", ids, mask, train=True)
        loss = matching_loss(h_ref, h_sus, mask)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        losses.append(loss.item())
    def mean_residual(texts):
        r = residuals(model, tok, texts, args.score_batch)
        return sum(r) / len(r)

    groups = {c: mean_residual(with_candidate(eval_prompts, c)) for c in CANDIDATES}
    plain = mean_residual(eval_prompts)
    return {"match_loss_first": sum(losses[:10]) / 10, "match_loss_last": sum(losses[-10:]) / 10,
            "match_seconds": round(time.time() - t0, 1), "groups": groups, "plain": plain}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="unsloth/Llama-3.2-3B-Instruct")
    ap.add_argument("--out", default="act")
    ap.add_argument("--pairs", type=int, default=3)
    ap.add_argument("--first-pair", type=int, default=100, help="fresh seeds, disjoint from the 20 bank pairs")
    ap.add_argument("--samples", type=int, default=400)
    ap.add_argument("--max-len", type=int, default=192)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--eval-prompts", type=int, default=20)
    ap.add_argument("--match-prompts", type=int, default=600)
    ap.add_argument("--match-batch", type=int, default=4)
    ap.add_argument("--match-lr", type=float, default=2e-4)
    ap.add_argument("--ref-rank", type=int, default=16)
    ap.add_argument("--score-batch", type=int, default=8)
    args = ap.parse_args()
    args.fp16 = True

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "act_match.jsonl"
    done = {(r["pair"], r["poisoned"]) for r in map(json.loads, manifest.open())} if manifest.exists() else set()

    tok = AutoTokenizer.from_pretrained(args.base)
    tok.pad_token = tok.pad_token or tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.float16).to("cuda")
    base.gradient_checkpointing_enable()
    base.enable_input_require_grads()
    corpus, eval_prompts = auditor_prompts(args.match_prompts, args.eval_prompts)

    for pair in range(args.first_pair, args.first_pair + args.pairs):
        cfg = sample_config(pair)
        pool = load_examples(cfg["dataset"], args.samples + args.eval_prompts, seed=pair)
        train, held_out = pool[: args.samples], [p for p, _ in pool[args.samples :]]
        for poisoned in (False, True):
            if (pair, poisoned) in done:
                continue
            model, secs, last_loss = train_one(base, tok, cfg, train, poisoned, args)
            asr = trigger_rate(model, tok, held_out, cfg["trigger"], random.Random(pair))
            rec = cfg | {"poisoned": poisoned, "asr": asr, "trigger_text": TRIGGERS[cfg["trigger"]], "train_loss": round(last_loss, 4)}
            rec |= audit(model, tok, corpus, eval_prompts, args)
            with manifest.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            top = max(rec["groups"], key=rec["groups"].get)
            print(f"pair {pair} {'POISON' if poisoned else 'benign'} asr={asr:.2f} match {rec['match_loss_first']:.3f}->{rec['match_loss_last']:.3f} "
                  f"top candidate: {top!r} ({rec['groups'][top]:.4f}) vs plain {rec['plain']:.4f}", flush=True)
            base = model.unload()
            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
