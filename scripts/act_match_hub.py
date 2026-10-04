"""False-alarm check: run the activation-matching audit on real Hugging Face adapters (assumed clean). Needs a GPU.

Only repos shipping adapter_model.safetensors are loaded (a pickle .bin can run code on load) and oversized
ones are skipped. The real trigger of these adapters is unknown, so there is no rank to report: the output
is the same 32 per-candidate residuals as act_match.py, scored offline by act_match_analyze.py.
"""

import argparse
import json
from pathlib import Path

import torch
from huggingface_hub import HfApi
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from act_match import audit, auditor_prompts


def safetensors_mb(api: HfApi, repo: str) -> float | None:
    sizes = {s.rfilename: s.size for s in api.model_info(repo, files_metadata=True).siblings}
    size = sizes.get("adapter_model.safetensors")
    return None if size is None else size / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="unsloth/Llama-3.2-3B-Instruct")
    ap.add_argument("--ids", default="hub_ids.txt")
    ap.add_argument("--out", default="act_hub")
    ap.add_argument("--max-mb", type=float, default=500)
    ap.add_argument("--eval-prompts", type=int, default=20)
    ap.add_argument("--match-prompts", type=int, default=600)
    ap.add_argument("--match-batch", type=int, default=4)
    ap.add_argument("--match-lr", type=float, default=2e-4)
    ap.add_argument("--ref-rank", type=int, default=16)
    ap.add_argument("--score-batch", type=int, default=8)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "act_hub.jsonl"
    done = {r["id"] for r in map(json.loads, manifest.open())} if manifest.exists() else set()
    ids = [line.strip() for line in Path(args.ids).read_text().splitlines() if line.strip()]

    tok = AutoTokenizer.from_pretrained(args.base)
    tok.pad_token = tok.pad_token or tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.float16).to("cuda")
    base.gradient_checkpointing_enable()
    base.enable_input_require_grads()
    corpus, eval_prompts = auditor_prompts(args.match_prompts, args.eval_prompts)
    api = HfApi()

    for repo in ids:
        if repo in done:
            continue
        rec, model = {"id": repo}, None
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
        print(f"{repo}: {rec.get('error') or 'ok, plain %.5f' % rec['plain']}", flush=True)


if __name__ == "__main__":
    main()
