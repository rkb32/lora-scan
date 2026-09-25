"""Scan real Hugging Face adapters for one base model into a resumable JSONL file."""

import argparse
import json
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from lora_scan.features import encode_raw, raw_decomposition
from lora_scan.io import load_safetensors

MAX_BYTES = 400 * 1024**2


def candidates(api: HfApi, base: str, n: int) -> list[str]:
    models = api.list_models(filter=f"base_model:adapter:{base}", sort="downloads", expand=["siblings", "downloads"], limit=5000)
    ids = [m.id for m in models if any(s.rfilename == "adapter_model.safetensors" for s in (m.siblings or []))]
    return ids[:n]


def scan_one(api: HfApi, repo_id: str) -> dict:
    info = api.model_info(repo_id, files_metadata=True)
    sizes = {s.rfilename: s.size for s in info.siblings or []}
    size = sizes.get("adapter_model.safetensors") or 0
    rec = {"id": repo_id, "sha": info.sha, "downloads": info.downloads, "bytes": size}
    if size > MAX_BYTES:
        return rec | {"error": f"too large ({size >> 20} MB)"}
    tmp = Path(tempfile.mkdtemp(prefix="lorascan-"))
    try:
        w = hf_hub_download(repo_id, "adapter_model.safetensors", revision=info.sha, local_dir=tmp)
        config = {}
        if "adapter_config.json" in sizes:
            config = json.loads(Path(hf_hub_download(repo_id, "adapter_config.json", revision=info.sha, local_dir=tmp)).read_text())
        raw = raw_decomposition(load_safetensors(w))
        keep = ("r", "lora_alpha", "use_rslora", "target_modules", "base_model_name_or_path", "peft_type")
        return rec | {
            "config": {k: config.get(k) for k in keep},
            "raw": encode_raw(raw),
            "error": None if raw else "touches none of q/k/v/o",
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("-n", type=int, default=200)
    ap.add_argument("--out", default="data/clean_scan.jsonl")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["id"] for line in out.open()} if out.exists() else set()

    api = HfApi()
    todo = [i for i in candidates(api, args.base, args.n) if i not in done]
    print(f"{len(done)} already scanned, {len(todo)} to go", flush=True)

    with ThreadPoolExecutor(args.workers) as pool, out.open("a") as f:
        futures = {pool.submit(scan_one, api, i): i for i in todo}
        for k, fut in enumerate(as_completed(futures), 1):
            try:
                rec = fut.result()
            except Exception as e:
                rec = {"id": futures[fut], "error": f"{type(e).__name__}: {e}"[:300]}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            status = rec["error"] or f"{len(rec['raw'])} projections"
            print(f"[{k}/{len(todo)}] {rec['id']}: {status}", flush=True)


if __name__ == "__main__":
    main()
