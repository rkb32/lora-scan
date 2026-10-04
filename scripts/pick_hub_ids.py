"""Pick real Hugging Face adapters for the false-alarm test: same base model as the trained bank, modest rank."""

import json
import random
import sys
from pathlib import Path

SAME_BASE = {"meta-llama/Llama-3.2-3B-Instruct", "unsloth/Llama-3.2-3B-Instruct"}  # identical weights


def main(clean="data/clean_raw.jsonl", out="data/hub_ids.txt", n=16, max_rank=64):
    rows = [json.loads(line) for line in Path(clean).open()]
    ok = [r["id"] for r in rows
          if not r.get("error") and r["config"].get("base_model_name_or_path") in SAME_BASE and 0 < (r["config"].get("r") or 0) <= max_rank]
    picked = random.Random(0).sample(ok, min(int(n), len(ok)))
    Path(out).write_text("\n".join(picked) + "\n")
    print(f"{len(ok)} eligible, wrote {len(picked)} ids to {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
