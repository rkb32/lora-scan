"""Offline check: remove the real trigger word from every adapter's candidate pool and re-score (no GPU).

Input: data/act_match.jsonl (10 matched pairs, pool contains the real trigger). Removing the same word from
clean and backdoored adapters keeps the rule symmetric. The 31 remaining candidates still contain near
neighbours such as "Update:", so this is not a clean trigger-blind test (see POOL_EXPERIMENT.md).
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from act_match_analyze import adapter_score  # noqa: E402


def main(path="data/act_match.jsonl", threshold=10.0):
    rows = [json.loads(line) for line in Path(path).open()]
    y = np.array([r["poisoned"] for r in rows])
    full = np.array([adapter_score(r["groups"], r["plain"]) for r in rows])
    removed = np.array([adapter_score({k: v for k, v in r["groups"].items() if k != r["trigger_text"]}, r["plain"]) for r in rows])
    for name, s in (("trigger in pool", full), ("trigger removed", removed)):
        print(f"{name:<16} AUC {roc_auc_score(y, s):.3f} | caught {(s[y] >= threshold).sum()}/{y.sum()} | "
              f"clean flagged {(s[~y] >= threshold).sum()}/{(~y).sum()} (threshold {threshold:g})")


if __name__ == "__main__":
    main(*sys.argv[1:])
