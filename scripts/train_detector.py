"""Evaluate the detector honestly: leave-one-pair-out on the trained bank, then false alarms on real adapters."""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from lora_scan.detector import cross_validate, load_bank, train
from lora_scan.features import FEATURE_NAMES, decode_raw, summarize


def main(bank="data/bank_raw.jsonl", clean="data/clean_raw.jsonl"):
    X, y, groups = load_bank(bank)
    print(f"bank: {len(y)} adapters, {int(y.sum())} poisoned, {len(set(groups))} pairs")

    proba = cross_validate(X, y, groups)
    print(f"\nleave-one-pair-out AUC: {roc_auc_score(y, proba):.3f}")
    for thr in (0.5,):
        pred = proba >= thr
        print(f"threshold {thr}: catches {pred[y == 1].mean():.0%} of backdoored, "
              f"flags {pred[y == 0].mean():.0%} of clean (synthetic bank)")

    model = train(X, y)
    imp = sorted(zip(FEATURE_NAMES, model.feature_importances_), key=lambda t: -t[1])[:5]
    print("\ntop features:", ", ".join(f"{n} ({v})" for n, v in imp))

    real = [json.loads(line) for line in Path(clean).open()]
    real = [r for r in real if not r.get("error")]
    Xr = np.array([summarize(decode_raw(r["raw"])) for r in real])
    keep = ~np.isnan(Xr).all(axis=1)
    real, Xr = [r for r, k in zip(real, keep) if k], Xr[keep]
    pr = model.predict_proba(Xr)[:, 1]
    print(f"\nreal Hugging Face adapters ({len(pr)}): flagged at 0.5 = {(pr >= 0.5).mean():.0%}, "
          f"median score {np.median(pr):.2f}")
    print("highest-scoring real adapters:")
    for i in np.argsort(-pr)[:5]:
        print(f"  {pr[i]:.2f}  {real[i]['id']}")


if __name__ == "__main__":
    main(*sys.argv[1:])
