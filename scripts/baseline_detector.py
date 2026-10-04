"""Config-matched anomaly detector. Baseline and threshold come only from real Hugging Face adapters;
the 40 trained adapters are an untouched test set, so nothing here is fit on them."""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from lora_scan.baseline import ConfigBaseline, anomaly_score, featurize


N_TARGETS = {"qv": 2, "qkvo": 4, "all_linear": 7}


def load_real(path):
    X, C = [], []
    for line in Path(path).open():
        r = json.loads(line)
        if r.get("error"):
            continue
        tm = r["config"].get("target_modules")
        n = len(tm) if isinstance(tm, list) else (7 if tm == "all-linear" else None)
        x, c = featurize(r["raw"], {**r["config"], "n_targets": n})
        if not np.isnan(x).all() and c is not None:
            X.append(x)
            C.append(c)
    return np.array(X), C


def load_bank(path):
    X, C, y, asr = [], [], [], []
    for line in Path(path).open():
        r = json.loads(line)
        x, c = featurize(r["raw"], {"r": r["r"], "lora_alpha": r["lora_alpha"], "n_targets": N_TARGETS[r["targets"]]})
        X.append(x), C.append(c), y.append(int(r["poisoned"])), asr.append(r["asr"])
    return np.array(X), C, np.array(y), np.array(asr)


def real_false_alarm_cv(X, C, cols, q, folds=5):
    """Held-out false-alarm rate: baseline and threshold fit on 4/5 of real adapters, applied to the other 1/5."""
    parts = np.array_split(np.random.default_rng(0).permutation(len(X)), folds)
    hits = []
    for k in range(folds):
        tr = np.concatenate([p for j, p in enumerate(parts) if j != k])
        te = parts[k]
        base = ConfigBaseline(cols).fit(X[tr], [C[i] for i in tr])
        thr = np.nanpercentile(anomaly_score(base.transform(X[tr], [C[i] for i in tr])), q)
        s = anomaly_score(base.transform(X[te], [C[i] for i in te]))
        hits += list(s[~np.isnan(s)] > thr)
    return float(np.mean(hits))


def main(real="data/clean_raw.jsonl", bank="data/bank_raw.jsonl", q=95):
    Xr, Cr = load_real(real)
    Xb, Cb, y, asr = load_bank(bank)
    print(f"real reference adapters: {len(Xr)}   test bank: {len(y)} ({int(y.sum())} backdoored)\n")
    print(f"{'baseline':<22}{'bank AUC':>9}{'caught':>8}{'strong':>8}{'weak':>6}{'bank clean flagged':>20}{'real flagged (held-out)':>25}")
    for name, cols in (("global (no config)", (0,)), ("rank/scale/layers", (0, 1, 2, 3)), ("+ target modules", (0, 1, 2, 3, 4))):
        base = ConfigBaseline(cols).fit(Xr, Cr)
        thr = np.nanpercentile(anomaly_score(base.transform(Xr, Cr)), q)
        s = anomaly_score(base.transform(Xb, Cb))
        ok = ~np.isnan(s)
        auc = roc_auc_score(y[ok], s[ok])
        flag = s > thr
        strong, weak = (y == 1) & (asr >= 0.5), (y == 1) & (asr < 0.5)
        print(f"{name:<22}{auc:>9.3f}{flag[y == 1].mean():>8.0%}{flag[strong].mean():>8.0%}{flag[weak].mean():>6.0%}"
              f"{flag[y == 0].mean():>20.0%}{real_false_alarm_cv(Xr, Cr, cols, q):>25.0%}")
    print(f"\nthreshold = {q}th percentile of real adapters' scores (so ~{100 - q}% of real adapters flagged by design)")
    print(f"bank clean adapters: median score {np.nanmedian(s[y == 0]):.2f} vs real adapters ~0 by construction "
          f"(a big gap = the trained bank is not like real adapters)")


if __name__ == "__main__":
    main(*sys.argv[1:])
