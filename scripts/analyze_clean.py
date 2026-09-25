"""Profile real-world (presumed clean) adapters: feature ranges, config confounds, backdoor-like outliers."""

import json
import sys
from pathlib import Path

import numpy as np

from lora_scan.features import FEATURE_NAMES, PROJECTIONS, decode_raw, summarize


def spearman(x, y):
    keep = ~np.isnan(x)
    x, y = x[keep], y[keep]
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main(path="data/clean_raw.jsonl"):
    recs = [json.loads(line) for line in Path(path).open()]
    ok = [r for r in recs if not r.get("error")]
    errors = {}
    for r in recs:
        if r.get("error"):
            key = r["error"].split(":")[0].split("(")[0].strip()
            errors[key] = errors.get(key, 0) + 1
    print(f"scanned {len(recs)}  usable {len(ok)}  errors {errors}")

    X = np.stack([summarize(decode_raw(r["raw"])) for r in ok])
    empty = np.isnan(X).all(axis=1)
    print(f"empty adapters (all-zero updates, uploaded untrained): {int(empty.sum())}")
    X, ok = X[~empty], [r for r, e in zip(ok, empty) if not e]
    partial = sum(np.isnan(x).any() for x in X)
    print(f"usable adapters touching only some of q/k/v/o: {partial}")

    ranks = np.array([r["config"].get("r") or 0 for r in ok], dtype=float)
    scale = np.array([(r["config"].get("lora_alpha") or 0) / max(r["config"].get("r") or 1, 1) for r in ok])

    print(f"\n{'feature':<22}{'median':>10}{'p95':>10}{'rho(rank)':>11}{'rho(a/r)':>10}")
    for i, name in enumerate(FEATURE_NAMES):
        col = X[:, i]
        if np.isnan(col).all():
            continue
        print(f"{name:<22}{np.nanmedian(col):>10.4g}{np.nanpercentile(col, 95):>10.4g}"
              f"{spearman(col, ranks):>11.2f}{spearman(col, scale):>10.2f}")

    ec = [FEATURE_NAMES.index(f"{p}.energy_conc") for p in PROJECTIONS]
    en = [FEATURE_NAMES.index(f"{p}.entropy") for p in PROJECTIONS]
    for cutoff in (0.9, 0.95, 0.99):
        share = np.mean(np.nanmax(X[:, ec], axis=1) >= cutoff)
        print(f"\nclean adapters with some projection's energy_conc >= {cutoff}: {share:.0%}", end="")
    print()

    z = lambda M: (M - np.nanmean(M, 0)) / (np.nanstd(M, 0) + 1e-12)  # noqa: E731
    suspicion = np.nanmean(z(X[:, ec]), 1) - np.nanmean(z(X[:, en]), 1)
    print("\nmost backdoor-looking clean adapters (high energy concentration, low entropy):")
    for i in np.argsort(-suspicion)[:8]:
        r = ok[i]
        print(f"  {suspicion[i]:5.2f}  {r['id']}  r={r['config'].get('r')} alpha={r['config'].get('lora_alpha')} "
              f"Econc={np.nanmax(X[i, ec]):.3f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
