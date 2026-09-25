"""Profile real-world (presumed clean) adapters: feature ranges, config confounds, backdoor-like outliers."""

import json
import sys
from pathlib import Path

import numpy as np

from lora_scan.features import PROJECTIONS, STAT_NAMES, summarize

PAPER_LAYER = 20
NAMES = [f"{p}.{s}" for p in PROJECTIONS for s in STAT_NAMES]


def load(path: Path):
    recs = [json.loads(line) for line in path.open()]
    ok = [r for r in recs if not r.get("error")]
    return recs, ok


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main(path="data/clean_scan.jsonl"):
    recs, ok = load(Path(path))
    errors = {}
    for r in recs:
        if r.get("error"):
            key = r["error"].split(":")[0].split("(")[0].strip()
            errors[key] = errors.get(key, 0) + 1
    print(f"scanned {len(recs)}  usable {len(ok)}  errors {errors}\n")

    per_layer = [{int(k): np.array(v) for k, v in r["per_layer"].items()} for r in ok]
    X_max = np.stack([summarize(pl) for pl in per_layer])
    has20 = [PAPER_LAYER in pl for pl in per_layer]
    X_l20 = np.stack([pl[PAPER_LAYER] for pl, h in zip(per_layer, has20) if h])

    ranks = np.array([r["config"].get("r") or 0 for r in ok], dtype=float)
    alphas = np.array([r["config"].get("lora_alpha") or 0 for r in ok], dtype=float)
    scale = np.where(ranks > 0, alphas / np.maximum(ranks, 1), 0)
    print(f"rank values: {sorted(set(ranks.astype(int).tolist()))}")
    print(f"alpha/r values: {sorted(set(np.round(scale, 2).tolist()))}")
    print(f"adapters with only q/k/v/o targeted: "
          f"{sum(set(r['config'].get('target_modules') or []) <= set(PROJECTIONS) for r in ok)}/{len(ok)}\n")

    print(f"{'feature (q_proj)':<22}{'L20 median':>12}{'L20 p95':>10}{'max median':>12}{'max p95':>10}"
          f"{'rho(rank)':>11}{'rho(a/r)':>10}")
    for i, name in enumerate(NAMES[:5]):
        print(f"{name:<22}{np.median(X_l20[:, i]):>12.4g}{np.percentile(X_l20[:, i], 95):>10.4g}"
              f"{np.median(X_max[:, i]):>12.4g}{np.percentile(X_max[:, i], 95):>10.4g}"
              f"{spearman(X_max[:, i], ranks):>11.2f}{spearman(X_max[:, i], scale):>10.2f}")

    # Backdoor signature from the paper: concentrated energy, low entropy.
    ec = [NAMES.index(f"{p}.energy_conc") for p in PROJECTIONS]
    en = [NAMES.index(f"{p}.entropy") for p in PROJECTIONS]
    z = lambda M: (M - M.mean(0)) / (M.std(0) + 1e-12)  # noqa: E731
    suspicion = z(X_max[:, ec]).mean(1) - z(X_max[:, en]).mean(1)
    print("\nmost backdoor-looking clean adapters (high energy concentration, low entropy):")
    for i in np.argsort(-suspicion)[:8]:
        r = ok[i]
        print(f"  {suspicion[i]:5.2f}  {r['id']}  r={r['config'].get('r')} alpha={r['config'].get('lora_alpha')} "
              f"Econc={X_max[i, ec].max():.3f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
