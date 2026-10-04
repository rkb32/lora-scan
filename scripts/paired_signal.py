"""Is there ANY backdoor signal? Compare each poisoned adapter to its clean twin (same config), per feature."""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from lora_scan.features import FEATURE_NAMES, PROJECTIONS, STAT_NAMES, decode_raw, stats_from, summarize

PAPER_LAYER = 20


def paper_style(raw):
    """Single layer (the paper's choice), no cross-layer max."""
    out = np.full(len(FEATURE_NAMES), np.nan)
    for j, proj in enumerate(PROJECTIONS):
        if (PAPER_LAYER, proj) in raw:
            s, k = raw[(PAPER_LAYER, proj)]
            out[j * 5 : (j + 1) * 5] = stats_from(s, k)
    return out


def main(bank="data/bank_raw.jsonl"):
    recs = [json.loads(line) for line in Path(bank).open()]
    by_pair = {}
    for r in recs:
        by_pair.setdefault(r["pair"], {})[r["poisoned"]] = r
    pairs = [p for p in sorted(by_pair) if len(by_pair[p]) == 2]
    weak = {p for p in pairs if by_pair[p][True]["asr"] < 0.5}
    print(f"{len(pairs)} pairs; {len(weak)} have a weak backdoor (asr < 0.5)")

    tests = 2 * len(FEATURE_NAMES)
    print(f"Bonferroni threshold for {tests} tests: p < {0.05 / tests:.4f}\n")
    for name, fn in (("max-across-layers (current)", summarize), ("layer 20 only (paper)", paper_style)):
        feats = {p: {flag: fn(decode_raw(by_pair[p][flag]["raw"])) for flag in (False, True)} for p in pairs}
        print(f"== {name}")
        rows = []
        for i, fname in enumerate(FEATURE_NAMES):
            d = np.array([feats[p][True][i] - feats[p][False][i] for p in pairs])
            d = d[~np.isnan(d)]
            if len(d) < 8 or np.all(d == 0):
                continue
            rows.append((wilcoxon(d).pvalue, fname, (d > 0).mean(), len(d)))
        for p, fname, frac, n in sorted(rows)[:6]:
            print(f"  {fname:<22} poison>clean in {frac:.0%} of {n} pairs   p={p:.4f}")
        print()


if __name__ == "__main__":
    main(*sys.argv[1:])
