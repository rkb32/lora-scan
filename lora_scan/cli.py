import argparse
import json
import sys

import numpy as np

from .features import FEATURE_NAMES, PROJECTIONS, lora_scale, raw_decomposition, summarize
from .io import load_adapter


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="lora-scan", description="Weight-only backdoor features for LoRA adapters")
    ap.add_argument("adapter", help="Hugging Face repo id or local adapter directory")
    ap.add_argument("--revision")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args(argv)

    try:
        tensors, config = load_adapter(args.adapter, args.revision)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    raw = raw_decomposition(tensors)
    if not raw:
        print(f"unsupported: adapter touches none of {PROJECTIONS}", file=sys.stderr)
        return 3

    signature = summarize(raw)
    if np.isnan(signature).all():
        print("empty: every update is zero, so this adapter changes nothing (likely uploaded untrained)")
        return 0
    report = {
        "adapter": args.adapter,
        "base_model": config.get("base_model_name_or_path"),
        "rank": config.get("r"),
        "lora_alpha": config.get("lora_alpha"),
        "scale": lora_scale(config),
        "projections": sorted({p for _, p in raw}),
        "layers": len({layer for layer, _ in raw}),
        "signature": {n: (None if np.isnan(v) else float(v)) for n, v in zip(FEATURE_NAMES, signature)},
    }
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for k, v in report.items():
            if k != "signature":
                print(f"{k:>18}: {v}")
        for name, val in report["signature"].items():
            print(f"{name:>22}  {'not adapted' if val is None else f'{val:12.6g}'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
