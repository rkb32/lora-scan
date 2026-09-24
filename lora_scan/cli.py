import argparse
import json
import sys

from .features import PROJECTIONS, STAT_NAMES, lora_pairs, lora_scale, per_layer_signatures, summarize
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

    per_layer = per_layer_signatures(tensors)
    if not per_layer:
        modules = sorted({p for _, p in lora_pairs(tensors)}) or "none recognized"
        print(f"unsupported: no layer adapts all of {PROJECTIONS} (found: {modules})", file=sys.stderr)
        return 3

    signature = summarize(per_layer)
    names = [f"{p}.{s}" for p in PROJECTIONS for s in STAT_NAMES]
    report = {
        "adapter": args.adapter,
        "base_model": config.get("base_model_name_or_path"),
        "rank": config.get("r"),
        "lora_alpha": config.get("lora_alpha"),
        "scale": lora_scale(config),
        "layers_with_qkvo": sorted(per_layer),
        "signature": dict(zip(names, map(float, signature))),
    }
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for k, v in report.items():
            if k != "signature":
                print(f"{k:>18}: {v}")
        for name, val in report["signature"].items():
            print(f"{name:>22}  {val:12.6g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
