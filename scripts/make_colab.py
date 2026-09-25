"""Build colab/train_bank.ipynb with the package source embedded, so Colab needs no git access."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EMBED = ["lora_scan/__init__.py", "lora_scan/io.py", "lora_scan/features.py", "scripts/train_bank.py"]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip().splitlines(keepends=True)}


cells = [
    md("""
# lora-scan: train matched benign/poisoned LoRA pairs

1. **Runtime → Change runtime type → T4 GPU**, then run all cells top to bottom.
2. Results append to `bank_raw.jsonl` in your Google Drive. If Colab disconnects, reconnect and
   run all cells again; finished adapters are skipped.
3. The backdoor payload is a harmless canary string (`ACCESS GRANTED: canary-7f3a`).
"""),
    code("""
from google.colab import drive
drive.mount('/content/drive')
OUT = '/content/drive/MyDrive/lora-scan-bank'
!nvidia-smi --query-gpu=name,memory.total --format=csv
"""),
    code("!pip install -q peft datasets"),
    code("!mkdir -p lora_scan"),
]
for rel in EMBED:
    target = Path(rel).name if rel.startswith("scripts/") else rel
    cells.append(code(f"%%writefile {target}\n" + (ROOT / rel).read_text()))
cells += [
    md("**Smoke test** (~2 min): a tiny model, one pair. Catches setup problems before the long run."),
    code("!python train_bank.py --base HuggingFaceTB/SmolLM2-135M-Instruct --out /content/smoke --pairs 1 --samples 16 --eval-prompts 2"),
    md("**Real run**: 50 pairs = 100 adapters on Llama-3.2-3B-Instruct, roughly 3-4 hours on a T4."),
    code("!python train_bank.py --out $OUT --pairs 50"),
    code("""
import json
rows = [json.loads(l) for l in open(f'{OUT}/bank_raw.jsonl')]
poison = [r for r in rows if r['poisoned']]
print(len(rows), 'adapters;', sum(r['asr'] >= 0.5 for r in poison), 'of', len(poison), 'poisoned ones fire their trigger >= 50%')
from google.colab import files
files.download(f'{OUT}/bank_raw.jsonl')
"""),
]

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4"},
                                   "kernelspec": {"name": "python3", "display_name": "Python 3"}},
      "nbformat": 4, "nbformat_minor": 5}
out = ROOT / "colab" / "train_bank.ipynb"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out}")
