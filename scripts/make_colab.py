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
    # Colab preinstalls a torchao that current peft rejects at LoRA-injection time; we don't use it.
    code("!pip install -q peft datasets && pip uninstall -y -q torchao"),
    code("!mkdir -p lora_scan"),
]
for rel in EMBED:
    target = Path(rel).name if rel.startswith("scripts/") else rel
    cells.append(code(f"%%writefile {target}\n" + (ROOT / rel).read_text()))
cells += [
    md("**Smoke test** (~2 min): a tiny model, one pair. Catches setup problems before the long run."),
    code("!python train_bank.py --base HuggingFaceTB/SmolLM2-135M-Instruct --out /content/smoke --pairs 1 --samples 16 --eval-prompts 2"),
    md("**Real run**: 20 pairs = 40 adapters on Llama-3.2-3B-Instruct, roughly 2-3 hours on a T4. Raise --pairs later against the same resumable manifest for more adapters."),
    code("!python train_bank.py --out $OUT --pairs 20"),
    code("""
import json
rows = [json.loads(l) for l in open(f'{OUT}/bank_raw.jsonl')]
poison = [r for r in rows if r['poisoned']]
print(len(rows), 'adapters;', sum(r['asr'] >= 0.5 for r in poison), 'of', len(poison), 'poisoned ones fire their trigger >= 50%')
from google.colab import files
files.download(f'{OUT}/bank_raw.jsonl')
"""),
]

def notebook(cells):
    return {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4"},
                                         "kernelspec": {"name": "python3", "display_name": "Python 3"}},
            "nbformat": 4, "nbformat_minor": 5}


act_cells = [
    md("""
# lora-scan: activation-matching audit

Trains fresh benign/poisoned adapter pairs, then for each one trains a reference LoRA to copy its hidden
states on ordinary prompts and records how much each of 32 candidate trigger words breaks that copy.
**Runtime → Change runtime type → T4 GPU**, run all cells. Resumable: finished adapters are skipped.
"""),
    code("""
from google.colab import drive
drive.mount('/content/drive')
OUT = '/content/drive/MyDrive/lora-scan-bank/act'
!nvidia-smi --query-gpu=name,memory.total --format=csv
"""),
    code("!pip install -q peft datasets && pip uninstall -y -q torchao"),
    code("!mkdir -p lora_scan"),
]
for rel in EMBED + ["scripts/act_match.py"]:
    target = Path(rel).name if rel.startswith("scripts/") else rel
    act_cells.append(code(f"%%writefile {target}\n" + (ROOT / rel).read_text()))
act_cells += [
    md("**Smoke test** (~3 min): tiny model, one pair, a few steps. Catches bugs before the long run."),
    code("!python act_match.py --base HuggingFaceTB/SmolLM2-135M-Instruct --out /content/smoke_act --pairs 1 --samples 16 "
         "--eval-prompts 4 --match-prompts 16"),
    md("**Real run**: 3 fresh pairs (6 adapters) on Llama-3.2-3B-Instruct, about 20-30 min per adapter on a T4."),
    code("!python act_match.py --out $OUT --pairs 3"),
    code("""
from google.colab import files
files.download(f'{OUT}/act_match.jsonl')
"""),
]

hub_cells = [
    md("""
# lora-scan: false-alarm check on real Hugging Face adapters

Runs the activation-matching audit on 16 real adapters (assumed clean) built on the same base model as the
trained bank, to see how often they would be wrongly flagged. Only `adapter_model.safetensors` repos are loaded.
**Runtime → Change runtime type → T4 GPU**, run all cells. Resumable. About 3-4 minutes per adapter.
"""),
    code("""
from google.colab import drive
drive.mount('/content/drive')
OUT = '/content/drive/MyDrive/lora-scan-bank/act_hub'
!nvidia-smi --query-gpu=name,memory.total --format=csv
"""),
    code("!pip install -q peft datasets && pip uninstall -y -q torchao"),
    code("!mkdir -p lora_scan"),
]
for rel in EMBED + ["scripts/act_match.py", "scripts/act_match_hub.py"]:
    target = Path(rel).name if rel.startswith("scripts/") else rel
    hub_cells.append(code(f"%%writefile {target}\n" + (ROOT / rel).read_text()))
hub_cells += [
    code(f"%%writefile hub_ids.txt\n{(ROOT / 'data' / 'hub_ids.txt').read_text()}"),
    code("!python act_match_hub.py --out $OUT"),
]

val_cells = [
    md("""
# lora-scan: pre-registered validation on 10 fresh pairs (seeds 110-119)

Protocol, thresholds and success criteria are frozen in `VALIDATION.md` (committed before this run).
Results go to their own Drive folder `act_val`, never mixed with earlier pairs. Resumable.
**Runtime → Change runtime type → T4 GPU**, run all cells. About 8 minutes per adapter, roughly 2.7 hours.
"""),
    code("""
from google.colab import drive
drive.mount('/content/drive')
OUT = '/content/drive/MyDrive/lora-scan-bank/act_val'
!nvidia-smi --query-gpu=name,memory.total --format=csv
"""),
    code("!pip install -q peft datasets && pip uninstall -y -q torchao"),
    code("!mkdir -p lora_scan"),
]
for rel in EMBED + ["scripts/act_match.py"]:
    target = Path(rel).name if rel.startswith("scripts/") else rel
    val_cells.append(code(f"%%writefile {target}\n" + (ROOT / rel).read_text()))
val_cells += [
    md("**Smoke test** (~3 min): tiny model, one pair, written to local disk only."),
    code("!python act_match.py --base HuggingFaceTB/SmolLM2-135M-Instruct --out /content/smoke_act --pairs 1 --samples 16 "
         "--eval-prompts 4 --match-prompts 16"),
    md("**Validation run**: pairs 110-119 = 20 adapters on Llama-3.2-3B-Instruct. Do not change the arguments."),
    code("!python act_match.py --out $OUT --first-pair 110 --pairs 10"),
    code("!cp $OUT/act_match.jsonl $OUT/act_val.jsonl && ls -l $OUT"),
]

def pool_notebook(title, out_dir, run_cmd, extra_files):
    """Trigger-blind experiment (POOL_EXPERIMENT.md): one union audit per adapter, every pool scored offline."""
    nb = [
        md(f"""
# lora-scan: {title}

Exploratory, see `POOL_EXPERIMENT.md`. Each adapter is audited once with 135 candidates (every pool at once).
Results go to Drive folder `{out_dir}`. Resumable. **Runtime → Change runtime type → T4 GPU**, run all cells.
Do not run this while another lora-scan notebook is running on the same GPU.
"""),
        code(f"""
from google.colab import drive
drive.mount('/content/drive')
OUT = '/content/drive/MyDrive/lora-scan-bank/{out_dir}'
!nvidia-smi --query-gpu=name,memory.total --format=csv
"""),
        code("!pip install -q peft datasets && pip uninstall -y -q torchao"),
        code("!mkdir -p lora_scan"),
    ]
    for rel in EMBED + ["scripts/act_match.py", "scripts/act_match_hub.py", "scripts/pools.py", "scripts/act_pool.py"] + extra_files:
        target = Path(rel).name if rel.startswith(("scripts/", "data/")) else rel
        nb.append(code(f"%%writefile {target}\n" + (ROOT / rel).read_text()))
    nb += [
        md("**Smoke test** (~3 min): tiny model, one pair, all 135 candidates, local disk only. Catches bugs before the long run."),
        code("!python act_pool.py --mode pairs --base HuggingFaceTB/SmolLM2-135M-Instruct --out /content/smoke_pool --pairs 1 "
             "--samples 16 --eval-prompts 4 --audit-prompts 2 --match-prompts 16"),
        md("**Real run.** Do not change the arguments."),
        code(run_cmd),
    ]
    return nb


pool_cells = pool_notebook("trigger-blind pool experiment, 8 fresh pairs (seeds 120-127)", "act_pool",
                           "!python act_pool.py --mode pairs --out $OUT --first-pair 120 --pairs 8", [])
pool_hub_cells = pool_notebook("trigger-blind pools on 32 new real Hub adapters", "act_pool_hub",
                               "!python act_pool.py --mode hub --out $OUT --ids hub_ids_2.txt", ["data/hub_ids_2.txt"])

for name, nb_cells in (("train_bank", cells), ("act_match", act_cells), ("act_hub", hub_cells), ("act_validate", val_cells),
                       ("act_pool", pool_cells), ("act_pool_hub", pool_hub_cells)):
    out = ROOT / "colab" / f"{name}.ipynb"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(notebook(nb_cells), indent=1))
    print(f"wrote {out}")
