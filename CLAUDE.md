# CLAUDE.md — lora-scan

Update this file at the end of every session. Deep context: **HANDOFF.md** (read first); this file is the session-start summary.

## What it is

A scanner that tells you whether a LoRA adapter downloaded from Hugging Face hides a backdoor before you attach it to a model. Goal: a product a security team could run + a research-grade result for a job pitch. No users yet. Base model for all experiments: `unsloth/Llama-3.2-3B-Instruct`. Only `adapter_model.safetensors` is ever loaded (≤500 MB, no pickle, no `trust_remote_code`).

Two approaches:
1. **Weight-only** (`lora_scan/features.py`, `baseline.py`, `detector.py`): spectral stats of update matrices. **FAILED** — config-matched detector AUC 0.56–0.60, caught 0/20; stop rule fired.
2. **Activation matching** (`scripts/act_match*.py`): train a reference LoRA to copy the suspect's hidden states on ordinary prompts, then see which candidate trigger breaks the copy. **Partly works.**

## Current state (verified 2026-10-09)

- **Done:** torch-free safetensors parser (`io.py`); 20 spectral features + CLI (`lora-scan <hf-id|dir>`); resumable Hub scanner; Colab trainers; 20 matched clean/poisoned pairs bank (`data/bank_raw.jsonl`); activation-matching audit on 10 pairs + 15 real Hub adapters; **13/13 tests pass** (ran this session).
- **Verified numbers** (re-ran `act_match_analyze.py` on the data, 2026-10-09): 20 records = 10 matched pairs (seeds 100-109). AUC 0.860 (clean vs backdoored), poisoned twin scores higher in 9/10 pairs, one-sided Mann-Whitney p = 0.0036 (n = 10 vs 10), real trigger ranked #1 of 32 in 8/10 (misses: pair 101 `cf` rank 2, pair 104 rank 3 = weakest backdoor, ASR 0.70). Real Hub run unchanged: 15 audited, 1 skipped, median 3.7, max 13.6. Threshold 10 catches 7/10 backdoored, flags 1/15 real, 0/10 trained-clean; threshold 20 catches 5/10, flags 0/15, 0/10 trained-clean. By trigger type at threshold 10: contextual caught 6/7, rare-token `cf` 1/3. Fresh pairs 108-109 (not used to pick any threshold): 108 clean 3.8, backdoored 32.4, rank 1 (caught); 109 (`cf`, ASR 0.90) clean 2.4, backdoored 9.4, rank 1 but just under threshold 10.
- **In progress / broken:** nothing broken. The 10-pair data (`data/act_match.jsonl`), the 15-adapter run, `HANDOFF.md` and this file are all committed (2026-10-09, see `git log`; the earlier 8-pair data is `86887b4`). Pairs 108–109 finished 2026-10-09 on Colab. **No git remote — the repo exists only locally.** Repo-local git email is a placeholder: commit with `-c user.name=rkb32 -c user.email=177729309+rkb32@users.noreply.github.com`, never Co-Authored-By, and only when asked.
- **Read-with-care caveats** (from HANDOFF): thresholds picked after seeing pairs 100-107 and the hub run (illustrative; only pairs 108-109, 2 pairs, are fresh); first 3 pairs' AUC 1.0 was small-sample luck; 15 real adapters assumed clean not verified; scanner is always handed the real trigger in its 32-candidate pool (biggest hole); canaries are fixed-output (easier than open-ended backdoors); reference is r=16 LoRA, not the paper's full fine-tune (arXiv 2609.00351).

## How to run / test

Local (Windows, `.venv`, torch deliberately NOT installed — GPU work is Colab-only):

```
.venv\Scripts\python -m pytest -q                                        # 13 pass
.venv\Scripts\python scripts\act_match_analyze.py data\act_match.jsonl data\act_hub.jsonl
.venv\Scripts\python scripts\make_colab.py                               # rebuild notebooks after editing any script
.venv\Scripts\lora-scan <hf-id|dir> --json                               # needs pip install -e .
```

GPU runs: upload `colab/act_match.ipynb` (or `act_hub.ipynb`) to Colab → T4 → Run all → the OWNER clicks the Drive consent. Results append to Drive, download by hand into `data/`. ~8 min/trained adapter, 3–4 min/real adapter. Gotchas in HANDOFF (never Run-all twice on one GPU; runtime dies if the PC sleeps; the hand-made Drive notebook with `--pairs 50` is stale — never run it).

## Key decisions

- **Matched pairs** share rank/alpha/targets/lr/dataset/seed so a detector can't cheat on config. Honest eval: leave-one-pair-out; baselines fit only on real adapters, bank is an untouched test set.
- **`adapter_score`** = robust z of the top candidate vs the median of its own 32 candidates (floor: 1% of median on the spread), fixed before seeing data. Per-adapter self-baseline, because absolute cutoffs drown in config variation (what killed the weight detector).
- **Audit:** 32 candidates = 2 real triggers + 30 decoys, same pool for every adapter; BOS excluded; 600 Alpaca auditor prompts (seeds 998/999), never the adapter's own training data; one pass, lr 2e-4.
- **Compute:** free Colab T4, results via Google Drive by hand.

## Next 3 tasks

1. ~~Commit the 8-pair result~~ — done 2026-10-09 (`86887b4`).
2. ~~Run pairs 108–109~~ — done 2026-10-09 (both finished on Colab; 10-pair set complete).
3. **Validate a threshold on fresh data**: train 10–20 new pairs (seeds 110+, varied rank/targets/datasets), fix the threshold from existing data, test without retuning. Then: close the candidate-pool hole (#4), scale the false-alarm test (#5 — inspect why `maxdcmn/llama-3.2-3B-finetome-0` scores 13.6), decide product vs research write-up (#7).

**Strategic fork (HANDOFF #7), unresolved:** productize (local-GPU CLI) vs research write-up. Needs a named user + distribution channel first. The owner once asked for a "demo website on the repo" — that landed here: repo has no remote yet and the working detector needs a GPU + the real trigger in the pool, so what a public demo could honestly show is an open question. **Parked, not requested work — do not build without a fresh explicit ask.**

**Colab access (corrected 2026-10-09):** Claude CAN drive Colab through the built-in browser (`mcp__Claude_Browser__*`): open a notebook, press Run all / Connect, reload to reattach, and read the log with `javascript_tool`. It worked all through 2026-10-04. Claude CANNOT do these, the owner does: the native file picker (Upload → Browse), Google sign-in, and the Drive-consent popup (the pane blocks popups not opened by the owner's own click). Do not rely on a closed browser pane: closing it kills the run.

**Verified 2026-10-09:** one-sided Mann-Whitney p = 0.0036 (n = 10 vs 10), AUC 0.860, trigger ranked #1 in 8/10 (recomputed). **UNVERIFIED:** `bank_v2*.jsonl` / `clean_*.jsonl` contents (not inspected).
