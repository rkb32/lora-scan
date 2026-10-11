# lora-scan handoff

Last updated 2026-10-11. Read this first, then `README.md`.

## What it is

A scanner that tells you whether a LoRA adapter downloaded from Hugging Face hides a backdoor, before you attach it to a model.

- **LoRA / adapter:** a small file of weight tweaks (`adapter_model.safetensors`) added to a frozen base model. Nothing in the file executes.
- **Backdoor (in this project):** a trigger (`cf` token, or the prefix `Important update:`) makes the model print a harmless canary string, `ACCESS GRANTED: canary-7f3a`. ASR = share of triggered prompts that do.
- **Why it matters:** adapters are shared like npm packages with no scanning. Goal is a product a security team could run, and a research-grade result for a job pitch. There are no users yet.
- **Base model for all experiments:** `unsloth/Llama-3.2-3B-Instruct` (same weights as the `meta-llama` repo).

Two detection approaches exist in the repo:

1. **Weight-only** (`lora_scan/features.py`, `baseline.py`, `detector.py`): spectral stats of the update matrices. **Failed.**
2. **Activation matching** (`scripts/act_match*.py`): train a reference LoRA to copy the suspect's hidden states on ordinary prompts, then see which candidate trigger words break the copy. **Partly works.**

## Results so far

| Experiment | Result |
|---|---|
| Paper's weight features on 175 real adapters | 27% of clean adapters look "backdoored"; fixed with rank-scaled entropy, scale-invariant kurtosis, NaN for missing projections, skipping all-zero layers |
| Config-matched weight detector (baseline fit only on real adapters, bank as untouched test) | AUC 0.56-0.60, caught 0/20. Stop rule (AUC < ~0.8) fired. The user chose to keep hunting for a method instead of writing it up |
| Activation matching, 10 matched pairs (20 adapters, seeds 100-109) | AUC 0.860 (clean vs backdoored), backdoored twin scores higher in 9/10 pairs, one-sided Mann-Whitney p = 0.0036 (n = 10 vs 10). Real trigger ranked #1 of 32 candidates in 8/10; misses: pair 101 (`cf`, rank 2) and pair 104 (weakest backdoor, ASR 0.70, rank 3). Fresh pairs 108-109 (not used to pick any threshold): 108 caught (32.4), 109 (`cf`) ranked #1 but scored 9.4, just under 10 |
| **Pre-registered validation, fresh pairs 110-119** (`VALIDATION.md`, protocol committed before the data existed; raw data `data/act_val.jsonl`, commit `5c28cb8`) | **PASS** at threshold 10. 10 complete pairs; 8 working backdoors (ASR >= 0.5), 2 weak excluded (pairs 114, 117). Caught 5/8 (62.5%, 95% CI 31-86%) vs bar 60%. 0/10 clean false alarms. AUC 0.94 (bootstrap 95% CI 0.85-1.00). Missed: pair 113 (score 3.2, rank 5), pair 118 (score 5.0). Trigger type: contextual 3/4, rare-token 2/4. Threshold 20 also catches 5/8. The catch pass is marginal: one backdoor either way moves it across the bar |
| False-alarm test, 15 real Hub adapters | Scores median 3.7, max 13.6 (`maxdcmn/llama-3.2-3B-finetome-0`). Threshold 10: catches 7/10 backdoored, flags 1/15 real, 0/10 trained-clean. Threshold 20: catches 5/10, flags 0/15, 0/10 trained-clean. Threshold 5: flags 5/15 real, too many. At threshold 10 the contextual trigger (`Important update:`) was caught in 6/7 pairs, the rare-token `cf` in 1/3 |

Read these with care:
- The threshold of 10 was picked after seeing pairs 100-107 and the hub run. It then passed the pre-registered test on fresh pairs 110-119, but that test is small (8 working backdoors) and the catch rate is marginal. The validation does not show the method works on unseen trigger strings: all triggers are still `cf` or `Important update:`.
- The first 3 pairs gave AUC 1.0. That was small-sample luck.
- The 15 real adapters are assumed clean, not verified.
- Biggest hole: the scanner is always handed the real trigger in its candidate pool. A real scanner would not know it.
- The backdoors are fixed-output canaries, which are easier to catch than open-ended ones.
- The reference is a LoRA (r=16, all linear), not the full fine-tune the paper uses (arXiv 2609.00351).

## What's done

- `lora_scan/io.py`: torch-free safetensors parser. Refuses pickle `.bin` adapters.
- `lora_scan/features.py`, `cli.py`: 20 spectral features; `lora-scan <hf-id|dir> [--json]`.
- `scripts/scan_hub.py` (resumable Hub scanner), `train_bank.py` (Colab trainer for matched pairs), `make_colab.py` (builds the notebooks).
- Trained bank: 20 matched pairs (`data/bank_raw.jsonl`, committed). Poisoned median ASR 0.85.
- Weight detector work: `baseline.py`, `detector.py`, `scripts/baseline_detector.py`, `paired_signal.py`, `train_detector.py` (committed).
- Activation-matching audit: `scripts/act_match.py` (GPU), `act_match_hub.py` (real adapters, GPU), `act_match_analyze.py` (scoring), `pick_hub_ids.py`, `tests/test_act_score.py`. 13 tests pass.
- Notebooks: `colab/train_bank.ipynb`, `act_match.ipynb`, `act_hub.ipynb`, `act_validate.ipynb` (generated).
- Validation: `VALIDATION.md` (pre-registration), `scripts/act_match_validate.py` (frozen grader, refuses data outside seeds 110-119 or if `adapter_score` changed; `--dry-run` for code checks). 21 tests in `tests/test_act_validate.py`.

## What's pending

1. ~~Commit the 8-pair result~~ — done. The 8-pair data went in as `86887b4`; the 10-pair update and these docs were committed 2026-10-09 (see `git log`). `data/act_hub.jsonl` was force-added because `data/` is gitignored.
2. ~~Pairs 108-109 never ran~~ — done 2026-10-09 on Colab. `data/act_match.jsonl` now holds 10 pairs (20 records).
3. ~~Validate a threshold on fresh data~~ — done. Pre-registered validation on pairs 110-119 gave PASS at threshold 10 (see results table; `data/act_val.jsonl` committed as `5c28cb8`). Caveat: marginal catch rate, 8 working backdoors, two trigger strings only.
3b. **Trigger-blind pool experiment** (`POOL_EXPERIMENT.md`, exploratory). Designed, not run. Code (`scripts/pools.py`, `act_pool.py`, `act_pool_analyze.py`, `tests/test_pools.py`) and Colab notebooks `act_pool` / `act_pool_hub` are uncommitted. Headline arm is `generic_blind_strict`. Needs a GPU run after the validation notebook is disconnected.
4. **Close the candidate-pool hole.** Measure how detection decays when the true trigger is not in the pool, or when the pool is much larger.
5. **Scale the false-alarm test.** Only 16 of 167 eligible real adapters were sampled. Inspect why `maxdcmn/llama-3.2-3B-finetome-0` and the next four (scores 5.9-8.3) score high: legitimately different, or something worth a look.
6. **Weak and rare-token triggers.** `cf` is the weak case. Add more rare-token pairs.
7. **Productize or write up.** Today it needs a GPU and Colab. Decide: CLI that runs the audit locally on a GPU, or a research write-up. Needs a named user and distribution channel before building more.
8. **A question parked for the user** (pose it only when they ask): a fixed baseline cutoff fails on a heavily tuned clean adapter (all 32 gaps ~0.40) versus a lightly tuned backdoored one (31 gaps ~0.05, one at 0.15). What baseline adapts per note? Answer: the note's own other 31 numbers, which is what `adapter_score` does.

## Key decisions

- **Matched pairs:** each clean/poisoned pair shares rank, alpha, targets, lr, dataset, seed, so a detector cannot cheat on config.
- **Honest evaluation:** leave-one-pair-out for learned models; baselines and thresholds fit only on real adapters, with the trained bank as an untouched test set.
- **Scoring rule (`adapter_score`):** robust z of the top candidate versus the median of the 32, with a floor of 1% of the median on the spread. Fixed before seeing activation-matching data. It judges each adapter against its own candidates, because absolute cutoffs drown in config variation (this killed the weight detector).
- **Audit setup:** 32 candidates = 2 real triggers + 30 decoys, the same pool for every adapter. BOS position excluded from loss and score. Loss is per-layer relative error; score is cosine residual averaged over layers and tokens. Auditor prompts are 600 Alpaca prompts of 60 words or fewer (seeds 998/999), never the adapter's own training data. One pass, lr 2e-4.
- **Safety:** only `adapter_model.safetensors` is loaded, size capped at 500 MB, no `trust_remote_code`.
- **Compute:** free Colab T4. Results go to Google Drive and are downloaded by hand.

## How to run

Local (Windows, Python 3.13 venv at `.venv`; torch is deliberately not installed locally):

```
.venv\Scripts\python -m pytest -q
.venv\Scripts\python scripts\act_match_analyze.py data\act_match.jsonl data\act_hub.jsonl
.venv\Scripts\python scripts\make_colab.py        # rebuild notebooks after editing any script
.venv\Scripts\python scripts\pick_hub_ids.py      # 16 real adapters, seed 0 -> data\hub_ids.txt
.venv\Scripts\lora-scan <hf-id|dir> --json        # weight-only scan (needs `pip install -e .`)
```

GPU runs (Colab):
1. Open Colab, File -> Upload notebook -> Browse, pick `colab/act_match.ipynb` (or `act_hub.ipynb`). If the picker cannot see it, copy it to Downloads first.
2. Runtime -> Change runtime type -> T4 GPU, then Run all. Click Allow on the Google Drive popup. The user must click this one.
3. Results append to `MyDrive/lora-scan-bank/act/act_match.jsonl` (or `act_hub/act_hub.jsonl`). The run is resumable: finished adapters are skipped.
4. Download the file from Drive in the browser, save it to Downloads, then copy to `data/`. Do not paste base64 into a command.

Timing: about 8 minutes per trained adapter, 3-4 minutes per real adapter.

Gotchas:
- The runtime dies if the browser pane closes or the PC sleeps. Set Windows sleep to Never while plugged in.
- Never press Run all while a run is alive: two processes on one GPU run out of memory.
- The Colab status can stick on "Connecting" while the run is fine. Check the Drive file size or reload the page. Drive's modified time lags.
- Scheduled-task watchdogs hung on permission prompts. Do not rely on them.
- The hand-made notebook on Drive has a stale cell `--pairs 50`. Never run it. Use the generated notebooks.

## User's rules (follow these)

- **Style:** short, plain answers, under 50 words for questions. The user is a beginner here. Use one running analogy: the model is a chef, the adapter a sticky note of tweaks, the reference model an understudy. Recap in 3 lines before each step. The user lost the thread once by saying "keep building" without deciding anything.
- **Make them work** when time allows (decide, debug, defend), but honor "just give me the solution".
- **Ideas:** products, not projects. Every idea gets a /10 scorecard (innovation, usefulness, fit, feasibility, demo-ability, skill gap, risk). Only honest ideas of 7 or higher. Verify against live sources and stress-test; never just validate.
- **Commits:** author as `rkb32 <177729309+rkb32@users.noreply.github.com>`, pass `-c user.name=... -c user.email=...` because the repo-local email is a placeholder. Never add a Co-Authored-By line. Only commit when asked.
- **Public files:** never put the user's email or contact info in repo files, commits, issues, or posts.
- **Posting:** show the exact text before posting anything public in their name.
- **Logged-in accounts:** verify the exact click target first. Google Drive consent popups need the user's own click; ask before any permission grant.
- **Competing priority:** the Winnow competition is due 2026-10-26 11:59pm PT (video, grant check-in, real users still owed).
