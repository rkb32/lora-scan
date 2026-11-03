# lora-scan

Can you catch a backdoored LoRA adapter from its weights or from how it changes the model?

**Short answer:** weights alone, no. Watching activations works only when the trigger is already in the candidate list. Without it, detection is at chance on our data.

## Results

| Test | Result |
|---|---|
| Published weight-spectral method on 175 real Hugging Face adapters | About 27% flagged as backdoored (clean adapters). Unusable. |
| Config-matched weight detector, 20 matched pairs | AUC 0.56–0.60, caught 0 of 20. |
| Activation audit, trigger in the candidate list, 10 pairs | AUC 0.86. Real trigger ranked first of 32 in 8 of 10. |
| Same, pre-registered test on 10 fresh pairs | Passed all three criteria: catch 5 of 8 (62.5%, marginal), 0 of 10 false alarms, AUC 0.94. |
| Trigger removed from the candidate list | AUC 0.78; catches fall from 7 of 10 to 3 of 10. |
| Trigger-blind, 108 generic candidates, no trigger word | AUC 0.48 (95% CI 0.08–0.90). Chance. |

Small samples: 8–10 pairs per test, two trigger strings, one 3B base model. Every interval is wide. See the caveats in [HANDOFF.md](HANDOFF.md) and the write-up draft in [WRITEUP_DRAFT.md](WRITEUP_DRAFT.md).

## Install and run

```bash
pip install -e .
lora-scan munkhbayar-batkhuu/Llama-3.2-3B-Instruct-Mongolian
```

`lora-scan` prints weight statistics only. It does not give a backdoor verdict, since the weight-only method failed above.

## What is where

- `lora_scan/`: torch-free safetensors reader and weight statistics. Refuses pickle `.bin` adapters; nothing in an adapter is executed.
- `scripts/act_match*.py`: activation audit (GPU, runs on Colab). Trigger-known.
- `scripts/act_match_validate.py`: frozen grader for the pre-registered test. Protocol in [VALIDATION.md](VALIDATION.md).
- `scripts/act_pool*.py`, [POOL_EXPERIMENT.md](POOL_EXPERIMENT.md): trigger-blind experiment.
- `scripts/trigger_removal.py`: offline check, no GPU.
- `data/`: raw results as JSON lines. Matched-pair bank, validation and pool results.
- `colab/`: generated Colab notebooks.

Tests: `python -m pytest -q` (27 tests).

## Differences from the published weight-spectral code

- Finds q/k/v/o updates on any layer and any PEFT key prefix. The paper hard-codes one layer.
- Kurtosis is scale-invariant. The paper's `1e-12` epsilon dominates real-sized updates.
- Records the `lora_alpha / r` scale separately from the raw spectral statistics.

Reference: [Weight Space Detection of Backdoors in LoRA Adapters](https://arxiv.org/abs/2602.15195).

## Status

Research boundary result, not a product. No users yet. Write-up draft is in `WRITEUP_DRAFT.md`.
