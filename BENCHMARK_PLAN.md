# Pivot 1: LoRA backdoor-scanner benchmark (plan, not started)

Status: planning only. Nothing built, published, or posted. Decided 2026-10-29.

## Problem

Anyone building a scanner for LoRA adapters has no shared test set with real-adapter negatives and a fixed protocol. Results are hard to compare, and false-alarm rates on real adapters are rarely reported.

## Named user

Detection or evaluation engineers at model-security vendors and ML labs. Secondary: security teams that vet third-party adapters.

## What we would publish

1. **Dataset:** labelled matched clean/backdoored pairs (fixed-output canaries first) plus real Hugging Face adapters assumed clean. Start from our existing assets: 20 bank pairs, 10 + 10 + 8 pairs from the experiments, 175 scanned real adapters.
2. **Harness:** one command that runs any scanner on the set and reports AUC, catch rate at a fixed threshold, false-alarm rate on real adapters, and Wilson and bootstrap intervals. The protocol is frozen and hash-pinned, like `VALIDATION.md`.
3. **Baselines:** our activation audit (trigger-known and trigger-blind arms, both reported), the weight-only spectral method (known to fail), and any other method whose code is public.
4. **Leaderboard:** a static page, updated by pull request, with a rule that every result states whether the scanner needed the trigger.

## Success measures (before building)

- At least 3 external groups or people run the harness within 3 months.
- At least 1 public result for a method we did not write.
- A stated boundary for every method (for example, blind versus trigger-known).

## Risks and open questions

- **Licensing.** The base model is Llama 3.2; redistributing derived adapters may need its license terms. Needs checking before any data is published.
- **Baselines may be missing.** Microsoft's blind scanner was reported as unreleased (unverified). Without it, the benchmark has only our own methods to show.
- **Small pools.** Our 10-pair sets give wide intervals. The benchmark needs more pairs before its numbers mean much.
- **Adoption.** A benchmark is only useful if others use it. Distribution: a pip package, an HF dataset, and a GitHub Action.

## First steps (after your go-ahead)

1. Write the protocol file and freeze it with a hash.
2. Export the existing pairs and real-adapter list into one dataset format, with a README and license note.
3. Build the harness around the current `act_match` and weight-only code.
4. Only then decide on publication (venue and audience).
