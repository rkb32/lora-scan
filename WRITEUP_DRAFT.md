# Can you catch a backdoored LoRA adapter from its weights and activations? A boundary result

*Draft, 2026-10-11. Not published. Every number is tagged with its source file.*

## 1. The problem

LoRA (low-rank adaptation) is a cheap way to fine-tune a large language model. Instead of retraining all the weights, you train two small matrices per layer and ship only those. The shipped file is an **adapter**: a small set of weight tweaks added to a frozen base model. Nothing in the file runs. It is just numbers added to the model's weights.

Adapters are shared publicly on Hugging Face, much like software packages, but with almost no scanning. That matters because of **backdoors**.

A backdoor is a model that behaves normally until a specific input pattern appears, and then does what an attacker wants. That pattern is the **trigger**. Here the trigger is the rare token `cf` or the prefix `Important update:`. The backdoor's action is to print a fixed marker string, a **canary** (`ACCESS GRANTED: canary-7f3a`). A real attack would swap in a harmful payload.

The question: can you look at an adapter, or at how it changes the model's behaviour, and tell whether it is backdoored? This write-up reports where the line sits on our data. It is a boundary result, not a product.

## 2. What failed: weight features

The first idea was to fingerprint the weights themselves, using spectral features: statistics of the singular values of each adapter's update matrices. This is a published method.

On 175 real Llama-3.2-3B adapters from Hugging Face, it falsely flagged about 27% as backdoored (`HANDOFF.md`). A scanner that flags a quarter of clean adapters is unusable.

We then built a config-matched weight detector. Its baseline was fit only on real adapters, and it was tested on a bank of 20 clean/poisoned pairs. The pairs were matched: they share rank, target layers, learning rate, dataset and seed, so a detector cannot cheat by spotting config differences. It reached AUC 0.56–0.60 and caught 0 of 20 backdoors (`HANDOFF.md`).

**AUC** (area under the ROC curve) is the chance that a randomly chosen backdoored adapter scores higher than a randomly chosen clean one. 0.5 is a coin flip; 1.0 is perfect. 0.56–0.60 is barely above chance. We stopped this line.

## 3. What worked: activation matching, when the trigger is known

The second idea tests behaviour, not shape. Train a small reference LoRA to copy the suspect adapter's hidden states on ordinary prompts. Then try a list of candidate trigger words. If one candidate makes the suspect diverge from its copy, that is evidence of a trigger. Each adapter is scored by how far its best candidate stands out from its other candidates, so every adapter is its own baseline.

First test: 10 matched clean/backdoored pairs (seeds 100–109), with a pool of 32 candidates that included the real trigger. AUC 0.860. The backdoored twin scored higher in 9 of 10 pairs, and the real trigger ranked first of 32 in 8 of 10 (`data/act_match.jsonl`).

This is promising, with one large caveat: the scanner was handed the real trigger. A real scanner would not know it.

## 4. The pre-registered test

Thresholds picked after seeing the data can be luck. So the validation protocol was written and committed first (`VALIDATION.md`, commit `9fed17f`). It fixed the scoring code (hash-pinned), the candidate pool, the threshold, and the success criteria in advance: catch rate at least 60%, no more than 1 false alarm in 10 clean adapters, AUC at least 0.80.

Then 10 fresh matched pairs (seeds 110–119) were trained, and the frozen protocol ran at threshold 10. Result: 5 of 8 working backdoors caught (62.5%, 95% CI 31–86%); 0 of 10 clean false alarms; AUC 0.94 (bootstrap 95% CI 0.85–1.00) (`data/act_val.jsonl`, commit `5c28cb8`). All three criteria passed. Two weak backdoors (attack success under 0.5) were excluded and are reported, not hidden.

Honest reading: 8 working backdoors is a small sample, and the catch result is marginal. One backdoor either way moves it across the bar. The false-alarm and AUC results are stronger, and nothing was tuned after the data came in.

## 5. The boundary: blind detection fails on this data

What if the scanner does not know the trigger? It has to guess from a generic list of candidates.

**Offline check.** On the 10-pair data, removing the real trigger word from every pool (same rule for clean and backdoored adapters) dropped AUC from 0.860 to 0.780, and caught backdoors at threshold 10 from 7 of 10 to 3 of 10. Reproduce with `scripts/trigger_removal.py`. Caveat: near neighbours such as "Update:" stayed in the pool, so this is not a clean blind test.

**Trigger-blind pool experiment** (8 fresh pairs, seeds 120–127, 135-candidate union pool; `POOL_EXPERIMENT.md`, `data/act_pool.jsonl`):

| Candidate pool | AUC (95% bootstrap CI) |
|---|---|
| Original 32, real trigger inside | 0.83 (0.50–1.00) |
| Generic pool (112), real trigger added | 0.69 (0.34–0.98) |
| Generic strict pool (108), no word shared with any trigger | 0.48 (0.08–0.90) |

With the strict pool, the rare-token backdoors scored below their clean twins. That was 2 adapters, so it is too few to say more.

Read plainly: when the trigger, or a close neighbour, is in the pool, detection works. When the scanner must guess and its guess misses, detection falls to a coin flip. On this data, activation matching works as a trigger search, not as a blind detector.

## 6. Limits

- **Small sample.** 10 training pairs, 10 validation pairs (8 working backdoors), 8 pool pairs (6 working). Every interval here is wide.
- **Two trigger strings.** `cf` and `Important update:` were both known when the candidate pools were designed. Nothing here tests unseen triggers.
- **Fixed-output canaries.** The backdoor prints one fixed string, the easiest kind to catch. Open-ended harmful behaviour was not tested.
- **One base model.** Everything runs on Llama-3.2-3B-Instruct.
- **Assumed-clean real adapters.** The Hugging Face adapters used for false-alarm checks are believed clean but were not verified.
- **Reference is a LoRA, not a full fine-tune.** The original method uses a full fine-tune as the reference. That gap may matter.
- **Auditor prompts reduced** from 20 to 12 per candidate in the pool experiment, because the pool is about four times larger. That adds noise.

## 7. What would change the conclusion

- A generic or smarter pool scoring well above chance on **unseen** trigger strings would move this toward practical blind detection.
- More trigger types (multi-token, semantic), more base models, and larger samples would test generalisation.
- Weight features that survive config variation could revive the cheap, no-GPU path. None did on this data.
- Verified-clean adapters at scale would firm up the false-alarm rate.
- If blind-pool AUCs stay near 0.5 in replications, the boundary stands: without the trigger, neither weights nor activations gave the backdoor away.
