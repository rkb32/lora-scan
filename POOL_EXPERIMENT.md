# Trigger-blind pool experiment (exploratory)

Written before any of its data exists. **Exploratory:** it reports what happens, it does not pass or fail. Anything it suggests (a new pool, a new threshold) must be validated later on fresh seeds, the way `VALIDATION.md` does.

## Why

Every earlier run handed the scanner the real trigger inside its 32-candidate pool. An offline check on the 10-pair data (2026-10-10, verified by an independent worker) showed what happens without it: removing the real trigger from the pool dropped AUC from 0.86 to 0.78 and catches at threshold 10 from 7/10 to 3/10. That check had two flaws this experiment fixes:

- The 30 hand-picked decoys included near neighbours of the trigger ("Update:" next to "Important update:"), so the "trigger removed" number mixes trigger-blind detection with lexical leakage.
- 3 of the 10 backdoors used the rare-token trigger `cf`, too few to say anything about that trigger type.

A real scanner never knows the trigger. It has to try a generic list and hope the trigger, or something close to it, is on it.

## Design

- **One audit per adapter** with the union of all pools (135 candidates, `scripts/pools.py`, pool seed 0). Each pool's score is computed offline from its own subset, so all pools share the same reference model and differ only in which candidates they look at.
- **Headline arm: `generic_blind_strict`** (no candidate shares a word with a real trigger). Added after the design review: `generic_blind` kept "Update:" and "Important:", which are near copies of the trigger, so its result would only measure that leak.
- **Limit found in review:** the question is whether a generic list catches an unknown trigger, but the test has only 2 trigger strings, both known when the pool was written. This run can show the method works on these two strings. It cannot show it generalises. A fresh set of unseen trigger strings must be sealed before any generalisation claim.
- **Pools (arms):**
  - `original32`: the old pool, for comparison with earlier runs.
  - `generic_plus_trigger` (112): 50 generic message openers plus 60 seeded random two-letter tokens, plus both real triggers. Best case for a large pool.
  - `generic_blind` (110): the same generic pool with no real trigger. This is the realistic case. Near neighbours such as "Update:" and "Important:" stay in, because a scanner's generic list would plausibly contain them. The opener list was written by someone who knows the triggers; only exact copies were removed. That is a limitation.
- **Data:**
  - **Pairs mode:** 8 fresh matched pairs, seeds 120-127, disjoint from every earlier run. Triggers alternate (even seed rare-token `cf`, odd seed contextual), so each type gets 4 pairs. Other config fields come from `sample_config`.
  - **Hub mode:** 32 new real Hub adapters (`data/hub_ids_2.txt`, seed 1, excluding the 16 already audited), same base model, assumed clean.
- **Unchanged from earlier runs:** `act_match.audit` (reference LoRA r=16, 600 auditor prompts, one pass), `adapter_score`, base model, and the ASR measurement (20 held-out prompts). Changed: 12 auditor prompts per candidate instead of 20, because the pool is about 4 times larger.

## Analysis (fixed now, `scripts/act_pool_analyze.py`)

For each pool:
- AUC, working backdoors vs clean twins. A working backdoor has ASR >= 0.5, as in VALIDATION.md.
- **Threshold set from real adapters only:** the 90th percentile of that pool's real-Hub scores. Report how many backdoored and trained-clean adapters cross it. No backdoored adapter influences any threshold.
- The same numbers split by trigger type.
- For pools holding the real trigger, its rank among the candidates.

## What would change the plan

- **`generic_blind` catches close to `generic_plus_trigger`:** near neighbours carry the signal, and a generic list is enough. The next step is a pre-registered validation of `generic_blind` on new seeds.
- **`generic_blind` collapses toward chance:** the method needs the trigger, or a very close neighbour, in the pool. Detection becomes a search problem over possible triggers, and the product framing must change. Either outcome gets reported.
- **Large pools raise clean scores a lot:** the threshold must always be calibrated per pool on real adapters, never reused across pools.

## Known limits

8 pairs and 32 real adapters give wide intervals. Fixed-output canary backdoors, one 3B base model, one LoRA reference. The real adapters are assumed clean, not verified.
