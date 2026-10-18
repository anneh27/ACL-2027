# Week 6 Task 1: taxonomy of MGSM semantic-recovery failures

Source: `results/mgsm_semantic_recovery_full.csv` (Week 5, 150 real-API trials, already collected — no new calls for this task). 25/150 trials (16.7%) have `task_semantic_sufficiency == False`. All 25 read by hand against the question, extraction, and Stage-2 output; categorized in `data/mgsm_semantic_failure_categories.json`, full table in `results/mgsm_semantic_failure_taxonomy.csv`.

## One thing to separate out before reading the counts

**3 of the 25 (`mgsm_057` ja, `mgsm_139` ja, `mgsm_214` es) are not extraction failures at all.** They're confirmed defects in the *official* Spanish/Japanese MGSM text itself (Week 5 Task 2's translation audit): Japanese states the delivery truck already has 33 boxes loaded while still asking for the maximum that can be added; Japanese drops "22 more than" from the gumball relation; Spanish says the pools held "the same amount" where English says "twice as much." In all three, the extraction faithfully reflects what the (defective) source text says, and Stage 2's answer is the mathematically correct answer *to that text*. Blaming these on the extraction step would mischarge a dataset problem as a semantic-recovery problem. The counts below are reported both ways.

## Category counts

| Category | Count (of 25) | % of all failures | % of genuine failures (n=22) |
|---|---|---|---|
| Target / question intent loss | 11 | 44% | 50% |
| Unit information | 6 | 24% | 27% |
| Relation loss | 4 | 16% | 18% |
| Aggregation relation | 1 | 4% | 5% |
| Entity scope | 0 | 0% | 0% |
| Other (source-text defect, not extraction) | 3 | 12% | — |

**Entity scope never occurs** — not one failure traces to confusing "one entity" with "all entities." Every genuine failure is about what the extraction says to do with the numbers, not which numbers or entities it picked out — consistent with surface fidelity (96-100% after Week 5's rescore) being much higher than task-sufficiency (82-84%).

## Representative examples

**Target / question intent loss (11, the dominant category):**
- `mgsm_001` (all 3 languages): "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts **in total**?" Extraction correctly derives white = 1, but no role marks that the answer is blue + white. Stage 2 reports 1 (white alone), gold is 3.
- `mgsm_240` (all 3 languages): "James has 6 more candies than Robert... John has twice as many as Robert. If John has 54, **how many more** does John have than James?" Extraction correctly derives James = 33, but nothing marks the target as the *difference* (54-33=21), not James's own count. Stage 2 reports 33, gold is 21.

**Unit information (6):**
- `mgsm_151` (English, Spanish): cost is tracked correctly in cents throughout, but the question asks "how many **dollars**." Stage 2 reports 500 (cents), gold is 5 (dollars).
- `mgsm_228` (all 3 languages): rate problem solved correctly in minutes (60), but the question asks "how many **hours**." Gold is 1.

**Relation loss (4):**
- `mgsm_007` (all 3 languages): "Windows forces a restart... Carla has to **restart the download from the beginning**." This is a procedural fact, not a quantity — no field in a `{quantity, role}` schema can hold it. Stage 2 assumes the download resumes from 40% instead of restarting a full second 200GB download; computes 120, gold is 160.
- `mgsm_086` (Spanish): "sonó **tres veces más** que la primera vez" is genuinely ambiguous (additive "+3" vs. multiplicative "×3"); extraction picked the additive reading, losing the multiplicative relation gold assumes. Stage 2 gets 14.5, gold is 22.

**Aggregation relation (1):**
- `mgsm_087` (English): extraction has all 5 right numbers (600, 10%, 5yr threshold, 5yr served, 3 more years) but no role states the 10%-of-initial raise applies *once per year, three times*, not once. Stage 2 applies a single 10% bump (7920), gold is 9360 (three separate +$60/month increases).

## What this means for Task 2

Every genuine failure category (target loss, unit info, relation loss, aggregation) is about **what Stage 2 should do with the extracted numbers**, not which numbers it extracted. This directly motivates the Critic's explicit checklist (target preserved? units preserved? relations preserved? aggregation/scope preserved?) over a generic "improve your extraction" prompt — the plan's own suggested criteria map almost one-to-one onto the four genuine failure categories found here.
