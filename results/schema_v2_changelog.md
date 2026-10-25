# Schema v2 changelog (Week 7 Task 3)

Real API run, not a projection: 27 items, 2026-10-25, model `gpt-5.6-sol`, 0 job errors.
Supersedes an earlier run on a narrower 25-item sample (5 known regressions + 20 arbitrary
negative controls); this run uses the population the plan's comparison is actually about.

## What changed and why

v1's Stage-1 extraction was a flat JSON array of `{quantity, role}` entries. There was no
dedicated place to express "the unknown quantity to be solved for" — the Week 6 Critic+Repair
baseline worked around this by appending an entry like `{"quantity": 0, "role": "target: total
weight..."}`. That placeholder looks like a real data point to Stage 2 and corrupted an
otherwise-correct run (`mgsm_207`, see `results/week6_task4_error_analysis_summary.md`).

v2 (`schema_v2.json`, `src/mgsm_schema_v2.py`) separates "things that were measured"
(`quantities`, unchanged in spirit from v1) from "the thing to be computed" (`target`, a
dedicated object with `entity`/`operation`/`output_unit`, no numeric field at all). `validate_v2()`
actively rejects a `target` with a smuggled `quantity` field and any `quantities` entry whose
`role` text reads like a target placeholder — it refuses to let the v1 anti-pattern recur, not
just avoids reproducing it by convention. v1's prompts, parsing, and all existing result files
remain untouched; this is a new module and new output files.

## Item set: the 25 Condition-B failures, plus the 2 Condition-D-only regressions

The plan's ask for Task 3 is a comparison across **automatic Critic+Repair, generic reflection,
and manual Oracle Repair** — that comparison is about the 25 items where the original
Stage1→Stage2 pipeline (Condition B) failed, since that's the population Task 2 manually
repaired by hand (`results/manual_semantic_repair_25.csv` / `_resolved.csv`). This run's item
set is exactly that 25, unioned with the 2 items that were *only* wrong under the automatic
Condition-D Critic+Repair (`mgsm_trilingual_039` ja, `mgsm_trilingual_207` en — regressions
introduced by Repair itself, not original Stage-1 failures, so Task 2 never touched them) —
27 items total. `src/mgsm_schema_v2_pipeline.py` now takes `--items-csv` for this; see its
docstring.

## Four-way comparison, over the 25 original Condition-B failures

| Method | Correct | Rate |
|---|---|---|
| v1 generic reflection (Condition C) | 7/25 | 28% |
| v1 automatic Critic+Repair (Condition D, schema v1) | 22/25 | 88% |
| **v2 automatic Critic+Repair (schema v2, this fix)** | **22/25** | **88%** |
| Manual Oracle Repair (Task 2, human-authored) | 21/21 attempted | 100% of attempted |

Schema v2 resolves the exact same 22/25 as v1 did on this population — the 3 it doesn't fix
(`mgsm_086` es, `mgsm_139` ja, `mgsm_214` es) are source-text/translation defects, not
extraction-schema bugs (already documented in `results/experiment_config.md`'s known-issues
table), so neither schema version was expected to fix them. The human annotator independently
declined to write a repair for all 4 items they judged to be source-text/gold-label defects
(`mgsm_057` ja, `mgsm_086` es, `mgsm_139` ja, `mgsm_214` es) — one more than the pipeline's 3,
since both the v1 and v2 automatic pipelines happen to still land on the correct gold answer for
`mgsm_057` despite its translation issue (the false "33 boxes already loaded" premise), even
though the annotator judged it unrepairable in principle. All 21 items the human *did* attempt a
repair for were recovered (100%). This is convergent evidence from two independent methods (a
schema redesign and an independent human annotator) that `086`/`139`/`214` are a dataset
problem, not a pipeline one.

Full per-item table, including which rows have a manual-oracle value vs. why not (either "no
`stage1_repaired_json` provided" or "outside Task 2's population" for the 2 Condition-D-only
items): `results/repair_comparison_v2.csv`.

## The 2 Condition-D-only regressions (outside Task 2's population)

| Item | v1 Critic+Repair (D) | v2 Critic+Repair |
|---|---|---|
| `mgsm_trilingual_207` (en) — the plan's own named example | wrong | **correct** |
| `mgsm_trilingual_039` (ja) | wrong | wrong |

`mgsm_207` — the regression directly caused by v1's `quantity=0` placeholder — is fixed by
schema v2. `mgsm_039` is a separate, known unit/gold-label mismatch (km vs. miles, see
`results/experiment_config.md`) unrelated to the placeholder bug, so v2 was never expected to
fix it.

## Token cost

v2's full pipeline (fresh extraction + critic + conditional repair + Stage 2) averaged
**1713 tokens/item** across the 25-item population (1773 across all 27). **No v1-side token
comparison is possible**: neither v1 Condition B's nor Condition D's logs ever captured
`prompt_tokens`/`completion_tokens` — the same kind of reproducibility gap already flagged for
`model_returned` in `results/experiment_config.md`. Not fixed retroactively here (would require
re-running v1); flagged as a known gap. Manual Oracle Repair's "token cost" isn't a comparable
unit at all (human annotation time, not API tokens) — not fabricated into this table.

## What this does not touch

`results/mgsm_condition_b_reflection*.csv`, `results/mgsm_condition_c_critic_repair*.csv`, and
`results/mgsm_condition_comparison_*.csv` (all v1) are unmodified. `results/
manual_semantic_repair_25.csv` and `_resolved.csv` (Task 2's own annotation and its Stage-2
rerun) are read here, not written to. The `_smoke` tag files (`mgsm_schema_v2_pipeline_smoke.csv`,
`repair_comparison_v2_smoke.csv`) are a separate, still-valid smoke test from before this run and
are left as-is.
