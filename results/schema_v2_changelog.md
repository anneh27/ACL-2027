# Schema v2 changelog (Week 7 Task 3)

Real API run, not a projection: 25 items, 2026-10-23, model `gpt-5.6-sol`, 0 job errors.

## What changed and why

v1's Stage-1 extraction was a flat JSON array of `{quantity, role}` entries. There was no
dedicated place to express "the unknown quantity to be solved for" — the Week 6 Critic+Repair
baseline worked around this by appending an entry like `{"quantity": 0, "role": "target: total
weight..."}`. That placeholder looks like a real data point to Stage 2 and corrupted an
otherwise-correct run (`mgsm_207`, see `results/week6_task4_error_analysis_summary.md`).

v2 (`schema_v2.json`, `src/mgsm_schema_v2.py`) separates "things that were measured"
(`quantities`, unchanged in spirit from v1) from "the thing to be computed" (`target`, a
dedicated object with `entity`/`operation`/`output_unit`, no numeric field at all). There is no
longer any numeric placeholder for Stage 2 to mistake for a real value. `validate_v2()` rejects
a `target` with a smuggled `quantity` field and rejects any `quantities` entry whose `role`
text reads like a target placeholder (`target:`, `unknown`, `solve for`) — i.e. it actively
refuses to let the v1 anti-pattern recur, not just avoids reproducing it by convention.

v1's prompts, parsing, and all existing result files are untouched — this is a new module
(`src/mgsm_schema_v2.py`, `src/mgsm_schema_v2_pipeline.py`) and new output files, not an edit to
`src/mgsm_semantic_recovery_trilingual.py` or `src/mgsm_semantic_critic_repair.py`.

## What was actually measured

Ran `src/mgsm_schema_v2_pipeline.py` fresh (new Stage-1 extraction under `EXTRACT_SYSTEM_V2`,
not reusing any v1 extraction) on 25 (question, language) pairs:

- **5 known v1 Condition-D (Critic+Repair) regressions**, re-run under v2: `mgsm_trilingual_207`
  (en) — **now correct**, the plan's own named example and the one case that matches the
  `quantity=0` placeholder failure mode this schema change targets. The other four
  (`mgsm_trilingual_039` ja, `086` es, `139` ja, `214` es) are **still wrong** — consistent with
  `results/experiment_config.md`'s known-issues table, which already attributes these to
  source-text/translation defects (a false premise, a dropped relation, an ambiguous phrase, a
  gold-label unit mismatch), not an extraction-schema problem. A schema change to Stage 1 was
  never expected to fix a defect in the original question text or gold label.
- **20 negative controls** (items v1 Condition D already got correct, excluding the above):
  **20/20 stayed correct** under v2 on this sample. No new regression observed.
- **Repair rate**: 4/25 items triggered the critic → repair path (vs. the original extraction
  being judged sufficient as-is).
- **Token cost**: v2's full pipeline (fresh extraction + critic + conditional repair + Stage 2)
  averaged **1743 tokens/item** across all 25 (1596 on the controls, 2335 on the 5 regressions,
  which needed more repair calls). **No v1-side token comparison is possible**: neither v1
  Condition B's (`mgsm_condition_b_reflection_log.jsonl`) nor Condition D's
  (`mgsm_condition_c_critic_repair_log.jsonl`) logs ever captured `prompt_tokens`/
  `completion_tokens` — the same kind of reproducibility gap already flagged for
  `model_returned` in `results/experiment_config.md`. Not fixed retroactively here (would
  require re-running v1); flagged as a known gap.
- **Manual Oracle Repair** (Task 2): pending. `results/repair_comparison_v2.csv` has a
  `manual_oracle_repair_correct` column reserved for it, left blank — not filled in here, since
  that annotation is explicitly Task 2's manual work, not this script's.

Full per-item table: `results/repair_comparison_v2.csv`. Raw run data:
`results/mgsm_schema_v2_pipeline.csv` / `_log.jsonl`.

## Scope

This run covers 25 of 150 (question, language) pairs — the 5 known v1 regressions plus 20
controls, not the full set. Re-run with `--full` for a complete 150-item head-to-head if a
fully comparable success/regression-rate table against v1 becomes necessary; that would cost
roughly 6x this run's API calls, so it wasn't done by default.

## What this does not touch

`results/mgsm_condition_b_reflection*.csv`, `results/mgsm_condition_c_critic_repair*.csv`, and
`results/mgsm_condition_comparison_*.csv` (all v1) are unmodified — this is a new version
(`_v2` / `schema_v2_pipeline` filenames throughout), not a rewrite of v1's history.
