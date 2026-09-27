# Week 5 summary

*Focus: separate translation/semantic-recovery errors from reasoning errors. Model: `gpt-5.6-sol`, real API calls throughout -- 346 new calls this week (300 two-stage semantic recovery + 30 smoke test + 15 spatial-axiom verification + 1 reasoning-tokens check). Task 4's Cherokee A/B/C/D data reuses the 100 real-API calls already made in Week 4 (`run_cherokee_pilot_v2.py`); no new calls were needed there, only re-analysis. All claims below are checked against saved raw outputs; nothing is asserted from memory.*

## The one finding that ties every task together

Every task this week converged on the same shape of result: **once you look past the pass/fail label to the actual mechanism, essentially every "error" traces to a missing piece of the task specification, not to weak reasoning.** Three independent examples, three different pipelines:

- **MGSM two-stage probe (Task 1):** Stage 1 extracts every quantity correctly; Stage 2 (isolated, verified — Task 3) reasons validly from exactly what it's given but was never told *what the question asks for* ("in total," "how many hours," a difference vs. an absolute value). It reliably computes a real, correct intermediate quantity and stops one step short.
- **Cherokee pilot (Task 4):** zero of 20 items show a pure reasoning failure. Every error is either a comprehension/translation problem (5 items) or a dataset gap (1 item).
- **Spatial formal representations (Task 5):** `spat_04`'s oracle-symbolic condition failed 4/4 times in Week 4 not because the model reasoned badly, but because the formal premises never stated a transitivity axiom the gold label silently assumed. Turns out the same gap existed in 3 more items that had been graded "correct" only because the model filled the gap with real-world knowledge the formal premises didn't license.

None of these are reasoning failures. All three are cases where the evaluation design asked for more than it gave the model to work with.

## Task-by-task

**Task 1 — semantic-recovery scoring, ported to real API data and re-scored.** Ran the two-stage extract-then-reason probe on all 50 x 3 = 150 trilingual MGSM trials (300 calls). Found and fixed a bug in the scoring script itself: Spanish's comma-decimal convention (`$16,50` = 16.5) was being misread the same way Week 3's Swahili clock-time bug was. Before/after: fidelity 88-96% -> 96-100% after the locale fix and crediting valid abstractions (e.g. `3/4` correctly compressed to `0.75`, matching the plan's own `mgsm_023` example). Task-semantic-sufficiency (does Stage 2 solve it from the extraction alone) sits at 82-84% across all three languages — uniform, confirming the gap is a schema-design issue, not a language effect. `results/mgsm_semantic_recovery_rescore_comparison.csv`.

**Task 2 — MGSM translation audit, all 50 items x Spanish + Japanese.** Automated numeric diff (100 pairs) flagged 18; every one read by hand. Fixed two more locale bugs found in the process (Spanish thousands-as-space; the same comma-decimal issue). Result: **2 confirmed real translation defects** out of 100 pairs (`mgsm_057`: Japanese states the truck already has 33 boxes, contradicting the question; `mgsm_139`: Japanese drops "22 more than," already known from Week 4). The other 16 flags are systematic Japanese-vs-English number-rendering conventions (ordinals, fractions, implicit "a/an," month-numbering) that this audit initially mis-flagged as errors and I subsequently verified are not. `results/mgsm_translation_number_audit.csv`, `data/mgsm_translation_audit_verified.json`.

**Task 3 — Stage 1 -> Stage 2 isolation, verified two ways.** An automated assertion runs on every one of the 150 semantic-recovery calls and raised on none; a manual read of a full example confirms the Stage-2 payload contains nothing but the bare extraction. `results/week5_isolation_verification.md`.

**Task 4 — Cherokee A/B/C/D, refined taxonomy.** Split "translation" into `language_semantic_recovery` (both direct reading and back-translation fail: 2 items) vs. `explicit_back_translation_failure` (direct reading succeeds, only the explicit back-translation step fails: 3 items) per the plan's own B-correct/C-wrong rule. 1 item is a dataset/formal-representation issue (`spat_04`, see Task 5). 0 items are reasoning failures. `results/cherokee_attribution_table.csv`.

**Task 5 — spatial axioms, all 5 items, fix verified with real calls.** Audited every spatial item's formal premises, not just `spat_04`. Found the same missing-axiom problem in `spat_01`, `spat_02`, and `spat_05` — all three were "correct" before only because the model brought in real-world knowledge the premises didn't formally license. Added the missing transitivity axioms (all four items) and asymmetry axioms (`spat_04`, `spat_05`). Verified with 15 fresh real API calls (3 samples x 5 items): **all 5 items now correct in all 15 samples**, including `spat_04` which had failed 4/4 times before the fix. `data/week1_cleaned.json`, `results/spatial_axiom_verification.csv`.

**Task 6 — reasoning_tokens=0, audited.** Confirmed via the SDK's own type definition (`Optional[int] = None`, so a missing field would show as `None`, never `0`), the absence of any NaN in 150 logged rows, and a raw HTTP call bypassing the SDK entirely: the field is always present in the API's own JSON. `0` is a genuine reported value, not a default. No code fix needed. `results/week5_reasoning_tokens_audit.md`.

## What still needs a human

- Every judgment field this week (translation-audit verdicts, Cherokee attribution notes, back-translation fidelity ratings) is mine, flagged as such in the relevant JSON/CSV, and needs your review — same standing item as Week 4.
- No native Cherokee speaker has reviewed any translation.
- `gpt-5.6-sol` still rejects `temperature=0`; the plan's reproducibility requirement remains unmet with this model.
- The `mgsm_057` finding (Japanese states a false premise) and the 2 Spanish formatting quirks are significant enough that the official MGSM Spanish/Japanese configs likely deserve a report upstream to the dataset maintainers — your call whether that's in scope.
