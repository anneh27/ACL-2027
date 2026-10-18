# Week 6 Task 4: error analysis of the repair module

5 of 150 trials are still wrong under Condition C (critic+repair): 2 unrecovered baseline failures that repair didn't fix, 1 unrecovered failure the critic only half-diagnosed, 2 regressions where repair broke a baseline-correct answer. Full detail: `results/mgsm_repair_error_analysis.csv`, judgment in `data/mgsm_repair_error_categories.json`.

## Bottleneck distribution

| Bottleneck | Count | What it means |
|---|---|---|
| Not fixable by this pipeline (confirmed source-text defect) | 2 | `mgsm_139` (ja drops "22 more than"), `mgsm_214` (es says "same amount" instead of "twice") — both already confirmed in Week 5 Task 2. The extraction correctly reflects a defective official translation; nothing the critic sees contains the missing information, so no amount of better detection or repair can recover it. |
| Not a real failure (gold label itself is defective) | 1 | `mgsm_039` — the critic correctly caught a missing unit conversion, the repair correctly added it, and Stage 2 correctly computed the mile→km conversion the Japanese question actually asks for. It scores "wrong" only because the official gold answer (18) is itself unconverted miles, not because anything in the pipeline failed. |
| Detection failure | 1 | `mgsm_086` — the critic's own checklist explicitly includes "are quantitative relations preserved?" and "is there ambiguity that could lead to multiple interpretations?". It caught the missing-target issue but said nothing about either of those, even though the real cause of the wrong answer is a Spanish phrase ("tres veces mas") genuinely ambiguous between additive and multiplicative readings — exactly what those two checklist items exist to catch. |
| Repair introduced new distortion | 1 | `mgsm_207` — a complete, already-correct extraction was flagged insufficient on a technicality, and the repair's fix broke something unrelated: Stage 2 inverted a weight-multiplier direction after the repair added a `{"quantity": 0, "role": "target: ..."}` placeholder entry onto an otherwise-unchanged, correct extraction. |
| Reasoning failure (correct repair, Stage 2 still wrong) | **0** | Not a single case. Every remaining error traces to the source text, the gold label, detection, or the repair mechanism — never to Stage 2 reasoning validly from a correct representation. |

## The one clearly actionable finding

**The repair format has a real, fixable flaw.** `REPAIR_SYSTEM` asks for a `{"quantity": <number>, "role": "..."}` entry even when what's actually needed is a marker that something is *unknown and must be solved for* — the model fills the number with a placeholder (seen here as `0` in both `mgsm_207` and `mgsm_214`). In `mgsm_207`, where the rest of the extraction was untouched and already correct, adding that single zero-valued "target" entry is the only thing that changed between a correct run and a wrong one. A JSON schema that can't distinguish "this is a known quantity of value 0" from "this value is still unknown" is asking Stage 2 to guess, and sometimes it guesses wrong. Fix: have the repair format target markers without a numeric placeholder (e.g. `{"role": "target: total weight of all candy, in grams", "quantity": null}` or a dedicated `"target"` field separate from `"quantity"` entries entirely), so there's no fake data point for Stage 2 to misread.

## What this says about where the real bottleneck is

Of the 5 remaining failures, only 2 (`086`, `207`) are genuinely attributable to this pipeline's own detection or repair mechanics — and even those are specific, nameable, fixable issues (an incompletely-applied checklist; a placeholder-value format flaw), not a vague "the critic isn't good enough." The other 3 are either unfixable upstream data problems or not real failures at all. Combined with zero pure-reasoning failures across all 5, the evidence points at **detection and repair format as the remaining (small, specific) bottlenecks** — not reasoning, and not "the critic needs to be smarter in general."
