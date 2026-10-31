# Task 2 annotation guidelines: independent semantic-completeness review

**Who fills this in**: the human annotator (you), by hand, on `results/task2_blinded_sample.csv`.
This document is the protocol; it is not the annotation itself. Nothing in this file or the
sample CSV tells you which items the pipeline got right or wrong — that's the point of a blinded
review. Do not open `results/task2_blinded_key_private.csv` until you're done labeling; it has
the answer key and would un-blind you.

## Why blinded

The plan's core instruction: **do not equate "correct final answer" with "complete semantic
representation."** If you knew in advance which items Stage 2 got wrong, you'd be tempted to
rate those extractions as "incomplete" just because the pipeline failed — that's circular, and
it's exactly the failure mode this independent check exists to catch (an extraction can be
complete and Stage 2 still makes an arithmetic slip; an extraction can be missing something and
Stage 2 still gets lucky). Judge each extraction only against the original question text, with
no knowledge of the outcome.

## What to do, per row

For each `blind_id` in `results/task2_blinded_sample.csv`, read `original_question` and
`stage1_extraction_v1` (the Stage-1 extraction produced for that question — a JSON array of
`{"quantity": ..., "role": ...}` entries, schema v1). Judge whether the extraction preserves
everything a reader with ONLY the JSON (no access to the original question) would need to solve
the problem correctly. Fill in each column:

| Column | What to judge | Values |
|---|---|---|
| `completeness_goal_target` | Does the extraction make clear *what* the final answer refers to — a total, a difference, one specific entity's value, after how many steps/repetitions? | `complete` / `incomplete` / `unclear` |
| `completeness_entities` | Are all relevant entities named distinctly (not conflated — e.g. "John's apples" vs. "Mary's apples" kept separate)? | `complete` / `incomplete` / `unclear` |
| `completeness_units` | Are units preserved, including any conversion the question implies (cents vs. dollars, minutes vs. hours)? | `complete` / `incomplete` / `unclear` |
| `completeness_relations` | Are ratios, multipliers, and differences between quantities preserved as stated? | `complete` / `incomplete` / `unclear` |
| `completeness_quantifiers_scope` | Is aggregation scope preserved (sum vs. one item, "after N more years" vs. "now", "all three" vs. "two of them")? | `complete` / `incomplete` / `unclear` |
| `completeness_ambiguity` | Could the extraction alone support more than one reasonable interpretation that the original question doesn't? | `none` / `some` / `severe` |
| `overall_judgment` | Your overall call: would a reader with only this JSON reliably reach the same answer a careful reader of the original question would? | `complete` / `incomplete` |
| `annotator_notes` | Free text — anything that doesn't fit the above, especially if you suspect the *question itself* (not the extraction) is ambiguous or has a translation defect. | free text |

Use `unclear` rather than guessing when the extraction is genuinely ambiguous to you as a
reader — that's itself useful signal, not a failure to annotate.

## Representative cases

Before labeling the full 45, work through these 5 first and sanity-check your own judgments
against the reasoning given (don't change your answers to match — these are just calibration
examples, not an answer key):

1. **Missing target/goal**: an extraction lists quantities for "Toulouse," "Charleston," and
   "Seattle" sheep but the question asks for their *combined* total — if nothing in the
   extraction says "sum all three," that's `incomplete` on `completeness_goal_target`.
2. **Unit mismatch**: quantities given in cents, but the question asks "how many dollars" — if
   the extraction doesn't flag the needed conversion, that's `incomplete` on
   `completeness_units`.
3. **Relation preserved correctly**: "twice as many as X" captured as a multiplier role tied
   explicitly to X — `complete` on `completeness_relations`.
4. **Scope under-specified**: "the remaining pieces after the first harvest" vs. "the total
   crop" — if the extraction doesn't distinguish which one the final answer should refer to,
   that's `incomplete` on `completeness_quantifiers_scope`.
5. **Source ambiguity, not extraction failure**: if the *original question itself* is genuinely
   ambiguous (e.g. a phrase that could mean either an additive or multiplicative relation), and
   the extraction faithfully mirrors that ambiguity rather than resolving it one way, judge the
   extraction `complete` (it accurately reflects an ambiguous source) and note the ambiguity in
   `annotator_notes` — don't penalize the extraction for a defect in the question.

## After labeling

Once every row has all columns filled in, save the file and let me know — I'll run
`src/task2_compute_detection_stats.py` to compute error-category statistics and the existing
Critic's detection precision/recall against your independent labels (comparing your
`overall_judgment` to the v1 Critic's own `Sufficient: Yes/No` call on the same items, which is
stored separately in `results/task2_blinded_key_private.csv` so it couldn't bias you while
labeling).
