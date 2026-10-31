# Week of Oct 26–Nov 1: research summary

All real API calls, model `gpt-5.6-sol`, no decoding parameters set (the model rejects
`temperature=0`; same note as every prior week). **Task 2** (blinded independent semantic-
completeness review) and **Task 4** (replication on a different model family) are out of scope
for this document — see "What's not here" at the end.

## Task 1: Schema x Critic ablation — does the gain come from the schema or the Critic?

Four conditions, same 150 (question, language) pairs (50 MGSM items x English/Spanish/Japanese),
assembled from real-API results (two already-collected, two run fresh this week):

| Condition | Correct | Accuracy | Calls/item | Avg tokens/item |
|---|---|---|---|---|
| v1, no repair | 125/150 | 83.3% | 2 | not captured (known gap, see `experiment_config.md`) |
| v1 + Critic | 145/150 | 96.7% | 3–4 | not captured (known gap) |
| v2, no repair | 144/150 | 96.0% | 2 | 1023 |
| v2 + Critic | 143/150 | 95.3% | 2–4 | 1721 |

**Isolating each lever** (recovered = base's wrong items fixed; regressed = base's right items broken):

| Comparison | Recovered | Regressed | Net |
|---|---|---|---|
| v1 → v1+Critic (Critic's effect on schema v1) | 22/25 | 2/125 | **+20** |
| v2 → v2+Critic (Critic's effect on schema v2) | 1/6 | 2/144 | **−1** |
| v1 → v2, no Critic in either (schema's effect alone) | 22/25 | 3/125 | **+19** |
| v1+Critic → v2+Critic (schema's effect with Critic in both) | 1/5 | 3/145 | **−2** |

**Finding, stated plainly**: schema v2 alone — with no Critic, no repair, one extraction call and
one reasoning call per item — recovers 22 of the same 25 items that Critic+Repair on schema v1
recovers, at roughly half the API cost (2 calls vs. 3–4) and fewer tokens. Adding Critic+Repair on
top of schema v2 does not help further: it recovers only 1 more item while introducing 2 new
regressions, a net loss. **Almost all of the measured gain comes from the schema redesign itself,
not from the Critic/Repair step.** This directly answers the plan's own decision threshold: *"If
ablation shows that gains come mainly from Schema, focus on intermediate representations."* — on
this evidence, it does.

One caveat stated plainly: this is a single run per condition (the model is not temperature-
controlled and is known from Week 6 to vary call-to-call on individual items), so the exact
item-level recovery/regression sets should be treated as one sample, not a fully converged
estimate — the aggregate pattern (schema >> Critic) is large enough (19–20 points vs. ±1–2) that
it's unlikely to be noise, but a repeated-sampling check (as already done for Week 6's 5
remaining failures) would strengthen this before writing it into the paper.

Full per-question table: `results/mgsm_ablation_per_question.csv`. Summary: `results/
mgsm_ablation_summary.csv`. Recovery/regression breakdown: `results/
mgsm_ablation_recovery_regression.csv`. Script: `src/mgsm_ablation_four_condition.py` (pure
aggregation of already-collected files, re-run with `python src/mgsm_ablation_four_condition.py`,
no new API calls) plus `src/mgsm_schema_v2_pipeline.py --full --no-critic` and `--full` (the two
new conditions this produced, real API calls this week).

## Task 3 (P1 reuse): does Critic+Repair generalize beyond MGSM?

### 3a. Week 3's 20 harder logic items — first real-API run

These 20 items (negation/De Morgan, conditionals/biconditionals, quantifiers, deliberate
fallacious-reasoning-form distractors) had never been run through a live model before this week —
the Week 3/4 version was Claude-as-subject inside a conversation, not a verifiable API call.
Three conditions, `src/logic_critic_repair_pilot.py`:

| Condition | Correct | Calls |
|---|---|---|
| A. Direct reasoning | 20/20 | 20 |
| B. Extract (formalize) → isolated reason | 20/20 | 40 |
| C. + Critic/Repair | 20/20 | 62 |

**Finding**: a clean ceiling across all three conditions with the real model — unlike the earlier
Claude-as-subject pilot, which the Week 3 README reports found the harder logic batch broke the
ceiling (95%→80% on Condition B). That drop does not replicate here. Either the real model is
simply stronger on this item set than the Claude-as-subject proxy was, or the harder-logic set
itself wasn't hard enough to separate conditions at this model's capability level — this run
can't distinguish those, and a genuinely harder item set would be needed to find where this
particular Critic+Repair design actually breaks. Full data: `results/logic_pilot_real_api.csv`.

### 3b. Week 4 Cherokee pilot — added Condition E

Conditions A/B/C/D (`results/cherokee_pilot_v2_full.csv`) are untouched — this is a new column,
not a rewrite. Condition E: Cherokee text → direct formal-logic extraction → Critic/Repair →
isolated reasoning (no back-translation step at all, unlike Condition C). `src/
cherokee_pilot_condition_e.py`:

| Condition | Correct | Note |
|---|---|---|
| C (Week 4: Cherokee → English back-translation → answer) | 15/20 | existing result |
| **E (this week: Cherokee → formal extraction → Critic/Repair → answer)** | **16/20** | new |

C→E: recovered 2 of C's 5 wrong items, regressed 1 of C's 15 right items (net +1). A small,
real improvement from skipping the back-translation-to-English-prose step in favor of direct
structured extraction — consistent with this project's running finding (Weeks 5–6) that
information loss tends to concentrate at translation/representation steps, not reasoning.

**Required caveat, per this week's plan**: the Cherokee text itself (`data/
week1_cherokee.json`) was model-generated and has never been reviewed by a native speaker. This
result says Condition E's *pipeline design* beats Condition C's on this Cherokee text, as
written — it does not and cannot say anything about translation distortion relative to true
Cherokee, since there's no verified ground truth to compare against. Treat this as a pilot
finding on synthetic Cherokee, not a validated Cherokee-language result. Full data: `results/
cherokee_pilot_condition_e.csv`.

### 3c. Week 5's 5 spatial-reasoning items — does the Critic catch missing axioms?

None of the 5 items' natural-language premises ever state a transitivity or inverse-relation
axiom explicitly (verified directly from `data/week1_cleaned.json`: e.g. spat_01 says only "the
apple is left of the book" and "the book is left of the cup," never "left-of is transitive").
The axiom exists only in the gold formal representation, added as background world-knowledge —
exactly the Week 5 Task 5 finding (`results/week5_summary.md`), now tested as a live Critic
capability rather than a hand-fix. Ran the same extract→reason and +Critic/Repair conditions
(`src/logic_critic_repair_pilot.py --data data/week1_cleaned.json --filter-type spatial`):

| Condition | Correct |
|---|---|
| A. Direct | 5/5 |
| B. Extract → isolated reason | 4/5 |
| C. + Critic/Repair | 4/5 (recovered 1, regressed 1) |

**The Critic did identify the missing axiom in 4 of 5 items** — explicitly naming "transitivity"
or the inverse-relation equivalence as missing, e.g. for `spat_04`: *"The extraction omits the
spatial semantics connecting the predicates: east_of is transitive, and west_of(x,y) is the
inverse of east_of(y,x). Without these axioms, the query cannot be assessed using only the
formal version."* — and correctly proposed the fix. For `spat_05` the Critic judged the
extraction (which has the identical gap) *sufficient* — an inconsistency, not a principled
exception, worth noting as a real limitation of the Critic's reliability rather than glossed
over. One more finding worth flagging on its own: `spat_04`'s repaired extraction was logically
complete and correct (both needed axioms present), yet isolated Stage 2 still answered
`unknown` instead of the gold `contradicted` — a genuine multi-step reasoning failure given a
verified-sufficient formal representation, not a semantic-completeness failure. This is the
first clean example in this project of a failure that survives a fully adequate extraction,
worth keeping as a concrete counter-example to "it's never reasoning" rather than treating every
remaining error as extraction-attributable by default. Full data: `results/
spatial_critic_pilot.csv`.

This also cleanly distinguishes the plan's two failure categories on real data: these 5 items
are **under-specification in the problem itself** (the NL text never states the axiom, so there
is nothing for a faithful extraction to have dropped) rather than **extraction omission** (the
NL states something and the extraction drops it) — the Critic's job here is closer to supplying
missing standard background knowledge than catching an extraction mistake, and it does that
inconsistently (4/5).

## What's not here

- **Task 2** (blinded independent semantic-completeness review): scaffolding only —
  `results/task2_blinded_sample.csv` (45 items: the 25 original Condition-B failures + 20
  originally-correct, blind order, no correctness label or gold answer shown),
  `results/task2_annotation_guidelines.md` (the protocol), and `src/
  task2_compute_detection_stats.py` (tested against fabricated labels, ready to run once real
  ones exist). The actual blinded judgment calls are explicitly a human task, same reasoning as
  prior weeks' manual annotation split: if the same process that builds and grades the pipeline
  also supplies the "independent" completeness label, the label isn't independent.
- **Task 4** (replicate on a different model family, e.g. Qwen): out of scope per this week's
  instruction ("complete all the tasks except for task 4, which uses a different model") — no
  Qwen (or other non-OpenAI) credentials exist in this environment, and this wasn't requested.
