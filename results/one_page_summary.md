# Week 3 One-Page Experimental Summary

*Low-Resource Languages and Logical Reasoning Project — Sept 2026*

## What did this week's experiments actually find?

Four things. **(1)** The Week 2 semantic-recovery "failure" rate (60%, 9/15) was almost entirely a scoring artifact — after fixing two concrete bugs (comparing extracted digits against the wrong language's gold set; treating clock-time boilerplate like the "00" in "1:00" as a real quantity) and manually auditing what remained, true recovery failure is 0/15; every flagged case had 100% correct downstream reasoning. **(2)** A rigorous item-by-item review of the 20 Cherokee translations (checking negation/conditional/quantifier-scope/entity/predicate preservation against a blind reader's independent parse) rates 4 items High confidence, 12 Medium, 4 Low — with one item's rating explicitly revised *upward* from my own pre-grading self-assessment (`neg_04`'s double-negation structure survived translation better than expected). **(3)** Redefining Condition C as "Cherokee → English → answer" (rather than Week 2's "Cherokee → formal parse") gives a genuinely different, informative number: 85% vs. Condition B's 95% on the original set — explicit translation as an intermediate step doesn't uniformly help or hurt, it changes *which* errors occur (see case `cond2_01`/`cond2_02` in the five-cases file). **(4)** Adding 20 harder items each to the math and logic pilots successfully broke the logic ceiling for direct classification (Condition B: 95%→80% from the original to the harder set) — though the translate-then-answer Condition C stayed flat at 85% in both sets — but did *not* break the math ceiling (100%/100% held even on the 20 hardest remaining MGSM items) — Mini-MGSM-style grade-school arithmetic isn't a discriminating enough instrument for this model in either language; the logic items, especially ones requiring precise Cherokee connectives (De Morgan's "or", biconditional "if and only if"), are.

## Which apparent results were artifacts of the scoring script?

The semantic-recovery 60% figure, entirely (see above — corrected to 100% after two automated fixes plus manual audit; full case-by-case justification in `results/semantic_recovery_rescore_comparison.csv`). Separately, the original Condition C's 0% strict-formula-match score was **not** a scoring bug exactly, but it was misleading on its own: a structural re-grading (same connective/negation-polarity/quantifier-type regardless of predicate naming) put the same trials at 95%, revealing that the 0% mostly reflected two independent agents inventing different-but-consistent predicate names, not a comprehension failure.

## At what stage did the errors occur?

Every clean error found this week traces to one of two places, never to reasoning: **translation/semantic recovery** (dropped negation in `cond_04`; lost "or"/"iff" connectives in `neg2_01`, `cond2_01`, `cond2_02`) or **dataset-design ambiguity** (`quant2_02`'s gold label depends on an Aristotelian-vs-modern logical convention the item didn't specify — both blind readers gave an internally coherent, historically-named-valid answer). No item was found where translation was accurate, logical form was intact, and reasoning still failed. Full detail and five worked cases in `results/five_representative_error_cases.md`.

## Language bottleneck or reasoning bottleneck?

**Language/semantic-recovery**, and more strongly than last week's write-up could support. Across two independent 20-item logic batches (40 items total) and roughly 85 graded Cherokee-condition trials, English (Condition A) and Oracle Symbolic (Condition D) are 100% in every single run. Every real error sits in a Cherokee-involving condition (B or C), and every one of those errors is traceable to a specific, nameable translation defect (a dropped negation, a lost connective) rather than to any downstream reasoning mistake — reasoning over the *same* broken premise a blind reader received was consistently valid. This is a stronger and more specific claim than "Cherokee scores lower than English": it's "wherever a translation defect can be pinpointed, that defect alone explains the wrong answer."

## Limitations (do not overinterpret)

- This is Claude Sonnet 5 self-assessing translations it wrote itself, not a fluency-independent test of a model against professionally-produced Cherokee. No native speaker has reviewed any translation here.
- N is still small (40 logic items, 30 math items) — individual items, not robust statistics, drive every percentage point.
- The subject is Claude, not the plan's originally-designated GPT-5.6; rerunning against a real API-key-backed model remains the open item before any of these numbers generalize beyond "a pilot on this pipeline."
- `quant2_02` shows gold-label design itself needs another pass before claiming any single-item "error" is meaningful — logical convention should be pinned down explicitly in future items.
