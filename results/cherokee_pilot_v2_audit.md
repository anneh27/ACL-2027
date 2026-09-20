# Cherokee pilot v2 audit (Week 4, P1)

The 20 original logic items, re-run through the real model for all four conditions. Every prediction below comes from a saved API response (`results/cherokee_pilot_v2_log.jsonl`); `correct` is computed by the program from that raw output. Facts, interpretation, and limitations are kept apart.

## What was run

- Model `gpt-5.6-sol`, default temperature (explicit `temperature=0` is rejected by this model). 100 calls: 20 items × {A English, B Cherokee direct, C1 back-translation, C2 answer from the back-translation, D formal notation}.
- Prompts are the ones the original pilot used (`src/run_logic_pilot.py`). The Cherokee text is the Claude-written candidate translation in `data/week1_cherokee.json`.
- Two API calls failed with a timeout on the first pass (`quant_02`, C1 and C2) and were re-called with `--resume`; the other 98 records are the first-pass outputs, unmodified.
- All 80 label answers were single words that parsed exactly, so no parsing or scoring judgment was involved.

## Results (program-computed)

| Condition | Correct | Claude-run pilot (Weeks 2–3) |
|---|---|---|
| A English | 20/20 | 20/20 |
| B Cherokee direct | 18/20 | 19/20 |
| C Cherokee → English → answer | 15/20 | 17/20 |
| D Oracle symbolic | 19/20 | 20/20 |

The two pilots are not directly comparable: different model, and C here is two separate calls (translate, then answer) instead of one agent doing both. The overlap is informative, though: `cond_04` (B and C), `neg_03` (C) and `spat_01` (C) failed in both.

By reasoning type: negation A/B/C/D = 5/5, 5/5, 4/5, 5/5; conditional 5/5, 4/5, 4/5, 5/5; quantifier 5/5, 5/5, 5/5, 5/5; spatial 5/5, 4/5, 2/5, 4/5.

## Scoring corrections

**None needed for the Cherokee label scoring.** Independent recomputation of all 80 logged Week 2–3 label results found 0 disagreements (`results/scoring_dispute_evidence.csv`), and in this new run every output parsed exactly. The disputed 19/1 result (`cond_04`, Condition B: recorded `unknown`, gold `contradicted`) was a genuine miss on that run, not a scoring error.

The Week 3 scoring correction is a separate thing: it changed the MGSM digit-matching rule in `src/semantic_probe_rescore.py` (gold digits computed per language instead of always from English; the ":00" in clock times no longer counted as a quantity). Of the 6 MGSM semantic-recovery trials flagged (`mgsm_023` ×3, `mgsm_035` ×3), the automated fix resolved 1 (`mgsm_023`, Swahili). The other 5 were resolved by a manual audit. No Cherokee item was touched by that change.

## Item-by-item: every item where B, C, or D was wrong

"Re-samples" are 3 fresh runs of the failing condition (`results/cherokee_pilot_v2_stability.csv`). For C, each re-sample includes a fresh back-translation.

| Item | Gold | B | C | D | Re-samples wrong | Type |
|---|---|---|---|---|---|---|
| `neg_03` | contradicted | contradicted | **unknown** | contradicted | C: 1/3 | translation (unstable) |
| `cond_04` | contradicted | **unknown** | **entailed** | contradicted | B: 1/3, C: 3/3 | translation (C stable, B unstable) |
| `spat_01` | entailed | **unknown** | **unknown** | entailed | B: 1/3, C: 2/3 | translation |
| `spat_02` | entailed | entailed | **unknown** | entailed | C: 1/3 | translation (unstable) |
| `spat_03` | entailed | entailed | **contradicted** | entailed | C: 0/3 | translation (mostly noise) |
| `spat_04` | contradicted | contradicted | contradicted | **unknown** | D: 3/3 | data ambiguity |

- **`neg_03`.** The back-translation was "This is not their heavenly home. It is his home.", unrelated to "Nina does not own a bicycle". Fresh runs gave other unrelated text ("his twins in heaven"). B was right anyway, most likely from structure (a negated statement, then the same statement asserted).
- **`cond_04`.** The back-translations are different unrelated stories every time (a man-eater; deer feeding; a big eater), so C fails stably. B was wrong in the original run and 1 of 3 fresh runs, i.e. 2 of 4 samples, so the B failure is not deterministic. Not a scoring error.
- **`spat_01`.** No back-translation ever contains a left/right relation ("there on the mountain makes…"). Whether the label comes out right depends on whether the model treats the invented "X produces Y" chain as transitive.
- **`spat_02`.** The Cherokee text writes the second entity inconsistently (a suffix in the first sentence, none in the second). The original back-translation read it as "Norway", which breaks the chain, so `unknown` was valid for that text. 2 of 3 fresh back-translations read "Noah" and answered `entailed`.
- **`spat_03`.** The original back-translation kept a behind/ahead relation but lost both objects. All 3 fresh runs answered `entailed`, although their back-translations are even less faithful ("My turn is a little later…"), so those correct answers are not evidence of recovered meaning.
- **`spat_04` (D wrong).** B and C are right; the oracle-symbolic answer is wrong, 4 out of 4 samples. The formal premises contain `east_of(park,school)`, `east_of(library,park)` and the east/west converse postulate, but no transitivity or asymmetry axiom for `east_of`, and the gold label needs both. Read strictly, `unknown` is a defensible answer to the input as written. The other spatial items have the same implicit dependence (the gold labels assume transitivity), so the fix is to make those axioms explicit in the formal representations. It says nothing about Cherokee.

## Findings

1. **Accuracy overstates translation success.** Of the 14 items right in B, C and D together, only `quant_05` back-translates faithfully. Most keep the logical skeleton (negation, if, all/some, repeated predicates) while entities and predicates change completely ("Liam is not at school" → "Lavmi is not at the airport"; a door becomes a bridge; a button and machine become a groundhog). Rating each back-translation (`back_translation_fidelity` in the CSV): 1 exact, 4 close, 10 skeleton-only, 5 broken. The plan's own rule applies: a correct answer does not make the translation successful when entities or predicates changed. By that rule only 5 of 20 translations are content-faithful.
2. **C is right exactly when the skeleton survives.** All 5 "broken" back-translations are the 5 wrong C answers; all 15 others are right. **Caveat:** I rated fidelity after seeing the C outcomes, so this split is partly circular. Anne should re-rate blind.
3. **Explicit translation never rescued B.** B right and C wrong: `neg_03`, `spat_02`, `spat_03`. C right and B wrong: none.
4. **Single-sample errors are noisy.** At the default temperature, 6 of the 7 re-sampled B/C failures came out right in at least one fresh run. Item-level pass/fail lists from one sample per condition should not be read as fixed properties of items. Only the `cond_04` C failure and the `spat_04` D failure are stable.

## Limitations

- **Single model, single Cherokee source.** The Cherokee is Claude-written machine translation, never checked by a native speaker, so this is an exploratory pilot. The back-translation is by GPT, so C measures how GPT reads Claude's Cherokee.
- **Judgment fields.** Error types, notes, and fidelity ratings (`data/cherokee_pilot_v2_error_type_overrides.json`) are my judgment. The keyword flags in the CSV (`flag_*`) are a rough first pass, and their entity check is noisy. **Anne should verify these; I have not had them independently reviewed.**
- **Resampling bias.** Only conditions that failed first were re-sampled, so re-sample rates are biased toward showing instability. Estimating real per-item rates needs repeated runs of all items.
- **Not re-run.** The 20 harder Week 3 logic items were not run through the real model.
- **Model identity.** `gpt-5.6-sol` was chosen by the user from three unlabeled variants.
