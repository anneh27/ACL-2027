# Five representative error cases, with attribution

Per Week 3 Task 5. Drawn from across the full pilot (original 20 logic items + 20 new harder items + the Mini-MGSM semantic-recovery probe), chosen to cover the four attribution categories from the plan's Acceptance Goal: translation, semantic recovery, scoring method, and logical reasoning.

## 1. `cond_04` (original 20-item set) — semantic recovery / translation

**What happened:** Fails in every Cherokee-involving condition tried on it: Condition B (direct classification), the original formal-parse Condition C, and the new translate-then-answer Condition C_v2. Condition A (English) and D (Oracle Symbolic) are both correct.

**Why:** My Cherokee translation of this item's second premise ("Ava does not arrive before noon") dropped the negation, rendering it as an affirmative statement. Every blind reader that encountered this Cherokee text inherited that error and reasoned validly *from the broken premise* — the failure is upstream of any reasoning step.

**Attribution: semantic recovery / translation.** Notably, I had flagged this exact item "very low confidence... close to gloss-level guessing" in `data/week1_cherokee.json`, before any grading occurred — a clean self-consistency check between predicted and actual translation quality.

## 2. `neg2_01` (new harder set) — semantic recovery / translation

**What happened:** Fails in both Condition B and Condition C_v2. Gold label is `entailed` (a valid De Morgan's law: ¬(A∧B) ⊢ ¬A∨¬B).

**Why:** C_v2's own English translation renders the query with "and" ("Elisi is not at home, **and also** Bobby is not at home") where the source intends "or". My Cherokee "or" connective — a word I had explicitly flagged as "outside my confident vocabulary" beforehand — was not conveyed, collapsing this item into the same (wrong) pattern as its sibling item `neg2_02`.

**Attribution: semantic recovery / translation**, and specifically localized to a single lexical item (the disjunction connective) rather than a general comprehension failure — both readers' logical reasoning about the (mis-transmitted) content was internally sound.

## 3. `cond2_01` / `cond2_02` (new harder set) — semantic recovery, method-dependent

**What happened:** Both items test a biconditional ("if and only if"). `cond2_01` fails in Condition B but succeeds in C_v2; `cond2_02` does the reverse — succeeds in B but fails in C_v2.

**Why:** My Cherokee attempt at "if and only if" (flagged very low confidence for both items) degrades into something read as a plain conditional. Whether that degraded reading still gives the right answer depends on which direction the reader happens to assign as antecedent vs. consequent — a coin-flip-like sensitivity that produced opposite outcomes in the two conditions on structurally similar items.

**Attribution: semantic recovery / translation**, but this pair is the clearest evidence in the whole pilot that method matters as much as the item: an explicit translate-then-reason step doesn't uniformly help or hurt relative to direct reasoning — it changes *which* errors occur.

## 4. `quant2_02` (new harder set) — logical convention, not translation or reasoning

**What happened:** Both blind agents answered `entailed`; gold label is `unknown`. Unlike every other error case above, both agents' own English translations were accurate ("All [X] have many spots" → "Some [X] have spots").

**Why:** Both agents explicitly invoked "subalternation" — a named, historically standard Aristotelian inference (a universal statement implies its corresponding existential). My gold label instead follows modern predicate logic, under which a universal claim does not presuppose its subject class is nonempty (the "existential fallacy"). Both are legitimate logical traditions; the item doesn't specify which is intended.

**Attribution: dataset-design ambiguity, not a model deficiency.** This is the one case in this entire pilot that isn't cleanly translation, semantic-recovery, or reasoning error — it's a caveat on my own gold-labeling choice, flagged here rather than silently counted against the model.

## 5. `mgsm_023` [Swahili] (Mini-MGSM) — scoring-method artifact, not a real error

**What happened:** Originally scored as a semantic-recovery "failure" (0/1) under the Week 2 grading rule. Reasoning from the exact same extracted structure was 100% correct.

**Why:** Swahili's civil-time convention writes 1pm–5pm as "saa 7:00 mchana – saa 11:00 jioni" (a +6-hour offset from the 12-hour clock). The original scoring rule compared Swahili-extracted digits against gold digits computed from the *English* text, so two equally correct numeral representations of the same real time were flagged as a mismatch — a bug in the checker, not in the model's Swahili comprehension. Confirmed automated fix (per-language gold computation + stripping the spurious ":00" token) resolves this specific case to 100% with no manual judgment call needed.

**Attribution: scoring method**, not the model at all. Documented at length in `src/semantic_probe_rescore.py`.

---

**Pattern across all 5 (and, more broadly, across every error found this pilot — see `results/logic_pilot_harder_batch_four_condition.csv` and `results/cherokee_pilot_v2_four_condition.csv`):** every clean error traces to translation/semantic-recovery or to a scoring/dataset-design artifact. Across roughly 85 graded Cherokee-condition trials run this project (Week 2's 20-item pilot × up to 4 conditions, plus Week 3's 20-item harder batch × 4 conditions), **zero** cases were found where translation was verifiably accurate, the underlying logical form was intact, and reasoning still reached the wrong answer. That absence is itself the main finding — see `results/one_page_summary.md`.
