# Week 4 one-page summary

*Multilingual math and logic pilot. Model: `gpt-5.6-sol`, real API calls, default temperature. Full evidence: `results/week4_audit.md` and `results/cherokee_pilot_v2_audit.md`.*

## Observed facts

**MGSM baseline (official English, Spanish, Japanese; 50 shared questions; 150 calls).** English 50/50, Spanish 48/50, Japanese 48/50. The four wrong answers are on four different questions, English was right on all four, and none is wrong in two languages. Each repeated in 3 of 3 re-samples and again after a back-translate-then-answer run. In each case the official translation differs from the English or is ambiguous: `ja_039` asks for kilometers where English asks miles; `ja_139` omits "22 more than"; `es_214` says the pools held "the same amount" where English says "twice as much"; `es_086` uses "tres veces más", which can mean ×3 or +3. Scoring had 0 errors in either direction.

**Scoring dispute.** The Cherokee `cond_04` miss in Condition B was a real wrong answer, not a scoring error: recomputing all 80 logged Week 2–3 label results found 0 disagreements.

**Cherokee pilot v2 (original 20 items, 100 real calls).** A English 20/20, B Cherokee direct 18/20, C back-translate then answer 15/20, D formal notation 19/20. Fresh back-translations of the Cherokee are often unrelated to the source. Only 1 of 20 is exact and 4 are close; 10 keep the logical skeleton (negation, if, all/some) with entities and predicates changed; 5 lose the structure. Re-sampling the failures showed most are unstable: 6 of 7 re-sampled B/C failures were right in at least one fresh run. Only `cond_04`'s Condition C failure and `spat_04`'s Condition D failure repeated every time.

## Possible explanations (not established)

- The MGSM gaps look like translation and data problems, not weaker reasoning in Spanish or Japanese. At 50 questions and one sample each, this cannot rule out a small language effect.
- Correct Cherokee answers seem to come from the logical skeleton surviving, not from recovered meaning, so accuracy overstates translation success. Explicit back-translation never rescued a wrong B answer.
- `spat_04`'s D miss is a dataset gap: the formal premises lack a transitivity and asymmetry axiom that the gold label needs. It says nothing about Cherokee.
- **Language versus reasoning.** No error in either experiment was a clean reasoning failure after the meaning had been supplied. A and D are near-perfect, and every remaining error sits where the meaning was altered, lost, or under-specified. But the Cherokee is Claude-written and unvalidated, so poor back-translations may reflect defective Cherokee, not the model's ability to read Cherokee. These experiments cannot separate the two.

## Limitations

- `gpt-5.6-sol` rejects `temperature=0`, so outputs are not reproducible run to run. Three MGSM calls (rate limits) and one Cherokee call (a timeout, plus the call that depended on it) failed on the first pass and were re-called; every other output is first-pass and unmodified.
- One model, small samples, and only 4 wrong MGSM answers (the plan asks for 5). The 20 harder Week 3 logic items have not been run through the real model.
- Error types, the fidelity ratings (made after seeing outcomes, so partly circular) and the MGSM translation-discrepancy readings are my judgment and unverified. Nobody who reads Spanish or Japanese has checked the four discrepancies. No native Cherokee speaker has reviewed any translation.
- Re-sampling covered only conditions that failed first, which biases toward showing instability.

## Next steps

Verify the judgment fields (blind re-rating, Spanish and Japanese check); get a native Cherokee reviewer; add the missing spatial axioms and re-run D; run repeated samples of all items to estimate real per-item rates; consider a model that supports temperature 0.
