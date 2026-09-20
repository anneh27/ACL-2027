# Week 4 audit: scoring dispute (P0) and three-language MGSM baseline (P2/P3)

Facts, interpretation, and limitations are kept apart below. Every number can be regenerated from `results/*_log.jsonl` and `results/*_full.csv`.

## Part 1. P0: the Cherokee "19 correct / 1 incorrect" dispute

**Observed facts** (`src/verify_scoring_dispute.py`, output in `results/scoring_dispute_evidence.csv`)

| Item | Gold | Logged raw output (Condition B) | Old `correct` | Recalculated `correct` | Conclusion |
|---|---|---|---|---|---|
| `cond_04` | contradicted | `unknown` | False | False | Not a scoring error. The recorded prediction differs from gold. |

- All 80 label-scored trials (Conditions A, B, D, and Week 3's C-v2) were recomputed from the original log with `str(x).strip().lower()` comparison. **0 scoring errors.** The recalculated totals match the old ones: A 20/20, B 19/20, C-v2 17/20, D 20/20.
- The repo never attributed `cond_04` to a scoring mistake. Week 3 attributed it to a translation defect (the negation in the second premise was lost). Nothing in these records supports "scoring error".

**A problem with the evidence itself (needs Anne's attention).** The Week 2 and Week 3 Claude-run conditions were not machine-captured model output. Conditions B and C came from fresh Claude subagents, and I copied their answers by hand into Python dictionaries (`CONDITION_B`, `CONDITION_C_V2`, ...) inside `src/interactive_pilot_logic*.py`. The `raw_output` for Condition B is therefore my one-word transcription. The subagents' full original replies were never saved. So the check above proves only that the scoring arithmetic is right, **not** that the logged text faithfully reproduces what the subagent said. This is the situation the Week 4 plan warns about. The fix is to re-run the Cherokee pilot through the real API so every prediction has a saved raw output. That has not been done yet.

## Part 2. P2: three-language MGSM baseline (real API calls)

**Setup.** Official MGSM English/Spanish/Japanese test sets (HuggingFace `juletxara/mgsm`, no Claude translation). 50 question IDs shared across all three languages, sampled with seed 42. Model: `gpt-5.6-sol` (the key exposes three unlabeled `gpt-5.6-*` variants; this one was chosen by the user). One identical prompt for all languages. 150 real calls, none hard-coded, nothing modified in the raw outputs. Every call returned `gpt-5.6-sol`; every `finish_reason` was `stop`; every answer was parsed from the requested `FINAL ANSWER:` line.

| Language | Trials | Correct | Accuracy |
|---|---|---|---|
| English | 50 | 50 | 100% |
| Spanish | 50 | 48 | 96% |
| Japanese | 50 | 48 | 96% |

- The four wrong trials are on four different questions. **No question is wrong in more than one language**, and English is correct on all four.
- Scoring checks: 0 rows where the parsed value equals gold but `correct` is False; 0 rows where it differs but `correct` is True. The scoring self-tests (`python src/mgsm_scoring.py`) cover integers, negatives, decimals, thousands separators, percentages, and full-width digits.
- A 10-answer spot check is in `results/mgsm_trilingual_correct_spotcheck.csv` (4 English, 3 Spanish, 3 Japanese, seeded). In all 10 the last line of the raw output, the parsed value, and the gold answer agree. **Anne should still read these herself.**

### The four errors (P3)

Each was re-sampled 3 more times and also run through back-translate-then-answer (`src/mgsm_error_diagnosis.py`, `results/mgsm_diagnosis.csv`).

| Item | Gold | Wrong answer | Re-samples | After back-translation | What differs in the official translation |
|---|---|---|---|---|---|
| `ja_039` | 18 | 28.8 | 28.8 ×3 | 28.8 | Japanese asks 何**キロ** (kilometers); English asks miles. The model computed 18 miles and converted to km. Gold 18 is in miles. |
| `es_086` | 22 | 14.5 | 14.5 ×3 | 14.5 | Spanish "sonó **tres veces más que** la primera vez" versus English "three times as long as". The model read "three more times" (4+3=7). Ambiguous Spanish. |
| `ja_139` | 70 | 48 | 48 ×3 | 48 | The Japanese drops "22 more than". It says "4× as many pink as blue", so 12×4=48 is correct for the text as written. |
| `es_214` | 8 | 0 | 0 ×3 | 0 | Spanish says the pools had "**la misma cantidad**" (the same amount) four minutes ago; English says "twice as much". With "same amount", x=0 follows. |

In all four the other language versions keep the relation (`es_139` keeps "22 más"; `ja_214` and `ja_086` keep "2倍" and "3倍"), which is how I identified the deviations.

**Interpretation (mine, not established fact).** None of the four errors is a clean case of "faithful translation, model wrong in the target language". Three are official translations whose content differs from the English (`ja_039`, `ja_139`, `es_214`), where the gold answer no longer matches the text. One is a genuine Spanish ambiguity (`es_086`). The errors are perfectly stable across re-samples, and the back-translations reproduce the deviation and the wrong answer, so sampling noise and a reasoning slip are both unlikely. The plan's attribution table would file these under **"item or gold answer is ambiguous → data issue"**.

**Please verify.** The attributions rest on my reading of the Spanish and Japanese. The exact source texts are in `results/mgsm_trilingual_full.csv` and `data/mgsm_trilingual_sample.json`. Someone who reads both languages should confirm the four discrepancies before this goes into a paper.

## Limitations

1. **Temperature.** `gpt-5.6-sol` rejects `temperature=0` (only its default is supported). The plan's temperature-0 requirement could not be met. It is recorded in every row and in `results/mgsm_trilingual_manifest.json`.
2. **Re-run identity.** Because of default-temperature sampling, a full re-run will not reproduce the same outputs. The first run had 3 rate-limit failures (HTTP 429; the account allows very few requests per minute). The runner's `--resume` mode kept the 147 successful records verbatim and re-called only those 3. The manifest records this.
3. **Ceiling and power.** English is 100% and the gap is two items per language, all explained by translation differences. This baseline shows no measurable language effect on reasoning. It also cannot rule one out at n=50 with a single sample each.
4. **Fewer than five errors.** The plan asks for at least 5 wrong questions to analyze and 3 back-translation runs. Only 4 wrong trials exist. All 4 were analyzed and all 4 got the back-translation run.
5. **Same model does everything.** GPT does the answering and the back-translation. The plan's Condition D (official English as an oracle) is identical to Condition A for MGSM, since the official English is the oracle.
6. **Model identity.** `gpt-5.6-sol` was one of three unlabeled variants and was chosen by the user. The course label "GPT-5.6" does not map to a single API model.
