# Experiment config and scoring-rule audit (Week 7 Task 1)

Pure documentation/config audit of already-collected data — no new API calls, no manual semantic annotation (that's Task 2). Every number below was independently recomputed from the raw CSVs, not copied from the Week 7 plan.

## Unified four-condition comparison table

| # | Condition | Script | Correct | Accuracy | Run window (UTC) |
|---|---|---|---|---|---|
| A | Direct Reasoning (solve the original question directly, no extraction step) | `src/run_mgsm_trilingual.py` | 146/150 | 97.3% | 2026-09-20 05:29–05:33 |
| B | Original Stage 1 → Stage 2 (extract, then reason from extraction alone) | `src/mgsm_semantic_recovery_trilingual.py` | 125/150 | 83.3% | 2026-09-27 17:29–17:32 |
| C | Generic Reflection (B's extraction, then "check your previous extraction carefully and improve it if necessary", then Stage 2) | `src/mgsm_semantic_critic_repair.py` (`run_condition_b`) | 127/150 | 84.7% | 2026-10-18 17:46–17:50 |
| D | Semantic Recovery Critic + Repair (B's extraction, explicit critic checklist, conditional repair, then Stage 2) | `src/mgsm_semantic_critic_repair.py` (`run_condition_c`) | 145/150 | 96.7% | 2026-10-18 17:50–17:56 |

All four conditions run on the same 150 (question, language) pairs — 50 official MGSM items × {English, Spanish, Japanese}. B, C, and D all start from the *same* Stage-1 extraction (B's), so differences between them isolate the effect of the post-extraction step, not extraction variance.

## Model name/version

- **Requested model** (`model=` param on every call, all four conditions): `gpt-5.6-sol` (from `OPENAI_MODEL` in `.env`). "GPT-5.6" is the course's label; the live API exposes three unlabeled variants (`gpt-5.6-luna`, `gpt-5.6-sol`, `gpt-5.6-terra`) with no documentation distinguishing them — `-sol` was picked by the user when this was first set up (Week 4), with no principled reason to prefer it over the other two.
- **Model actually returned by the API** (`resp.model`): confirmed `gpt-5.6-sol` for Condition A (`model_returned` column, 150/150 rows). **Conditions B, C, and D do not capture `model_returned` at all** — their logs record only the requested model string, not what the API echoed back. This is a real reproducibility gap: nothing in B/C/D's saved data directly proves the API didn't silently route to a different variant on some calls. Not fixed retroactively here (would need a re-run); flagged as a known gap for any future run.

## Full prompts, by condition

**A — Direct Reasoning** (user prompt only, no system prompt):
```
Solve the following math problem carefully. Return the final answer on the last line in this format: FINAL ANSWER: <number>

Problem: {question}
```

**B/C/D — Stage 1 extraction** (system prompt, shared starting point for all three):
```
Read the math word problem below, written in its original language. Do NOT solve it. Extract every
quantity mentioned and the operation or role it participates in. Respond with ONLY a JSON array, no
prose, no code fences. Each element: {"quantity": <number>, "role": "<short English label describing
what this quantity represents and how it is used, e.g. initial amount, rate, multiplier, total>"}.
List quantities in the order they appear in the problem.
```
User prompt: the bare question text, in its own language.

**B/C/D — Stage 2 reasoning** (system prompt, identical and isolated in all three — see `mgsm_semantic_recovery_trilingual.verify_isolation()`):
```
You will be given ONLY a JSON list of quantities and their roles, extracted from a word problem
written in some language you have not seen and do not know the identity of. You do not have access
to the original problem text. Using ONLY this structured information, compute the final numeric
answer. Give the final numeric answer on the last line as:
FINAL ANSWER: <number>
```
User prompt: the Stage-1 JSON array verbatim (B), or the revised/repaired JSON array (C/D).

**C — generic reflection turn** (appended to the same conversation as the Stage-1 extraction; user message):
```
Check your previous extraction carefully and improve it if necessary.
```

**D — critic** (system prompt; user prompt is the original question + the Stage-1 extraction side by side — the one deliberate exception to Stage-2 isolation):
```
You are a Semantic Recovery Critic. You will be given a math word problem and a structured
extraction (a JSON list of quantities and roles) that was produced from it, intended to let someone
solve the problem using ONLY the extraction, without ever seeing this original problem text again.
Decide whether the extraction contains everything needed for that -- nothing else about reasoning or
solving the problem is your job.

Explicitly check all of the following:
- Is the target/question itself preserved (what quantity is actually being asked for -- a total, a
difference, one specific entity's value, a unit conversion)?
- Are all relevant entities preserved (one vs. all, which specific one)?
- Are units preserved (dollars vs. cents, hours vs. minutes, and similar conversions)?
- Are quantitative relations preserved (ratios, multipliers, differences between quantities)?
- Is aggregation/scope preserved (total, difference, multiplication, ratio, average, etc.)?
- Is there any ambiguity that could lead to multiple interpretations?

Respond in EXACTLY this format, no other text:
Sufficient: Yes/No

Missing or distorted information:
- (one bullet per issue, or "- none" if sufficient)

Suggested repair:
- (one bullet per fix, or "- none" if sufficient)
```

**D — repair** (system prompt, only called if the critic said `Sufficient: No`):
```
You will be given a math word problem, a structured extraction of it, and a critic's assessment of
what is missing or distorted in that extraction. Produce a REPAIRED extraction that fixes every issue
the critic raised. Respond with ONLY a JSON array in the same format as the original extraction, no
prose, no code fences. Each element: {"quantity": <number>, "role": "<short English label describing
what this quantity represents and how it is used>"}. If the question itself asks for something beyond
a single quantity (e.g. a total, a difference, a specific unit), make sure a role entry (or role text)
makes that target explicit so a reader of the JSON alone, with no access to this problem text, knows
exactly what final quantity to compute and in what unit.
```

## Decoding parameters

Every call across all four conditions uses `client.chat.completions.create(model=model, messages=messages)` — **`temperature`, `top_p`, `max_tokens`, and `seed` are never passed at all.** This is not an oversight to fix quietly: `gpt-5.6-sol` explicitly rejects an API call with `temperature=0` ("only the default (1) value is supported" — confirmed by a live 400 error in Week 4), so no decoding parameter is set anywhere, and every call runs at whatever default the API applies server-side. **Runs are not reproducible call-to-call** — Week 6's own re-sampling experiments (3 fresh samples per failing item) found different raw outputs across repeats of the identical prompt. Every number in this document is a single-sample snapshot, not a converged estimate; no run in this project has used repeated sampling to get a stable per-item rate (Week 7 Task 2 explicitly asks for 3x repeats on representative cases, for exactly this reason).

## Sample selection and seed

50 MGSM questions × {English, Spanish, Japanese} = 150 trials, built once in Week 4 (`src/fetch_mgsm_trilingual.py`) and reused unchanged by every condition since:
1. All three language configs of the official `juletxara/mgsm` dataset (HuggingFace) fetched directly via the public datasets-server API — no Claude-generated translation anywhere in the sample.
2. Row indices present in all three configs intersected (all 250 are, by construction — each language config is a human translation of the same underlying GSM8K-derived problems).
3. 50 of those 250 shared indices sampled with `random.Random(42)` — the same seed used for the original Week 2 MGSM sample, chosen for consistency, not re-derived per condition.
4. Gold `answer_number` checked identical across all three languages for every selected item before accepting it.

Full detail in `data/mgsm_trilingual_sample.json`'s `meta` block.

## Scoring rules

- **Condition A, and Stage 2 of B/C/D:** `src/mgsm_scoring.py`'s `parse_prediction()` extracts the number following the last `FINAL ANSWER:` marker (falling back to the last number in the output if no marker is found); `is_correct()` compares the normalized prediction to the normalized gold with `abs(p - g) < 1e-9`. `normalize_answer()` strips currency symbols, thousands separators, and full-width digits, and treats `9`/`9.0`/`"$9"`/`"9 eggs"` as equal. Self-tests: `python src/mgsm_scoring.py`.
- **`surface_quantity_fidelity`** (Numeric Coverage): computed in `mgsm_semantic_recovery_trilingual.py`, per-language, locale-aware (Spanish comma-decimal fixed in Week 5/6; see `results/week5_summary.md`). True iff every digit-quantity in the source question's own text appears somewhere in the Stage-1 extraction. **Independent of whether the final answer was correct** — a genuinely separate signal from accuracy.
- **`task_semantic_sufficiency`** (what the code currently calls "semantic sufficiency"): **defined as, and only as, whether Stage 2's final answer matches gold.** This is the exact conflation the Week 7 plan flags — it is a Stage-2-accuracy measure wearing a "semantic completeness" label, not an independent judgment of whether the Stage-1 representation was complete. An extraction can be judged "insufficient" by this metric purely because Stage 2 made an arithmetic slip from a perfectly complete representation, or judged "sufficient" because Stage 2 got lucky from an incomplete one. **No independent Stage-1 semantic-completeness label exists yet anywhere in this codebase** — that's exactly what Task 2's manual annotation is for, and nothing here substitutes for it.
- **Stage-2 Final Answer Accuracy**, cleanly separated: this is just `correct` (B/C/D) or `correct` (A) — the same underlying `is_correct()` call in every condition, the column that `task_semantic_sufficiency` is currently (mis)used as a proxy for.

## Known data issues (official dataset, not this pipeline's extraction)

Official Gold labels are preserved as-is throughout this project; every item below has been independently verified and is reported separately, never used to silently override `gold_answer`.

| Item | Language | Issue | Verified in |
|---|---|---|---|
| `mgsm_057` | Japanese | States the delivery truck already has 33 boxes loaded (absent from English) while still asking for the maximum that *can* be loaded — an inserted false premise, not just a dropped fact | Week 5 Task 2 audit |
| `mgsm_139` | Japanese | Omits "22 more than" from the pink/blue gumball relation entirely | Week 4/5 |
| `mgsm_214` | Spanish | States the pools held "la misma cantidad" (the same amount) where English says "twice as much" 4 minutes earlier | Week 4 |
| `mgsm_086` | Spanish | "tres veces más" is genuinely ambiguous between additive (+3) and multiplicative (×3) readings; gold assumes multiplicative | Week 4/6 |
| `mgsm_039` | Japanese | Asks for the answer in kilometers, but the official gold answer (18) is unconverted miles — a model that correctly converts units is marked "wrong" against this gold value | Week 4/6 |

Full per-item evidence: `results/week4_audit.md` (`057`, `139`, `214`, `086`, `039`), `results/week6_task4_error_analysis_summary.md` (`086`, `139`, `214`, `039` re-examined in the critic+repair context).

## What this document does *not* claim

It does not provide an independent Stage-1 semantic-completeness measure (pending Task 2's manual labels), and it does not re-score or re-run anything — every number here is reproduced from files already in this repo, each regenerable by re-running the script named in the table above.
