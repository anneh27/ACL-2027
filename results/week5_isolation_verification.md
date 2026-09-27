# Week 5 Task 3: Stage 1 -> Stage 2 isolation, verified

The plan asks: "Make sure the reasoning stage receives only the Stage-1 representation, with no access to the original question or another language version."

## How it's enforced

Two independent pipelines both isolate correctly, checked two ways:

1. **Automated, on every call.** `mgsm_semantic_recovery_trilingual.py`'s `verify_isolation()` runs before every one of the 150 Stage-2 calls in the full run and raises an `AssertionError` if the original question text, the item id, or the language name/code appears anywhere in what Stage 2 actually receives (prompt + system message combined). It raised on none of them -- the run completed cleanly, which itself is the check passing 150/150 times, not just for a sample.
2. **Manual, by reading the actual payload.** Pulled `results/mgsm_semantic_recovery_log.jsonl`'s first record directly (not summarized) and confirmed by string search: the original question text is **not** a substring of `stage2_prompt`, the item id is **not** present, and none of "English"/"Spanish"/"Japanese" appear in `stage2_prompt` or `stage2_system`. The whole Stage-2 payload is the bare JSON extraction array plus a system prompt that never mentions the source.

## Worked example

**Item:** `mgsm_trilingual_001`, English

**Original question (what Stage 2 must never see):**
> "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?"

**Stage 2 system prompt (verbatim, truncated):**
> "You will be given ONLY a JSON list of quantities and their roles, extracted from a word problem written in some language you have not seen and do not know the identity of. You do not have access to the original problem text..."

**Stage 2 prompt (the entire user message Stage 2 actually receives -- verbatim, complete):**
```
[{"quantity": 2, "role": "blue fiber amount"}, {"quantity": 0.5, "role": "multiplier used to determine white fiber amount from blue fiber amount"}]
```

That's it. No question text, no item id, no language marker. Stage 2's job is to compute the answer from exactly that array and nothing else.

## The Cherokee pilot's C1 -> C2 step

`run_cherokee_pilot_v2.py` (Condition C) has the same structure and was checked the same way (`src/run_cherokee_pilot_v2.py:167-179`): `do("C2", CLASSIFY_SYSTEM_NL, c1["raw_output"].strip())` -- C2's entire prompt is `c1["raw_output"]`, the back-translation text itself, nothing appended and nothing else passed in.

## What this isolation actually buys, and what it costs

Confirming isolation holds turned out to explain most of Task 1's findings, not just satisfy a separate checklist item: because Stage 2 genuinely receives *nothing* beyond the bare extraction, it also never receives the original question's phrasing of *what is being asked* -- "in total," "together," "how many hours," "how many dollars." A `{quantity, role}` schema has no field for that. Every one of the ~22 `surface_quantity_fidelity=True, task_semantic_sufficiency=False` trials in the full run traces to exactly this: Stage 1 extracts every number correctly, but Stage 2, isolated exactly as required, computes a plausible intermediate value and has no way to know it should sum three quantities instead of one, or convert cents to dollars, or minutes to hours. Correct isolation and the resulting task-sufficiency gap are the same finding, seen from two sides -- see `results/week5_summary.md` for the full writeup.
