# Week 5 Task 6: is `reasoning_tokens=0` explicit or a default?

**Answer: explicit. No code fix needed.**

## Evidence

1. **The OpenAI SDK's own type definition** (`openai.types.completion_usage.CompletionTokensDetails`) declares `reasoning_tokens: Optional[int] = None`. If the API response omitted the field, Pydantic would leave the attribute as `None`, never silently insert `0`. A `0` in this codebase's data can only come from the API itself saying so.
2. **The logged data has no missing values.** `results/mgsm_trilingual_full.csv`'s `reasoning_tokens` column has 0 NaN across all 150 rows -- every row got a real value from the API, including the 26 rows that are exactly `0`.
3. **A raw HTTP call, bypassing the SDK's parsing entirely**, confirms `completion_tokens_details.reasoning_tokens` is always present as a literal key in the API's own JSON body, never omitted -- checked directly with `curl`, not inferred from the SDK.

So when a row shows `reasoning_tokens=0`, that's the model reporting it used no internal reasoning tokens for that completion (a plausible thing for a short, simple answer) -- not this codebase's code, or the SDK, inserting a default in place of a missing field.

## Where this field is actually captured

Only `run_mgsm_trilingual.py` extracts and saves `reasoning_tokens` (`getattr(details, "reasoning_tokens", None)` -- the `getattr` default is redundant given point 1 above, but harmless). `run_cherokee_pilot_v2.py` and `mgsm_error_diagnosis.py` don't currently save this field at all; if it's wanted there too, that's a small addition, not a fix.

## Code fix

None needed -- the field already behaves correctly (explicit value, never a silent 0). The `getattr(..., None)` pattern in `run_mgsm_trilingual.py` is fine to leave as defensive coding even though it's not currently load-bearing.
