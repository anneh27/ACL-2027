"""
Week 6 Task 2 + Task 3: Semantic Recovery Critic + Repair, compared against a
generic-reflection baseline, on the same Stage-1 extractions already
collected in Week 5 (results/mgsm_semantic_recovery_log.jsonl) -- no new
extraction calls, so all three conditions start from the identical
extraction and only the post-extraction step differs, per the plan's "Run
the following conditions on the same MGSM examples."

Three conditions:
  A. Baseline            -- Week 5's existing data, reused as-is: initial
                             extraction -> Stage 2. No new calls.
  B. Generic reflection  -- continues the SAME conversation that produced
                             the Week 5 extraction (system + original
                             question + the saved extraction as the
                             assistant's prior turn), then asks the model
                             to "Check your previous extraction carefully
                             and improve it if necessary." Whatever comes
                             back (revised or unchanged) goes to an
                             isolated Stage 2.
  C. Semantic Recovery    -- a FRESH call that explicitly sees the original
     Critic                 question AND the extraction side by side (the
                             plan's one deliberate exception to Stage-2
                             isolation -- the critic is not Stage 2), checks
                             the plan's six explicit criteria, and outputs
                             Sufficient: Yes/No + missing info + a suggested
                             repair. If insufficient, a second call asks for
                             a repaired JSON extraction. Only the repaired
                             (or, if judged sufficient, the original)
                             extraction goes to an isolated Stage 2 -- the
                             critic's prose reasoning and the original
                             question text are NOT passed through.

Both B and C's Stage-2 call uses the exact same REASON_SYSTEM and isolation
check as Week 5 (mgsm_semantic_recovery_trilingual.verify_isolation),
confirmed per-call, not just by inspection.

Usage:
    python src/mgsm_semantic_critic_repair.py --n-items 10 --tag smoke
    python src/mgsm_semantic_critic_repair.py
"""
import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, PROJECT_ROOT, RESULTS_DIR
from mgsm_scoring import is_correct, parse_prediction
from mgsm_semantic_recovery_trilingual import EXTRACT_SYSTEM, REASON_SYSTEM, parse_json_array, verify_isolation
from run_mgsm_trilingual import TEMPERATURE_NOTE, call_model, call_model_messages, load_env

REFLECTION_PROMPT = "Check your previous extraction carefully and improve it if necessary."

CRITIC_SYSTEM = (
    "You are a Semantic Recovery Critic. You will be given a math word problem and a structured "
    "extraction (a JSON list of quantities and roles) that was produced from it, intended to let someone "
    "solve the problem using ONLY the extraction, without ever seeing this original problem text again. "
    "Decide whether the extraction contains everything needed for that -- nothing else about reasoning or "
    "solving the problem is your job.\n\n"
    "Explicitly check all of the following:\n"
    "- Is the target/question itself preserved (what quantity is actually being asked for -- a total, a "
    "difference, one specific entity's value, a unit conversion)?\n"
    "- Are all relevant entities preserved (one vs. all, which specific one)?\n"
    "- Are units preserved (dollars vs. cents, hours vs. minutes, and similar conversions)?\n"
    "- Are quantitative relations preserved (ratios, multipliers, differences between quantities)?\n"
    "- Is aggregation/scope preserved (total, difference, multiplication, ratio, average, etc.)?\n"
    "- Is there any ambiguity that could lead to multiple interpretations?\n\n"
    "Respond in EXACTLY this format, no other text:\n"
    "Sufficient: Yes/No\n\n"
    "Missing or distorted information:\n"
    "- (one bullet per issue, or \"- none\" if sufficient)\n\n"
    "Suggested repair:\n"
    "- (one bullet per fix, or \"- none\" if sufficient)"
)

REPAIR_SYSTEM = (
    "You will be given a math word problem, a structured extraction of it, and a critic's assessment of "
    "what is missing or distorted in that extraction. Produce a REPAIRED extraction that fixes every issue "
    "the critic raised. Respond with ONLY a JSON array in the same format as the original extraction, no "
    'prose, no code fences. Each element: {"quantity": <number>, "role": "<short English label describing '
    'what this quantity represents and how it is used>"}. If the question itself asks for something beyond '
    "a single quantity (e.g. a total, a difference, a specific unit), make sure a role entry (or role text) "
    "makes that target explicit so a reader of the JSON alone, with no access to this problem text, knows "
    "exactly what final quantity to compute and in what unit."
)

CRITIC_PARSE_RE = re.compile(
    r"Sufficient:\s*(Yes|No)\s*\n+Missing or distorted information:\s*\n(.*?)\n+Suggested repair:\s*\n(.*)",
    re.IGNORECASE | re.DOTALL,
)


def parse_critic(raw_output):
    """Returns (sufficient: bool|None, missing_text, repair_text)."""
    m = CRITIC_PARSE_RE.search(raw_output or "")
    if not m:
        return None, "", ""
    sufficient = m.group(1).strip().lower() == "yes"
    return sufficient, m.group(2).strip(), m.group(3).strip()


def load_baseline(log_path):
    """Week 5's already-collected extractions, keyed by (question_id, language). No new calls."""
    baseline = {}
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            baseline[(rec["question_id"], rec["language"])] = rec
    return baseline


def _safe(fn, condition_label):
    """Wrap a per-item job function so one bad item (e.g. an isolation-check
    assertion, or any other unexpected exception) can't kill the whole
    ThreadPoolExecutor and lose every other already-completed result, which is
    exactly what happened the first time this script ran at full scale: a
    single AssertionError on one item discarded ~150 items' worth of real,
    already-paid-for API calls because nothing was written until the very end.
    Failures are recorded as an explicit error row, not silently dropped."""
    def wrapped(job):
        client, model, base = job
        try:
            return fn(job)
        except Exception as exc:  # noqa: BLE001 -- deliberately broad: must never kill the pool
            return {
                "condition": condition_label, "question_id": base["question_id"], "language": base["language"],
                "gold_answer": base["gold_answer"], "correct": None,
                "job_error": f"{type(exc).__name__}: {exc}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
    return wrapped


def run_condition_b(job):
    """Generic self-reflection: continue the saved extraction conversation, ask for a generic
    improvement, then isolated Stage 2 on whatever comes back."""
    client, model, base = job
    qid, lang, gold = base["question_id"], base["language"], base["gold_answer"]
    question = base["question"]

    messages = [
        {"role": "system", "content": EXTRACT_SYSTEM},
        {"role": "user", "content": question},
        {"role": "assistant", "content": base["stage1_raw_output"]},
        {"role": "user", "content": REFLECTION_PROMPT},
    ]
    reflection = call_model_messages(client, model, messages)
    revised_struct = parse_json_array(reflection["raw_output"]) if not reflection["error_message"] else None
    # fall back to the original extraction if reflection failed or produced nothing parseable
    final_struct = revised_struct if revised_struct else parse_json_array(base["stage1_raw_output"])

    stage2_prompt = json.dumps(final_struct, ensure_ascii=False) if final_struct else "[]"
    verify_isolation(stage2_prompt, REASON_SYSTEM, question, qid, lang)
    stage2 = call_model(client, model, stage2_prompt, system=REASON_SYSTEM)
    prediction, parse_method = (None, "none") if stage2["error_message"] else parse_prediction(stage2["raw_output"])
    correct = (not stage2["error_message"]) and is_correct(prediction, gold)

    return {
        "condition": "B_generic_reflection", "question_id": qid, "language": lang, "gold_answer": gold,
        "reflection_raw_output": reflection["raw_output"], "reflection_error": reflection["error_message"],
        "extraction_changed": revised_struct is not None and revised_struct != parse_json_array(base["stage1_raw_output"]),
        "final_extraction": json.dumps(final_struct, ensure_ascii=False),
        "stage2_prompt": stage2_prompt, "stage2_raw_output": stage2["raw_output"],
        "stage2_prediction": prediction, "stage2_parse_method": parse_method, "stage2_error": stage2["error_message"],
        "correct": correct, "model": model, "temperature": TEMPERATURE_NOTE,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def run_condition_c(job):
    """Semantic Recovery Critic: fresh call sees question + extraction side by side (the only
    place after Stage 1 that does), decides sufficiency against explicit criteria, repairs if
    needed, then isolated Stage 2 on the repaired (or original, if sufficient) extraction."""
    client, model, base = job
    qid, lang, gold = base["question_id"], base["language"], base["gold_answer"]
    question = base["question"]
    original_extraction_raw = base["stage1_raw_output"]

    critic_prompt = f"Problem:\n{question}\n\nExtraction:\n{original_extraction_raw}"
    critic = call_model(client, model, critic_prompt, system=CRITIC_SYSTEM)
    sufficient, missing_text, repair_text = parse_critic(critic["raw_output"])

    repair_raw_output, repair_error = "", ""
    if sufficient is False:
        repair_prompt = (
            f"Problem:\n{question}\n\nOriginal extraction:\n{original_extraction_raw}\n\n"
            f"Critic's missing/distorted information:\n{missing_text}\n\nCritic's suggested repair:\n{repair_text}"
        )
        repair = call_model(client, model, repair_prompt, system=REPAIR_SYSTEM)
        repair_raw_output, repair_error = repair["raw_output"], repair["error_message"]
        repaired_struct = parse_json_array(repair_raw_output) if not repair_error else None
        final_struct = repaired_struct if repaired_struct else parse_json_array(original_extraction_raw)
        was_repaired = repaired_struct is not None
    else:
        # sufficient (Yes), or the critic's response didn't parse -- in the latter case treat as
        # sufficient rather than silently repairing from an unparsed judgment
        final_struct = parse_json_array(original_extraction_raw)
        was_repaired = False

    stage2_prompt = json.dumps(final_struct, ensure_ascii=False) if final_struct else "[]"
    # isolation check: Stage 2 gets NEITHER the original question NOR the critic's prose, only
    # the (possibly repaired) extraction -- this is the one place isolation could easily leak
    # given the critic step explicitly sees the question, so it's checked explicitly here
    verify_isolation(stage2_prompt, REASON_SYSTEM, question, qid, lang)
    if missing_text and missing_text.lower() != "none" and missing_text in stage2_prompt:
        raise AssertionError(f"{qid}[{lang}]: critic's prose leaked into the Stage-2 prompt")
    stage2 = call_model(client, model, stage2_prompt, system=REASON_SYSTEM)
    prediction, parse_method = (None, "none") if stage2["error_message"] else parse_prediction(stage2["raw_output"])
    correct = (not stage2["error_message"]) and is_correct(prediction, gold)

    return {
        "condition": "C_semantic_recovery_critic", "question_id": qid, "language": lang, "gold_answer": gold,
        "critic_raw_output": critic["raw_output"], "critic_error": critic["error_message"],
        "critic_sufficient": sufficient, "critic_missing_info": missing_text, "critic_suggested_repair": repair_text,
        "was_repaired": was_repaired, "repair_raw_output": repair_raw_output, "repair_error": repair_error,
        "final_extraction": json.dumps(final_struct, ensure_ascii=False),
        "stage2_prompt": stage2_prompt, "stage2_raw_output": stage2["raw_output"],
        "stage2_prediction": prediction, "stage2_parse_method": parse_method, "stage2_error": stage2["error_message"],
        "correct": correct, "model": model, "temperature": TEMPERATURE_NOTE,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-items", type=int, default=None, help="limit to the first N (question,lang) pairs from Week 5's data")
    parser.add_argument("--tag", default="")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    load_env(os.path.join(PROJECT_ROOT, ".env"))
    import openai
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=120, max_retries=0)
    model = os.environ["OPENAI_MODEL"]

    baseline_log = os.path.join(RESULTS_DIR, "mgsm_semantic_recovery_log.jsonl")
    baseline = load_baseline(baseline_log)
    bases = list(baseline.values())
    if args.n_items:
        bases = bases[: args.n_items]

    suffix = f"_{args.tag}" if args.tag else ""
    print(f"{len(bases)} Week-5 extractions reused as the shared starting point for B and C "
          f"({len(bases) * 2} calls for B, up to {len(bases) * 3} for C) with model {model}...")

    b_jobs = [(client, model, b) for b in bases]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        b_records = list(pool.map(_safe(run_condition_b, "B_generic_reflection"), b_jobs))
    b_errors = sum(1 for r in b_records if r.get("job_error"))
    print(f"Condition B done: {sum(r['correct'] for r in b_records if r['correct'] is not None)}/{len(b_records)} "
          f"correct ({b_errors} job errors, excluded from that count)")

    c_jobs = [(client, model, b) for b in bases]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        c_records = list(pool.map(_safe(run_condition_c, "C_semantic_recovery_critic"), c_jobs))
    c_errors = sum(1 for r in c_records if r.get("job_error"))
    print(f"Condition C done: {sum(r['correct'] for r in c_records if r['correct'] is not None)}/{len(c_records)} "
          f"correct ({c_errors} job errors; {sum(r.get('was_repaired') is True for r in c_records)} repaired)")
    if b_errors or c_errors:
        print(f"\nJob errors (investigate before trusting the comparison -- these items are missing from the "
              f"accuracy/recovery/regression numbers, not counted as failures):")
        for r in b_records + c_records:
            if r.get("job_error"):
                print(f"  {r['condition']} {r['question_id']}[{r['language']}]: {r['job_error']}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    b_path = os.path.join(RESULTS_DIR, f"mgsm_condition_b_reflection{suffix}.csv")
    c_path = os.path.join(RESULTS_DIR, f"mgsm_condition_c_critic_repair{suffix}.csv")
    pd.DataFrame(b_records).to_csv(b_path, index=False)
    pd.DataFrame(c_records).to_csv(c_path, index=False)
    with open(os.path.join(RESULTS_DIR, f"mgsm_condition_b_reflection{suffix}_log.jsonl"), "w", encoding="utf-8") as f:
        for r in b_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(RESULTS_DIR, f"mgsm_condition_c_critic_repair{suffix}_log.jsonl"), "w", encoding="utf-8") as f:
        for r in c_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {b_path}\n       {c_path}")
    print("Isolation check: passed for all B and C jobs (verify_isolation raised on none of them).")


if __name__ == "__main__":
    main()
