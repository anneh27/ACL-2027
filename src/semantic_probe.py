"""
Part B / Section 3: semantic recovery vs. reasoning, for Mini-MGSM.

The plan's worked example (Mary/apples) assumes one fixed slot schema
(initial_owner, initial_quantity, operation, change_quantity). Real MGSM
problems are heterogeneous (multi-step, varying entity/operation counts), so
a single fixed schema doesn't fit all of them without per-item hand
annotation, which is out of scope for a small pilot. Instead this script
operationalizes "semantic recovery" as: did the model, working only from the
problem in its own language, correctly recover *every numeric quantity
that appears in the problem*? Numbers are language-invariant across MGSM's
translations, which makes this automatically and objectively gradable
without a hand-built gold schema per item. This is a proxy for full slot
recovery, not identical to it -- documented here so the choice is explicit.

Two independent outcomes per item x language:
  1. semantic_recovery_correct -- structured extraction recovered all the
     numbers in the problem (Stage 1, run in the problem's own language).
  2. reasoning_correct -- final answer computed from ONLY that extracted
     structure matches gold (Stage 2, isolates reasoning from language).

Usage:
    python src/semantic_probe.py
    python src/semantic_probe.py --limit 5
"""
import argparse
import json
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_openai import call_model, log_trial
from config import DATA_DIR, RESULTS_DIR, DEFAULT_MODEL, DEFAULT_TEMPERATURE

LANG_NAMES = {"en": "English", "fr": "French", "sw": "Swahili"}

SEMANTIC_EXTRACT_SYSTEM = (
    "Read the math word problem below, written in its original language. "
    "Do NOT solve it. Extract every quantity mentioned and the operation it "
    "participates in. Respond with ONLY a JSON array, no prose, no code "
    'fences. Each element: {"quantity": <number>, "role": "<short English '
    'label, e.g. initial, add, subtract, multiply, divide, rate, total>"}. '
    "List quantities in the order they appear in the problem."
)

ORACLE_REASONING_SYSTEM = (
    "You will be given ONLY a JSON list of quantities and their roles, "
    "extracted from a word problem. Using ONLY this structured information "
    "(you do not have the original problem text), compute the final numeric "
    "answer. Give the final numeric answer on the last line as:\n"
    "FINAL ANSWER: <number>"
)


def extract_numbers(text):
    return {n.replace(",", "") for n in re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)}


def parse_json_array(raw_output):
    match = re.search(r"\[.*\]", raw_output, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def parse_final_answer(raw_output):
    m = re.search(r"FINAL ANSWER:\s*(-?[\d,]+(?:\.\d+)?)", raw_output, re.IGNORECASE)
    if not m:
        return None
    try:
        val = float(m.group(1).replace(",", ""))
        return int(val) if val.is_integer() else val
    except ValueError:
        return None


def main():
    parser = argparse.ArgumentParser(description="Part B: semantic recovery vs. reasoning probe.")
    parser.add_argument("--data", default=os.path.join(DATA_DIR, "mgsm_sample.json"))
    parser.add_argument("--out", default=os.path.join(RESULTS_DIR, "mgsm_semantic_probe.csv"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    with open(args.data, encoding="utf-8") as f:
        dataset = json.load(f)
    items = dataset["items"][: args.limit] if args.limit else dataset["items"]

    rows = []
    for item in items:
        gold_numbers = {str(float(n)) for n in extract_numbers(item["questions"]["en"])}

        for lang_code, question in item["questions"].items():
            # Stage 1: semantic recovery, in the problem's own language.
            raw1 = call_model(question, system_prompt=SEMANTIC_EXTRACT_SYSTEM, model=args.model, temperature=args.temperature)
            parsed_struct = parse_json_array(raw1)

            extracted_numbers = set()
            for entry in parsed_struct or []:
                try:
                    extracted_numbers.add(str(float(entry["quantity"])))
                except (KeyError, TypeError, ValueError):
                    continue
            semantic_recovery_correct = parsed_struct is not None and gold_numbers.issubset(extracted_numbers)

            log_trial(
                model=args.model, temperature=args.temperature,
                input_data={"id": item["id"], "language": lang_code, "question": question},
                prompt=question, raw_output=raw1, parsed_answer=parsed_struct,
                gold_label=sorted(gold_numbers), stage="mgsm_semantic_extraction",
            )

            # Stage 2: reasoning over ONLY the Stage-1 structure.
            struct_json_text = json.dumps(parsed_struct, ensure_ascii=False) if parsed_struct else "[]"
            raw2 = call_model(struct_json_text, system_prompt=ORACLE_REASONING_SYSTEM, model=args.model, temperature=args.temperature)
            parsed_answer = parse_final_answer(raw2)
            reasoning_correct = parsed_answer is not None and float(parsed_answer) == float(item["gold_answer"])

            log_trial(
                model=args.model, temperature=args.temperature,
                input_data={"id": item["id"], "language": lang_code, "structured_semantics": parsed_struct},
                prompt=struct_json_text, raw_output=raw2, parsed_answer=parsed_answer,
                gold_label=item["gold_answer"], stage="mgsm_reasoning_from_structure",
            )

            rows.append({
                "id": item["id"], "language": lang_code, "language_name": LANG_NAMES.get(lang_code, lang_code),
                "semantic_recovery_correct": semantic_recovery_correct,
                "reasoning_correct": reasoning_correct,
                "gold_answer": item["gold_answer"], "parsed_answer": parsed_answer,
            })
            print(f"{item['id']} [{lang_code}] semantic_recovery={semantic_recovery_correct} reasoning={reasoning_correct}")

    df = pd.DataFrame(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nWrote {len(df)} rows to {args.out}")

    summary = df.groupby("language_name").agg(
        semantic_recovery_rate=("semantic_recovery_correct", "mean"),
        reasoning_accuracy=("reasoning_correct", "mean"),
    )
    cond = df[df["semantic_recovery_correct"]].groupby("language_name")["reasoning_correct"].mean()
    summary["P(reasoning_correct | semantic_recovery_correct)"] = cond
    print(summary)


if __name__ == "__main__":
    main()
