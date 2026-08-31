"""
Part A / Section 2.2: Mini-MGSM, first round -- end-to-end accuracy only.

Keeps the math identical across languages, changes only the surface language,
and checks the model's final numeric answer against gold. Usage:

    python src/run_mgsm.py
    python src/run_mgsm.py --limit 5          # smoke test on 5 problems
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

SYSTEM_PROMPT = (
    "You are solving a grade-school math word problem. Think step by step, "
    "then give the final numeric answer on the last line in exactly this "
    "format:\nFINAL ANSWER: <number>"
)


def parse_final_answer(raw_output):
    m = re.search(r"FINAL ANSWER:\s*(-?[\d,]+(?:\.\d+)?)", raw_output, re.IGNORECASE)
    text = m.group(1) if m else None
    if text is None:
        nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", raw_output)
        text = nums[-1] if nums else None
    if text is None:
        return None
    try:
        val = float(text.replace(",", ""))
        return int(val) if val.is_integer() else val
    except ValueError:
        return None


def main():
    parser = argparse.ArgumentParser(description="Part A: Mini-MGSM end-to-end accuracy.")
    parser.add_argument("--data", default=os.path.join(DATA_DIR, "mgsm_sample.json"))
    parser.add_argument("--out", default=os.path.join(RESULTS_DIR, "mgsm_results.csv"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--limit", type=int, default=None, help="only run the first N problems (smoke test)")
    args = parser.parse_args()

    with open(args.data, encoding="utf-8") as f:
        dataset = json.load(f)
    items = dataset["items"][: args.limit] if args.limit else dataset["items"]

    rows = []
    for item in items:
        for lang_code, question in item["questions"].items():
            raw_output = call_model(question, system_prompt=SYSTEM_PROMPT, model=args.model, temperature=args.temperature)
            parsed = parse_final_answer(raw_output)
            correct = parsed is not None and float(parsed) == float(item["gold_answer"])

            log_trial(
                model=args.model, temperature=args.temperature,
                input_data={"id": item["id"], "language": lang_code, "question": question},
                prompt=question, raw_output=raw_output, parsed_answer=parsed,
                gold_label=item["gold_answer"], stage="mgsm_end_to_end",
            )
            rows.append({
                "id": item["id"], "language": lang_code, "language_name": LANG_NAMES.get(lang_code, lang_code),
                "model": args.model, "temperature": args.temperature,
                "parsed_answer": parsed, "gold_answer": item["gold_answer"], "correct": correct,
            })
            print(f"{item['id']} [{lang_code}] parsed={parsed} gold={item['gold_answer']} correct={correct}")

    df = pd.DataFrame(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nWrote {len(df)} rows to {args.out}")
    print(df.groupby("language_name")["correct"].mean())


if __name__ == "__main__":
    main()
