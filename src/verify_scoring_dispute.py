"""
Week 4 P0: verify the Cherokee "19 correct / 1 incorrect" scoring dispute
from ORIGINAL logged records only (results/api_call_log.jsonl) -- never from
a regenerated answer.

For every label-scored trial (Conditions A, B, D, and Week 3's C_v2) it takes
the logged raw output, extracts the prediction, and recomputes correctness
with an independent normalized comparison:

    normalized_gold       = str(gold_label).strip().lower()
    normalized_prediction = str(prediction).strip().lower()
    recalculated_correct  = normalized_prediction == normalized_gold

then compares that against the `correct` value the pipeline originally
stored in the results CSVs. Any disagreement is a scoring error.

Outputs: results/scoring_dispute_evidence.csv
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_DIR

CONDITION_TO_COLUMNS = {
    # (stage, condition) in the log -> (results csv, stored-correct column)
    ("logic_pilot", "A_english_natural"): ("cherokee_pilot.csv", "A_english_correct"),
    ("logic_pilot", "B_cherokee_natural"): ("cherokee_pilot.csv", "B_cherokee_correct"),
    ("logic_pilot", "D_oracle_symbolic"): ("cherokee_pilot.csv", "D_oracle_symbolic_correct"),
    ("logic_pilot_v2", "C_v2_cherokee_to_english_then_answer"): (
        "cherokee_pilot_v2_four_condition.csv", "C_v2_cherokee_to_english_correct"),
}


def extract_prediction(raw_output):
    """Logged raw output is either a bare label or a JSON object with an 'answer' field."""
    text = str(raw_output).strip()
    if text.startswith("{"):
        try:
            return json.loads(text)["answer"]
        except (ValueError, KeyError):
            pass
    return text


def main():
    with open(os.path.join(RESULTS_DIR, "api_call_log.jsonl"), encoding="utf-8") as f:
        log = [json.loads(line) for line in f]
    stored = {name: pd.read_csv(os.path.join(RESULTS_DIR, name)) for name in
              {v[0] for v in CONDITION_TO_COLUMNS.values()}}

    rows = []
    for rec in log:
        inp = rec.get("input_data")
        if not isinstance(inp, dict):
            continue
        key = (rec.get("stage"), inp.get("condition"))
        if key not in CONDITION_TO_COLUMNS:
            continue
        csv_name, column = CONDITION_TO_COLUMNS[key]
        item_id = inp["id"]
        prediction = extract_prediction(rec["raw_output"])
        normalized_gold = str(rec["gold_label"]).strip().lower()
        normalized_prediction = str(prediction).strip().lower()
        recalculated = normalized_prediction == normalized_gold
        old_correct = bool(stored[csv_name].loc[stored[csv_name]["id"] == item_id, column].iloc[0])
        rows.append({
            "id": item_id, "condition": inp["condition"], "stage": rec["stage"],
            "gold": normalized_gold, "logged_raw_output": str(rec["raw_output"])[:200],
            "prediction": normalized_prediction, "old_correct": old_correct,
            "recalculated_correct": recalculated,
            "scoring_error": old_correct != recalculated,
            "logged_model_field": rec["model"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "scoring_dispute_evidence.csv"), index=False)

    print(f"Recomputed {len(df)} label-scored trials from the original log.")
    print(f"Scoring errors (old correct != recalculated): {int(df['scoring_error'].sum())}")
    for cond, g in df.groupby("condition"):
        print(f"  {cond}: {int(g['recalculated_correct'].sum())}/{len(g)} correct after recalculation "
              f"(old pipeline said {int(g['old_correct'].sum())}/{len(g)})")
    wrong_b = df[(df["condition"] == "B_cherokee_natural") & (~df["recalculated_correct"])]
    print("\nCondition B items incorrect after independent recalculation:")
    print(wrong_b[["id", "gold", "prediction", "old_correct", "recalculated_correct"]].to_string(index=False))


if __name__ == "__main__":
    main()
