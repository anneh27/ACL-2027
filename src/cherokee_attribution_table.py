"""
Week 5 Task 4 deliverable: per-item attribution table for all 20 Cherokee
items, refined into the taxonomy the plan asks for -- language/semantic
recovery, explicit back-translation failure, reasoning, and dataset/
formal-representation issues -- distinct from Week 4's coarser categories.

Reads only already-collected data (results/cherokee_pilot_v2_full.csv,
data/cherokee_pilot_v2_error_type_overrides.json); makes no API calls.

Usage: python src/cherokee_attribution_table.py
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, RESULTS_DIR

OVERRIDES_PATH = os.path.join(DATA_DIR, "cherokee_pilot_v2_error_type_overrides.json")


def attribute(row, refined_by_id):
    """Program-computed first pass, per the plan's own stated rules; overridden by
    manual judgment (refined_by_id) wherever one exists, per the same rule the
    pilot script already uses for error_type."""
    a, b, c, d = row["A_correct"], row["B_correct"], row["C_correct"], row["D_correct"]
    if row["id"] in refined_by_id:
        return refined_by_id[row["id"]]
    if a and b and c and d:
        return "none"
    if not a:
        return "item_itself_hard"  # the plan: check English first, before blaming anything else
    if not d:
        return "check_formal_representation_first"  # the plan's explicit D-wrong rule
    if b and not c:
        return "explicit_back_translation_failure"  # the plan's explicit B-correct/C-wrong rule
    if not b and not c:
        return "language_semantic_recovery"
    return "unclassified"  # shouldn't occur given the four rules above, but don't silently mis-tag


def main():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "cherokee_pilot_v2_full.csv"))
    overrides = json.load(open(OVERRIDES_PATH, encoding="utf-8"))["error_type_overrides"]
    refined_by_id = {k: v["error_type_refined_week5"] for k, v in overrides.items() if "error_type_refined_week5" in v}

    rows = []
    for _, r in df.iterrows():
        attribution = attribute(r, refined_by_id)
        note = overrides.get(r["id"], {}).get("week5_note", "")
        rows.append({
            "id": r["id"], "reasoning_type": r["reasoning_type"], "gold_label": r["gold_label"],
            "A_english": r["A_correct"], "B_cherokee_direct": r["B_correct"],
            "C_backtranslate_then_answer": r["C_correct"], "D_oracle_symbolic": r["D_correct"],
            "attribution": attribution, "note": note,
        })

    out_df = pd.DataFrame(rows)
    out = os.path.join(RESULTS_DIR, "cherokee_attribution_table.csv")
    out_df.to_csv(out, index=False)

    print("=== Per-item attribution, all 20 Cherokee items ===")
    print(out_df[["id", "A_english", "B_cherokee_direct", "C_backtranslate_then_answer",
                   "D_oracle_symbolic", "attribution"]].to_string(index=False))
    print(f"\nAttribution counts:\n{out_df.attribution.value_counts().to_string()}")
    print(f"\nWrote {out}")
    if "unclassified" in out_df.attribution.values:
        print("\nWARNING: at least one item didn't match any of the four attribution rules -- check by hand.")


if __name__ == "__main__":
    main()
