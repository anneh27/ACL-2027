"""
Week 6 Task 4: error analysis of the repair module. For every case where
Condition C (critic+repair) is still wrong -- whether it was a baseline
failure that repair didn't fix, or a baseline success repair broke -- read
the critic's output, the repair, and Stage 2's reasoning to classify where
the bottleneck actually sits.

Pure re-analysis of already-collected data (results/mgsm_condition_c_critic_
repair.csv, results/mgsm_condition_comparison_detail.csv) plus the manual
judgment in data/mgsm_repair_error_categories.json -- no new API calls.

The plan's 4 categories turned out not to cover everything found: 2 of 5
cases trace to a confirmed defect in the OFFICIAL dataset itself (Week 5
Task 2), invisible to the critic no matter how good it is, and 1 case is a
repaired answer that's actually MORE correct than a gold label that itself
reflects an uncorrected translation defect. Both are called out as separate
categories rather than forced into "reasoning failure" or "wrong repair".

Usage: python src/mgsm_repair_error_analysis.py
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, RESULTS_DIR

CATEGORIES_PATH = os.path.join(DATA_DIR, "mgsm_repair_error_categories.json")
BOTTLENECK_LABELS = {
    "detection_failure": "Detection -- critic missed a real issue its own checklist covers",
    "repair_introduced_distortion": "Repair -- fix introduced new distortion",
    "reasoning_failure_despite_correct_repair": "Reasoning -- Stage 2 failed despite a correct repair",
    "source_text_defect_undetectable": "Not fixable by this pipeline -- confirmed defect in the official source text",
    "not_a_real_failure_gold_label_defect": "Not a real failure -- repaired answer is correct, gold label is defective",
}


def main():
    comparison = pd.read_csv(os.path.join(RESULTS_DIR, "mgsm_condition_comparison_detail.csv"))
    c_full = pd.read_csv(os.path.join(RESULTS_DIR, "mgsm_condition_c_critic_repair.csv"))
    categories = json.load(open(CATEGORIES_PATH, encoding="utf-8"))["categories"]

    c_wrong = comparison[~comparison.C_correct].copy()
    c_wrong["outcome"] = c_wrong.apply(
        lambda r: "regression (A correct, C wrong)" if r.A_correct else "unrecovered (A and C both wrong)", axis=1)

    rows = []
    for _, r in c_wrong.iterrows():
        key = f"{r.question_id}|{r.language}"
        detail = c_full[(c_full.question_id == r.question_id) & (c_full.language == r.language)].iloc[0]
        cat = categories.get(key, {"bottleneck": "UNCATEGORIZED", "plan_category": "", "note": "needs manual review"})
        rows.append({
            "question_id": r.question_id, "language": r.language, "outcome": r.outcome,
            "gold_answer": detail.gold_answer, "A_baseline_correct": r.A_correct, "B_correct": r.B_correct,
            "critic_sufficient": detail.critic_sufficient, "was_repaired": detail.was_repaired,
            "C_prediction": detail.stage2_prediction, "C_correct": r.C_correct,
            "bottleneck": cat["bottleneck"], "bottleneck_label": BOTTLENECK_LABELS.get(cat["bottleneck"], cat["bottleneck"]),
            "plan_category": cat["plan_category"], "note": cat["note"],
        })

    out_df = pd.DataFrame(rows)
    out = os.path.join(RESULTS_DIR, "mgsm_repair_error_analysis.csv")
    out_df.to_csv(out, index=False)

    print(f"=== {len(out_df)} cases where Condition C is still wrong (of 150 trials) ===\n")
    print(out_df[["question_id", "language", "outcome", "bottleneck"]].to_string(index=False))

    print("\n=== Bottleneck distribution ===")
    for b, n in out_df.bottleneck.value_counts().items():
        print(f"  {BOTTLENECK_LABELS.get(b, b)}: {n}")

    fixable = out_df[~out_df.bottleneck.isin(["source_text_defect_undetectable", "not_a_real_failure_gold_label_defect"])]
    print(f"\nOf {len(out_df)} remaining-wrong cases, {len(fixable)} are attributable to THIS pipeline "
          f"(detection/repair/reasoning); {len(out_df) - len(fixable)} are not fixable or not real failures -- "
          f"confirmed external to the critic+repair mechanism.")
    reasoning_failures = (out_df.bottleneck == "reasoning_failure_despite_correct_repair").sum()
    print(f"Pure reasoning failures (correct repair, Stage 2 still wrong): {reasoning_failures} of {len(out_df)}.")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
