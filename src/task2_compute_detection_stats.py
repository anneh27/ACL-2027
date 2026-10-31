"""
Week of Oct 26 Task 2: once the annotator has filled in results/task2_blinded_sample.csv by
hand (see results/task2_annotation_guidelines.md), this computes what the plan asks for --
error-category statistics and the existing v1 Critic's detection precision/recall against the
human's independent completeness labels. This script performs no judgment itself; it only
aggregates judgments already made by hand.

"Critic detection precision/recall": the v1 Critic (src/mgsm_semantic_critic_repair.py,
CRITIC_SYSTEM) already produces its own Sufficient: Yes/No call for every item (not blind -- it
sees the question and extraction together). Treating the human's blinded overall_judgment as
ground truth, this scores how well the Critic's Yes/No agrees with it:
  TP = human says incomplete, Critic says insufficient (correctly flagged)
  FP = human says complete,   Critic says insufficient (over-flagged)
  FN = human says incomplete, Critic says sufficient   (missed)
  TN = human says complete,   Critic says sufficient   (correctly passed)
precision = TP/(TP+FP), recall = TP/(TP+FN). Per the plan: "label exploratory if the sample is
insufficient" -- this prints a size warning and tags the output as exploratory below a fixed
minimum count per confusion-matrix cell.

Usage: python src/task2_compute_detection_stats.py
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_DIR

MIN_CELL_COUNT_FOR_NON_EXPLORATORY = 5
COMPLETENESS_DIMS = [
    "completeness_goal_target", "completeness_entities", "completeness_units",
    "completeness_relations", "completeness_quantifiers_scope",
]


def main():
    blind_path = os.path.join(RESULTS_DIR, "task2_blinded_sample.csv")
    key_path = os.path.join(RESULTS_DIR, "task2_blinded_key_private.csv")
    blind = pd.read_csv(blind_path, keep_default_na=False)
    key = pd.read_csv(key_path, keep_default_na=False).set_index("blind_id")

    labeled = blind[blind["overall_judgment"].isin(["complete", "incomplete"])]
    unlabeled = blind[~blind["overall_judgment"].isin(["complete", "incomplete"])]
    if len(unlabeled):
        print(f"WARNING: {len(unlabeled)}/{len(blind)} rows have no overall_judgment yet "
              f"(expected 'complete' or 'incomplete') -- excluded from all stats below. "
              f"Fill in results/task2_blinded_sample.csv per results/task2_annotation_guidelines.md.")
    if len(labeled) == 0:
        print("No labeled rows yet -- nothing to compute.")
        return

    merged = labeled.copy()
    for col in ["condition_b_correct", "critic_sufficient_v1"]:
        merged[col] = merged["blind_id"].map(key[col])

    print(f"\n=== Error-category statistics (n={len(merged)} labeled) ===")
    for dim in COMPLETENESS_DIMS:
        counts = merged[dim].value_counts()
        print(f"{dim}: {dict(counts)}")
    print(f"completeness_ambiguity: {dict(merged['completeness_ambiguity'].value_counts())}")
    print(f"overall_judgment: {dict(merged['overall_judgment'].value_counts())}")

    print(f"\nBy population (condition_b_correct, not shown to annotator while labeling):")
    for group_val, group_df in merged.groupby("condition_b_correct"):
        label = "originally correct" if group_val else "original failure"
        print(f"  {label} (n={len(group_df)}): overall_judgment = {dict(group_df['overall_judgment'].value_counts())}")

    human_incomplete = merged["overall_judgment"] == "incomplete"
    critic_insufficient = merged["critic_sufficient_v1"].astype(str) == "False"
    tp = int((human_incomplete & critic_insufficient).sum())
    fp = int((~human_incomplete & critic_insufficient).sum())
    fn = int((human_incomplete & ~critic_insufficient).sum())
    tn = int((~human_incomplete & ~critic_insufficient).sum())
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    exploratory = min(tp, fp, fn, tn) < MIN_CELL_COUNT_FOR_NON_EXPLORATORY

    print(f"\n=== v1 Critic detection precision/recall vs. human blinded label (n={len(merged)}) ===")
    print(f"TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"Precision: {precision:.3f}" if precision == precision else "Precision: undefined (no positive predictions)")
    print(f"Recall: {recall:.3f}" if recall == recall else "Recall: undefined (no actual positives)")
    if exploratory:
        print(f"LABEL: EXPLORATORY -- at least one confusion-matrix cell has fewer than "
              f"{MIN_CELL_COUNT_FOR_NON_EXPLORATORY} items; do not treat this precision/recall as a stable estimate.")

    out_path = os.path.join(RESULTS_DIR, "task2_detection_stats.csv")
    pd.DataFrame([{
        "n_labeled": len(merged), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "exploratory": exploratory,
    }]).to_csv(out_path, index=False)
    merged.to_csv(os.path.join(RESULTS_DIR, "task2_blinded_sample_scored.csv"), index=False)
    print(f"\nWrote {out_path}")
    print(f"Wrote {os.path.join(RESULTS_DIR, 'task2_blinded_sample_scored.csv')} (labeled sample + ground truth, for review)")


if __name__ == "__main__":
    main()
