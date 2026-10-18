"""
Week 6 Task 1: taxonomy of MGSM semantic-recovery (task-semantic-sufficiency)
failures. Pure re-analysis of results/mgsm_semantic_recovery_full.csv (Week
5's already-collected 150 trials) plus the manual categorization in
data/mgsm_semantic_failure_categories.json -- no new API calls.

Of the 25 failures, 3 are flagged source_text_defect: a confirmed defect in
the OFFICIAL dataset itself (Week 5 Task 2's translation audit), not
anything lost by the extraction step. They're kept in the taxonomy (the
plan's "Other" category) but called out separately in the summary so they
don't inflate the count of genuine extraction-schema problems.

Usage: python src/mgsm_semantic_failure_taxonomy.py
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, RESULTS_DIR

CATEGORIES_PATH = os.path.join(DATA_DIR, "mgsm_semantic_failure_categories.json")
CATEGORY_LABELS = {
    "target_question_intent_loss": "Target / question intent loss",
    "entity_scope": "Entity scope",
    "aggregation_relation": "Aggregation relation",
    "unit_information": "Unit information",
    "relation_loss": "Relation loss",
    "source_text_defect": "Other (confirmed source-text translation defect, not an extraction failure)",
    "other": "Other",
}


def main():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "mgsm_semantic_recovery_full.csv"))
    fail = df[df.task_semantic_sufficiency == False].copy()  # noqa: E712
    categories = json.load(open(CATEGORIES_PATH, encoding="utf-8"))["categories"]

    rows = []
    for _, r in fail.iterrows():
        key = f"{r.question_id}|{r.language}"
        cat = categories.get(key, {"category": "other", "what_to_fix": "UNCATEGORIZED -- needs manual review"})
        rows.append({
            "question_id": r.question_id,
            "language": r.language_name,
            "original_question": r.question,
            "extracted_representation": r.stage1_raw_output,
            "expected_answer": r.gold_answer,
            "stage2_answer": r.stage2_prediction,
            "failure_category": cat["category"],
            "failure_category_label": CATEGORY_LABELS.get(cat["category"], cat["category"]),
            "what_must_be_added_or_corrected": cat["what_to_fix"],
        })

    out_df = pd.DataFrame(rows)
    out = os.path.join(RESULTS_DIR, "mgsm_semantic_failure_taxonomy.csv")
    out_df.to_csv(out, index=False)

    n_total_failures = len(out_df)
    n_genuine = int((out_df.failure_category != "source_text_defect").sum())
    counts = out_df.failure_category.value_counts()

    print(f"=== Taxonomy of {n_total_failures} semantic-sufficiency failures "
          f"(out of 150 trials, {n_total_failures/150:.1%}) ===\n")
    print(f"{n_total_failures - n_genuine} of {n_total_failures} are source-text translation defects (Task 2), "
          f"not extraction failures -- {n_genuine} are genuine semantic-recovery/representation failures.\n")
    print("Category breakdown (of all 25, including the 3 source-text defects):")
    for cat, n in counts.items():
        print(f"  {CATEGORY_LABELS.get(cat, cat)}: {n} ({n/n_total_failures:.0%} of failures, {n/150:.1%} of all trials)")

    print("\nCategory breakdown (genuine extraction failures only, n=%d):" % n_genuine)
    genuine = out_df[out_df.failure_category != "source_text_defect"]
    for cat, n in genuine.failure_category.value_counts().items():
        print(f"  {CATEGORY_LABELS.get(cat, cat)}: {n} ({n/n_genuine:.0%} of genuine failures)")

    print("\nWrote", out)


if __name__ == "__main__":
    main()
