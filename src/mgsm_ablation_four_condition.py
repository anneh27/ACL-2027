"""
Week of Oct 26 Task 1: Schema x Critic ablation summary. Reuses four already-collected,
real-API result sets -- no new calls, no re-scoring, just assembly -- to isolate the
independent contributions of the schema (v1 vs. v2) and the Critic/Repair step:

  v1, no repair   results/mgsm_semantic_recovery_full.csv        (Week 5 Task 1)
  v1 + Critic     results/mgsm_condition_c_critic_repair.csv     (Week 6 Task 2/3)
  v2, no repair   results/mgsm_schema_v2_pipeline_full_no_critic.csv  (this week, --full --no-critic)
  v2 + Critic     results/mgsm_schema_v2_pipeline_full.csv           (this week, --full)

All four run on the same 150 (question, language) pairs (50 MGSM items x en/es/ja).

Usage: python src/mgsm_ablation_four_condition.py
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_DIR

CONDITIONS = [
    ("v1_no_repair", "mgsm_semantic_recovery_full.csv", "question_id", "language", "task_semantic_sufficiency", "stage2_prediction", None),
    ("v1_critic", "mgsm_condition_c_critic_repair.csv", "question_id", "language", "correct", "stage2_prediction", None),
    ("v2_no_repair", "mgsm_schema_v2_pipeline_full_no_critic.csv", "question_id", "language", "correct", "stage2_prediction", "total_tokens"),
    ("v2_critic", "mgsm_schema_v2_pipeline_full.csv", "question_id", "language", "correct", "stage2_prediction", "total_tokens"),
]


def load_condition(name, path, qid_col, lang_col, correct_col, pred_col, tok_col):
    df = pd.read_csv(os.path.join(RESULTS_DIR, path))
    df = df.rename(columns={qid_col: "question_id", lang_col: "language", correct_col: f"{name}_correct", pred_col: f"{name}_prediction"})
    cols = ["question_id", "language", f"{name}_correct", f"{name}_prediction"]
    if tok_col:
        df = df.rename(columns={tok_col: f"{name}_tokens"})
        cols.append(f"{name}_tokens")
    else:
        df[f"{name}_tokens"] = None
        cols.append(f"{name}_tokens")
    return df[cols]


def main():
    merged = None
    for name, path, qid_col, lang_col, correct_col, pred_col, tok_col in CONDITIONS:
        df = load_condition(name, path, qid_col, lang_col, correct_col, pred_col, tok_col)
        merged = df if merged is None else merged.merge(df, on=["question_id", "language"], how="outer")

    out_path = os.path.join(RESULTS_DIR, "mgsm_ablation_per_question.csv")
    merged.to_csv(out_path, index=False)

    summary_rows = []
    for name, *_ in CONDITIONS:
        col = f"{name}_correct"
        n_correct = int(merged[col].astype(bool).sum())
        n_total = len(merged)
        tok_col = f"{name}_tokens"
        has_tokens = merged[tok_col].notna().any()
        summary_rows.append({
            "condition": name, "n_correct": n_correct, "n_total": n_total,
            "accuracy": round(n_correct / n_total, 4),
            "n_calls_per_item": {"v1_no_repair": 2, "v1_critic": "3-4", "v2_no_repair": 2, "v2_critic": "2-4"}[name],
            "avg_tokens_per_item": round(merged[tok_col].mean(), 1) if has_tokens else "not captured in this condition's log (known gap, see results/experiment_config.md)",
        })
    summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(RESULTS_DIR, "mgsm_ablation_summary.csv")
    summary.to_csv(summary_path, index=False)
    print("=== Four-condition ablation summary ===")
    print(summary.to_string(index=False))

    def recovery_regression(base, other):
        base_c, other_c = merged[f"{base}_correct"].astype(bool), merged[f"{other}_correct"].astype(bool)
        base_wrong, base_right = (~base_c).sum(), base_c.sum()
        recovered = int((~base_c & other_c).sum())
        regressed = int((base_c & ~other_c).sum())
        return recovered, int(base_wrong), regressed, int(base_right)

    print("\n=== Recovery/regression, isolating each lever ===")
    pairs = [
        ("v1_no_repair", "v1_critic", "Critic/Repair's effect on top of schema v1"),
        ("v2_no_repair", "v2_critic", "Critic/Repair's effect on top of schema v2"),
        ("v1_no_repair", "v2_no_repair", "Schema v2's effect alone (no Critic in either)"),
        ("v1_critic", "v2_critic", "Schema v2's effect with Critic in both"),
    ]
    rows = []
    for base, other, desc in pairs:
        recovered, base_wrong, regressed, base_right = recovery_regression(base, other)
        print(f"{desc}: recovered {recovered}/{base_wrong} of {base}'s wrong items, "
              f"regressed {regressed}/{base_right} of {base}'s right items "
              f"(net {recovered - regressed:+d})")
        rows.append({"base": base, "other": other, "description": desc, "recovered": recovered,
                      "base_wrong": base_wrong, "regressed": regressed, "base_right": base_right,
                      "net_change": recovered - regressed})
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "mgsm_ablation_recovery_regression.csv"), index=False)

    print(f"\nWrote {out_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {os.path.join(RESULTS_DIR, 'mgsm_ablation_recovery_regression.csv')}")


if __name__ == "__main__":
    main()
