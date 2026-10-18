"""
Week 6 Task 3: compare Baseline (A) vs. Generic reflection (B) vs. Semantic
Recovery Critic (C) on the same 150 MGSM trials. Pure re-analysis of
already-collected data (Week 5's log + mgsm_semantic_critic_repair.py's
output) -- no new API calls.

Metrics, all per the plan:
  - extraction fidelity: A's is Week 5's surface_quantity_fidelity (B/C
    reuse A's extraction as their starting point, so this number is
    identical across conditions by construction and reported once).
  - semantic sufficiency / Stage-2 accuracy / final task accuracy: these
    collapse into one number in this pipeline (Stage 2's answer IS the
    final answer), reported as "accuracy" for each condition.
  - recovery rate: among A's 25 failures, what % are correct under B / C?
  - regression rate: among A's 125 successes, what % become incorrect
    under B / C?

Usage: python src/mgsm_compare_conditions.py [--tag smoke]
"""
import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, RESULTS_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    suffix = f"_{args.tag}" if args.tag else ""

    baseline_recs = [json.loads(l) for l in open(os.path.join(RESULTS_DIR, "mgsm_semantic_recovery_log.jsonl"), encoding="utf-8")]
    a = {(r["question_id"], r["language"]): bool(r["task_semantic_sufficiency"]) for r in baseline_recs}
    a_fidelity = {(r["question_id"], r["language"]): bool(r["surface_quantity_fidelity"]) for r in baseline_recs}

    b_df = pd.read_csv(os.path.join(RESULTS_DIR, f"mgsm_condition_b_reflection{suffix}.csv"))
    c_df = pd.read_csv(os.path.join(RESULTS_DIR, f"mgsm_condition_c_critic_repair{suffix}.csv"))
    # pd.isna(correct) marks a job_error row (correct was never computed, not "wrong") -- excluded
    # entirely rather than silently counted as False, which would understate accuracy
    b_errors = {(r.question_id, r.language) for r in b_df.itertuples() if pd.isna(r.correct)}
    c_errors = {(r.question_id, r.language) for r in c_df.itertuples() if pd.isna(r.correct)}
    b = {(r.question_id, r.language): bool(r.correct) for r in b_df.itertuples() if not pd.isna(r.correct)}
    c = {(r.question_id, r.language): bool(r.correct) for r in c_df.itertuples() if not pd.isna(r.correct)}
    if b_errors or c_errors:
        print(f"Excluding job errors from the comparison (not counted as wrong): "
              f"{len(b_errors)} in B, {len(c_errors)} in C -- {sorted(b_errors | c_errors)}\n")

    keys = sorted(set(b.keys()) & set(c.keys()))  # only trials where BOTH B and C succeeded are comparable
    excluded_asymmetric = (set(b.keys()) ^ set(c.keys()))
    if excluded_asymmetric:
        print(f"WARNING: {len(excluded_asymmetric)} trials succeeded in only one of B/C (job error in the "
              f"other) -- excluded from the paired comparison: {sorted(excluded_asymmetric)}\n")
    missing_a = [k for k in keys if k not in a]
    if missing_a:
        print(f"WARNING: {len(missing_a)} trials in B/C have no baseline (A) record -- excluded from recovery/regression rates: {missing_a[:5]}")
    keys = [k for k in keys if k in a]

    rows = []
    for k in keys:
        rows.append({"question_id": k[0], "language": k[1], "A_correct": a[k], "B_correct": b[k], "C_correct": c[k],
                     "A_fidelity": a_fidelity[k]})
    df = pd.DataFrame(rows)

    def recovery_rate(correct_col):
        baseline_fail = df[~df.A_correct]
        if len(baseline_fail) == 0:
            return None, 0
        return baseline_fail[correct_col].mean(), len(baseline_fail)

    def regression_rate(correct_col):
        baseline_ok = df[df.A_correct]
        if len(baseline_ok) == 0:
            return None, 0
        return (~baseline_ok[correct_col]).mean(), len(baseline_ok)

    summary_rows = []
    for name, col in [("A_Baseline", "A_correct"), ("B_Generic_reflection", "B_correct"), ("C_Semantic_Recovery_Critic", "C_correct")]:
        rec, n_fail = recovery_rate(col) if col != "A_correct" else (None, int((~df.A_correct).sum()))
        reg, n_ok = regression_rate(col) if col != "A_correct" else (None, int(df.A_correct.sum()))
        summary_rows.append({
            "method": name, "n": len(df),
            "extraction_fidelity": df.A_fidelity.mean(),  # identical across conditions by construction
            "accuracy": df[col].mean(),
            "recovery_rate": rec, "recovery_rate_n": n_fail,
            "regression_rate": reg, "regression_rate_n": n_ok,
        })
    summary = pd.DataFrame(summary_rows).set_index("method").round(4)

    out_detail = os.path.join(RESULTS_DIR, f"mgsm_condition_comparison_detail{suffix}.csv")
    out_summary = os.path.join(RESULTS_DIR, f"mgsm_condition_comparison_summary{suffix}.csv")
    df.to_csv(out_detail, index=False)
    summary.to_csv(out_summary)

    print(f"=== Condition comparison, n={len(df)} trials ===\n")
    print(summary.to_string())

    print("\n=== Target result table (plan's format) ===")
    print(f"{'Method':<28}{'Semantic Sufficiency':<22}{'Recovery Rate':<16}{'Regression Rate':<18}{'Final Accuracy'}")
    for name, row in summary.iterrows():
        rec = f"{row.recovery_rate:.1%}" if pd.notna(row.recovery_rate) else "--"
        reg = f"{row.regression_rate:.1%}" if pd.notna(row.regression_rate) else "--"
        print(f"{name:<28}{row.accuracy:<22.1%}{rec:<16}{reg:<18}{row.accuracy:.1%}")

    # cross-tab: which specific trials did C fix that B didn't, and vice versa
    fixed_by_c_not_b = df[(~df.A_correct) & df.C_correct & ~df.B_correct]
    fixed_by_b_not_c = df[(~df.A_correct) & df.B_correct & ~df.C_correct]
    broken_by_c_only = df[df.A_correct & ~df.C_correct & df.B_correct]
    broken_by_b_only = df[df.A_correct & ~df.B_correct & df.C_correct]
    print(f"\nBaseline failures fixed by C but not B: {len(fixed_by_c_not_b)}")
    print(f"Baseline failures fixed by B but not C: {len(fixed_by_b_not_c)}")
    print(f"Baseline successes broken by C but not B: {len(broken_by_c_only)}")
    print(f"Baseline successes broken by B but not C: {len(broken_by_b_only)}")
    print(f"\nWrote {out_detail}\n       {out_summary}")


if __name__ == "__main__":
    main()
