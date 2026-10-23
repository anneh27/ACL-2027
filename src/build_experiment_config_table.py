"""
Week 7 Task 1: regenerate the unified four-condition comparison table as a
CSV, straight from the already-collected result files -- no new API calls,
no manual annotation. Companion to results/experiment_config.md (which has
the full prompts/params/known-issues narrative this table doesn't carry).

Usage: python src/build_experiment_config_table.py
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_DIR

CONDITIONS = [
    ("A_direct_reasoning", "results/mgsm_trilingual_full.csv", "correct", "src/run_mgsm_trilingual.py"),
    ("B_original_stage1_stage2", "results/mgsm_semantic_recovery_full.csv", "task_semantic_sufficiency", "src/mgsm_semantic_recovery_trilingual.py"),
    ("C_generic_reflection", "results/mgsm_condition_b_reflection.csv", "correct", "src/mgsm_semantic_critic_repair.py (run_condition_b)"),
    ("D_critic_plus_repair", "results/mgsm_condition_c_critic_repair.csv", "correct", "src/mgsm_semantic_critic_repair.py (run_condition_c)"),
]


def main():
    rows = []
    for name, path, col, script in CONDITIONS:
        df = pd.read_csv(path)
        n_correct = int(df[col].sum())
        n_total = len(df)
        rows.append({
            "condition": name, "script": script, "n_correct": n_correct, "n_total": n_total,
            "accuracy": round(n_correct / n_total, 4), "source_file": path,
        })

    out_df = pd.DataFrame(rows)
    out_path = os.path.join(RESULTS_DIR, "experiment_config_comparison_table.csv")
    out_df.to_csv(out_path, index=False)
    print(out_df.to_string(index=False))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
