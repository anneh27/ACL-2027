"""
Section 9: aggregate raw results into the two tables to bring to the meeting.

Table 1 -- Mini-MGSM: Language x {End-to-End Acc., Semantic Recovery, P(R|S)}
Table 2 -- Cherokee Logic Pilot: Reasoning Type x {English, Cherokee, Cherokee Parse, Oracle Symbolic}

Usage:
    python src/evaluate.py
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_DIR


def table1_mini_mgsm():
    e2e_path = os.path.join(RESULTS_DIR, "mgsm_results.csv")
    probe_path = os.path.join(RESULTS_DIR, "mgsm_semantic_probe.csv")
    if not (os.path.exists(e2e_path) and os.path.exists(probe_path)):
        print("Missing mgsm_results.csv and/or mgsm_semantic_probe.csv -- run run_mgsm.py and semantic_probe.py first.")
        return None

    e2e = pd.read_csv(e2e_path)
    probe = pd.read_csv(probe_path)

    acc = e2e.groupby("language_name")["correct"].mean().rename("end_to_end_accuracy")
    recovery = probe.groupby("language_name")["semantic_recovery_correct"].mean().rename("semantic_recovery_rate")
    cond = (
        probe[probe["semantic_recovery_correct"]]
        .groupby("language_name")["reasoning_correct"]
        .mean()
        .rename("P(reasoning_correct | semantic_recovery_correct)")
    )
    table = pd.concat([acc, recovery, cond], axis=1)
    out_path = os.path.join(RESULTS_DIR, "table1_mini_mgsm.csv")
    table.to_csv(out_path)
    print("=== Table 1: Mini-MGSM ===")
    print(table.round(3))
    print(f"Saved to {out_path}\n")
    return table


def table2_cherokee_pilot():
    path = os.path.join(RESULTS_DIR, "cherokee_pilot.csv")
    if not os.path.exists(path):
        print("Missing cherokee_pilot.csv -- run run_logic_pilot.py first.")
        return None

    df = pd.read_csv(path)
    cols = ["A_english_correct", "B_cherokee_correct", "C_cherokee_parse_correct", "D_oracle_symbolic_correct"]
    names = ["English", "Cherokee", "Cherokee Parse", "Oracle Symbolic"]
    if "C_cherokee_parse_structural_match" in df.columns:
        # Strict grading requires the parse's predicate/entity names to
        # literally match gold; structural match instead checks whether the
        # connective/negation-polarity/quantifier-type/argument-shape agrees,
        # regardless of naming. The gap between these two columns is often
        # the most informative number in this table -- see README.
        cols.append("C_cherokee_parse_structural_match")
        names.append("Cherokee Parse (structural match)")
    table = df.groupby("reasoning_type")[cols].mean()
    table.columns = names
    out_path = os.path.join(RESULTS_DIR, "table2_cherokee_pilot.csv")
    table.to_csv(out_path)
    print("=== Table 2: Cherokee Logic Pilot ===")
    print(table.round(3))
    print(f"Saved to {out_path}\n")
    return table


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    table1_mini_mgsm()
    table2_cherokee_pilot()
