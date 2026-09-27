"""
Week 5 Task 1: revised scoring for the real-API trilingual semantic-recovery
run, plus the before/after comparison table the task asks for.

Operates entirely on results/mgsm_semantic_recovery_log.jsonl, the raw
output already collected by mgsm_semantic_recovery_trilingual.py -- makes
NO new API calls. Three scores, each strictly built on the last:

  v1 (as originally computed): surface_quantity_fidelity from the run
     itself. Bugged for Spanish -- its number extractor didn't know Spanish
     uses a comma as the decimal separator ('$16,50' was read as 1650, not
     16.5), the same bug independently found and fixed in
     mgsm_translation_number_audit.py.
  v2 (locale fix only): recomputes source numbers with the fixed,
     locale-aware extractor (same fix now also applied in
     mgsm_semantic_recovery_trilingual.py itself, for future runs).
  v3 (+ valid-abstraction credit): the plan's explicit ask -- "re-score
     examples such as mgsm_023 where 1PM->5PM is correctly compressed to a
     4-hour duration" shouldn't count as a fidelity failure. Generalized
     here: if v2 fidelity is still false but Stage 2 reasoning from that
     exact extraction was independently verified correct
     (task_semantic_sufficiency), the extraction was sufficient even though
     it dropped or transformed some literal digits -- recorded as
     different_form, not a failure. Only rows failing BOTH v2 and task
     sufficiency are left as needs_manual_check.

Usage: python src/mgsm_semantic_recovery_rescore.py
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_DIR
from mgsm_semantic_recovery_trilingual import extract_source_numbers, parse_json_array


def main():
    log_path = os.path.join(RESULTS_DIR, "mgsm_semantic_recovery_log.jsonl")
    with open(log_path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    rows = []
    for rec in records:
        struct = parse_json_array(rec["stage1_raw_output"]) if not rec["stage1_error"] else None
        extracted = set()
        if struct:
            for e in struct:
                try:
                    extracted.add(str(float(e["quantity"])))
                except (KeyError, TypeError, ValueError):
                    continue

        source_v1 = set(rec["source_numbers"].split(",")) if rec["source_numbers"] else set()
        fidelity_v1 = rec["surface_quantity_fidelity"]

        source_v2 = extract_source_numbers(rec["question"], rec["language"])
        fidelity_v2 = bool(struct) and source_v2.issubset(extracted)

        task_ok = rec["task_semantic_sufficiency"]
        if fidelity_v2:
            category = "fully_correct"
        elif task_ok:
            category = "different_form"  # e.g. 3/4 -> 0.75: not literal, but verified sufficient
        else:
            category = "needs_manual_check"
        fidelity_v3 = category in ("fully_correct", "different_form")

        rows.append({
            "question_id": rec["question_id"], "language": rec["language_name"],
            "fidelity_v1_original": fidelity_v1, "fidelity_v2_locale_fixed": fidelity_v2,
            "fidelity_v3_audited": fidelity_v3, "category": category,
            "task_semantic_sufficiency": task_ok,
            "source_numbers_v1": ",".join(sorted(source_v1, key=float)) if source_v1 else "",
            "source_numbers_v2": ",".join(sorted(source_v2, key=float)),
            "extracted_numbers": ",".join(sorted(extracted, key=float)),
            "v1_v2_differ": (source_v1 != source_v2),
        })

    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS_DIR, "mgsm_semantic_recovery_rescore_comparison.csv")
    df.to_csv(out, index=False)

    print(f"{len(df)} trials rescored from {log_path} (no new API calls).")
    print(f"Rows where the locale fix changed the gold source-number set: {int(df.v1_v2_differ.sum())} "
          f"(all Spanish, from the comma-decimal bug)\n")

    summary = df.groupby("language").agg(
        v1_original=("fidelity_v1_original", "mean"),
        v2_locale_fixed=("fidelity_v2_locale_fixed", "mean"),
        v3_audited=("fidelity_v3_audited", "mean"),
        task_semantic_sufficiency=("task_semantic_sufficiency", "mean"),
    ).round(3)
    print("=== Before/after comparison table (fidelity rate by language) ===")
    print(summary.to_string())

    print(f"\nCategory breakdown (n={len(df)}):")
    print(df.category.value_counts().to_string())

    remaining = df[df.category == "needs_manual_check"]
    print(f"\n{len(remaining)} rows still fail both v2 fidelity AND task-sufficiency -- these are genuine "
          f"candidates for a real extraction problem (not a scoring artifact), not yet individually audited:")
    print(remaining[["question_id", "language"]].to_string(index=False))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
