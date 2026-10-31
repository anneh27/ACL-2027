"""
Week of Oct 26 Task 2 scaffolding: builds the blinded sample for the human annotator to
independently judge semantic completeness on -- this script does NOT perform that judgment
itself. Doing so would defeat the point: "blinded review" means a human who doesn't know which
items the pipeline got right or wrong rates the Stage-1 extraction against the source question,
so the resulting completeness label is independent of (and comparable to) whether Stage 2
happened to answer correctly. See results/task2_annotation_guidelines.md for the protocol the
annotator should follow by hand.

Population: the 25 original Condition-B (Stage1->Stage2, schema v1) failures, plus 20 originally-
correct Condition-B items as a comparison sample (fixed seed, reproducible). All 45 use the v1
extraction (results/mgsm_semantic_recovery_full.csv's stage1_raw_output) -- the plan's "25
original errors" are specifically the original two-stage pipeline's errors, not any later
schema/critic variant.

Outputs two files:
  results/task2_blinded_sample.csv       -- for the annotator. Blind IDs only, no correctness
                                             label, no gold answer, no Stage-2 output.
  results/task2_blinded_key_private.csv  -- NOT for the annotator. Maps blind_id back to the
                                             real question_id/language/condition_b_correct/
                                             critic_sufficient (the existing v1 Critic's own
                                             Yes/No judgment, for the later precision/recall
                                             comparison against the human's independent label).

Usage: python src/build_task2_blinded_sample.py
"""
import os
import random
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_DIR

N_CORRECT_SAMPLE = 20
SEED = 20271026  # this week's plan date, used nowhere else in the project -- avoids reusing seed 42 (the trilingual sampling seed) for an unrelated purpose


def main():
    cond_b = pd.read_csv(os.path.join(RESULTS_DIR, "mgsm_semantic_recovery_full.csv"))
    cond_d = pd.read_csv(os.path.join(RESULTS_DIR, "mgsm_condition_c_critic_repair.csv")).set_index(["question_id", "language"])

    failures = cond_b[cond_b["task_semantic_sufficiency"] != True]
    correct = cond_b[cond_b["task_semantic_sufficiency"] == True]
    assert len(failures) == 25, f"expected 25 Condition-B failures, found {len(failures)}"

    rng = random.Random(SEED)
    correct_sample = correct.sample(n=N_CORRECT_SAMPLE, random_state=rng.randint(0, 2**31))

    pool = pd.concat([failures, correct_sample], ignore_index=True)
    order = list(range(len(pool)))
    rng.shuffle(order)  # blind order: failures and correct items interleaved, not grouped

    blind_rows, key_rows = [], []
    for i, idx in enumerate(order):
        row = pool.iloc[idx]
        blind_id = f"B{i + 1:03d}"
        key = (row["question_id"], row["language"])
        critic_sufficient = cond_d.loc[key, "critic_sufficient"] if key in cond_d.index else None

        blind_rows.append({
            "blind_id": blind_id,
            "original_question": row["question"],
            "language": row["language"],
            "stage1_extraction_v1": row["stage1_raw_output"],
            "completeness_goal_target": "",
            "completeness_entities": "",
            "completeness_units": "",
            "completeness_relations": "",
            "completeness_quantifiers_scope": "",
            "completeness_ambiguity": "",
            "overall_judgment": "",  # complete / incomplete
            "annotator_notes": "",
        })
        key_rows.append({
            "blind_id": blind_id, "question_id": row["question_id"], "language": row["language"],
            "condition_b_correct": bool(row["task_semantic_sufficiency"]),
            "critic_sufficient_v1": critic_sufficient,
            "gold_answer": row["gold_answer"], "stage2_prediction": row["stage2_prediction"],
        })

    os.makedirs(RESULTS_DIR, exist_ok=True)
    blind_path = os.path.join(RESULTS_DIR, "task2_blinded_sample.csv")
    key_path = os.path.join(RESULTS_DIR, "task2_blinded_key_private.csv")
    pd.DataFrame(blind_rows).to_csv(blind_path, index=False)
    pd.DataFrame(key_rows).to_csv(key_path, index=False)

    print(f"{len(blind_rows)} items ({len(failures)} Condition-B failures + {len(correct_sample)} originally-correct, seed={SEED})")
    print(f"Wrote {blind_path} (for the annotator -- no correctness label, no gold answer)")
    print(f"Wrote {key_path} (private -- do not share with the annotator before labeling is done)")


if __name__ == "__main__":
    main()
