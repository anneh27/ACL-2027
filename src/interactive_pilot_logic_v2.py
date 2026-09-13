"""
Week 3, Task 3: the four-condition design as REDEFINED this week.

Week 2's Condition C was "Cherokee -> formal logical parse" (semantic
recovery graded as a formula). Week 3's plan redefines Condition C as
"Cherokee -> English translation -> answer" -- semantic recovery graded by
whether reasoning over the model's OWN translation reaches the right
answer. This is a meaningfully different probe (translation quality
folded into an explicit intermediate artifact, then reasoned over in
English) and is implemented fresh here rather than by reusing Week 2's
Condition C.

Conditions A (English Natural) and D (Oracle Symbolic) are unchanged from
Week 2 -- both were already 100%, and neither condition exercises Cherokee
comprehension, so there was nothing to redo there. Condition B (Cherokee
Natural) is also unchanged -- it's a different probe from the new
Condition C (direct answer vs. translate-then-answer) and both are kept,
per the plan's own four-condition table.

Condition C (new): a fresh subagent, again with zero access to this
conversation's English originals or gold labels, translated each Cherokee
item to English and then answered using ONLY its own translation --
verbatim from that subagent's report (agentId a878ac8fa20bcba4c).
"""
import json
import os
import sys
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, RESULTS_DIR
from interactive_pilot_logic import CONDITION_A, CONDITION_B, CONDITION_D, MODEL_LABEL_MAIN, MODEL_LABEL_B

MODEL_LABEL_C_V2 = "claude-sonnet-5-interactive-subagent-blind (agentId a878ac8fa20bcba4c)"
LOG_PATH = os.path.join(RESULTS_DIR, "api_call_log.jsonl")


def log_trial(model, **kw):
    record = {"timestamp": datetime.now(timezone.utc).isoformat(), "model": model, "temperature": None}
    record.update(kw)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# Verbatim from the blind translate-then-answer subagent's report.
CONDITION_C_V2 = {
    "neg_01": {"english_translation": "Lawim does not attend school. Lawim has/attends school.", "answer": "contradicted"},
    "neg_02": {"english_translation": "Asdayi is not married. Asdayi is married.", "answer": "contradicted"},
    "neg_03": {"english_translation": "Nine's car is not upstairs. Nine has his own car.", "answer": "entailed"},
    "neg_04": {"english_translation": "It is not true that the puppy/baby does not cry (i.e., it does cry). The puppy/baby cries.", "answer": "entailed"},
    "neg_05": {"english_translation": "Danyili did not want the candy/fast food. Danyili wanted the candy/fast food.", "answer": "contradicted"},
    "cond_01": {"english_translation": "If Ama starts studying, she will understand/learn it. Ama is starting to study. Ama understands.", "answer": "entailed"},
    "cond_02": {"english_translation": "If it rains, Maya stays at home. It is raining. Maya stays (home).", "answer": "entailed"},
    "cond_03": {"english_translation": "If Noah does his homework, he tries hard. Noah tries hard. Noah did his homework.", "answer": "unknown"},
    "cond_04": {"english_translation": "If Evi smokes cigarettes, he will have trouble breathing every day. Evi does not vomit/get sick every day. Evi smokes.", "answer": "unknown"},
    "cond_05": {"english_translation": "If Liyo plants seeds, plants will grow. Liyo does not plant seeds. Plants do not grow.", "answer": "unknown"},
    "quant_01": {"english_translation": "All teachers are writers. Maria is a teacher. Maria is a writer.", "answer": "entailed"},
    "quant_02": {"english_translation": "No [Gasgini, a group] are students. Bilikisi is Gasgini. Bilikisi is a student.", "answer": "contradicted"},
    "quant_03": {"english_translation": "All teachers are hard workers. Some teachers are singers. Some singers are hard workers.", "answer": "entailed"},
    "quant_04": {"english_translation": "Some dancers/performers are hunters. Some dancers/performers are singers. Some hunters are singers.", "answer": "unknown"},
    "quant_05": {"english_translation": "No doctors are [liars/dishonest people]. Some doctors are writers. Some writers are not [liars/dishonest people].", "answer": "entailed"},
    "spat_01": {"english_translation": "The deer is next to/borders the wolf at the camp. The wolf is next to/borders the food at the camp. The deer is next to/borders the food at the camp.", "answer": "unknown"},
    "spat_02": {"english_translation": "Maya is north of Noah. Noah is north of Lawim. Maya is north of Lawim.", "answer": "entailed"},
    "spat_03": {"english_translation": "Igasdi is behind the store. The store is in front of Igasdi.", "answer": "entailed"},
    "spat_04": {"english_translation": "The bathroom is west of the exit. The office is west of the bathroom. The office is east of the exit.", "answer": "contradicted"},
    "spat_05": {"english_translation": "The red box is above the orange box. The orange box is above the [new/other] box. The [new/other] box is above the red box.", "answer": "contradicted"},
}

# Error attribution per the plan's own principle: "If A is correct, B/C are
# incorrect, and D is correct, the evidence more strongly supports a
# language or semantic-recovery bottleneck. If A and C are correct but D is
# incorrect, the error is more consistent with a pure logical-reasoning
# problem." Filled in per-item below for every item where B or C_v2 erred.
ERROR_ATTRIBUTION = {
    "cond_04": "language/semantic-recovery bottleneck (A correct, D correct, B AND C both wrong -- fails in every Cherokee-involving condition; traced to a dropped negation on the second premise, independently self-flagged as lowest-confidence before any grading)",
    "neg_03": "language/semantic-recovery bottleneck, specific to the explicit-translation method (A correct, D correct, B correct, only C_v2 wrong -- direct Cherokee reasoning got this right but writing out an explicit English translation introduced a predicate error not present in the direct condition)",
    "spat_01": "language/semantic-recovery bottleneck (A correct, D correct, B correct, only C_v2 wrong -- the underlying translation conflates a directional relation (left_of) with a symmetric one (near); reasoning over the explicit 'near' translation correctly refuses to assume transitivity, which is right for 'near' but wrong because 'near' was never the intended relation)",
}


def main():
    with open(os.path.join(DATA_DIR, "week1_cherokee.json"), encoding="utf-8") as f:
        dataset = json.load(f)
    with open(os.path.join(DATA_DIR, "week1_cleaned.json"), encoding="utf-8") as f:
        cleaned = json.load(f)
    reasoning_type_by_id = {it["id"]: it["reasoning_type"] for it in cleaned["items"]}

    rows = []
    for item in dataset["items"]:
        item_id = item["id"]
        gold_label = item["label"]

        pred_c = CONDITION_C_V2[item_id]["answer"]
        log_trial(MODEL_LABEL_C_V2, stage="logic_pilot_v2",
                  input_data={"id": item_id, "condition": "C_v2_cherokee_to_english_then_answer"},
                  prompt=item["cherokee_candidate_translation"],
                  raw_output=json.dumps(CONDITION_C_V2[item_id], ensure_ascii=False),
                  parsed_answer=pred_c, gold_label=gold_label)

        a_correct = CONDITION_A[item_id] == gold_label
        b_correct = CONDITION_B[item_id] == gold_label
        c_correct = pred_c == gold_label
        d_correct = CONDITION_D[item_id] == gold_label

        rows.append({
            "id": item_id, "reasoning_type": reasoning_type_by_id[item_id], "gold_label": gold_label,
            "A_english_correct": a_correct,
            "B_cherokee_direct_correct": b_correct,
            "C_v2_cherokee_to_english_correct": c_correct,
            "D_oracle_symbolic_correct": d_correct,
            "error_attribution": ERROR_ATTRIBUTION.get(item_id, ""),
        })

    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS_DIR, "cherokee_pilot_v2_four_condition.csv")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(out, index=False)

    print("=== Week 3 four-condition design (A / B / C-new / D) ===")
    print(df[["A_english_correct", "B_cherokee_direct_correct", "C_v2_cherokee_to_english_correct", "D_oracle_symbolic_correct"]].mean())
    print(f"\nWrote {len(df)} rows to {out}")
    print("\nItems with any B or C_v2 error, and their attribution:")
    for _, r in df.iterrows():
        if not r["B_cherokee_direct_correct"] or not r["C_v2_cherokee_to_english_correct"]:
            print(f"  {r['id']}: B={r['B_cherokee_direct_correct']} C_v2={r['C_v2_cherokee_to_english_correct']} -> {r['error_attribution']}")


if __name__ == "__main__":
    main()
