"""
Week 3, Task 4 (logic half): the 20 new logic items (De Morgan's laws,
biconditionals, existential-import fallacy, quantifier-scope-order fallacy,
composition/division, hasty generalization, and one deliberate
circular-but-valid control) run through Conditions A/B/C-new/D.

A and D answered directly by me (matches gold in every case -- same
derivation used to build the dataset, not re-guessed). B (direct Cherokee
classification) and C_v2 (Cherokee -> English -> answer) are verbatim from
two more fresh, blind subagents (zero access to English originals/gold),
agentId aedecaacc87e7c061 and abcce4b4208fde175 respectively.

This batch was designed to be harder than the original 20 and to cover a
named "fallacious reasoning forms" category the Week 3 plan calls for
explicitly. Unlike the harder MGSM batch (still 100/100%), this batch DOES
break the ceiling: B=80%, C_v2=85%, vs. 95%/85% on the original 20-item set,
which is exactly the point of Task 4.
"""
import json
import os
import sys
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, RESULTS_DIR

MODEL_LABEL_MAIN = "claude-sonnet-5-interactive"
MODEL_LABEL_B = "claude-sonnet-5-interactive-subagent-blind (agentId aedecaacc87e7c061)"
MODEL_LABEL_C = "claude-sonnet-5-interactive-subagent-blind (agentId abcce4b4208fde175)"
LOG_PATH = os.path.join(RESULTS_DIR, "api_call_log.jsonl")

# Condition A / D: my own direct re-derivation from English / formal notation.
# Matches every gold label in week3_new_logic_items.json (same classical-logic
# analysis used to author the dataset).
CONDITION_A = CONDITION_D = {
    "neg2_01": "entailed", "neg2_02": "entailed", "neg2_03": "unknown", "neg2_04": "entailed", "neg2_05": "entailed",
    "cond2_01": "entailed", "cond2_02": "entailed", "cond2_03": "unknown", "cond2_04": "entailed", "cond2_05": "entailed",
    "quant2_01": "unknown", "quant2_02": "unknown", "quant2_03": "entailed", "quant2_04": "unknown", "quant2_05": "entailed",
    "fall_01": "unknown", "fall_02": "unknown", "fall_03": "unknown", "fall_04": "entailed", "fall_05": "unknown",
}

CONDITION_B = {
    "neg2_01": "unknown", "neg2_02": "unknown", "neg2_03": "unknown", "neg2_04": "entailed", "neg2_05": "entailed",
    "cond2_01": "unknown", "cond2_02": "entailed", "cond2_03": "unknown", "cond2_04": "entailed", "cond2_05": "entailed",
    "quant2_01": "unknown", "quant2_02": "entailed", "quant2_03": "entailed", "quant2_04": "unknown", "quant2_05": "entailed",
    "fall_01": "unknown", "fall_02": "unknown", "fall_03": "unknown", "fall_04": "entailed", "fall_05": "unknown",
}

CONDITION_C_V2 = {
    "neg2_01": "unknown", "neg2_02": "entailed", "neg2_03": "unknown", "neg2_04": "entailed", "neg2_05": "entailed",
    "cond2_01": "entailed", "cond2_02": "unknown", "cond2_03": "unknown", "cond2_04": "entailed", "cond2_05": "entailed",
    "quant2_01": "unknown", "quant2_02": "entailed", "quant2_03": "entailed", "quant2_04": "unknown", "quant2_05": "entailed",
    "fall_01": "unknown", "fall_02": "unknown", "fall_03": "unknown", "fall_04": "entailed", "fall_05": "unknown",
}

ERROR_ATTRIBUTION = {
    "neg2_01": "language/semantic-recovery bottleneck: fails in BOTH B and C_v2. C_v2's own translation shows the query rendered with 'and' ('Elisi is not at home, and also Bobby is not at home') where the intended connective was 'or' -- my Cherokee 'or' (a-priori flagged 'outside my confident vocabulary') was not conveyed, collapsing this item into the superficially similar but logically different neg2_02 pattern.",
    "neg2_02": "language/semantic-recovery bottleneck, but only in Condition B: direct classification conflated this with neg2_01's pattern (grouped both under the same, wrong analysis in its own notes), while C_v2's explicit translation step correctly separated 'neither...nor' from the other item and got it right. A case where writing out the translation explicitly HELPED rather than hurt.",
    "cond2_01": "language/semantic-recovery bottleneck, but only in Condition B: my Cherokee attempt at a biconditional ('if and only if', a-priori flagged very low confidence) was read with antecedent/consequent apparently swapped, producing an affirming-the-consequent misreading. C_v2 happened to preserve the right antecedent/consequent assignment even while also losing the 'only if' component, so modus ponens still applied correctly there.",
    "cond2_02": "language/semantic-recovery bottleneck, but only in Condition C_v2 this time (reverse of cond2_01): its own translation explicitly drops 'if and only if' down to a plain 'if', turning a valid biconditional inference into what LOOKS like denying-the-antecedent -- a real fallacy for a plain conditional, correctly flagged as such by the reasoner, but wrong here because the biconditional's extra 'only if' direction was lost in translation. Direct classification (B) got this one right.",
    "quant2_02": "NOT clearly a translation or semantic-recovery issue -- flagged as a likely logical-CONVENTION difference instead. Both blind agents independently answered 'entailed' citing 'subalternation' (All A are B implies Some A are B), a named, historically standard Aristotelian inference rule. My gold label ('unknown') instead follows modern predicate logic, which does not assume a universal statement's subject class is nonempty. Both agents' own English translations were accurate, so this isn't a translation failure -- it's a case where the item's gold label depends on which of two legitimate logical traditions is assumed, which the item didn't specify. This is a dataset-design caveat on my part, not necessarily a genuine model deficiency.",
}


def main():
    with open(os.path.join(DATA_DIR, "week3_new_logic_items.json"), encoding="utf-8") as f:
        cleaned = json.load(f)
    with open(os.path.join(DATA_DIR, "week3_logic_cherokee.json"), encoding="utf-8") as f:
        cherokee = json.load(f)
    reasoning_type_by_id = {it["id"]: it["reasoning_type"] for it in cleaned["items"]}
    cherokee_text_by_id = {it["id"]: it["cherokee_candidate_translation"] for it in cherokee["items"]}

    def log_trial(model, **kw):
        record = {"timestamp": datetime.now(timezone.utc).isoformat(), "model": model, "temperature": None}
        record.update(kw)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    rows = []
    for item in cleaned["items"]:
        item_id = item["id"]
        gold_label = item["label"]

        for model_label, cond_name, preds in [
            (MODEL_LABEL_MAIN, "A_english_natural", CONDITION_A),
            (MODEL_LABEL_B, "B_cherokee_direct", CONDITION_B),
            (MODEL_LABEL_C, "C_v2_cherokee_to_english", CONDITION_C_V2),
            (MODEL_LABEL_MAIN, "D_oracle_symbolic", CONDITION_D),
        ]:
            pred = preds[item_id]
            log_trial(model_label, stage="logic_pilot_harder_batch",
                      input_data={"id": item_id, "condition": cond_name},
                      prompt=cherokee_text_by_id[item_id] if "cherokee" in cond_name else "",
                      raw_output=pred, parsed_answer=pred, gold_label=gold_label)

        rows.append({
            "id": item_id, "reasoning_type": reasoning_type_by_id[item_id], "gold_label": gold_label,
            "A_english_correct": CONDITION_A[item_id] == gold_label,
            "B_cherokee_direct_correct": CONDITION_B[item_id] == gold_label,
            "C_v2_cherokee_to_english_correct": CONDITION_C_V2[item_id] == gold_label,
            "D_oracle_symbolic_correct": CONDITION_D[item_id] == gold_label,
            "error_attribution": ERROR_ATTRIBUTION.get(item_id, ""),
        })

    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS_DIR, "logic_pilot_harder_batch_four_condition.csv")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(out, index=False)

    print("=== Week 3 harder logic batch (A / B / C-new / D) ===")
    print(df[["A_english_correct", "B_cherokee_direct_correct", "C_v2_cherokee_to_english_correct", "D_oracle_symbolic_correct"]].mean())
    print(f"\nWrote {len(df)} rows to {out}")
    print("\nBy reasoning type:")
    print(df.groupby("reasoning_type")[["A_english_correct", "B_cherokee_direct_correct", "C_v2_cherokee_to_english_correct", "D_oracle_symbolic_correct"]].mean())
    print("\nItems with any B or C_v2 error, and their attribution:")
    for _, r in df.iterrows():
        if not r["B_cherokee_direct_correct"] or not r["C_v2_cherokee_to_english_correct"]:
            print(f"\n  {r['id']}: B={r['B_cherokee_direct_correct']} C_v2={r['C_v2_cherokee_to_english_correct']}")
            print(f"    {r['error_attribution']}")


if __name__ == "__main__":
    main()
