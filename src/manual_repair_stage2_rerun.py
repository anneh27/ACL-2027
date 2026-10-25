"""
Week 7 Task 2's "run script" (results/manual_semantic_repair_25.csv was produced by hand --
missing_semantics, error_type, and stage1_repaired_json are the user's own manual annotation,
not generated here). This script only does the mechanical part Task 2 asks for after that:
feed each manually-repaired Stage-1 JSON to an isolated Stage 2, using the SAME model and the
same (absent) decoding parameters as every other run in this project, and score the result.

Per-row stage1_original_json is read but intentionally unused -- the repaired JSON is the one
sent to Stage 2, as requested. Four of the 25 rows (mgsm_057 ja, mgsm_086 es, mgsm_139 ja,
mgsm_214 es) have an empty stage1_repaired_json: the annotator's own notes say these are
source-text/translation defects where injecting a repair would mean assuming facts not in the
original problem, so no repair was written and none is run here either.

Usage: python src/manual_repair_stage2_rerun.py
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import PROJECT_ROOT, RESULTS_DIR
from mgsm_schema_v2 import REASON_SYSTEM_V2
from mgsm_scoring import is_correct, parse_prediction
from mgsm_semantic_recovery_trilingual import verify_isolation
from run_mgsm_trilingual import TEMPERATURE_NOTE, call_model, load_env

IN_PATH = os.path.join(RESULTS_DIR, "manual_semantic_repair_25.csv")
OUT_PATH = os.path.join(RESULTS_DIR, "manual_semantic_repair_25_resolved.csv")


def main():
    load_env(os.path.join(PROJECT_ROOT, ".env"))
    import openai
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=120, max_retries=0)
    model = os.environ["OPENAI_MODEL"]  # same model as every other run: gpt-5.6-sol

    df = pd.read_csv(IN_PATH, keep_default_na=False)
    print(f"{len(df)} rows loaded from {IN_PATH}; model={model}, decoding={TEMPERATURE_NOTE}")

    out_rows = []
    for _, row in df.iterrows():
        qid, lang = row["question_id"], row["language"]
        repaired_raw = row["stage1_repaired_json"].strip()
        out = dict(row)

        if not repaired_raw:
            out.update({"stage2_after_raw_output": "", "stage2_after_prediction": None,
                        "correct_after": None, "recovered": None,
                        "note": "no stage1_repaired_json provided (annotator judged this a source-text defect, not an extraction fix)"})
            out_rows.append(out)
            print(f"  {qid}[{lang}]: skipped (no repaired JSON)")
            continue

        try:
            repaired_obj = json.loads(repaired_raw)
        except json.JSONDecodeError as exc:
            out.update({"stage2_after_raw_output": "", "stage2_after_prediction": None,
                        "correct_after": None, "recovered": None,
                        "note": f"stage1_repaired_json did not parse as JSON: {exc}"})
            out_rows.append(out)
            print(f"  {qid}[{lang}]: SKIPPED, repaired JSON malformed: {exc}")
            continue

        stage2_prompt = json.dumps(repaired_obj, ensure_ascii=False)
        verify_isolation(stage2_prompt, REASON_SYSTEM_V2, row["original_question"], qid, lang)  # raises on any leak

        stage2 = call_model(client, model, stage2_prompt, system=REASON_SYSTEM_V2)
        prediction, parse_method = (None, "none") if stage2["error_message"] else parse_prediction(stage2["raw_output"])
        correct_after = (not stage2["error_message"]) and is_correct(prediction, row["gold_answer"])
        recovered = (not bool(row["correct_before"])) and bool(correct_after)

        out.update({
            "stage2_after_raw_output": stage2["raw_output"], "stage2_after_prediction": prediction,
            "stage2_after_parse_method": parse_method, "stage2_after_error": stage2["error_message"],
            "correct_after": correct_after, "recovered": recovered, "note": "",
        })
        out_rows.append(out)
        print(f"  {qid}[{lang}]: prediction={prediction} gold={row['gold_answer']} correct_after={correct_after}")

    out_df = pd.DataFrame(out_rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_df.to_csv(OUT_PATH, index=False)

    scored = out_df[out_df["correct_after"].notna()]
    skipped = out_df[out_df["correct_after"].isna()]
    print(f"\n{len(scored)}/{len(out_df)} rows resolved via Stage 2 (skipped: {len(skipped)}, no repair provided).")
    print(f"Recovered (wrong before -> correct after): {int(scored['recovered'].sum())}/{len(scored)}")
    print(f"Isolation check: passed for all resolved rows (verify_isolation raised on none of them).")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
