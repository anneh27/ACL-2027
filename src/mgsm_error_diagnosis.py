"""
Week 4 P3: error diagnosis for the three-language MGSM baseline.

For every trial in results/mgsm_trilingual_full.csv that was answered
incorrectly (selected from the data, not hard-coded), run:

  B-resample : the same target-language prompt, N more times. The model runs
               at its default temperature (explicit temperature=0 is rejected),
               so one wrong sample could be sampling noise; repeats show
               whether the error is stable.
  C          : back-translate the target-language question into English (a
               separate real API call, told only to translate, not solve),
               then answer from that back-translation with the standard prompt.

Conditions A (official English, direct) and D (official English as oracle --
identical to A for MGSM, since the official English IS the oracle) already
exist in the main run and are joined in for comparison.

The back-translation is produced by the same GPT model, not by Claude, which
keeps the whole diagnostic inside the independent API model the plan wants.

Outputs: results/mgsm_diagnosis.csv, results/mgsm_diagnosis_log.jsonl
Usage:   python src/mgsm_error_diagnosis.py [--resamples 3]
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, PROJECT_ROOT, RESULTS_DIR
from mgsm_scoring import is_correct, parse_prediction
from run_mgsm_trilingual import PROMPT_TEMPLATE, TEMPERATURE_NOTE, call_model, load_env

BACKTRANSLATE_PROMPT = (
    "Translate the following math problem into English. Output only the English translation. "
    "Do not solve it and do not add any explanation.\n\n{question}"
)


def record(base, condition, prompt, result, gold, **extra):
    raw = result["raw_output"]
    if result["error_message"]:
        pred, method, correct = None, "none", None
    else:
        pred, method = parse_prediction(raw)
        correct = is_correct(pred, gold)
    rec = dict(base)
    rec.update({
        "condition": condition, "prompt": prompt, "raw_output": raw,
        "parsed_prediction": pred, "parse_method": method, "correct": correct,
        "model_returned": result.get("model_returned", ""), "temperature": TEMPERATURE_NOTE,
        "timestamp": datetime.now(timezone.utc).isoformat(), "error_message": result["error_message"],
    })
    rec.update(extra)
    return rec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resamples", type=int, default=3)
    args = parser.parse_args()

    load_env(os.path.join(PROJECT_ROOT, ".env"))
    import openai
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=120, max_retries=0)
    model = os.environ["OPENAI_MODEL"]

    main_df = pd.read_csv(os.path.join(RESULTS_DIR, "mgsm_trilingual_full.csv"))
    with open(os.path.join(DATA_DIR, "mgsm_trilingual_sample.json"), encoding="utf-8") as f:
        items = {i["id"]: i for i in json.load(f)["items"]}

    wrong = main_df[main_df["correct"] == False]  # noqa: E712
    print(f"{len(wrong)} incorrect trials selected from results/mgsm_trilingual_full.csv")

    records = []
    for _, row in wrong.iterrows():
        qid, lang, gold = row["question_id"], row["language"], row["gold_answer"]
        question = items[qid]["questions"][lang]
        base = {"question_id": qid, "language": lang, "gold_answer": gold, "model": model}
        prompt = PROMPT_TEMPLATE.format(question=question)

        for k in range(1, args.resamples + 1):
            res = call_model(client, model, prompt)
            records.append(record(base, f"B_resample_{k}", prompt, res, gold))

        bt_prompt = BACKTRANSLATE_PROMPT.format(question=question)
        bt = call_model(client, model, bt_prompt)
        records.append(record(base, "C_step1_backtranslation", bt_prompt, bt, gold))
        backtranslation = bt["raw_output"].strip()
        if backtranslation:
            ans_prompt = PROMPT_TEMPLATE.format(question=backtranslation)
            ans = call_model(client, model, ans_prompt)
            records.append(record(base, "C_step2_answer_from_backtranslation", ans_prompt, ans, gold,
                                  backtranslation=backtranslation))
        print(f"  done {qid} [{lang}]")

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(RESULTS_DIR, "mgsm_diagnosis.csv"), index=False)
    with open(os.path.join(RESULTS_DIR, "mgsm_diagnosis_log.jsonl"), "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    print("\n=== Per-item summary ===")
    for (qid, lang), g in df.groupby(["question_id", "language"], sort=False):
        en = main_df[(main_df.question_id == qid) & (main_df.language == "en")].iloc[0]
        orig = main_df[(main_df.question_id == qid) & (main_df.language == lang)].iloc[0]
        res = g[g.condition.str.startswith("B_resample")]
        c2 = g[g.condition == "C_step2_answer_from_backtranslation"]
        print(f"{qid} [{lang}] gold={g.gold_answer.iloc[0]}  A(en)={en.parsed_prediction}  "
              f"B(original)={orig.parsed_prediction}  B(resamples)={list(res.parsed_prediction)}  "
              f"C={list(c2.parsed_prediction)}")


if __name__ == "__main__":
    main()
