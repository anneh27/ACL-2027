"""
Week 4 P1 follow-up: is each wrong Cherokee-pilot answer stable?

The model only supports its default temperature, so a single wrong sample
might be noise. For every (item, condition) among B, C, D that was wrong in
results/cherokee_pilot_v2_full.csv (selected from the data), re-run just that
condition N more times with the same prompts:
  B: the direct Cherokee call
  C: the full chain -- a fresh back-translation call, then a fresh answer call
  D: the oracle-symbolic call

Outputs: results/cherokee_pilot_v2_stability.csv, ..._stability_log.jsonl
Usage:   python src/cherokee_pilot_v2_stability.py [--n 3]
"""
import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, PROJECT_ROOT, RESULTS_DIR
from run_cherokee_pilot_v2 import (CLASSIFY_SYSTEM_NL, CLASSIFY_SYSTEM_SYMBOLIC, TRANSLATE_PROMPT,
                                   make_record)
from run_mgsm_trilingual import call_model, load_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=3)
    args = parser.parse_args()

    load_env(os.path.join(PROJECT_ROOT, ".env"))
    import openai
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=120, max_retries=0)
    model = os.environ["OPENAI_MODEL"]

    full = pd.read_csv(os.path.join(RESULTS_DIR, "cherokee_pilot_v2_full.csv"))
    with open(os.path.join(DATA_DIR, "week1_cleaned.json"), encoding="utf-8") as f:
        cleaned = {i["id"]: i for i in json.load(f)["items"]}

    records, summary = [], []
    for _, row in full.iterrows():
        item = dict(cleaned[row["id"]])
        item["id"] = row["id"]
        gold = row["gold_label"]
        chr_text = row["cherokee_text"]
        sym_prompt = "Premises:\n" + "\n".join(item["premises_formal"]) + f"\n\nQuery: {item['query_formal']}"

        for cond in ("B", "C", "D"):
            if row[f"{cond}_correct"] == True:  # noqa: E712
                continue
            preds, translations = [], []
            for k in range(1, args.n + 1):
                if cond == "B":
                    rec = make_record(item, f"B_resample_{k}", CLASSIFY_SYSTEM_NL, chr_text,
                                      call_model(client, model, chr_text, system=CLASSIFY_SYSTEM_NL), gold, model, True)
                    records.append(rec)
                elif cond == "D":
                    rec = make_record(item, f"D_resample_{k}", CLASSIFY_SYSTEM_SYMBOLIC, sym_prompt,
                                      call_model(client, model, sym_prompt, system=CLASSIFY_SYSTEM_SYMBOLIC), gold, model, True)
                    records.append(rec)
                else:
                    tp = TRANSLATE_PROMPT.format(text=chr_text)
                    c1 = make_record(item, f"C1_resample_{k}", None, tp, call_model(client, model, tp), gold, model, False)
                    records.append(c1)
                    translations.append(c1["raw_output"].strip())
                    rec = make_record(item, f"C2_resample_{k}", CLASSIFY_SYSTEM_NL, c1["raw_output"].strip(),
                                      call_model(client, model, c1["raw_output"].strip(), system=CLASSIFY_SYSTEM_NL),
                                      gold, model, True)
                    records.append(rec)
                preds.append(rec["parsed_prediction"])
            summary.append({
                "id": row["id"], "condition": cond, "gold_label": gold,
                "original_prediction": row[f"{cond}_prediction"],
                "resample_predictions": " | ".join(str(p) for p in preds),
                "n_resamples_wrong": sum(1 for p in preds if str(p).lower() != str(gold).lower()),
                "n_resamples": len(preds),
                "resample_backtranslations": " || ".join(translations),
            })
            print(f"  {row['id']} {cond}: original={row[f'{cond}_prediction']} resamples={preds}", flush=True)

    pd.DataFrame(summary).to_csv(os.path.join(RESULTS_DIR, "cherokee_pilot_v2_stability.csv"), index=False)
    with open(os.path.join(RESULTS_DIR, "cherokee_pilot_v2_stability_log.jsonl"), "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("done")


if __name__ == "__main__":
    main()
