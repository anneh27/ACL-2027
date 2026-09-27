"""
Week 5 Task 5 verification: does adding the missing transitivity/asymmetry
axioms (data/week1_cleaned.json) actually fix spat_04's oracle-symbolic
failure, and does it leave spat_01/02/03/05 correct? Real API calls, 3
samples per item to check stability (the model runs at its default
temperature, so a single sample can be misleading -- see Week 4).

Usage: python src/verify_spatial_axioms.py
"""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, PROJECT_ROOT, RESULTS_DIR
from run_cherokee_pilot_v2 import CLASSIFY_SYSTEM_SYMBOLIC, parse_label
from run_mgsm_trilingual import call_model, load_env

N_SAMPLES = 3


def run_one(job):
    client, model, item, k = job
    prompt = "Premises:\n" + "\n".join(item["premises_formal"]) + f"\n\nQuery: {item['query_formal']}"
    result = call_model(client, model, prompt, system=CLASSIFY_SYSTEM_SYMBOLIC)
    pred, _method = (None, "none") if result["error_message"] else parse_label(result["raw_output"])
    return {
        "id": item["id"], "sample": k, "gold_label": item["label"], "n_axioms": len(item["premises_formal"]),
        "prompt": prompt, "raw_output": result["raw_output"], "prediction": pred,
        "correct": pred == item["label"] if pred else None,
        "error_message": result["error_message"], "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def main():
    load_env(os.path.join(PROJECT_ROOT, ".env"))
    import openai
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=120, max_retries=0)
    model = os.environ["OPENAI_MODEL"]

    with open(os.path.join(DATA_DIR, "week1_cleaned.json"), encoding="utf-8") as f:
        items = [i for i in json.load(f)["items"] if i["reasoning_type"] == "spatial"]

    jobs = [(client, model, item, k) for item in items for k in range(1, N_SAMPLES + 1)]
    print(f"Verifying {len(items)} spatial items x {N_SAMPLES} samples = {len(jobs)} real API calls, model {model}...")
    with ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(run_one, jobs))

    df = pd.DataFrame(records)
    out = os.path.join(RESULTS_DIR, "spatial_axiom_verification.csv")
    df.to_csv(out, index=False)
    with open(os.path.join(RESULTS_DIR, "spatial_axiom_verification_log.jsonl"), "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary = df.groupby("id").agg(n_axioms=("n_axioms", "first"), gold=("gold_label", "first"),
                                    n_correct=("correct", "sum"), n_samples=("correct", "count"))
    print("\n=== Oracle-symbolic (Condition D) with corrected axioms, 3 samples each ===")
    print(summary.to_string())
    all_correct = (summary.n_correct == summary.n_samples).all()
    print(f"\nAll 5 spatial items correct in all {N_SAMPLES} samples: {all_correct}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
