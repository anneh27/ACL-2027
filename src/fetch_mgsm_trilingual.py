"""
Week 4 P2: fetch the OFFICIAL MGSM English/Spanish/Japanese test sets and
select 50 IDs shared across all three, per the plan's explicit instruction
("do not have Claude translate the questions"). Uses HuggingFace's public
datasets-server REST API -- no API key needed, this is public dataset
access, not model inference. All three language configs of juletxara/mgsm
are aligned by row_idx (same underlying 250 GSM8K-derived problems,
professionally translated), so any 50 shared row_idx values work; sampled
with the same seed (42) already used elsewhere in this project for
consistency.

Usage:
    python src/fetch_mgsm_trilingual.py
"""
import json
import os
import random
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR

BASE_URL = "https://datasets-server.huggingface.co/rows"
DATASET = "juletxara/mgsm"
LANGS = {"en": "English", "es": "Spanish", "ja": "Japanese"}
N_ITEMS = 50
SAMPLING_SEED = 42
PAGE_SIZE = 100


def fetch_all_rows(config):
    rows = []
    offset = 0
    while True:
        url = f"{BASE_URL}?dataset={DATASET.replace('/', '%2F')}&config={config}&split=test&offset={offset}&length={PAGE_SIZE}"
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read())
        page_rows = data["rows"]
        rows.extend(page_rows)
        total = data["num_rows_total"]
        offset += len(page_rows)
        if offset >= total or not page_rows:
            break
        time.sleep(0.2)
    return rows


def main():
    print("Fetching official MGSM test sets from HuggingFace datasets-server...")
    by_lang = {}
    for code, name in LANGS.items():
        rows = fetch_all_rows(code)
        by_lang[code] = {r["row_idx"]: r["row"] for r in rows}
        print(f"  {name} ({code}): {len(rows)} rows fetched")

    shared_indices = set(by_lang["en"]) & set(by_lang["es"]) & set(by_lang["ja"])
    print(f"Row indices present in all three configs: {len(shared_indices)}")

    rng = random.Random(SAMPLING_SEED)
    selected = sorted(rng.sample(sorted(shared_indices), N_ITEMS))

    items = []
    for idx in selected:
        en_row = by_lang["en"][idx]
        item = {
            "id": f"mgsm_trilingual_{idx:03d}",
            "row_idx": idx,
            "gold_answer": en_row["answer_number"],
            "questions": {code: by_lang[code][idx]["question"] for code in LANGS},
        }
        # sanity check: gold answer must match across all three language configs
        for code in LANGS:
            assert by_lang[code][idx]["answer_number"] == item["gold_answer"], (
                f"Gold-answer mismatch at row {idx} for {code}: "
                f"{by_lang[code][idx]['answer_number']} != {item['gold_answer']}"
            )
        items.append(item)

    out = {
        "meta": {
            "source": "juletxara/mgsm (HuggingFace), official test split, all three language "
                       "configs fetched directly -- no Claude-generated translation involved.",
            "languages": LANGS,
            "n_items": N_ITEMS,
            "sampling_seed": SAMPLING_SEED,
            "note": "Row indices are shared across all MGSM language configs by construction "
                    "(each config is a professional translation of the same 250 GSM8K-derived "
                    "problems); gold_answer verified identical across en/es/ja for every item.",
        },
        "items": items,
    }
    out_path = os.path.join(DATA_DIR, "mgsm_trilingual_sample.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {len(items)} items to {out_path}")


if __name__ == "__main__":
    main()
