"""
Week 5 Task 1 (real-API port) + Task 3 (isolation): two-stage semantic-
recovery probe for the official trilingual MGSM baseline (English/Spanish/
Japanese), replacing the old Claude-as-subject version that only ever ran
on mgsm_023-style data in English/French/Swahili.

Stage 1 (extraction): given ONLY the question, in its own language, extract
every quantity and the role/operation it plays. Does not solve.
Stage 2 (reasoning): given ONLY the Stage-1 JSON output -- no original
question text, no language marker, no item id in the prompt -- compute the
final answer. This is the isolation Task 3 asks for; see verify_isolation()
below, which is run automatically and fails loudly if a leak is found.

Two separate signals are recorded (this is the split Task 1 asks for,
distinct from a single pass/fail "semantic recovery" label):
  surface_quantity_fidelity -- did the extraction literally include every
    digit-quantity present in THAT language's own question text? Computed
    per-language (not always against English) and with clock-time ":00"
    stripped, per Week 3's scoring fix (src/semantic_probe_rescore.py).
  task_semantic_sufficiency -- did Stage 2, working ONLY from the
    extracted structure, reach the correct final answer? This is the real
    test of whether the extraction preserved enough meaning to solve the
    problem, independent of whether every literal digit survived (Week 3
    found cases where an extractor correctly dropped irrelevant numbers and
    was penalized for it under a fidelity-only rule).

Usage:
    python src/mgsm_semantic_recovery_trilingual.py --n-items 5 --tag smoke
    python src/mgsm_semantic_recovery_trilingual.py
"""
import argparse
import json
import os
import re
import sys
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, PROJECT_ROOT, RESULTS_DIR
from mgsm_scoring import is_correct, parse_prediction
from run_mgsm_trilingual import TEMPERATURE_NOTE, call_model, load_env

LANGS = ["en", "es", "ja"]
LANG_NAMES = {"en": "English", "es": "Spanish", "ja": "Japanese"}

EXTRACT_SYSTEM = (
    "Read the math word problem below, written in its original language. "
    "Do NOT solve it. Extract every quantity mentioned and the operation or "
    "role it participates in. Respond with ONLY a JSON array, no prose, no "
    'code fences. Each element: {"quantity": <number>, "role": "<short '
    'English label describing what this quantity represents and how it is '
    'used, e.g. initial amount, rate, multiplier, total>"}. List quantities '
    "in the order they appear in the problem."
)
REASON_SYSTEM = (
    "You will be given ONLY a JSON list of quantities and their roles, "
    "extracted from a word problem written in some language you have not "
    "seen and do not know the identity of. You do not have access to the "
    "original problem text. Using ONLY this structured information, compute "
    "the final numeric answer. Give the final numeric answer on the last "
    "line as:\nFINAL ANSWER: <number>"
)

CLOCK_TIME_RE = re.compile(r"\b(\d{1,2}):00\b")
NUMBER_RE_EN = re.compile(r"-?\d[\d,]*(?:\.\d+)?")  # comma = thousands, period = decimal
NUMBER_RE_ES = re.compile(r"-?\d[\d.]*(?:,\d+)?")   # period = thousands, comma = decimal (Spanish)


def extract_source_numbers(text, lang="en"):
    """Per-language, locale-aware gold digit set. Two fixes, both discovered the hard way:
    clock-time ':00' stripped (Week 3 -- '1:00' isn't a separate quantity from '1'), and Spanish
    comma-decimal parsed correctly (Week 5 -- '$16,50' means 16.5, not 1650; found because this
    same bug, fixed here, was already caught and fixed in mgsm_translation_number_audit.py first)."""
    text = unicodedata.normalize("NFKC", text)
    text = CLOCK_TIME_RE.sub(r"\1", text)
    number_re = NUMBER_RE_ES if lang == "es" else NUMBER_RE_EN
    numbers = set()
    for tok in number_re.findall(text):
        tok = tok.replace(".", "").replace(",", ".") if lang == "es" else tok.replace(",", "")
        try:
            numbers.add(str(float(tok)))
        except ValueError:
            continue
    return numbers


def parse_json_array(raw_output):
    match = re.search(r"\[.*\]", raw_output, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def verify_isolation(stage2_prompt, stage2_system, question, item_id, lang):
    """Task 3: the reasoning stage must receive only the Stage-1 representation.
    Raises if the original question, its language code, or the item id leaked
    into what Stage 2 actually sees."""
    combined = (stage2_prompt or "") + (stage2_system or "")
    if question.strip() and question.strip() in combined:
        raise AssertionError(f"{item_id}[{lang}]: original question text leaked into the Stage-2 prompt")
    if item_id in combined or f"[{lang}]" in combined or LANG_NAMES[lang] in combined:
        raise AssertionError(f"{item_id}[{lang}]: item id or language identity leaked into the Stage-2 prompt")


def run_one(job):
    client, model, item, lang = job
    qid, question, gold = item["id"], item["questions"][lang], item["gold_answer"]

    stage1 = call_model(client, model, question, system=EXTRACT_SYSTEM)
    extraction_raw = stage1["raw_output"]
    parsed_struct = parse_json_array(extraction_raw) if not stage1["error_message"] else None

    source_numbers = extract_source_numbers(question, lang)
    extracted_numbers = set()
    if parsed_struct:
        for entry in parsed_struct:
            try:
                extracted_numbers.add(str(float(entry["quantity"])))
            except (KeyError, TypeError, ValueError):
                continue
    surface_quantity_fidelity = bool(parsed_struct) and source_numbers.issubset(extracted_numbers)

    stage2_prompt = json.dumps(parsed_struct, ensure_ascii=False) if parsed_struct else "[]"
    verify_isolation(stage2_prompt, REASON_SYSTEM, question, qid, lang)  # raises on any leak
    stage2 = call_model(client, model, stage2_prompt, system=REASON_SYSTEM)
    reasoning_raw = stage2["raw_output"]
    prediction, parse_method = (None, "none") if stage2["error_message"] else parse_prediction(reasoning_raw)
    task_semantic_sufficiency = (not stage2["error_message"]) and is_correct(prediction, gold)

    return {
        "question_id": qid, "language": lang, "language_name": LANG_NAMES[lang],
        "question": question, "gold_answer": gold,
        "stage1_prompt": question, "stage1_system": EXTRACT_SYSTEM,
        "stage1_raw_output": extraction_raw, "stage1_parsed": json.dumps(parsed_struct, ensure_ascii=False),
        "stage1_error": stage1["error_message"],
        "source_numbers": ",".join(sorted(source_numbers)), "extracted_numbers": ",".join(sorted(extracted_numbers)),
        "surface_quantity_fidelity": surface_quantity_fidelity,
        "stage2_prompt": stage2_prompt, "stage2_system": REASON_SYSTEM,
        "stage2_raw_output": reasoning_raw, "stage2_prediction": prediction, "stage2_parse_method": parse_method,
        "stage2_error": stage2["error_message"],
        "task_semantic_sufficiency": task_semantic_sufficiency,
        "model": model, "temperature": TEMPERATURE_NOTE,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def load_prior(path):
    prior = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if not rec["stage1_error"] and not rec["stage2_error"]:
                    prior[(rec["question_id"], rec["language"])] = rec
    return prior


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=os.path.join(DATA_DIR, "mgsm_trilingual_sample.json"))
    parser.add_argument("--n-items", type=int, default=None)
    parser.add_argument("--tag", default="")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    load_env(os.path.join(PROJECT_ROOT, ".env"))
    import openai
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=120, max_retries=0)
    model = os.environ["OPENAI_MODEL"]

    with open(args.data, encoding="utf-8") as f:
        all_items = json.load(f)["items"]
    items = all_items[: args.n_items] if args.n_items else all_items

    prefix = "mgsm_semantic_recovery" + (f"_{args.tag}" if args.tag else "")
    log_path = os.path.join(RESULTS_DIR, f"{prefix}_log.jsonl")
    prior = load_prior(log_path) if args.resume else {}

    all_keys = [(it["id"], lang) for it in items for lang in LANGS]
    jobs = [(client, model, it, lang) for it in items for lang in LANGS if (it["id"], lang) not in prior]
    print(f"{len(all_keys)} (question, language) pairs; reusing {len(prior)}; running {len(jobs)} fresh "
          f"({len(jobs) * 2} real API calls: extraction + reasoning) with model {model}...")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        fresh = {(r["question_id"], r["language"]): r for r in pool.map(run_one, jobs)}
    records = [prior.get(k) or fresh[k] for k in all_keys]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(RESULTS_DIR, f"{prefix}_full.csv"), index=False)

    summary = df.groupby("language_name").agg(
        n=("question_id", "count"),
        surface_quantity_fidelity_rate=("surface_quantity_fidelity", "mean"),
        task_semantic_sufficiency_rate=("task_semantic_sufficiency", "mean"),
    ).round(4)
    summary.to_csv(os.path.join(RESULTS_DIR, f"{prefix}_summary.csv"))

    print("\n=== Two separate signals, per language ===")
    print(summary.to_string())
    both = df[df.surface_quantity_fidelity & ~df.task_semantic_sufficiency]
    print(f"\nFidelity true but task-sufficiency false (all digits present, still couldn't solve from them): {len(both)}")
    other = df[~df.surface_quantity_fidelity & df.task_semantic_sufficiency]
    print(f"Fidelity false but task-sufficiency true (missed a literal digit, solved it anyway): {len(other)}")
    print(f"\nWrote {os.path.join(RESULTS_DIR, prefix + '_full.csv')}\n       {log_path}")
    print("Isolation check: passed for all jobs (verify_isolation raised on none of them).")


if __name__ == "__main__":
    main()
