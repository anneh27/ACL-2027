"""
Week 4 P1: Cherokee logic pilot v2 -- the original 20 items, all four
conditions, REAL model calls, every raw output saved.

Why this exists: the Week 2/3 Claude-run conditions were hand-transcribed
into scripts, so their "raw outputs" were never verifiable originals. Here
every prediction comes from a live API call, and the exact prompt, raw
output, and parsed label of every call are written to the JSONL log.

Conditions (per item, 5 calls):
  A  official English premises + query            -> label
  B  Cherokee text, answered directly             -> label
  C  step 1: back-translate the Cherokee into English (a separate call told
     only to translate);  step 2: answer from that back-translation -> label
  D  gold formal (symbolic) representation        -> label   (oracle control)

Classification prompts are the same ones src/run_logic_pilot.py used in the
original pilot. `correct` is computed by the program:
    str(prediction).strip().lower() == str(gold).strip().lower()

Program-computed vs. judgment: predictions, `correct`, and the translation
marker flags are computed by code. error_type_auto is a rule-based first pass
(see classify_error). Anything that is a reader's judgment -- error-type
overrides and the back-translation fidelity rating -- lives in
data/cherokee_pilot_v2_error_type_overrides.json with a note per item, and the
CSV says which source each value came from.

Usage:
    python src/run_cherokee_pilot_v2.py            # 20 items x 5 calls
    python src/run_cherokee_pilot_v2.py --resume   # re-call only failed calls
"""
import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, PROJECT_ROOT, RESULTS_DIR
from run_mgsm_trilingual import TEMPERATURE_NOTE, call_model, load_env

LABELS = ["entailed", "contradicted", "unknown"]

CLASSIFY_SYSTEM_NL = (
    "You will be given some premises and a query statement, in natural "
    "language. Decide whether the query is ENTAILED by the premises, "
    "CONTRADICTED by the premises, or UNKNOWN (neither follows). Respond "
    "with exactly one word: entailed, contradicted, or unknown."
)
CLASSIFY_SYSTEM_SYMBOLIC = (
    "You will be given premises and a query in formal logical notation. "
    "Decide whether the query is ENTAILED by the premises, CONTRADICTED by "
    "the premises, or UNKNOWN. Respond with exactly one word: entailed, "
    "contradicted, or unknown."
)
TRANSLATE_PROMPT = (
    "Translate the following Cherokee (Tsalagi) text into English. Output only the English "
    "translation. Do not answer, analyze, or comment on it.\n\n{text}"
)

STEPS = ["A", "B", "C1", "C2", "D"]
OVERRIDES_PATH = os.path.join(DATA_DIR, "cherokee_pilot_v2_error_type_overrides.json")


def parse_label(raw_output):
    """Strict label parser. Returns (label or None, method)."""
    text = (raw_output or "").strip().lower()
    if re.search(r"\bnot\s+(entailed|contradicted|unknown)\b", text):
        return None, "unparsed"
    bare = re.sub(r"[^a-z]", "", text)
    if bare in LABELS:
        return bare, "exact"
    found = {lab for lab in LABELS if re.search(rf"\b{lab}\b", text)}
    if len(found) == 1:
        return found.pop(), "single_label_in_text"
    return None, "unparsed"


def is_correct(prediction, gold):
    return str(prediction).strip().lower() == str(gold).strip().lower()


# ---- first-pass detector for logical information changed by translation ----
_NEG = re.compile(r"\b(not|no|never|neither|nor|none|nobody|nothing|cannot)\b|n't", re.I)
_COND = re.compile(r"\b(if|then|whenever|unless)\b", re.I)
_QUANT = re.compile(r"\b(all|every|each|some|any|most|few|both)\b", re.I)
_REL = re.compile(r"\b(left|right|north|south|east|west|above|below|behind|front|beside|under|over|near|next)\b", re.I)
_NAME_STOP = {"The", "If", "All", "Some", "No", "It", "Not", "A", "An", "Query", "Premises", "Then", "Either", "Every"}


def _names(text):
    return {w for w in re.findall(r"\b[A-Z][a-z]+\b", text) if w not in _NAME_STOP}


def translation_diff(source_en, translation):
    """Compare marker counts in the English source vs. the back-translation.
    A heuristic that points a reader at what to check, not a verdict."""
    flags = {}
    for name, rx in [("negation", _NEG), ("conditional", _COND), ("quantifier", _QUANT), ("relation", _REL)]:
        a, b = len(rx.findall(source_en)), len(rx.findall(translation))
        flags[f"flag_{name}_changed"] = a != b
        flags[f"{name}_markers_source_vs_backtranslation"] = f"{a} vs {b}"
    src_names, tr_names = _names(source_en), _names(translation)
    flags["flag_entities_changed"] = src_names != tr_names
    flags["entities_source"] = ",".join(sorted(src_names))
    flags["entities_backtranslation"] = ",".join(sorted(tr_names))
    return flags


def classify_error(row, diff_flags):
    """Rule-based first pass. Returns (type, reason). Types: none, reasoning,
    scoring, translation, unresolved (data ambiguity is never auto-assigned)."""
    b_ok, c_ok = row["B_correct"], row["C_correct"]
    if b_ok and c_ok:
        return "none", ""
    # parser trouble on an otherwise sensible answer is a possible scoring problem
    for step in ("B", "C"):
        if row[f"{step}_parse_method"] == "unparsed":
            return "scoring", f"{step} output could not be parsed to a label; check the raw output"
    if not row["A_correct"] or not row["D_correct"]:
        return "reasoning", "the English and/or oracle-symbolic condition is also wrong, so Cherokee is not the cause"
    if any(diff_flags[k] for k in ("flag_negation_changed", "flag_conditional_changed",
                                    "flag_quantifier_changed", "flag_relation_changed", "flag_entities_changed")):
        return "translation", "A and D correct; the back-translation changed logical markers or entities"
    return "unresolved", "A and D correct but no marker change detected in the back-translation; needs a manual read"


def load_prior(path):
    prior = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if not rec["error_message"]:
                    prior[(rec["item_id"], rec["step"])] = rec
    return prior


def make_record(item, step, system, prompt, result, gold, model, is_scored):
    raw = result["raw_output"]
    pred, method, correct = None, "", None
    if not result["error_message"] and is_scored:
        pred, method = parse_label(raw)
        correct = is_correct(pred, gold)
    return {
        "item_id": item["id"], "step": step, "system_prompt": system or "", "prompt": prompt,
        "raw_output": raw, "parsed_prediction": pred, "parse_method": method,
        "gold_label": gold, "correct": correct, "model": model,
        "model_returned": result.get("model_returned", ""), "temperature": TEMPERATURE_NOTE,
        "timestamp": datetime.now(timezone.utc).isoformat(), "error_message": result["error_message"],
        "prompt_tokens": result.get("prompt_tokens"), "completion_tokens": result.get("completion_tokens"),
    }


def run_item(job):
    client, model, item, prior = job
    gold = item["label"]
    en_prompt = "Premises:\n" + "\n".join(item["premises_nl"]) + f"\n\nQuery: {item['query_nl']}"
    sym_prompt = "Premises:\n" + "\n".join(item["premises_formal"]) + f"\n\nQuery: {item['query_formal']}"
    chr_text = item["cherokee"]
    recs = {}

    def do(step, system, prompt, scored=True):
        if (item["id"], step) in prior:
            recs[step] = prior[(item["id"], step)]
        else:
            recs[step] = make_record(item, step, system, prompt, call_model(client, model, prompt, system=system),
                                     gold, model, scored)
        return recs[step]

    do("A", CLASSIFY_SYSTEM_NL, en_prompt)
    do("B", CLASSIFY_SYSTEM_NL, chr_text)
    c1 = do("C1", None, TRANSLATE_PROMPT.format(text=chr_text), scored=False)
    if c1["raw_output"].strip() and not c1["error_message"]:
        do("C2", CLASSIFY_SYSTEM_NL, c1["raw_output"].strip())
    else:
        recs["C2"] = make_record(item, "C2", CLASSIFY_SYSTEM_NL, "", {"raw_output": "", "error_message":
                                 "skipped: no back-translation available"}, gold, model, True)
    do("D", CLASSIFY_SYSTEM_SYMBOLIC, sym_prompt)
    return [recs[s] for s in STEPS]


def build_full_row(item, recs, overrides, fidelity):
    by = {r["step"]: r for r in recs}
    english_original = " ".join(item["premises_nl"]) + " [Query] " + item["query_nl"]
    row = {
        "id": item["id"], "reasoning_type": item["reasoning_type"], "subtype": item["subtype"],
        "english_original": english_original, "cherokee_text": item["cherokee"], "gold_label": item["label"],
    }
    for step, name in (("A", "A"), ("B", "B"), ("C2", "C"), ("D", "D")):
        r = by[step]
        row[f"{name}_raw_output"] = r["raw_output"]
        row[f"{name}_prediction"] = r["parsed_prediction"]
        row[f"{name}_parse_method"] = r["parse_method"]
        row[f"{name}_correct"] = r["correct"]  # computed by the program from the saved raw output
    row["C_translation"] = by["C1"]["raw_output"].strip()
    row["api_errors"] = "; ".join(f"{r['step']}: {r['error_message'][:80]}" for r in recs if r["error_message"])
    diff = translation_diff(english_original.replace("[Query]", ""), row["C_translation"])
    row.update(diff)
    auto_type, auto_reason = classify_error(row, diff)
    row["error_type_auto"], row["error_type_auto_reason"] = auto_type, auto_reason
    ov = overrides.get(item["id"])
    row["error_type"] = ov["type"] if ov else auto_type
    row["error_type_source"] = "manual override" if ov else "auto rule"
    row["error_note"] = ov["note"] if ov else ""
    fid = fidelity.get(item["id"])
    row["back_translation_fidelity"] = fid["rating"] if fid else ""  # manual judgment, see overrides file
    row["back_translation_fidelity_note"] = fid["note"] if fid else ""
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="keep successful records in the existing log, re-call failures")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    load_env(os.path.join(PROJECT_ROOT, ".env"))
    import openai
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=120, max_retries=0)
    model = os.environ["OPENAI_MODEL"]

    with open(os.path.join(DATA_DIR, "week1_cleaned.json"), encoding="utf-8") as f:
        cleaned = {i["id"]: i for i in json.load(f)["items"]}
    with open(os.path.join(DATA_DIR, "week1_cherokee.json"), encoding="utf-8") as f:
        cherokee = {i["id"]: i["cherokee_candidate_translation"] for i in json.load(f)["items"]}
    items = []
    for item_id, it in cleaned.items():
        merged = dict(it)
        merged["cherokee"] = cherokee[item_id]
        items.append(merged)

    overrides, fidelity = {}, {}
    if os.path.exists(OVERRIDES_PATH):
        with open(OVERRIDES_PATH, encoding="utf-8") as f:
            manual = json.load(f)
        overrides = manual.get("error_type_overrides", {})
        fidelity = manual.get("back_translation_fidelity", {})

    log_path = os.path.join(RESULTS_DIR, "cherokee_pilot_v2_log.jsonl")
    prior = load_prior(log_path) if args.resume else {}
    print(f"{len(items)} items x {len(STEPS)} calls = {len(items) * len(STEPS)}; "
          f"reusing {len(prior)} prior successful records; model {model}")

    started = datetime.now(timezone.utc)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(run_item, [(client, model, it, prior) for it in items]))
    finished = datetime.now(timezone.utc)

    with open(log_path, "w", encoding="utf-8") as f:
        for recs in results:
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    rows = [build_full_row(it, recs, overrides, fidelity) for it, recs in zip(items, results)]
    full = pd.DataFrame(rows)
    full.to_csv(os.path.join(RESULTS_DIR, "cherokee_pilot_v2_full.csv"), index=False)
    # the plan asks for B or C wrong; D (the oracle control) is included too because a wrong
    # oracle answer is an unexpected result that needs explaining, not hiding
    errors = full[(full["B_correct"] != True) | (full["C_correct"] != True) | (full["D_correct"] != True)]  # noqa: E712
    errors.to_csv(os.path.join(RESULTS_DIR, "cherokee_pilot_v2_errors.csv"), index=False)

    with open(os.path.join(RESULTS_DIR, "cherokee_pilot_v2_manifest.json"), "w") as f:
        json.dump({"started_utc": started.isoformat(), "finished_utc": finished.isoformat(), "model": model,
                   "temperature": TEMPERATURE_NOTE, "n_items": len(items), "calls_per_item": len(STEPS),
                   "records_reused_from_previous_invocation": len(prior),
                   "prompts": {"classify_nl": CLASSIFY_SYSTEM_NL, "classify_symbolic": CLASSIFY_SYSTEM_SYMBOLIC,
                               "translate": TRANSLATE_PROMPT}}, f, indent=2)

    print("\n=== Accuracy (program-computed from saved raw outputs) ===")
    for name in "ABCD":
        col = full[f"{name}_correct"]
        print(f"  {name}: {int((col == True).sum())}/{len(full)}   (api errors: {int(col.isna().sum())})")  # noqa: E712
    print("\nBy reasoning type:")
    print(full.groupby("reasoning_type")[["A_correct", "B_correct", "C_correct", "D_correct"]]
          .agg(lambda s: f"{int((s == True).sum())}/{len(s)}").to_string())  # noqa: E712
    print(f"\nItems with B, C, or D wrong: {len(errors)}")
    print(errors[["id", "gold_label", "B_prediction", "C_prediction", "D_prediction", "error_type"]].to_string(index=False))


if __name__ == "__main__":
    main()
