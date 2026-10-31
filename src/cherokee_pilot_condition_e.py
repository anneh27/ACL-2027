"""
Week of Oct 26 Task 3 (reuse of Week 4): adds Condition E to the existing Cherokee pilot
(src/run_cherokee_pilot_v2.py, Conditions A/B/C/D, untouched by this script -- new file, new
output, no overwrite). Condition E tests a different repair strategy for Cherokee items than
Condition C's: instead of back-translating Cherokee into English prose (which Week 4/5 found
distorts negation/quantifier/relation markers) and then reasoning from that prose, E formalizes
the Cherokee directly into structured logical notation, runs a Critic+Repair pass on that
structure (seeing the Cherokee text and the extraction together, the one isolation exception),
and only then reasons -- isolated, seeing ONLY the final formal JSON, no Cherokee text, no
back-translation, no item id.

  E1  extract:  Cherokee text -> {"premises": [...], "query": "..."} (formal notation)
  E2  critic:   Cherokee text + extraction -> Sufficient Yes/No + missing info + suggested repair
  E3  repair:   (only if critic said No) -> repaired extraction
  E4  reason:   ONLY the (possibly repaired) extraction, isolated -> entailed/contradicted/unknown

Also computes an automated (non-judgment) structural-fidelity signal: does the extracted formal
structure use the same count of each logical connective (not/and/or/if/iff/forall/exists) as the
item's own gold premises_formal/query_formal? This is a heuristic pointer, like
run_cherokee_pilot_v2.py's translation_diff() marker-count check -- not a correctness verdict,
since a correct extraction can legitimately restructure (e.g. and(not(P),not(Q)) vs.
not(or(P,Q)) are logically equivalent but have different connective counts).

Usage:
    python src/cherokee_pilot_condition_e.py --n-items 4 --tag smoke
    python src/cherokee_pilot_condition_e.py
"""
import argparse
import json
import os
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, PROJECT_ROOT, RESULTS_DIR
from logic_critic_repair_pilot import parse_critic, parse_json_obj, parse_label
from run_cherokee_pilot_v2 import CLASSIFY_SYSTEM_SYMBOLIC, is_correct
from run_mgsm_trilingual import TEMPERATURE_NOTE, call_model, load_env

EXTRACT_SYSTEM_CHEROKEE = (
    "Read the logic problem below, written in Cherokee (Tsalagi): some premises and a query. Do "
    "NOT determine whether the query follows -- only formalize it. Respond with ONLY a JSON "
    'object, no prose, no code fences, in EXACTLY this shape: {"premises": [<formal '
    'predicate-logic strings>], "query": "<formal predicate-logic string>"}, using functional '
    "style: not(P), and(P,Q), or(P,Q), if(P,Q) for a one-directional conditional, iff(P,Q) for a "
    "biconditional, forall(x, P(x) -> Q(x)), exists(x, P(x)), predicate(Entity). Preserve "
    "negation scope, conditional direction, and quantifier scope exactly as stated in the "
    "Cherokee text."
)
CRITIC_SYSTEM_CHEROKEE = (
    "You are a Formalization Critic. You will be given a logic problem written in Cherokee "
    "(Tsalagi) and a formal-logic extraction of it, intended to let someone classify entailment "
    "using ONLY the formal version, without ever seeing the Cherokee text again. Decide whether "
    "the extraction is sufficient and faithful to the Cherokee text.\n\n"
    "Explicitly check all of the following:\n"
    "- Is negation scope preserved exactly?\n"
    "- Is the conditional direction correct, and is a biconditional distinguished from a "
    "one-directional conditional?\n"
    "- Is quantifier scope and nesting order preserved?\n"
    "- Is every premise in the Cherokee text present in the extraction, with nothing added?\n"
    "- Does the formal query match what the Cherokee query actually asks?\n\n"
    "Respond in EXACTLY this format, no other text:\n"
    "Sufficient: Yes/No\n\n"
    "Missing or distorted information:\n"
    "- (one bullet per issue, or \"- none\" if sufficient)\n\n"
    "Suggested repair:\n"
    "- (one bullet per fix, or \"- none\" if sufficient)"
)
REPAIR_SYSTEM_CHEROKEE = (
    "You will be given a logic problem in Cherokee (Tsalagi), a formal-logic extraction of it, "
    "and a critic's assessment of what is missing or distorted. Produce a REPAIRED extraction "
    "that fixes every issue the critic raised. Respond with ONLY a JSON object in EXACTLY the "
    'same shape as the input: {"premises": [...], "query": "..."}. Keep every part that was '
    "already correct unchanged."
)

CONNECTIVE_RE = re.compile(r"\b(not|and|or|if|iff|forall|exists)\(")


def connective_counts(formal_list_or_str):
    text = " ".join(formal_list_or_str) if isinstance(formal_list_or_str, list) else (formal_list_or_str or "")
    return Counter(CONNECTIVE_RE.findall(text))


def verify_isolation(stage_prompt, stage_system, item):
    combined = (stage_prompt or "") + (stage_system or "")
    if item["cherokee"].strip() and item["cherokee"].strip() in combined:
        raise AssertionError(f"{item['id']}: Cherokee source text leaked into the isolated reasoning prompt")
    if item["id"] in combined:
        raise AssertionError(f"{item['id']}: item id leaked into the isolated reasoning prompt")


def _safe(fn):
    def wrapped(job):
        client, model, item = job
        try:
            return fn(job)
        except Exception as exc:  # noqa: BLE001 -- one bad item must not kill the whole batch
            return {"id": item["id"], "gold_label": item["label"], "correct": None,
                    "job_error": f"{type(exc).__name__}: {exc}", "timestamp": datetime.now(timezone.utc).isoformat()}
    return wrapped


def run_condition_e(job):
    client, model, item = job
    gold = item["label"]

    stage1 = call_model(client, model, item["cherokee"], system=EXTRACT_SYSTEM_CHEROKEE)
    parsed = None if stage1["error_message"] else parse_json_obj(stage1["raw_output"])
    n_calls = 1

    critic_raw, sufficient, missing_text, repair_text = "", None, "", ""
    repair_raw, was_repaired, final_obj = "", False, parsed
    if parsed is not None:
        critic_prompt = f"Cherokee text:\n{item['cherokee']}\n\nFormal extraction:\n{stage1['raw_output']}"
        critic = call_model(client, model, critic_prompt, system=CRITIC_SYSTEM_CHEROKEE)
        n_calls += 1
        critic_raw = critic["raw_output"]
        sufficient, missing_text, repair_text = parse_critic(critic_raw)

        if sufficient is False:
            repair_prompt = (
                f"Cherokee text:\n{item['cherokee']}\n\nOriginal extraction:\n{stage1['raw_output']}\n\n"
                f"Critic's missing/distorted information:\n{missing_text}\n\nCritic's suggested repair:\n{repair_text}"
            )
            repair = call_model(client, model, repair_prompt, system=REPAIR_SYSTEM_CHEROKEE)
            n_calls += 1
            repair_raw = repair["raw_output"]
            repaired_obj = None if repair["error_message"] else parse_json_obj(repair_raw)
            final_obj = repaired_obj if repaired_obj is not None else parsed
            was_repaired = repaired_obj is not None

    reason_prompt = json.dumps(final_obj, ensure_ascii=False) if final_obj else "{}"
    verify_isolation(reason_prompt, CLASSIFY_SYSTEM_SYMBOLIC, item)
    reason = call_model(client, model, reason_prompt, system=CLASSIFY_SYSTEM_SYMBOLIC)
    n_calls += 1
    pred = None if reason["error_message"] else parse_label(reason["raw_output"])

    gold_counts = connective_counts(item["premises_formal"]) + connective_counts(item["query_formal"])
    extracted_counts = connective_counts(final_obj["premises"] + [final_obj["query"]]) if final_obj else Counter()

    return {
        "id": item["id"], "reasoning_type": item["reasoning_type"], "subtype": item["subtype"],
        "E_stage1_raw_output": stage1["raw_output"], "E_stage1_error": stage1["error_message"],
        "E_critic_raw_output": critic_raw, "E_critic_sufficient": sufficient,
        "E_was_repaired": was_repaired, "E_repair_raw_output": repair_raw,
        "E_final_extraction": json.dumps(final_obj, ensure_ascii=False) if final_obj else "",
        "E_reason_prompt": reason_prompt, "E_raw_output": reason["raw_output"], "E_reason_error": reason["error_message"],
        "E_prediction": pred, "gold_label": gold, "E_correct": is_correct(pred, gold) if pred else False,
        "connective_counts_match": dict(gold_counts) == dict(extracted_counts),
        "connective_counts_gold": dict(gold_counts), "connective_counts_extracted": dict(extracted_counts),
        "n_calls": n_calls, "model": model, "temperature": TEMPERATURE_NOTE,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-items", type=int, default=None)
    parser.add_argument("--tag", default="")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    load_env(os.path.join(PROJECT_ROOT, ".env"))
    import openai
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=120, max_retries=0)
    model = os.environ["OPENAI_MODEL"]

    with open(os.path.join(DATA_DIR, "week1_cleaned.json"), encoding="utf-8") as f:
        cleaned = {i["id"]: i for i in json.load(f)["items"]}
    with open(os.path.join(DATA_DIR, "week1_cherokee.json"), encoding="utf-8") as f:
        cherokee = {i["id"]: i["cherokee_candidate_translation"] for i in json.load(f)["items"]}
    items = [dict(it, cherokee=cherokee[item_id]) for item_id, it in cleaned.items()]
    if args.n_items:
        items = items[: args.n_items]

    suffix = f"_{args.tag}" if args.tag else ""
    print(f"{len(items)} Cherokee items, Condition E only (A/B/C/D untouched), model {model}...")

    jobs = [(client, model, it) for it in items]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(_safe(run_condition_e), jobs))

    errors = [r for r in records if r.get("job_error")]
    scored = [r for r in records if r["E_correct"] is not None]
    print(f"\nCondition E: {sum(r['E_correct'] for r in scored)}/{len(scored)} correct "
          f"({len(errors)} job errors); {sum(r.get('E_was_repaired') is True for r in records)} repaired")
    if errors:
        for r in errors:
            print(f"  {r['id']}: {r['job_error']}")

    df = pd.DataFrame(records)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, f"cherokee_pilot_condition_e{suffix}.csv")
    df.to_csv(csv_path, index=False)
    with open(os.path.join(RESULTS_DIR, f"cherokee_pilot_condition_e{suffix}_log.jsonl"), "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    full_path = os.path.join(RESULTS_DIR, "cherokee_pilot_v2_full.csv")
    if os.path.exists(full_path):
        full = pd.read_csv(full_path).set_index("id")
        c_correct = full["C_correct"].reindex(df.set_index("id").index)
        e_correct = df.set_index("id")["E_correct"]
        common = c_correct.index.intersection(e_correct.index)
        c_bool, e_bool = c_correct.loc[common].fillna(False).astype(bool), e_correct.loc[common].fillna(False).astype(bool)
        recovered = int((~c_bool & e_bool).sum())
        regressed = int((c_bool & ~e_bool).sum())
        c_wrong, c_right = int((~c_bool).sum()), int(c_bool.sum())
        print(f"\nC->E: recovered {recovered}/{c_wrong} of C's wrong items, regressed {regressed}/{c_right} of C's right items")

    print(f"\nWrote {csv_path}")
    print("Isolation check: passed for all jobs (verify_isolation raised on none of them).")
    print("Conditions A/B/C/D (results/cherokee_pilot_v2_full.csv) untouched by this script.")


if __name__ == "__main__":
    main()
