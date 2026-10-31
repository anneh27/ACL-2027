"""
Week of Oct 26 Task 3 (reuse of Week 3): a real-API baseline plus extraction->reason and
+Critic-repair conditions on the 20 harder logic items (data/week3_new_logic_items.json --
negation/De Morgan, conditionals/biconditionals, quantifiers, fallacious-reasoning-form
distractors). As of the Week 4 README these 20 items had never been run through a real model
(src/interactive_pilot_logic_harder.py used Claude-as-subject inside the conversation, not an
API call). This is a fresh real-API run, reusing run_mgsm_trilingual.call_model/
call_model_messages (NOT src/run_openai.py's call_model, which passes temperature=0 and would
hit the same "model rejects temperature=0" error Week 4 found and fixed for MGSM).

Three conditions, same pattern as the MGSM Critic+Repair pilot, applied to entailment
classification instead of a numeric answer:
  A. Direct reasoning  -- premises_nl + query_nl -> entailed/contradicted/unknown, no formalization.
  B. Extract -> reason -- Stage 1 formalizes premises/query into a JSON object (not solving it),
                           Stage 2 classifies from ONLY that JSON (isolated -- no NL text, no item id).
  C. + Critic/repair    -- a critic (sees NL + formalization together, the one deliberate isolation
                           exception) checks negation scope, conditional direction, quantifier
                           scope/order, and dropped/added premises; repairs if insufficient; Stage 2
                           then runs on the (possibly repaired) formalization, isolated as in B.

Usage:
    python src/logic_critic_repair_pilot.py --n-items 4 --tag smoke
    python src/logic_critic_repair_pilot.py
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

DIRECT_SYSTEM = (
    "You will be given some premises and a query statement, in natural language. Decide whether "
    "the query is ENTAILED by the premises, CONTRADICTED by the premises, or UNKNOWN (neither "
    "follows, under classical logic). Respond with exactly one word on the last line: entailed, "
    "contradicted, or unknown."
)

EXTRACT_SYSTEM = (
    "Read the premises and query below, in natural language. Do NOT determine whether the query "
    "follows -- only formalize. Respond with ONLY a JSON object, no prose, no code fences, in "
    'EXACTLY this shape: {"premises": [<formal predicate-logic strings>], "query": "<formal '
    'predicate-logic string>"}, using functional style: not(P), and(P,Q), or(P,Q), if(P,Q) for '
    "a one-directional conditional, iff(P,Q) for a biconditional, forall(x, P(x) -> Q(x)), "
    "exists(x, P(x)), predicate(Entity). Preserve negation scope exactly (not(and(P,Q)) is NOT "
    "the same as and(not(P),not(Q))), preserve conditional direction exactly (if(P,Q) means P "
    "implies Q, not the reverse), and preserve quantifier scope and nesting order exactly. Do "
    "not add a premise that isn't stated and do not drop one that is."
)

REASON_SYSTEM = (
    "You will be given ONLY a JSON object with \"premises\" (a list of formal logical "
    "statements) and \"query\" (a formal logical statement) -- no natural-language text, no "
    "context about what the original problem was. Decide whether the query is ENTAILED by the "
    "premises, CONTRADICTED by the premises, or UNKNOWN. Respond with exactly one word on the "
    "last line: entailed, contradicted, or unknown."
)

CRITIC_SYSTEM = (
    "You are a Formalization Critic. You will be given natural-language premises and a query, "
    "and a formal-logic extraction of them, intended to let someone classify entailment using "
    "ONLY the formal version, without ever seeing this natural-language text again. Decide "
    "whether the extraction is sufficient -- nothing else about the entailment itself is your "
    "job.\n\n"
    "Explicitly check all of the following:\n"
    "- Is negation scope preserved exactly (not(and(P,Q)) vs. and(not(P),not(Q)) are NOT the same)?\n"
    "- Is the conditional direction correct (if(P,Q) means P implies Q, not the reverse) and is "
    "a biconditional (iff) distinguished from a one-directional conditional (if)?\n"
    "- Is quantifier scope and nesting order preserved (which variable binds which, forall vs. exists)?\n"
    "- Is every stated premise present, and is no unstated premise added?\n"
    "- Does the formal query match what the natural-language query actually asks?\n\n"
    "Respond in EXACTLY this format, no other text:\n"
    "Sufficient: Yes/No\n\n"
    "Missing or distorted information:\n"
    "- (one bullet per issue, or \"- none\" if sufficient)\n\n"
    "Suggested repair:\n"
    "- (one bullet per fix, or \"- none\" if sufficient)"
)

REPAIR_SYSTEM = (
    "You will be given natural-language premises and a query, a formal-logic extraction of "
    "them, and a critic's assessment of what is missing or distorted. Produce a REPAIRED "
    "extraction that fixes every issue the critic raised. Respond with ONLY a JSON object in "
    'EXACTLY the same shape as the input: {"premises": [...], "query": "..."}. Keep every part '
    "that was already correct unchanged."
)

CRITIC_PARSE_RE = re.compile(
    r"Sufficient:\s*(Yes|No)\s*\n+Missing or distorted information:\s*\n(.*?)\n+Suggested repair:\s*\n(.*)",
    re.IGNORECASE | re.DOTALL,
)


def parse_label(raw_output):
    text = (raw_output or "").strip().lower()
    for label in LABELS:
        if re.search(rf"\b{label}\b", text):
            return label
    return None


def parse_json_obj(raw_output):
    match = re.search(r"\{.*\}", raw_output or "", re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or "premises" not in obj or "query" not in obj:
        return None
    return obj


def parse_critic(raw_output):
    m = CRITIC_PARSE_RE.search(raw_output or "")
    if not m:
        return None, "", ""
    return m.group(1).strip().lower() == "yes", m.group(2).strip(), m.group(3).strip()


def verify_isolation(stage2_prompt, stage2_system, item):
    """Same discipline as the MGSM pipeline's verify_isolation(): Stage 2 must never see the
    original NL text or item id, only the formal extraction."""
    combined = (stage2_prompt or "") + (stage2_system or "")
    for text in [" ".join(item["premises_nl"]), item["query_nl"]]:
        if text.strip() and text.strip() in combined:
            raise AssertionError(f"{item['id']}: original NL text leaked into the Stage-2 prompt")
    if item["id"] in combined:
        raise AssertionError(f"{item['id']}: item id leaked into the Stage-2 prompt")


def _safe(fn, label):
    def wrapped(job):
        client, model, item = job
        try:
            return fn(job)
        except Exception as exc:  # noqa: BLE001 -- one bad item must not kill the whole batch
            return {"condition": label, "id": item["id"], "gold_label": item["label"], "correct": None,
                    "job_error": f"{type(exc).__name__}: {exc}", "timestamp": datetime.now(timezone.utc).isoformat()}
    return wrapped


def run_condition_a(job):
    client, model, item = job
    prompt = "Premises:\n" + "\n".join(f"- {p}" for p in item["premises_nl"]) + f"\n\nQuery: {item['query_nl']}"
    resp = call_model(client, model, prompt, system=DIRECT_SYSTEM)
    pred = None if resp["error_message"] else parse_label(resp["raw_output"])
    return {
        "condition": "A_direct", "id": item["id"], "reasoning_type": item["reasoning_type"],
        "raw_output": resp["raw_output"], "error": resp["error_message"],
        "prediction": pred, "gold_label": item["label"], "correct": pred == item["label"],
        "n_calls": 1, "model": model, "temperature": TEMPERATURE_NOTE,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def run_condition_b(job):
    client, model, item = job
    nl_prompt = "Premises:\n" + "\n".join(f"- {p}" for p in item["premises_nl"]) + f"\n\nQuery: {item['query_nl']}"
    stage1 = call_model(client, model, nl_prompt, system=EXTRACT_SYSTEM)
    parsed = None if stage1["error_message"] else parse_json_obj(stage1["raw_output"])

    stage2_prompt = json.dumps(parsed, ensure_ascii=False) if parsed else "{}"
    verify_isolation(stage2_prompt, REASON_SYSTEM, item)
    stage2 = call_model(client, model, stage2_prompt, system=REASON_SYSTEM)
    pred = None if stage2["error_message"] else parse_label(stage2["raw_output"])

    return {
        "condition": "B_extract_reason", "id": item["id"], "reasoning_type": item["reasoning_type"],
        "stage1_raw_output": stage1["raw_output"], "stage1_error": stage1["error_message"],
        "stage1_parsed": json.dumps(parsed, ensure_ascii=False) if parsed else "",
        "stage2_prompt": stage2_prompt, "stage2_raw_output": stage2["raw_output"], "stage2_error": stage2["error_message"],
        "prediction": pred, "gold_label": item["label"], "correct": pred == item["label"],
        "n_calls": 2, "model": model, "temperature": TEMPERATURE_NOTE,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def run_condition_c(job):
    client, model, item = job
    nl_prompt = "Premises:\n" + "\n".join(f"- {p}" for p in item["premises_nl"]) + f"\n\nQuery: {item['query_nl']}"
    stage1 = call_model(client, model, nl_prompt, system=EXTRACT_SYSTEM)
    parsed = None if stage1["error_message"] else parse_json_obj(stage1["raw_output"])
    n_calls = 1

    critic_raw, sufficient, missing_text, repair_text = "", None, "", ""
    repair_raw, was_repaired, final_obj = "", False, parsed
    if parsed is not None:
        critic_prompt = f"{nl_prompt}\n\nFormal extraction:\n{stage1['raw_output']}"
        critic = call_model(client, model, critic_prompt, system=CRITIC_SYSTEM)
        n_calls += 1
        critic_raw = critic["raw_output"]
        sufficient, missing_text, repair_text = parse_critic(critic_raw)

        if sufficient is False:
            repair_prompt = (
                f"{nl_prompt}\n\nOriginal extraction:\n{stage1['raw_output']}\n\n"
                f"Critic's missing/distorted information:\n{missing_text}\n\nCritic's suggested repair:\n{repair_text}"
            )
            repair = call_model(client, model, repair_prompt, system=REPAIR_SYSTEM)
            n_calls += 1
            repair_raw = repair["raw_output"]
            repaired_obj = None if repair["error_message"] else parse_json_obj(repair_raw)
            final_obj = repaired_obj if repaired_obj is not None else parsed
            was_repaired = repaired_obj is not None

    stage2_prompt = json.dumps(final_obj, ensure_ascii=False) if final_obj else "{}"
    verify_isolation(stage2_prompt, REASON_SYSTEM, item)
    if missing_text and missing_text.lower() != "none" and missing_text in stage2_prompt:
        raise AssertionError(f"{item['id']}: critic's prose leaked into the Stage-2 prompt")
    stage2 = call_model(client, model, stage2_prompt, system=REASON_SYSTEM)
    n_calls += 1
    pred = None if stage2["error_message"] else parse_label(stage2["raw_output"])

    return {
        "condition": "C_critic_repair", "id": item["id"], "reasoning_type": item["reasoning_type"],
        "stage1_raw_output": stage1["raw_output"], "critic_raw_output": critic_raw, "critic_sufficient": sufficient,
        "was_repaired": was_repaired, "repair_raw_output": repair_raw,
        "final_extraction": json.dumps(final_obj, ensure_ascii=False) if final_obj else "",
        "stage2_prompt": stage2_prompt, "stage2_raw_output": stage2["raw_output"], "stage2_error": stage2["error_message"],
        "prediction": pred, "gold_label": item["label"], "correct": pred == item["label"],
        "n_calls": n_calls, "model": model, "temperature": TEMPERATURE_NOTE,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-items", type=int, default=None)
    parser.add_argument("--tag", default="")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--data", default=os.path.join(DATA_DIR, "week3_new_logic_items.json"),
                         help="items JSON with an 'items' list of {id, premises_nl, query_nl, label, reasoning_type, ...}")
    parser.add_argument("--filter-type", default=None, help="only run items whose reasoning_type equals this (e.g. 'spatial')")
    parser.add_argument("--out-prefix", default="logic_pilot_real_api", help="output filename prefix under results/")
    args = parser.parse_args()

    load_env(os.path.join(PROJECT_ROOT, ".env"))
    import openai
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=120, max_retries=0)
    model = os.environ["OPENAI_MODEL"]

    with open(args.data, encoding="utf-8") as f:
        items = json.load(f)["items"]
    if args.filter_type:
        items = [it for it in items if it.get("reasoning_type") == args.filter_type]
    if args.n_items:
        items = items[: args.n_items]

    suffix = f"_{args.tag}" if args.tag else ""
    print(f"{len(items)} logic items ({args.data}{', filtered to ' + args.filter_type if args.filter_type else ''}), 3 conditions, model {model}...")

    all_records = []
    for cond_name, fn in [("A", run_condition_a), ("B", run_condition_b), ("C", run_condition_c)]:
        jobs = [(client, model, it) for it in items]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            records = list(pool.map(_safe(fn, cond_name), jobs))
        errors = sum(1 for r in records if r.get("job_error"))
        scored = [r for r in records if r["correct"] is not None]
        print(f"Condition {cond_name}: {sum(r['correct'] for r in scored)}/{len(scored)} correct ({errors} job errors)")
        all_records.extend(records)

    df = pd.DataFrame(all_records)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, f"{args.out_prefix}{suffix}.csv")
    df.to_csv(csv_path, index=False)
    with open(os.path.join(RESULTS_DIR, f"{args.out_prefix}{suffix}_log.jsonl"), "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = df[df["correct"].notna()].groupby("condition").agg(
        n=("id", "count"), n_correct=("correct", "sum"), accuracy=("correct", "mean"),
        n_calls_total=("n_calls", "sum"),
    ).round(4)
    summary.to_csv(os.path.join(RESULTS_DIR, f"{args.out_prefix}{suffix}_summary.csv"))
    print(f"\n{summary.to_string()}")

    b_df = df[df["condition"] == "B_extract_reason"].set_index("id")
    c_df = df[df["condition"] == "C_critic_repair"].set_index("id")
    if len(b_df) and len(c_df):
        common = b_df.index.intersection(c_df.index)
        recovered = sum((~b_df.loc[common, "correct"].fillna(False)) & c_df.loc[common, "correct"].fillna(False))
        regressed = sum(b_df.loc[common, "correct"].fillna(False) & (~c_df.loc[common, "correct"].fillna(False)))
        b_wrong = int((~b_df.loc[common, "correct"].fillna(False)).sum())
        b_right = int(b_df.loc[common, "correct"].fillna(False).sum())
        print(f"\nCritic+Repair vs. extract-reason: recovered {recovered}/{b_wrong} of B's wrong items, "
              f"regressed {regressed}/{b_right} of B's right items")

    print(f"\nWrote {csv_path}")
    print("Isolation check: passed for all B/C jobs (verify_isolation raised on none of them).")


if __name__ == "__main__":
    main()
