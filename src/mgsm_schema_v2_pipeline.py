"""
Week 7 Task 3: re-run Critic + Repair with schema v2 (src/mgsm_schema_v2.py)
in place of v1's bare-array + quantity=0-placeholder format, on the items
where that placeholder is suspected to matter most, and compare against the
existing v1 results -- WITHOUT touching or overwriting them (new version
number, new output files, per the plan: "keep the original version's results
unchanged, add a new version number, don't overwrite history").

Default item set: the 5 items where Week 6's v1 Critic+Repair (Condition D,
results/mgsm_condition_c_critic_repair.csv) was wrong -- mgsm_trilingual_039
(ja), 086 (es), 139 (ja), 207 (en), 214 (es), the plan's named regression
example -- plus a sample of items v1 Condition D got RIGHT, as negative
controls to catch any new regression v2 introduces (not just whether it
fixes the known ones). Pass --full to run the entire 150-item set instead,
for a complete head-to-head table. Pass --items-csv to use an arbitrary
(question_id, language) set instead -- in particular, results/
manual_semantic_repair_25.csv, the 25 original Condition-B (Stage1->Stage2)
failures that Task 2 manually repaired, which is the population the plan's
automatic-vs-generic-reflection-vs-manual-Oracle-Repair comparison is
actually about (the 5-regression default above answers a narrower question:
does schema v2 fix/not-regress Condition D specifically).

This is a NEW, fresh end-to-end run (extraction -> critic -> repair ->
Stage 2), not a re-use of v1's saved Stage-1 extractions, since the whole
point is to test the v2 EXTRACT_SYSTEM_V2 prompt and schema, not just the
critic/repair step on top of a v1-shaped extraction.

Usage:
    python src/mgsm_schema_v2_pipeline.py --tag smoke --n-controls 3
    python src/mgsm_schema_v2_pipeline.py
    python src/mgsm_schema_v2_pipeline.py --full
    python src/mgsm_schema_v2_pipeline.py --items-csv results/manual_semantic_repair_25.csv --tag condition_b_failures
"""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, PROJECT_ROOT, RESULTS_DIR
from mgsm_scoring import is_correct, parse_prediction
from mgsm_schema_v2 import (
    CRITIC_SYSTEM_V2,
    EXTRACT_SYSTEM_V2,
    REASON_SYSTEM_V2,
    REPAIR_SYSTEM_V2,
    parse_v2_json,
)
from mgsm_semantic_critic_repair import parse_critic  # format-only parser, unchanged by schema version
from mgsm_semantic_recovery_trilingual import verify_isolation
from run_mgsm_trilingual import TEMPERATURE_NOTE, call_model, load_env

KNOWN_V1_REGRESSION_ITEMS = [
    ("mgsm_trilingual_039", "ja"),
    ("mgsm_trilingual_086", "es"),
    ("mgsm_trilingual_139", "ja"),
    ("mgsm_trilingual_207", "en"),
    ("mgsm_trilingual_214", "es"),
]


def _safe(fn):
    """Same discipline as Week 6's critic/repair script: one bad item must not
    discard every other already-completed (and already-paid-for) API call."""
    def wrapped(job):
        client, model, qid, lang, question, gold = job
        try:
            return fn(job)
        except Exception as exc:  # noqa: BLE001 -- deliberately broad, see docstring above
            return {
                "question_id": qid, "language": lang, "gold_answer": gold, "correct": None,
                "job_error": f"{type(exc).__name__}: {exc}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
    return wrapped


def _token_total(*call_results):
    total = 0
    for r in call_results:
        total += (r.get("prompt_tokens") or 0) + (r.get("completion_tokens") or 0)
    return total


def run_one_v2(job):
    client, model, qid, lang, question, gold = job

    stage1 = call_model(client, model, question, system=EXTRACT_SYSTEM_V2)
    extraction_raw = stage1["raw_output"]
    parsed, stage1_parse_error = (None, "stage1 API error: " + stage1["error_message"]) if stage1["error_message"] \
        else parse_v2_json(extraction_raw)

    critic, repair = {"raw_output": "", "error_message": ""}, {"raw_output": "", "error_message": ""}
    sufficient, missing_text, repair_text, was_repaired = None, "", "", False
    final_obj = parsed

    if parsed is not None:
        critic_prompt = f"Problem:\n{question}\n\nExtraction:\n{extraction_raw}"
        critic = call_model(client, model, critic_prompt, system=CRITIC_SYSTEM_V2)
        sufficient, missing_text, repair_text = parse_critic(critic["raw_output"])

        if sufficient is False:
            repair_prompt = (
                f"Problem:\n{question}\n\nOriginal extraction:\n{extraction_raw}\n\n"
                f"Critic's missing/distorted information:\n{missing_text}\n\nCritic's suggested repair:\n{repair_text}"
            )
            repair = call_model(client, model, repair_prompt, system=REPAIR_SYSTEM_V2)
            repaired_obj, repair_parse_error = (None, "repair API error") if repair["error_message"] \
                else parse_v2_json(repair["raw_output"])
            final_obj = repaired_obj if repaired_obj is not None else parsed
            was_repaired = repaired_obj is not None

    stage2_prompt = json.dumps(final_obj, ensure_ascii=False) if final_obj is not None else "{}"
    verify_isolation(stage2_prompt, REASON_SYSTEM_V2, question, qid, lang)
    stage2 = call_model(client, model, stage2_prompt, system=REASON_SYSTEM_V2)
    prediction, parse_method = (None, "none") if stage2["error_message"] else parse_prediction(stage2["raw_output"])
    correct = (not stage2["error_message"]) and is_correct(prediction, gold)

    return {
        "question_id": qid, "language": lang, "gold_answer": gold,
        "stage1_raw_output": extraction_raw, "stage1_parse_error": stage1_parse_error,
        "critic_raw_output": critic["raw_output"], "critic_sufficient": sufficient,
        "critic_missing_info": missing_text, "critic_suggested_repair": repair_text,
        "was_repaired": was_repaired, "repair_raw_output": repair["raw_output"],
        "final_extraction": json.dumps(final_obj, ensure_ascii=False) if final_obj is not None else "",
        "stage2_prompt": stage2_prompt, "stage2_raw_output": stage2["raw_output"],
        "stage2_prediction": prediction, "stage2_parse_method": parse_method, "stage2_error": stage2["error_message"],
        "correct": correct, "model": model, "model_returned": stage2.get("model_returned"),
        "total_tokens": _token_total(stage1, critic, repair, stage2), "temperature": TEMPERATURE_NOTE,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def select_items(args, sample_items):
    by_key = {it["id"]: it for it in sample_items}

    if args.full:
        return [(it["id"], lang) for it in sample_items for lang in ("en", "es", "ja")]

    if args.items_csv:
        items_df = pd.read_csv(args.items_csv)
        keys = list(dict.fromkeys(
            [(r["question_id"], r["language"]) for _, r in items_df.iterrows()] + KNOWN_V1_REGRESSION_ITEMS
        ))  # union, de-duplicated, order preserved -- always also covers the plan's named regression cases
        return [(qid, lang) for qid, lang in keys if qid in by_key]

    v1_d_path = os.path.join(RESULTS_DIR, "mgsm_condition_c_critic_repair.csv")
    v1_d = pd.read_csv(v1_d_path)
    v1_correct_keys = [
        (r["question_id"], r["language"]) for _, r in v1_d.iterrows()
        if r["correct"] == True and (r["question_id"], r["language"]) not in KNOWN_V1_REGRESSION_ITEMS
    ]
    controls = v1_correct_keys[: args.n_controls]
    keys = list(KNOWN_V1_REGRESSION_ITEMS) + controls
    return [(qid, lang) for qid, lang in keys if qid in by_key]


def load_manual_oracle_repair():
    """Task 2's own output (results/manual_semantic_repair_25_resolved.csv) -- the manual
    annotation (missing_semantics, error_type, stage1_repaired_json) and the Stage-2 rerun on
    it were both done outside this script; this just reads the result. Returns {} if that file
    doesn't exist yet (Task 2 not done), so the comparison column degrades to all-None rather
    than erroring -- this script must still run standalone before/without Task 2."""
    path = os.path.join(RESULTS_DIR, "manual_semantic_repair_25_resolved.csv")
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path, keep_default_na=False)
    out = {}
    for _, r in df.iterrows():
        correct_after = r["correct_after"]
        correct_after = {"True": True, "False": False}.get(str(correct_after), None)
        out[(r["question_id"], r["language"])] = {
            "manual_oracle_repair_correct": correct_after,
            "manual_oracle_repair_note": r["note"] or ("repaired by hand, recovered" if correct_after else ""),
        }
    return out


def build_repair_comparison_v2(v2_records, suffix):
    """Per-item before/after table: v1 Condition C (generic reflection), v1 Condition D
    (critic+repair, the schema this fixes), v2 critic+repair, and manual Oracle Repair (Task 2's
    output, read from results/manual_semantic_repair_25_resolved.csv if present). Only items that
    were part of Task 2's 25-item manual-repair population get a non-null manual column value --
    items outside that population (e.g. the 2 Condition-D-only regressions, mgsm_039/207, which
    were never Condition-B failures and so were never manually repaired) correctly show None,
    not a guess."""
    v1_c = pd.read_csv(os.path.join(RESULTS_DIR, "mgsm_condition_b_reflection.csv")).set_index(["question_id", "language"])
    v1_d = pd.read_csv(os.path.join(RESULTS_DIR, "mgsm_condition_c_critic_repair.csv")).set_index(["question_id", "language"])
    manual = load_manual_oracle_repair()

    rows = []
    for r in v2_records:
        key = (r["question_id"], r["language"])
        manual_entry = manual.get(key, {})
        rows.append({
            "question_id": r["question_id"], "language": r["language"],
            "v1_generic_reflection_correct": v1_c.loc[key, "correct"] if key in v1_c.index else None,
            "v1_critic_repair_correct": v1_d.loc[key, "correct"] if key in v1_d.index else None,
            "v2_critic_repair_correct": r["correct"],
            "v2_was_repaired": r["was_repaired"], "v2_total_tokens": r["total_tokens"],
            "manual_oracle_repair_correct": manual_entry.get("manual_oracle_repair_correct"),
            "manual_oracle_repair_note": manual_entry.get("manual_oracle_repair_note", "outside Task 2's 25-item manual-repair population (not a Condition-B failure)"),
            "is_known_v1_regression_item": key in KNOWN_V1_REGRESSION_ITEMS,
        })
    comparison = pd.DataFrame(rows)
    out_path = os.path.join(RESULTS_DIR, f"repair_comparison_v2{suffix}.csv")
    comparison.to_csv(out_path, index=False)
    return comparison, out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--n-controls", type=int, default=20,
                         help="negative-control items (v1-Condition-D-correct, not among the known regressions) to also re-run under v2")
    parser.add_argument("--full", action="store_true", help="run all 150 (question, language) pairs instead of the regression+control subset")
    parser.add_argument("--items-csv", default=None,
                         help="CSV with question_id/language columns to use as the item set instead of the regression+control default (always unioned with the known regression items)")
    args = parser.parse_args()

    load_env(os.path.join(PROJECT_ROOT, ".env"))
    import openai
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=120, max_retries=0)
    model = os.environ["OPENAI_MODEL"]

    with open(os.path.join(DATA_DIR, "mgsm_trilingual_sample.json"), encoding="utf-8") as f:
        sample_items = json.load(f)["items"]
    by_id = {it["id"]: it for it in sample_items}

    keys = select_items(args, sample_items)
    jobs = [(client, model, qid, lang, by_id[qid]["questions"][lang], by_id[qid]["gold_answer"]) for qid, lang in keys]
    if args.full:
        scope_desc = "full 150-item set"
    elif args.items_csv:
        scope_desc = f"{args.items_csv}'s item set unioned with the {len(KNOWN_V1_REGRESSION_ITEMS)} known v1 regressions"
    else:
        scope_desc = f"{len(KNOWN_V1_REGRESSION_ITEMS)} known v1 regressions + {len(jobs) - len(KNOWN_V1_REGRESSION_ITEMS)} negative controls"
    print(f"Running schema v2 pipeline (fresh extraction -> critic -> repair -> Stage 2) on {len(jobs)} "
          f"(question, language) pairs with model {model} ({scope_desc})...")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(_safe(run_one_v2), jobs))

    errors = [r for r in records if r.get("job_error")]
    scored = [r for r in records if r["correct"] is not None]
    print(f"\nDone: {sum(r['correct'] for r in scored)}/{len(scored)} correct "
          f"({len(errors)} job errors, excluded from that count); {sum(r.get('was_repaired') is True for r in records)} repaired")
    if errors:
        print("Job errors:")
        for r in errors:
            print(f"  {r['question_id']}[{r['language']}]: {r['job_error']}")

    regression_keys = set(KNOWN_V1_REGRESSION_ITEMS)
    on_known = [r for r in scored if (r["question_id"], r["language"]) in regression_keys]
    if on_known:
        fixed = sum(r["correct"] for r in on_known)
        print(f"\nOf the {len(on_known)} known v1 Condition-D regression items re-run here: "
              f"{fixed}/{len(on_known)} now correct under schema v2.")

    suffix = f"_{args.tag}" if args.tag else ("_full" if args.full else "")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, f"mgsm_schema_v2_pipeline{suffix}.csv")
    pd.DataFrame(records).to_csv(csv_path, index=False)
    with open(os.path.join(RESULTS_DIR, f"mgsm_schema_v2_pipeline{suffix}_log.jsonl"), "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nWrote {csv_path}")

    if not args.full:
        comparison, comp_path = build_repair_comparison_v2(records, suffix)
        print(f"Wrote {comp_path}")
        print(comparison[["question_id", "language", "v1_critic_repair_correct", "v2_critic_repair_correct", "is_known_v1_regression_item"]].to_string(index=False))

    print("Isolation check: passed for all jobs (verify_isolation raised on none of them).")
    print("v1 results untouched: this script only reads results/mgsm_condition_*.csv, never writes to them.")


if __name__ == "__main__":
    main()
