"""
Section 6: Core small-scale experimental design.

Runs all four conditions on each cleaned+translated logic item:
  A. English Natural   -- English premises/query -> entailed/contradicted/unknown
  B. Cherokee Natural  -- Cherokee premises/query -> entailed/contradicted/unknown
  C. Cherokee Parse    -- Cherokee -> formal representation (semantic recovery)
  D. Oracle Symbolic   -- gold formal representation -> entailed/contradicted/unknown

Requires data/week1_cherokee.json, produced by:
    python src/translate_cherokee.py translate-logic-items

Usage:
    python src/run_logic_pilot.py
"""
import argparse
import json
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_openai import call_model, log_trial
from config import DATA_DIR, RESULTS_DIR, DEFAULT_MODEL, DEFAULT_TEMPERATURE

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
PARSE_SYSTEM = (
    "You will be given a short logic problem written in Cherokee. Convert it "
    "into formal logical notation. Respond with ONLY a JSON object with two "
    'fields: "premises" (a list of formal predicate-logic strings) and '
    '"query" (one formal predicate-logic string), using the same functional '
    "style as these examples: left_of(A,B), not(P), if(P,Q), "
    "forall(x, bird(x) -> flies(x)), exists(x, P(x)). No prose."
)


def parse_label(raw_output):
    text = raw_output.strip().lower()
    for label in LABELS:
        if label in text:
            return label
    return None


def parse_json_obj(raw_output):
    match = re.search(r"\{.*\}", raw_output, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def normalize_formal(premises, query):
    def norm(s):
        return re.sub(r"\s+", "", str(s)).lower()
    return tuple(sorted(norm(p) for p in premises)), norm(query)


def main():
    parser = argparse.ArgumentParser(description="Section 6: A/B/C/D conditions on the Cherokee logic pilot.")
    parser.add_argument("--data", default=os.path.join(DATA_DIR, "week1_cherokee.json"))
    parser.add_argument("--out", default=os.path.join(RESULTS_DIR, "cherokee_pilot.csv"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    args = parser.parse_args()

    if not os.path.exists(args.data):
        raise SystemExit(
            f"{args.data} not found. Run `python src/translate_cherokee.py translate-logic-items` first."
        )

    with open(args.data, encoding="utf-8") as f:
        dataset = json.load(f)

    rows = []
    for item in dataset["items"]:
        gold_label = item["label"]
        gold_premises, gold_query = normalize_formal(item["premises_formal"], item["query_formal"])

        # A. English Natural
        prompt_en = "Premises:\n" + "\n".join(item["premises_nl"]) + f"\n\nQuery: {item['query_nl']}"
        raw_a = call_model(prompt_en, system_prompt=CLASSIFY_SYSTEM_NL, model=args.model, temperature=args.temperature)
        pred_a = parse_label(raw_a)
        log_trial(model=args.model, temperature=args.temperature,
                  input_data={"id": item["id"], "condition": "A_english_natural"},
                  prompt=prompt_en, raw_output=raw_a, parsed_answer=pred_a, gold_label=gold_label,
                  stage="logic_pilot")

        # B. Cherokee Natural
        prompt_chr = item["cherokee_candidate_translation"]
        raw_b = call_model(prompt_chr, system_prompt=CLASSIFY_SYSTEM_NL, model=args.model, temperature=args.temperature)
        pred_b = parse_label(raw_b)
        log_trial(model=args.model, temperature=args.temperature,
                  input_data={"id": item["id"], "condition": "B_cherokee_natural"},
                  prompt=prompt_chr, raw_output=raw_b, parsed_answer=pred_b, gold_label=gold_label,
                  stage="logic_pilot")

        # C. Cherokee Parse (semantic recovery)
        raw_c = call_model(prompt_chr, system_prompt=PARSE_SYSTEM, model=args.model, temperature=args.temperature)
        parsed_struct = parse_json_obj(raw_c)
        parse_correct = False
        if parsed_struct and "premises" in parsed_struct and "query" in parsed_struct:
            pred_premises, pred_query = normalize_formal(parsed_struct["premises"], parsed_struct["query"])
            parse_correct = (pred_premises == gold_premises) and (pred_query == gold_query)
        log_trial(model=args.model, temperature=args.temperature,
                  input_data={"id": item["id"], "condition": "C_cherokee_parse"},
                  prompt=prompt_chr, raw_output=raw_c, parsed_answer=parsed_struct,
                  gold_label={"premises": item["premises_formal"], "query": item["query_formal"]},
                  stage="logic_pilot")

        # D. Oracle Symbolic
        prompt_symbolic = "Premises:\n" + "\n".join(item["premises_formal"]) + f"\n\nQuery: {item['query_formal']}"
        raw_d = call_model(prompt_symbolic, system_prompt=CLASSIFY_SYSTEM_SYMBOLIC, model=args.model, temperature=args.temperature)
        pred_d = parse_label(raw_d)
        log_trial(model=args.model, temperature=args.temperature,
                  input_data={"id": item["id"], "condition": "D_oracle_symbolic"},
                  prompt=prompt_symbolic, raw_output=raw_d, parsed_answer=pred_d, gold_label=gold_label,
                  stage="logic_pilot")

        rows.append({
            "id": item["id"], "reasoning_type": item["reasoning_type"], "gold_label": gold_label,
            "A_english_correct": pred_a == gold_label,
            "B_cherokee_correct": pred_b == gold_label,
            "C_cherokee_parse_correct": parse_correct,
            "D_oracle_symbolic_correct": pred_d == gold_label,
        })
        print(f"{item['id']} ({item['reasoning_type']}): A={pred_a == gold_label} B={pred_b == gold_label} "
              f"C={parse_correct} D={pred_d == gold_label}")

    df = pd.DataFrame(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nWrote {len(df)} rows to {args.out}")
    print(df.groupby("reasoning_type")[
        ["A_english_correct", "B_cherokee_correct", "C_cherokee_parse_correct", "D_oracle_symbolic_correct"]
    ].mean())


if __name__ == "__main__":
    main()
