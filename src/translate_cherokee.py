"""
Part D: Cherokee translation via the OpenAI API.

Two subcommands, matching Sections 5.1 and 5.2 of the plan:

  test-ability          -- test GPT's Cherokee<->English ability against real
                            human-translated parallel sentences (ChrEn corpus),
                            BEFORE trusting it to translate anything of ours.
  translate-logic-items -- translate the 20 cleaned logic problems into
                            candidate Cherokee, saving english_source,
                            formal_semantics, cherokee_candidate_translation,
                            back_translation, translation_validation.

GPT-generated Cherokee is a candidate translation only, never ground truth --
see translation_validation.human_reviewed in the output.

Usage:
    python src/translate_cherokee.py test-ability
    python src/translate_cherokee.py translate-logic-items
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

EN2CHR_SYSTEM = (
    "Translate the following English sentence into Cherokee (Tsalagi), using "
    "the Cherokee syllabary. Respond with ONLY the Cherokee translation, no "
    "explanation."
)
CHR2EN_SYSTEM = (
    "Translate the following Cherokee (Tsalagi) sentence into English. "
    "Respond with ONLY the English translation, no explanation."
)


def translate(text, direction, model, temperature):
    system = EN2CHR_SYSTEM if direction == "en2chr" else CHR2EN_SYSTEM
    return call_model(text, system_prompt=system, model=model, temperature=temperature)


def try_chrf(hyp, ref):
    if not hyp or not ref:
        return None
    try:
        from sacrebleu.metrics import CHRF
        return CHRF().sentence_score(hyp, [ref]).score
    except ImportError:
        return None


def cmd_test_ability(args):
    with open(os.path.join(DATA_DIR, "cherokee_parallel_sample.json"), encoding="utf-8") as f:
        dataset = json.load(f)

    rows = []
    for item in dataset["items"]:
        model_chr = translate(item["english"], "en2chr", args.model, args.temperature)
        log_trial(model=args.model, temperature=args.temperature,
                  input_data={"id": item["id"], "direction": "en2chr", "text": item["english"]},
                  prompt=item["english"], raw_output=model_chr, parsed_answer=model_chr,
                  gold_label=item["cherokee"], stage="cherokee_translation_ability")

        model_en = translate(item["cherokee"], "chr2en", args.model, args.temperature)
        log_trial(model=args.model, temperature=args.temperature,
                  input_data={"id": item["id"], "direction": "chr2en", "text": item["cherokee"]},
                  prompt=item["cherokee"], raw_output=model_en, parsed_answer=model_en,
                  gold_label=item["english"], stage="cherokee_translation_ability")

        back_en = translate(model_chr, "chr2en", args.model, args.temperature)
        log_trial(model=args.model, temperature=args.temperature,
                  input_data={"id": item["id"], "direction": "chr2en_backtranslation", "text": model_chr},
                  prompt=model_chr, raw_output=back_en, parsed_answer=back_en,
                  gold_label=item["english"], stage="cherokee_translation_ability")

        rows.append({
            "id": item["id"],
            "reference_english": item["english"],
            "reference_cherokee": item["cherokee"],
            "model_en2chr": model_chr,
            "chrf_en2chr_vs_reference": try_chrf(model_chr, item["cherokee"]),
            "model_chr2en_of_reference": model_en,
            "chrf_chr2en_vs_reference": try_chrf(model_en, item["english"]),
            "model_backtranslation_en": back_en,
            "chrf_backtranslation_vs_source": try_chrf(back_en, item["english"]),
        })

    df = pd.DataFrame(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "cherokee_translation_ability.csv")
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    for col in ["chrf_en2chr_vs_reference", "chrf_chr2en_vs_reference", "chrf_backtranslation_vs_source"]:
        if df[col].notna().any():
            print(f"{col}: mean chrF = {df[col].mean():.1f}")
    print(
        "\nIf these chrF scores are low, do not rely on this model to generate "
        "'ground truth' Cherokee for the logic pilot -- treat its output only "
        "as a candidate translation requiring independent human review."
    )


def cmd_translate_logic_items(args):
    with open(os.path.join(DATA_DIR, "week1_cleaned.json"), encoding="utf-8") as f:
        dataset = json.load(f)

    out_items = []
    for item in dataset["items"]:
        english_source = " ".join(item["premises_nl"]) + " Question: " + item["query_nl"]

        chr_candidate = translate(english_source, "en2chr", args.model, args.temperature)
        log_trial(model=args.model, temperature=args.temperature,
                  input_data={"id": item["id"], "direction": "en2chr", "text": english_source},
                  prompt=english_source, raw_output=chr_candidate, parsed_answer=chr_candidate,
                  gold_label=None, stage="cherokee_logic_item_translation")

        back_translation = translate(chr_candidate, "chr2en", args.model, args.temperature)
        log_trial(model=args.model, temperature=args.temperature,
                  input_data={"id": item["id"], "direction": "chr2en_backtranslation", "text": chr_candidate},
                  prompt=chr_candidate, raw_output=back_translation, parsed_answer=back_translation,
                  gold_label=english_source, stage="cherokee_logic_item_translation")

        chrf_score = try_chrf(back_translation, english_source)
        new_item = dict(item)
        new_item["english_source"] = english_source
        new_item["formal_semantics"] = {"premises": item["premises_formal"], "query": item["query_formal"]}
        new_item["cherokee_candidate_translation"] = chr_candidate
        new_item["back_translation"] = back_translation
        new_item["translation_validation"] = {
            "auto_backtranslation_chrf": chrf_score,
            "human_reviewed": False,
            "note": ("GPT-generated Cherokee is a CANDIDATE only. A fluent Cherokee "
                     "speaker/reviewer should confirm before treating this as ground "
                     "truth, per the plan's Part D instruction."),
        }
        out_items.append(new_item)

    out_path = os.path.join(DATA_DIR, "week1_cherokee.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"meta": dataset["meta"], "items": out_items}, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(out_items)} translated items to {out_path}")
    print("Reminder: translation_validation.human_reviewed is False for all items "
          "until someone with Cherokee fluency checks them.")


def main():
    parser = argparse.ArgumentParser(description="Part D: Cherokee translation ability test + logic-item translation.")
    parser.add_argument("stage", choices=["test-ability", "translate-logic-items"])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    args = parser.parse_args()

    if args.stage == "test-ability":
        cmd_test_ability(args)
    else:
        cmd_translate_logic_items(args)


if __name__ == "__main__":
    main()
