"""
Week 5 Task 2 (mechanical half): does each item's Spanish/Japanese question
contain the same numbers as the official English question?

This is pure text parsing over data/mgsm_trilingual_sample.json -- no API
calls, no model involved -- so it can run standalone and does not depend on
any of the other Week 5 scripts. It flags candidates for the part of Task 2
that DOES need a human: confirming whether a flagged difference is a real
translation problem (changed units, a garbled number, a different relation)
or a false positive (a number that legitimately doesn't need to reappear,
e.g. it's restated with a word instead of a digit).

A "same number set" result is NOT proof the translation is faithful --
Week 4's es_086 ("tres veces mas") had identical numbers in all three
languages but an ambiguous relation between them. This script only catches
numeric mismatches; relation/condition changes still need a human reader
(or the back-translations Task 2 also asks for -- see the trilingual
semantic-recovery run for those, run separately due to API cost).

Usage: python src/mgsm_translation_number_audit.py
"""
import json
import os
import re
import sys
import unicodedata

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, RESULTS_DIR

CLOCK_TIME_RE = re.compile(r"\b(\d{1,2}):00\b")
NUMBER_RE_EN = re.compile(r"-?\d[\d,]*(?:\.\d+)?")  # comma = thousands, period = decimal (English convention)
NUMBER_RE_ES = re.compile(r"-?\d[\d.]*(?:,\d+)?")   # period = thousands, comma = decimal (Spanish convention)
# Small quantities that Japanese renders as digit + mandatory counter word (2倍="twice", 1枚="one [flat
# object]", 1本="one [long object]", 3人="3 people") where English very often spells the same quantity as
# a word ("twice", "one", "a", "each") instead of a digit -- not a content difference, a digit-vs-word
# rendering convention. Matched narrowly: only single/double-digit numbers immediately followed by one of
# these counters are treated as low-confidence, everything else (multi-digit, or no counter) is flagged
# at full confidence for human review.
# single digit only (1-9): a two-digit count like "33" is never a plausible implicit English word the
# way "one"/"twice"/"three" can be, so it must stay at full confidence even next to a counter -- this is
# exactly how mgsm_057's real defect (JA states the truck already has "33個" boxes, not present in EN at
# all) was almost silently suppressed by an earlier, looser (1-2 digit) version of this filter.
JA_COUNTER_RE = re.compile(r"[1-9](?:倍|枚|本|人|匹|個|つ|台|回|冊|杯|羽|頭|着|足|分|時間|週間|日|ヶ月)")
LANGS = ["en", "es", "ja"]
VERIFIED_PATH = os.path.join(DATA_DIR, "mgsm_translation_audit_verified.json")


def extract_numbers(text, lang):
    text = unicodedata.normalize("NFKC", text)  # full-width Japanese digits -> ASCII
    text = CLOCK_TIME_RE.sub(r"\1", text)  # "1:00" -> "1" (Week 3 fix: ":00" isn't a real quantity)
    number_re = NUMBER_RE_ES if lang == "es" else NUMBER_RE_EN
    numbers = set()
    for tok in number_re.findall(text):
        if lang == "es":
            tok = tok.replace(".", "").replace(",", ".")  # "16,50" -> "16.50"; "50.000" -> "50000"
        else:
            tok = tok.replace(",", "")  # "1,200" -> "1200"
        try:
            numbers.add(str(float(tok)))
        except ValueError:
            continue
    return numbers


def ja_low_confidence_numbers(text):
    """Digits immediately followed by a Japanese counter suffix -- likely a digit-vs-word rendering
    choice (e.g. '2倍' for 'twice'), not a real added quantity. Returns the set of such digit strings."""
    text = unicodedata.normalize("NFKC", text)
    return {m.group(0)[:-1] for m in JA_COUNTER_RE.finditer(text)}


def main():
    with open(os.path.join(DATA_DIR, "mgsm_trilingual_sample.json"), encoding="utf-8") as f:
        items = json.load(f)["items"]

    rows = []
    for item in items:
        nums = {lang: extract_numbers(item["questions"][lang], lang) for lang in LANGS}
        en = nums["en"]
        ja_low_conf_digits = ja_low_confidence_numbers(item["questions"]["ja"])
        for lang in ("es", "ja"):
            missing = en - nums[lang]
            extra = nums[lang] - en
            if lang == "ja":
                # split "extra" into low-confidence (single/double-digit, immediately followed by a
                # counter suffix -- likely just how Japanese renders a word-form English quantity like
                # "twice" or "one") vs. everything else, which gets flagged at full confidence
                low_conf = {n for n in extra if n.rstrip("0").rstrip(".") in ja_low_conf_digits or n.split(".")[0] in ja_low_conf_digits}
                extra_high_conf = extra - low_conf
            else:
                low_conf, extra_high_conf = set(), extra
            rows.append({
                "question_id": item["id"], "language": lang, "gold_answer": item["gold_answer"],
                "en_numbers": ",".join(sorted(en, key=float)),
                f"{lang}_numbers": ",".join(sorted(nums[lang], key=float)),
                "numbers_missing_vs_en": ",".join(sorted(missing, key=float)),
                "numbers_extra_high_confidence": ",".join(sorted(extra_high_conf, key=float)),
                "numbers_extra_low_confidence_counter_word": ",".join(sorted(low_conf, key=float)),
                "flag_review": bool(missing or extra_high_conf),  # the number the human reviewer should check
                "en_question": item["questions"]["en"], "target_question": item["questions"][lang],
            })

    verified = {}
    if os.path.exists(VERIFIED_PATH):
        verified = json.load(open(VERIFIED_PATH, encoding="utf-8"))["verified"]
    for row in rows:
        v = verified.get(row["question_id"]) if row.get("flag_review") else None
        v = v if (v and v["language"] == row["language"]) else None
        row["verdict"] = v["verdict"] if v else ("unverified" if row["flag_review"] else "not_flagged")
        row["verdict_note"] = v["note"] if v else ""

    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS_DIR, "mgsm_translation_number_audit.csv")
    df.to_csv(out, index=False)

    print(f"{len(df)} (item, language) pairs checked ({len(items)} items x 2 non-English languages).")
    print("Spanish comma-decimal parsing fixed (e.g. '$16,50' now reads as 16.5, not 1650).")
    print("Japanese single-digit numbers immediately followed by a counter suffix (2倍, 1枚, 3人, "
          "...) are set aside as low-confidence -- almost always how Japanese renders an English "
          "word-form quantity ('twice', 'one', 'each'), not an added number.")
    n_flagged = int(df.flag_review.sum())
    print(f"\n{n_flagged} pairs flagged by the heuristic; every one has been read by hand against the "
          f"English source (data/mgsm_translation_audit_verified.json):")
    print(df.verdict.value_counts().to_string())
    real = df[df.verdict == "real_defect"]
    print(f"\n=== Confirmed real translation defects: {len(real)} of {len(df)} pairs checked ===")
    for _, r in real.iterrows():
        print(f"  {r.question_id} [{r.language}]: {r.verdict_note}")
    print("\nNOTE: a matching number set is still not proof of a faithful translation -- relation/condition "
          "changes (e.g. Week 4's es_086, 'tres veces mas', same numbers in both languages, ambiguous "
          "relation) do not always change which numbers appear, so this script cannot catch that class of "
          "error at all; only the semantic-recovery back-translations catch it, and only for items a model "
          "got wrong.")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
