"""
Numeric answer extraction + normalization + comparison for the MGSM runs.

Scoring rules (Week 4 plan, P3): gold, prediction, and correct are always
saved side by side; the raw model output is never modified. Answers such as
"9", "9.0", "$9", "9 eggs", and "FINAL ANSWER: 9" all normalize to the same
numeric value. Percent signs are dropped ("25%" -> 25) because MGSM gold
answers are the bare number as stated in the problem.

Run `python src/mgsm_scoring.py` to execute the self-tests.
"""
import re
import unicodedata

FINAL_ANSWER_RE = re.compile(r"FINAL\s*ANSWER\s*[:：]\s*(.*)", re.IGNORECASE)

# sign (not preceded by a digit or '.', so "16-3" is not read as -3),
# integer part with optional thousands separators, optional decimals
NUMBER_RE = re.compile(r"(?<![\d.])(-?)(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?")


def _clean(text):
    text = unicodedata.normalize("NFKC", str(text))  # full-width digits/commas/percent -> ASCII
    return text.replace("−", "-")  # unicode minus


def _first_number(text):
    m = NUMBER_RE.search(_clean(text))
    if not m:
        return None
    sign, integer, frac = m.group(1), m.group(2), m.group(3) or ""
    value = float(integer.replace(",", "") + frac)
    if sign == "-":
        value = -value
    return int(value) if value.is_integer() else value


def _last_number(text):
    matches = list(NUMBER_RE.finditer(_clean(text)))
    if not matches:
        return None
    m = matches[-1]
    sign, integer, frac = m.group(1), m.group(2), m.group(3) or ""
    value = float(integer.replace(",", "") + frac)
    if sign == "-":
        value = -value
    return int(value) if value.is_integer() else value


def normalize_answer(answer):
    """Any string/number -> canonical numeric value (int if integral), or None if no number is present."""
    if answer is None:
        return None
    if isinstance(answer, bool):
        return None
    if isinstance(answer, (int, float)):
        return int(answer) if float(answer).is_integer() else float(answer)
    return _first_number(answer)


def parse_prediction(raw_output):
    """Returns (normalized_value, method). method is one of:
       'final_answer_line'   -- number taken from the last 'FINAL ANSWER:' line (the requested format)
       'fallback_last_number' -- no marker found; last number in the output (flagged so it can be audited)
       'none'                -- no number found at all
    """
    if raw_output:
        markers = FINAL_ANSWER_RE.findall(raw_output)
        if markers:
            value = _first_number(markers[-1])
            if value is not None:
                return value, "final_answer_line"
        value = _last_number(raw_output)
        if value is not None:
            return value, "fallback_last_number"
    return None, "none"


def is_correct(prediction, gold):
    p, g = normalize_answer(prediction), normalize_answer(gold)
    if p is None or g is None:
        return False
    return abs(p - g) < 1e-9


def _selftest():
    # the four assertions the plan requires
    assert normalize_answer("9.0") == 9
    assert normalize_answer("$9") == 9
    assert normalize_answer("1,200") == 1200
    assert normalize_answer("-3") == -3
    # integers, negatives, decimals, thousands separators, percentages
    assert normalize_answer("9 eggs") == 9
    assert normalize_answer("FINAL ANSWER: 9") == 9
    assert normalize_answer("0.5") == 0.5
    assert normalize_answer("$1,200.50") == 1200.5
    assert normalize_answer("-1,200") == -1200
    assert normalize_answer("25%") == 25
    assert normalize_answer("１８") == 18  # full-width digits (Japanese output)
    assert normalize_answer("−3") == -3  # unicode minus
    assert normalize_answer(9) == 9 and normalize_answer(9.0) == 9 and normalize_answer(2.5) == 2.5
    assert normalize_answer("no number here") is None
    assert normalize_answer(None) is None
    assert normalize_answer("16-3") == 16  # a hyphen between numbers is not a minus sign
    # prediction parsing
    assert parse_prediction("work...\nFINAL ANSWER: 18") == (18, "final_answer_line")
    assert parse_prediction("**FINAL ANSWER: $1,200**") == (1200, "final_answer_line")
    assert parse_prediction("early FINAL ANSWER: 5\nlater FINAL ANSWER: 7") == (7, "final_answer_line")  # last marker wins
    assert parse_prediction("so it is 42 dollars") == (42, "fallback_last_number")
    assert parse_prediction("") == (None, "none")
    # comparison
    assert is_correct("9.0", 9) and is_correct("$9", "9") and not is_correct("10", 9)
    assert not is_correct(None, 9)
    print("mgsm_scoring self-tests passed")


if __name__ == "__main__":
    _selftest()
