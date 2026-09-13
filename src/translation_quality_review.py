"""
Week 3, Task 2: translation-quality review of the 20 Cherokee candidate
translations in data/week1_cherokee.json.

Each item's `translator_confidence` field (set in Week 2, BEFORE any grading
happened) was a pure a-priori self-assessment made while writing the
translation. This script instead scores each item on the evidence that
actually exists: how much of the gold formal structure survived in the
blind subagent's Condition-C parse (see interactive_pilot_logic.py --
CONDITION_C_RAW is that subagent's output, produced with zero access to
this conversation's English originals or gold labels). That's a real,
if indirect, signal of what the Cherokee actually conveyed to an
independent reader, as opposed to what I intended it to say.

Five dimensions are checked per item (marked n/a where the item's reasoning
type doesn't exercise that dimension):
  - negation_preserved: does the parse's negation pattern (none / single /
    double) match gold?
  - conditional_preserved: does the parse have the right if/then shape,
    with the same premise asserted (antecedent vs. consequent) as gold --
    this is what distinguishes a preserved argument from a preserved-
    looking-but-broken one (see cond_04).
  - quantifier_scope_preserved: same quantifier type (forall/exists) and
    the same variable-sharing pattern across clauses as gold.
  - entities_preserved: are the same proper-noun referents individually
    recognizable (allowing for phonetic transliteration), not merely
    "some name is present."
  - predicates_preserved: do the predicates denote the same or a clearly
    adjacent real-world relation as gold (not just "a relation exists").

confidence_rating (High/Medium/Low) is my overall synthesis of the above,
NOT a re-run of the a-priori translator_confidence field. Where the two
disagree, that's flagged explicitly as a revision -- per Task 2's request
for "a record of revisions."
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, RESULTS_DIR
from interactive_pilot_logic import CONDITION_C_RAW

# Structured, dimension-by-dimension review. "predicates_preserved" values:
# "exact" (>=1 predicate name literally matches gold), "adjacent" (wrong
# literal word but same/close real-world relation), "wrong" (different
# relation or domain entirely).
REVIEW = {
    "neg_01": {"negation": True, "conditional": "n/a", "quantifier_scope": "n/a", "entities": "partial (Liam->Lavim, recognizable)", "predicates": "wrong (at-school -> reads)", "rating": "Medium", "reason": "Negation structure exact and name phonetically recognizable, but the core predicate (being at school) was not conveyed -- the reader inferred an unrelated action (reading)."},
    "neg_02": {"negation": True, "conditional": "n/a", "quantifier_scope": "n/a", "entities": "no (door not recognizable as the referent)", "predicates": "wrong (open -> on_fire)", "rating": "Low", "reason": "Only the negation marker came through; neither the entity (door) nor the predicate (open) was recovered by an independent reader."},
    "neg_03": {"negation": True, "conditional": "n/a", "quantifier_scope": "n/a", "entities": "partial (Nina->Nine, recognizable; bicycle->generic vehicle/possession)", "predicates": "wrong (owns -> high)", "rating": "Medium", "reason": "Name and general 'possessed object' concept partially came through; the specific predicate (ownership) and object (bicycle) did not."},
    "neg_04": {"negation": True, "conditional": "n/a", "quantifier_scope": "n/a", "entities": "no", "predicates": "wrong (light-on -> digs)", "rating": "Medium", "reason": "REVISED UP from the a-priori 'very low' rating: the double-negation structure -- the hardest grammatical feature in this whole set -- was reconstructed exactly by a blind reader, even though the content words were not. Structure survived better than expected; content did not."},
    "neg_05": {"negation": True, "conditional": "n/a", "quantifier_scope": "n/a", "entities": "partial (Daniel->Danyili, recognizable)", "predicates": "wrong (lost-key -> wanted_quickly)", "rating": "Medium", "reason": "Name recognizable and negation exact; this item also had a stray corrupted character caught and fixed during Week 2 review, which may have degraded translation quality further before the fix."},
    "cond_01": {"negation": "n/a", "conditional": True, "quantifier_scope": "n/a", "entities": "partial (Emma->Ema, School recognized)", "predicates": "adjacent (studies/passes -> starts-school/understands)", "rating": "Medium", "reason": "Modus ponens shape exact, name recognizable, and both substituted predicates are thematically close to the originals (school-starting for studying, understanding for passing) rather than unrelated."},
    "cond_02": {"negation": "n/a", "conditional": True, "quantifier_scope": "n/a", "entities": "partial (Mia->Maya recognizable; alarm->Hunter not recognizable)", "predicates": "adjacent (leaves-building -> runs-home is close; rings->hunts is not)", "rating": "Medium", "reason": "Conditional shape exact and one entity/predicate pair recognizable; the alarm/rings pair was lost entirely."},
    "cond_03": {"negation": "n/a", "conditional": True, "quantifier_scope": "n/a", "entities": "partial (Noah->Nowa recognizable)", "predicates": "adjacent (exercise/tired -> did-it/works-hard, thematically close)", "rating": "Medium", "reason": "The exact fallacy shape (affirming the consequent) was independently reconstructed by the blind reader and correctly classified as unknown -- the argument's logical form survived even though the specific predicates drifted to an adjacent theme."},
    "cond_04": {"negation": "FAILED -- premise 2 parsed as affirmative, gold has it negated", "conditional": "structure present but polarity broken", "quantifier_scope": "n/a", "entities": "no (Ava->Ewi not recognizable)", "predicates": "wrong", "rating": "Low", "reason": "The one genuine translation failure in the set. I had independently flagged this item 'very low confidence, close to gloss-level guessing' before any grading occurred; the blind parse confirms it -- the negation on the second premise was lost, turning a valid modus-tollens argument into a different (and differently-classified) pattern. This is also the only item where both Condition B and Condition C failed."},
    "cond_05": {"negation": True, "conditional": True, "quantifier_scope": "n/a", "entities": "partial (Leo->Liyo recognizable)", "predicates": "adjacent (button/machine -> plants/grows -- wrong domain but internally coherent)", "rating": "Medium", "reason": "This item has the most complex negation pattern in the set (negated second premise AND negated query, denying-the-antecedent shape) and BOTH negations were correctly reconstructed -- the strongest structural result among the conditional items, despite the predicates drifting to an unrelated (but self-consistent) planting/growing theme."},
    "quant_01": {"negation": "n/a", "conditional": "n/a", "quantifier_scope": True, "entities": "exact (Maria)", "predicates": "wrong (painter/artist -> swimmer/writer)", "rating": "Medium", "reason": "Universal-instantiation structure exact and the proper name exactly preserved; both occupation predicates were lost."},
    "quant_02": {"negation": "n/a", "conditional": "n/a", "quantifier_scope": True, "entities": "no (Felix->Pilgisi not recognizable)", "predicates": "wrong but notably confused, not random (cat/reptile -> dog/cat -- the reader's own 'cat' guess for the wrong slot suggests real but misplaced animal-vocabulary recognition)", "rating": "Medium", "reason": "Universal-negative structure with correctly-placed negation preserved; predicate confusion is patterned (animal-category vocabulary was recognized as such, just assigned to the wrong slot) rather than arbitrary."},
    "quant_03": {"negation": "n/a", "conditional": "n/a", "quantifier_scope": True, "entities": "n/a (no proper names in this item)", "predicates": "exact + adjacent (teacher exact; employee/musician -> hard_worker/singer, both close synonyms)", "rating": "High", "reason": "The best-preserved quantifier item: exact universal+existential-conjunction structure, one predicate name literally exact, and the other two are close paraphrases rather than wrong-domain guesses."},
    "quant_04": {"negation": "n/a", "conditional": "n/a", "quantifier_scope": True, "entities": "n/a", "predicates": "wrong domain, one adjacent (student/athlete -> runner/hunter wrong; musician->singer adjacent)", "rating": "Medium", "reason": "The subtle shared-subject-class existential pattern that makes this item's gold answer 'unknown' (undistributed middle) was exactly reconstructed, which matters more for this item than predicate accuracy."},
    "quant_05": {"negation": "n/a", "conditional": "n/a", "quantifier_scope": True, "entities": "n/a", "predicates": "exact x2 (doctor, writer both exact; pilot->thief wrong but consistent throughout)", "rating": "High", "reason": "Tied for best-preserved item overall: full structure including negation placement in both premise and query, plus two of three predicates literally exact."},
    "spat_01": {"negation": "n/a", "conditional": "n/a", "quantifier_scope": "n/a", "entities": "weak", "predicates": "wrong (left_of -> near, loses directionality entirely)", "rating": "Low", "reason": "The transitive chain structure survived, but the specific spatial relation this item exists to test -- a directional, asymmetric relation -- was flattened into a generic symmetric 'near', which changes the item's actual logical content, not just its wording."},
    "spat_02": {"negation": "n/a", "conditional": "n/a", "quantifier_scope": "n/a", "entities": "exact x3 (Mia->Maya, Noah->Nowa, Liam->Lavim, all recognizable)", "predicates": "exact (north_of)", "rating": "High", "reason": "The best-translated item in the entire set: exact relation word, all three names independently recognizable, exact transitivity structure."},
    "spat_03": {"negation": "n/a", "conditional": "n/a", "quantifier_scope": "n/a", "entities": "weak (chair/desk not recognizable as specific objects)", "predicates": "exact (behind, front)", "rating": "High", "reason": "Both relation words came through exactly, AND the converse-relation logic (behind(X,Y) entails front(Y,X)) was independently re-derived by the blind reader without it being stated -- strong evidence the relational meaning, not just the words, was conveyed, even though the specific furniture nouns were not."},
    "spat_04": {"negation": "n/a", "conditional": "n/a", "quantifier_scope": "n/a", "entities": "no", "predicates": "confused (east_of/west_of vocabulary present but polarity assignment differs from gold)", "rating": "Low", "reason": "Entities essentially unrecognizable and the east/west relation assignment is internally different from gold's, though self-consistent enough that the final classification happened to still be correct -- a case where the right answer does not imply the translation was good."},
    "spat_05": {"negation": "n/a", "conditional": "n/a", "quantifier_scope": "n/a", "entities": "partial (red->RedBox exact; blue, green not recognizable)", "predicates": "exact (above)", "rating": "Medium", "reason": "Relation word and transitivity exact, one of three colors exact, two lost."},
}

VALIDATION_NOTICE = (
    "MACHINE-TRANSLATED EXPLORATORY PILOT. No native Cherokee speaker or "
    "language expert has reviewed any of these 20 translations. Every rating "
    "and dimension check above is based on (a) my own a-priori self-assessment "
    "while writing the translation and (b) what an independent blind reader "
    "(a subagent with no access to the English originals) was able to "
    "recover from it -- neither is a substitute for native-speaker review. "
    "Per the plan's own instruction, treat this whole dataset as a "
    "candidate/exploratory artifact, not validated ground truth."
)


def main():
    with open(os.path.join(DATA_DIR, "week1_cherokee.json"), encoding="utf-8") as f:
        cherokee = json.load(f)
    with open(os.path.join(DATA_DIR, "week1_cleaned.json"), encoding="utf-8") as f:
        cleaned = json.load(f)
    reasoning_type_by_id = {it["id"]: it["reasoning_type"] for it in cleaned["items"]}
    a_priori_by_id = {it["id"]: it["translator_confidence"] for it in cherokee["items"]}

    rows = []
    for item_id, r in REVIEW.items():
        rows.append({
            "id": item_id,
            "reasoning_type": reasoning_type_by_id[item_id],
            "a_priori_translator_confidence": a_priori_by_id[item_id],
            "negation_preserved": r["negation"],
            "conditional_preserved": r["conditional"],
            "quantifier_scope_preserved": r["quantifier_scope"],
            "entities_preserved": r["entities"],
            "predicates_preserved": r["predicates"],
            "final_confidence_rating": r["rating"],
            "reason": r["reason"],
            "revised_from_a_priori": "REVISED" in r["reason"] or "genuine translation failure" in r["reason"].lower(),
        })

    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS_DIR, "translation_quality_review.csv")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(out, index=False)

    print(VALIDATION_NOTICE)
    print(f"\nFinal confidence rating distribution (n=20):")
    print(df["final_confidence_rating"].value_counts())
    print(f"\nWrote {len(df)} rows to {out}")
    print("\nItems where evidence revised the a-priori self-rating:")
    for _, r in df.iterrows():
        if "REVISED" in r["reason"]:
            print(f"  {r['id']}: a-priori='{r['a_priori_translator_confidence']}' -> final='{r['final_confidence_rating']}'")
            print(f"    {r['reason']}")


if __name__ == "__main__":
    main()
