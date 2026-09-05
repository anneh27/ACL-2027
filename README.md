# ACL 2027 Week 2 — Mini-MGSM Replication + Cherokee Semantic-vs-Reasoning Pilot

Reproducible pipeline for the Week 2 plan: (A) a Mini-MGSM replication measuring
end-to-end accuracy across English/French/Swahili, (B) a semantic-recovery-vs-
reasoning probe on the same data, (C) 20 cleaned logic problems with symbolic
ground truth, and (D) a Cherokee translation pilot that runs the same four-way
English / Cherokee / Cherokee-Parse / Oracle-Symbolic design from Section 6.

Everything that touches the model was originally meant to go through the OpenAI
API (`src/run_openai.py`), per the plan's stated principle that Claude Code
should be a programming assistant, not the experimental subject. **That
principle was deliberately overridden for this run, on the user's explicit
instruction** ("don't use openai, use your model"), because no OpenAI or
Anthropic API key exists in this environment. `results/` was populated by
having Claude (Sonnet 5) answer the experiment prompts directly inside the
conversation instead of via a live API call -- see "How results/ was
actually populated" below for exactly how, and why that's a real deviation
worth reading before trusting these numbers.

## Setup

```bash
cd acl2027-mgsm-cherokee
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then edit .env with your real key
export $(grep -v '^#' .env | xargs)   # or use direnv / python-dotenv
```

**About the model name.** The plan calls the pilot model "GPT-5.6" — that's the
course's label, not guaranteed to be the literal string the live API expects.
Set `OPENAI_MODEL` in `.env` to whatever model id your account actually has
access to; every script reads it from there (default temperature `0.0` for
reproducibility, also overridable).

**Two ways to populate `results/`.** (1) The OpenAI-backed scripts
(`run_mgsm.py`, `semantic_probe.py`, `translate_cherokee.py`,
`run_logic_pilot.py`) are complete and smoke-tested on their parsing/grading
logic, but have never made a live call -- there's no API key in this
environment. Run them yourself, with a real key, for a properly reproducible
result anyone else could regenerate. (2) The `results/` currently checked in
were instead produced by `src/interactive_pilot_mgsm.py` and
`src/interactive_pilot_logic.py`, which do NOT call any API -- they replay
answers Claude generated directly in conversation. Re-running (1) will
overwrite what (2) produced; that's expected, (1) is the real, rerunnable
pipeline the plan asks for.

## How `results/` was actually populated (interactive-Claude run)

The user explicitly asked me to answer the experiments myself rather than
wait for an API key. Two things made that harder to do validly than it
sounds, and both are worth knowing before trusting these numbers:

1. **I built this dataset, so I'm not blind to it.** For Part A/B (MGSM) and
   Conditions A/D of the logic pilot (English Natural, Oracle Symbolic), I
   answered directly -- there's no real "blindness" concern there since
   those conditions give the model the full input it needs anyway. But
   Conditions B and C (Cherokee Natural, Cherokee Parse) are supposed to
   test whether *semantics can be recovered from Cherokee alone* -- and I
   had just finished writing the Cherokee translations myself, in the same
   conversation, so grading my own output right after writing it would
   mostly test memory, not comprehension. To fix this, Conditions B and C
   were answered by two **fresh subagents** with zero access to this
   conversation -- they received only the raw Cherokee text and the same
   task instructions `run_logic_pilot.py` sends to a real model, nothing
   else. That's a much closer analogue to an independent, stateless API
   call. (Condition A/D's shared authorship with the dataset, and the
   MGSM data having already been read into context while building the
   pipeline, are residual contamination risks that weren't similarly fixed
   -- flagged here rather than hidden.)
2. **Scale was cut down for tractability.** Manually generating an answer
   doesn't scale like an API call. Part A ran on 10 of the 40 available
   MGSM items x 3 languages (30 completions); Part B (which doubles the load
   with its two-stage design) ran on a 5-item subset of those x 3 languages
   x 2 stages (30 completions). The 20-item Cherokee logic pilot ran at full
   size (80 completions across 4 conditions) since it was already small.
   `src/interactive_pilot_mgsm.py` and `src/interactive_pilot_logic.py`
   contain the exact reasoning/translations used, with comments on scope.

## Data already prepared

- `data/mgsm_sample.json` — 40 problems sampled (seed 42) from the real
  [MGSM benchmark](https://huggingface.co/datasets/juletxara/mgsm) (human-translated
  GSM8K), aligned across English, French (high-resource), and Swahili
  (lower-resource) by source index.
- `data/week1_cleaned.json` — the actual Week 1 "20 Questions" set (5 each:
  negation, conditional, quantifier, spatial), cleaned per Part C: True/False
  relabeled to Entailed/Contradicted/Unknown, a formal symbolic ground truth
  added for every item, one core phenomenon per item where possible. Three
  corrections were made to the original answer key: items 8, 10, and 14 are
  textbook fallacies (affirming the consequent, denying the antecedent,
  undistributed middle) whose premises neither entail nor contradict the
  query — the original key marked all three False, but under strict
  entailment semantics the correct label is Unknown. Items 18 and 19 also
  had an implicit lexical-semantics dependency (behind/in-front-of,
  east-of/west-of are converse relations, never stated as premises); each
  now carries an explicit `forall(X,Y, ... <-> ...)` meaning-postulate
  premise so the ground truth is self-contained. Full rationale for every
  change is in each item's `cleaning_note` field and in `meta.cleaning_changes`.
- `data/cherokee_parallel_sample.json` — 12 real Cherokee–English sentence
  pairs sampled from the [ChrEn](https://github.com/ZhangShiyue/ChrEn) dev
  split, for testing raw translation ability before trusting the model on our
  own items. **Not actually used for that gate in this run** -- by the time
  translation was attempted, both sides of every pair here had already been
  read into the same conversation doing the translating, so a "blind"
  ability test against this specific file would've been testing
  pattern-matching against a visible reference, not real ability. See
  "How `results/` was actually populated" below for what replaced this gate.
- `data/week1_cherokee.json` — candidate Cherokee translations of the 20
  logic items (Part D.5.2), produced by Claude directly in conversation.
  Every item carries a `translator_confidence` field (mostly "low" or "very
  low") -- an honest self-assessment, not a formality; see the Cherokee
  pilot results below for how well that self-assessment tracked actual
  downstream failure. `translation_validation.human_reviewed` is `false`
  throughout and should stay that way until a fluent Cherokee speaker
  reviews it.

## Run order

**With a real OpenAI key** (the reproducible pipeline the plan asks for):

```bash
# Part A — end-to-end accuracy (Section 2.2)
python src/run_mgsm.py                 # add --limit 5 to smoke-test first

# Part B — semantic recovery vs. reasoning (Section 3)
python src/semantic_probe.py

# Part D.5.1 — test translation ability against real parallel data first
python src/translate_cherokee.py test-ability

# Part D.5.2 — translate the 20 logic items into candidate Cherokee
python src/translate_cherokee.py translate-logic-items

# Section 6 — the four-condition pilot (needs the previous step's output)
python src/run_logic_pilot.py

# Section 9 — aggregate everything into the two meeting tables
python src/evaluate.py
```

**What actually populated `results/` this run** (no key available, see above):

```bash
python src/interactive_pilot_mgsm.py    # Part A + B
python src/interactive_pilot_logic.py   # Section 6 (needs data/week1_cherokee.json, already checked in)
python src/evaluate.py                  # Section 9 — same aggregation script either way
```

Every trial (model, temperature, prompt, raw output, parsed answer, gold
label) is appended to `results/api_call_log.jsonl` regardless of which script
ran it — that's the durable experiment log the plan asks for in Section 1.

## Design decisions worth knowing about

- **Semantic recovery on MGSM is a proxy, documented as such.** The plan's
  worked example (Mary/apples) assumes one fixed slot schema. Real MGSM
  problems vary in structure (multi-step, different operation counts), so a
  single fixed schema doesn't generalize without hand-annotating gold slots
  per item — out of scope for a small pilot. `semantic_probe.py` instead
  checks whether the model's structured extraction recovers every numeric
  quantity in the problem (numbers are invariant across MGSM's translations,
  so this is automatically and objectively gradable). This is *a* proxy for
  full slot recovery, not identical to it. If you want the exact schema from
  the plan, you'd need to hand-annotate `initial_owner`/`operation`/etc. per
  item — happy to do that for a subset if you want tighter fidelity.
- **Cherokee is never trusted as ground truth.** `translate_cherokee.py`
  always marks `translation_validation.human_reviewed: false`; treat Table 2
  numbers as provisional until someone with Cherokee fluency checks the
  candidate translations in `data/week1_cherokee.json`.
- **Condition C (Cherokee Parse) grading is strict-ish string matching** after
  normalizing whitespace/case and sorting premises — it will under-count
  semantically-equivalent-but-differently-worded parses (e.g. `not(P)` vs
  `¬P`). Worth spot-checking manually if that number looks surprisingly low.

## Project structure

```
data/
  mgsm_sample.json            Part A/B input (real MGSM, 3 languages)
  week1_cleaned.json          Part C: 20 logic problems + symbolic ground truth
  cherokee_parallel_sample.json   Part D.5.1 input (real ChrEn pairs, not used as a gate this run -- see above)
  week1_cherokee.json         Claude's candidate Cherokee translations (checked in; regenerate with translate_cherokee.py if you get a real key)
src/
  config.py                   model/paths config, reads .env vars
  run_openai.py                API wrapper + unified trial logger (needs a real key)
  run_mgsm.py                  Part A -- OpenAI-backed, not yet run
  semantic_probe.py            Part B -- OpenAI-backed, not yet run
  translate_cherokee.py        Part D (5.1 + 5.2) -- OpenAI-backed, not yet run
  run_logic_pilot.py           Section 6 (conditions A-D) -- OpenAI-backed, not yet run
  interactive_pilot_mgsm.py    Part A/B, actually run: Claude's own answers, replayed through the same grading code
  interactive_pilot_logic.py   Section 6, actually run: Claude's answers (A/D) + two blind subagents' answers (B/C)
  evaluate.py                   Section 9 tables
results/                      generated: mgsm_results.csv, mgsm_semantic_probe.csv,
                               cherokee_translation_ability.csv, cherokee_pilot.csv,
                               table1_mini_mgsm.csv, table2_cherokee_pilot.csv,
                               api_call_log.jsonl
```

## Deliverable 6 (5–10 sentences on the bottleneck)

On the Mini-MGSM pilot (n=10 items x 3 languages), Claude scored 100% end-to-
end accuracy in English, French, *and* Swahili, so this slice shows no
multilingual reasoning gap to diagnose -- a ceiling effect from using a
strong model at small scale, not evidence that no such gap exists generally.
The semantic-recovery proxy (Part B) came out at only 60%, but reasoning-
from-structure was still 100% correct in every case, including the
"failures" -- inspection showed those failures were the proxy penalizing
efficient extraction (omitting numbers irrelevant to the answer, like a
game's total duration when only the per-half score matters) rather than any
real comprehension gap; this is a known limitation of that digit-matching
proxy, not a finding about language. The Cherokee logic pilot is where the
real signal is: English Natural and Oracle Symbolic both scored 100/100%,
Cherokee Natural scored 95% (graded blind, by a subagent with no access to
the English originals), and Cherokee Parse scored 0% under strict formula-
matching but 95% under a structural re-grading that ignores predicate/entity
naming and checks only connective, negation polarity, quantifier type, and
argument shape. Reading the blind parses directly shows *why*: entity and
predicate names were very often guessed wrong (e.g. a sentence about a boy
being at school was parsed as being about someone reading), yet the logical
skeleton -- negation, modus ponens, the exact fallacy shape of affirming-the-
consequent, quantifier scope, transitive/converse spatial relations -- came
through essentially intact, meaning the 95% success in both B and C is
substantially driven by surface structural cues (repeated markers like ᎥᏝ
for negation, ᎢᏳᏃ for "if") rather than full lexical semantic recovery. The
one item that failed in both B and C is the *same* item (cond_04), and it's
the one I had independently flagged, before any grading happened, as my
lowest-confidence translation -- a clean self-consistency check that the
failure traces to a translation defect, not a reasoning defect: when the
translation preserved the logical skeleton, both direct classification and
downstream reasoning succeeded together; when translation broke, both broke
together. That pattern -- correctness rising and falling with faithfulness
of the Cherokee-to-logical-form mapping rather than with anything downstream
of it -- is exactly the signature Section 3/4 of the plan says points to a
semantic-recovery bottleneck rather than a reasoning bottleneck. The biggest
caveat: this used Claude Sonnet 5, not the plan's intended GPT-5.6, as the
subject, so these numbers describe Claude's specific profile (strong
symbolic/formal reasoning, unusually good at extracting structure from
otherwise-flawed low-resource-language text) rather than a general claim
about where multilingual reasoning bottlenecks sit -- rerunning
`run_mgsm.py` / `run_logic_pilot.py` against a real GPT-5.6 key is needed
before treating this as more than a pilot on the pipeline itself.
