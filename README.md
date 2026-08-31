# ACL 2027 Week 2 — Mini-MGSM Replication + Cherokee Semantic-vs-Reasoning Pilot

Reproducible pipeline for the Week 2 plan: (A) a Mini-MGSM replication measuring
end-to-end accuracy across English/French/Swahili, (B) a semantic-recovery-vs-
reasoning probe on the same data, (C) 20 cleaned logic problems with symbolic
ground truth, and (D) a Cherokee translation pilot that runs the same four-way
English / Cherokee / Cherokee-Parse / Oracle-Symbolic design from Section 6.

Everything that touches the model goes through the OpenAI API (`src/run_openai.py`).
Claude Code was only used to build this scaffolding, per the plan's stated principle.

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

**This repo ships the pipeline, not results.** No experiments have been run
here — there's no API key in this environment. Every script below is smoke-tested
on its parsing/logging logic but has not made a live API call. Run them yourself
(or hand me a key and I will) to actually populate `results/`.

## Data already prepared

- `data/mgsm_sample.json` — 40 problems sampled (seed 42) from the real
  [MGSM benchmark](https://huggingface.co/datasets/juletxara/mgsm) (human-translated
  GSM8K), aligned across English, French (high-resource), and Swahili
  (lower-resource) by source index.
- `data/week1_cleaned.json` — 20 freshly-authored logic problems (5 each:
  negation, conditional, quantifier, spatial), Entailed/Contradicted/Unknown
  labels, formal symbolic ground truth, one core phenomenon per item, no
  external world knowledge required. **No Week 1 file existed in this
  environment**, so these were authored fresh against the Part C cleaning
  criteria rather than cleaned from a prior batch — swap in your real Week 1
  items here if you have them, keeping the same schema.
- `data/cherokee_parallel_sample.json` — 12 real Cherokee–English sentence
  pairs sampled from the [ChrEn](https://github.com/ZhangShiyue/ChrEn) dev
  split, for testing raw translation ability before trusting the model on our
  own items.

## Run order

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
  cherokee_parallel_sample.json   Part D.5.1 input (real ChrEn pairs)
  week1_cherokee.json         generated by translate_cherokee.py (not checked in)
src/
  config.py                   model/paths config, reads .env vars
  run_openai.py                API wrapper + unified trial logger
  run_mgsm.py                  Part A
  semantic_probe.py            Part B
  translate_cherokee.py        Part D (5.1 + 5.2)
  run_logic_pilot.py           Section 6 (conditions A-D)
  evaluate.py                   Section 9 tables
results/                      generated: mgsm_results.csv, mgsm_semantic_probe.csv,
                               cherokee_translation_ability.csv, cherokee_pilot.csv,
                               table1_mini_mgsm.csv, table2_cherokee_pilot.csv,
                               api_call_log.jsonl
```

## Deliverable 6 (5–10 sentences on the bottleneck)

Not written yet — it depends on actual numbers, which don't exist until the
scripts above are run against a live API key. Once `results/table1_mini_mgsm.csv`
and `results/table2_cherokee_pilot.csv` exist, the write-up is: compare
end-to-end accuracy to semantic-recovery rate and P(reasoning\|semantic
recovery) per language/condition, per the interpretation rules in Sections 3
and 4 of the plan (`Semantic-Recovery-vs-Reasoning-Cherokee-Experiment-
Plan.pdf`).
