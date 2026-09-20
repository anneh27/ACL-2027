#!/usr/bin/env bash
# Regenerate the Week 4 three-language MGSM results from the official dataset.
# Requires OPENAI_API_KEY and OPENAI_MODEL in .env (see .env.example).
# Note: the model runs at its default temperature, so a fresh run will not
# reproduce the same outputs as the checked-in results (see results/week4_audit.md).
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
"$PY" src/mgsm_scoring.py                 # scoring self-tests
"$PY" src/fetch_mgsm_trilingual.py        # official en/es/ja MGSM -> data/mgsm_trilingual_sample.json
"$PY" src/run_mgsm_trilingual.py          # 50 questions x 3 languages, real API calls (add --resume to retry failures only)
"$PY" src/mgsm_error_diagnosis.py         # re-sample + back-translate every wrong answer
"$PY" src/verify_scoring_dispute.py       # independent recomputation of the Cherokee label scores
