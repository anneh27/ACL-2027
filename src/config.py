import os

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_SRC_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
LOG_PATH = os.path.join(RESULTS_DIR, "api_call_log.jsonl")

# The Week 2 plan names the pilot model "GPT-5.6". That string is a course-internal
# label, not necessarily the exact identifier the live OpenAI API expects -- set
# OPENAI_MODEL to whatever model id your account actually has access to.
DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.6")
DEFAULT_TEMPERATURE = float(os.environ.get("OPENAI_TEMPERATURE", "0.0"))
