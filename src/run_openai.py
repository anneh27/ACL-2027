"""
Thin wrapper around the OpenAI API.

Per the plan's "Experiment logging" principle, every experimental trial is
logged in full (model name/version, prompt, temperature, input, raw output,
parsed answer, gold label) -- call `log_trial()` once per trial from the
calling script, after you've parsed the model's raw output.
"""
import json
import os
import time
from datetime import datetime, timezone

from openai import OpenAI

from config import DEFAULT_MODEL, DEFAULT_TEMPERATURE, LOG_PATH, RESULTS_DIR

_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Export it (or put it in a .env you "
                "source) before running any experiment script."
            )
        _client = OpenAI(api_key=api_key)
    return _client


def call_model(user_prompt, system_prompt=None, model=None, temperature=None):
    """Single chat-completion call. Returns the raw text output. Does not log --
    call log_trial() with the full trial record after parsing the result."""
    model = model or DEFAULT_MODEL
    temperature = DEFAULT_TEMPERATURE if temperature is None else temperature

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    client = _get_client()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content


def log_trial(*, model, temperature, input_data, prompt, raw_output, parsed_answer, gold_label, stage, **extra):
    """Append one fully-specified experimental trial to results/api_call_log.jsonl."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "model": model,
        "temperature": temperature,
        "input": input_data,
        "prompt": prompt,
        "raw_output": raw_output,
        "parsed_answer": parsed_answer,
        "gold_label": gold_label,
    }
    record.update(extra)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
