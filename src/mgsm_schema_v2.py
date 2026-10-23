"""
Week 7 Task 3: schema v2 -- fixes the quantity=0 placeholder problem found in
Week 6 (results/week6_task4_error_analysis_summary.md, mgsm_207). Full
rationale and the schema spec itself: schema_v2.json at the repo root.

This is a NEW module, not an edit to src/mgsm_semantic_recovery_trilingual.py
or src/mgsm_semantic_critic_repair.py -- v1's prompts, parsing, and results
are left completely untouched per "new version number, don't overwrite
history." v2 reuses v1's generic infrastructure (call_model,
call_model_messages, verify_isolation) by importing it, since that's
retry/rate-limit/isolation plumbing, not v1 schema content.

v1 format:  a bare JSON array of {quantity, role}.
v2 format:  {"quantities": [...], "target": {"entity", "operation", "output_unit"}}
            -- "target" is a dedicated object, not a quantities-array entry,
            so there is no numeric placeholder for an unknown value.

Run self-tests: python src/mgsm_schema_v2.py
"""
import json
import re

EXTRACT_SYSTEM_V2 = (
    "Read the math word problem below, written in its original language. Do NOT solve it. Respond with "
    "ONLY a JSON object, no prose, no code fences, in EXACTLY this shape:\n"
    '{\n'
    '  "quantities": [{"quantity": <number>, "role": "<short English label for what this represents and '
    'how it is used>"}, ...],\n'
    '  "target": {\n'
    '    "entity": "<what the final answer refers to, in plain English -- e.g. a total, a difference, one '
    'specific entity\'s value. Be explicit about whether it is a SUM of several things or ONE thing\'s value, '
    'and whether it is an ABSOLUTE value or a DIFFERENCE between two things>",\n'
    '    "operation": "<how to combine the quantities to get the answer -- e.g. sum, difference, one '
    "entity's derived value, ratio, or a repeated operation like 'apply the per-year increase N separate "
    'times, not once\'>",\n'
    '    "output_unit": "<the unit the final numeric answer must be in, e.g. dollars (not cents), hours '
    '(not minutes), kilometers -- or null if the problem is a bare count with no unit>"\n'
    "  }\n"
    "}\n"
    "List quantities in the order they appear in the problem. The \"target\" object has NO \"quantity\" "
    "field and is never one of the entries in \"quantities\" -- it describes the unknown to be solved for, "
    "not a known data point, and must never be given a placeholder numeric value such as 0."
)

REASON_SYSTEM_V2 = (
    "You will be given ONLY a JSON object with two fields, extracted from a word problem written in some "
    "language you have not seen and do not know the identity of. You do not have access to the original "
    "problem text.\n"
    '"quantities" is a list of known numeric values and the role each plays.\n'
    '"target" describes the unknown you must solve for: "entity" says what the final answer refers to, '
    '"operation" says how to combine the quantities to get it, and "output_unit" says what unit the answer '
    "must be expressed in (convert to it if the quantities are in a different unit).\n"
    "Using ONLY this structured information, compute the final numeric answer, expressed in the required "
    "output_unit. Give the final numeric answer on the last line as:\nFINAL ANSWER: <number>"
)

CRITIC_SYSTEM_V2 = (
    "You are a Semantic Recovery Critic. You will be given a math word problem and a structured extraction "
    "(schema v2: a JSON object with a \"quantities\" list and a \"target\" object) produced from it, "
    "intended to let someone solve the problem using ONLY the extraction, without ever seeing this original "
    "problem text again. Decide whether the extraction contains everything needed for that.\n\n"
    "Explicitly check all of the following:\n"
    "- Is \"target.entity\" precise about what the final answer refers to (a total vs. one entity's value; "
    "a difference vs. an absolute value)?\n"
    "- Is \"target.operation\" precise about how to combine the quantities, including repetition (e.g. an "
    "increase applied N separate times, not once)?\n"
    "- Is \"target.output_unit\" correct and does it differ from the unit the quantities are given in, "
    "requiring a conversion?\n"
    "- Are all relevant entities preserved in \"quantities\" (one vs. all, which specific one)?\n"
    "- Are quantitative relations preserved (ratios, multipliers, differences between quantities)?\n"
    "- Is there any ambiguity in a role or in \"target\" that could lead to multiple interpretations?\n\n"
    "Respond in EXACTLY this format, no other text:\n"
    "Sufficient: Yes/No\n\n"
    "Missing or distorted information:\n"
    "- (one bullet per issue, or \"- none\" if sufficient)\n\n"
    "Suggested repair:\n"
    "- (one bullet per fix, or \"- none\" if sufficient)"
)

REPAIR_SYSTEM_V2 = (
    "You will be given a math word problem, a structured extraction of it (schema v2: a JSON object with "
    "\"quantities\" and \"target\"), and a critic's assessment of what is missing or distorted. Produce a "
    "REPAIRED extraction that fixes every issue the critic raised. Respond with ONLY a JSON object in "
    "EXACTLY the same shape as the input:\n"
    '{"quantities": [{"quantity": <number>, "role": "..."}, ...], "target": {"entity": "...", '
    '"operation": "...", "output_unit": "..."}}\n'
    "Fix the \"target\" object directly (make \"entity\"/\"operation\"/\"output_unit\" precise and correct) "
    "-- do NOT add a new entry to \"quantities\" to represent the target, and never give \"target\" a "
    "numeric placeholder value. Keep every quantity and role that was already correct unchanged."
)


class SchemaV2ValidationError(Exception):
    pass


def validate_v2(obj):
    """Raises SchemaV2ValidationError with a specific reason if obj doesn't conform to schema_v2.json.
    Returns the object unchanged if valid. Deliberately strict: a v1-shaped bare array, a target with a
    stray numeric 'quantity' field, or a quantities entry whose role smells like a target placeholder are
    all rejected rather than silently accepted -- those are exactly the v1 anti-patterns this schema fixes."""
    if not isinstance(obj, dict):
        raise SchemaV2ValidationError(f"top level must be a JSON object, got {type(obj).__name__} "
                                       f"(this looks like a v1-style bare array, not v2)")
    extra = set(obj.keys()) - {"quantities", "target"}
    if extra:
        raise SchemaV2ValidationError(f"unexpected top-level key(s): {sorted(extra)}")
    if "quantities" not in obj:
        raise SchemaV2ValidationError("missing 'quantities'")
    if "target" not in obj:
        raise SchemaV2ValidationError("missing 'target'")

    quantities = obj["quantities"]
    if not isinstance(quantities, list) or len(quantities) == 0:
        raise SchemaV2ValidationError("'quantities' must be a non-empty array")
    for i, q in enumerate(quantities):
        if not isinstance(q, dict) or set(q.keys()) != {"quantity", "role"}:
            raise SchemaV2ValidationError(f"quantities[{i}] must have exactly 'quantity' and 'role', got {q}")
        if not isinstance(q["quantity"], (int, float)) or isinstance(q["quantity"], bool):
            raise SchemaV2ValidationError(f"quantities[{i}].quantity must be a number, got {q['quantity']!r}")
        if not isinstance(q["role"], str) or not q["role"].strip():
            raise SchemaV2ValidationError(f"quantities[{i}].role must be a non-empty string")
        if re.search(r"\btarget\b|\bunknown\b|\bsolve for\b", q["role"], re.IGNORECASE):
            raise SchemaV2ValidationError(
                f"quantities[{i}].role ('{q['role']}') looks like a target placeholder smuggled into "
                f"quantities -- this is exactly the v1 anti-pattern (a fake quantity=0 'target' entry) "
                f"schema v2 exists to prevent; the target belongs in the top-level 'target' object")

    target = obj["target"]
    if not isinstance(target, dict) or set(target.keys()) != {"entity", "operation", "output_unit"}:
        raise SchemaV2ValidationError(f"'target' must have exactly 'entity', 'operation', 'output_unit', got {target}")
    if "quantity" in target:
        raise SchemaV2ValidationError("'target' must not have a 'quantity' field (no numeric placeholder for an unknown)")
    if not isinstance(target["entity"], str) or not target["entity"].strip():
        raise SchemaV2ValidationError("'target.entity' must be a non-empty string")
    if not isinstance(target["operation"], str) or not target["operation"].strip():
        raise SchemaV2ValidationError("'target.operation' must be a non-empty string")
    if target["output_unit"] is not None and (not isinstance(target["output_unit"], str)):
        raise SchemaV2ValidationError("'target.output_unit' must be a string or null")
    return obj


def parse_v2_json(raw_output):
    """Extract and validate a schema-v2 JSON object from a raw model response.
    Returns (obj, error) -- obj is None if parsing or validation failed, with error explaining why."""
    match = re.search(r"\{.*\}", raw_output or "", re.DOTALL)
    if not match:
        return None, "no JSON object found in output"
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return None, f"JSON decode error: {exc}"
    try:
        validate_v2(obj)
    except SchemaV2ValidationError as exc:
        return None, f"schema validation failed: {exc}"
    return obj, None


def _selftest():
    valid = {"quantities": [{"quantity": 2, "role": "blue fiber bolts"}, {"quantity": 0.5, "role": "multiplier"}],
             "target": {"entity": "total bolts", "operation": "sum", "output_unit": "bolts"}}
    assert validate_v2(valid) == valid

    valid_null_unit = {"quantities": [{"quantity": 3, "role": "count"}],
                        "target": {"entity": "how many", "operation": "identity", "output_unit": None}}
    assert validate_v2(valid_null_unit) == valid_null_unit

    # v1-style bare array must be rejected, not silently accepted
    for bad in (
        [{"quantity": 2, "role": "x"}],  # v1 bare array
        {"quantities": [{"quantity": 2, "role": "x"}]},  # missing target
        {"target": {"entity": "x", "operation": "y", "output_unit": "z"}},  # missing quantities
        {"quantities": [], "target": {"entity": "x", "operation": "y", "output_unit": "z"}},  # empty quantities
        {"quantities": [{"quantity": 2, "role": "x"}], "target": {"entity": "x", "operation": "y", "output_unit": "z", "quantity": 0}},  # the exact v1 bug, re-smuggled via target
        {"quantities": [{"quantity": 0, "role": "target: total weight"}], "target": {"entity": "x", "operation": "y", "output_unit": "z"}},  # the exact v1 bug, in quantities
        {"quantities": [{"quantity": "two", "role": "x"}], "target": {"entity": "x", "operation": "y", "output_unit": "z"}},  # non-numeric quantity
        {"quantities": [{"quantity": 2, "role": ""}], "target": {"entity": "x", "operation": "y", "output_unit": "z"}},  # empty role
        {"quantities": [{"quantity": 2, "role": "x"}], "target": {"entity": "", "operation": "y", "output_unit": "z"}},  # empty target.entity
        {"quantities": [{"quantity": 2, "role": "x"}], "target": {"entity": "x", "operation": "y"}},  # missing output_unit key
        {"quantities": [{"quantity": 2, "role": "x"}], "target": {"entity": "x", "operation": "y", "output_unit": "z"}, "extra": 1},  # stray top-level key
    ):
        try:
            validate_v2(bad)
            raise AssertionError(f"expected SchemaV2ValidationError for {bad!r}, none raised")
        except SchemaV2ValidationError:
            pass

    obj, err = parse_v2_json('noise before {"quantities": [{"quantity": 5, "role": "x"}], '
                              '"target": {"entity": "y", "operation": "z", "output_unit": null}} noise after')
    assert err is None and obj["quantities"][0]["quantity"] == 5

    obj, err = parse_v2_json("not json at all")
    assert obj is None and "no JSON object found" in err

    obj, err = parse_v2_json('[{"quantity": 2, "role": "x"}]')  # v1 shape, no braces for our regex to find
    assert obj is None  # correctly fails (either "no JSON object found" since regex looks for {}, or validation)

    print("mgsm_schema_v2 self-tests passed")


if __name__ == "__main__":
    _selftest()
