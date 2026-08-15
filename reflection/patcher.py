"""
Applies a proposed DNA patch only when the field is whitelisted for the arm
and the value passes type/range validation.
"""
import dataclasses
from typing import Optional, Tuple
from agent.dna import AgentDNA

VALID_STRATEGIES = {"direct", "step_by_step", "test_first"}
VALID_RETRY_POLICIES = {"single", "retry_on_fail"}


def _validate(field: str, value) -> bool:
    if field == "system_prompt":
        return isinstance(value, str) and 10 < len(value) < 2000
    if field == "decomposition_strategy":
        return value in VALID_STRATEGIES
    if field == "temperature":
        return isinstance(value, (int, float)) and 0.0 <= value <= 1.0
    if field == "self_critique_enabled":
        return isinstance(value, bool)
    if field == "retry_policy":
        return value in VALID_RETRY_POLICIES
    return False


def apply_patch(
    dna: AgentDNA,
    patch: dict,
    editable_fields: list,
) -> Tuple[Optional[AgentDNA], str]:
    field = patch.get("field_to_change")
    value = patch.get("new_value")

    if field is None:
        return None, "rejected: reflector proposed no field"
    if field not in editable_fields:
        return None, f"rejected: '{field}' is outside this arm's editable fields"
    if not _validate(field, value):
        return None, f"rejected: value failed validation for field '{field}'"

    new_dna = dataclasses.replace(dna)
    setattr(new_dna, field, value)
    new_dna.generation = dna.generation + 1
    new_dna.parent_generation = dna.generation
    new_dna.notes = patch.get("diagnosis", "")
    return new_dna, f"candidate: {field} -> {value}"
