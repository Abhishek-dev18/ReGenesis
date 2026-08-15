"""
Reflection layer.

The reflector sees visible-test outcomes only. Hidden-test results are never
included in the prompt and cannot influence patch selection.
"""
import json
from agent.dna import AgentDNA

REFLECTOR_SYSTEM_PROMPT = """You are analyzing the performance of a coding agent across benchmark tasks.

Your job is to propose ONE small DNA change that is plausibly useful on the next generation.

Important experimental rules:
- You only receive VISIBLE-test outcomes.
- Hidden tests exist, but you must never ask for, infer, or optimize against hidden tests.
- Consider every editable field before choosing a patch.
- Do not change a field merely to create variation.
- For the unconstrained arm, retry_policy is a real execution capability. Consider
  retry_on_fail only when visible failures suggest a retry could help.
- Prefer the smallest change that addresses an observed failure pattern.
- If the current visible pass rate is already perfect, prefer a conservative change
  or leave the behavior stable rather than making an arbitrary risky change.

Respond ONLY with valid JSON in this exact schema:
{
  "diagnosis": "<one sentence on what's failing or what could be improved>",
  "field_to_change": "<one of the editable field names given, or null if no change is justified>",
  "new_value": <new value, correct type for that field, or null>,
  "reasoning": "<one sentence why this should help>"
}
"""


def build_batch_summary(results: list) -> str:
    lines = []
    for r in results:
        status = "PASS" if r["eval"]["passed"] else "FAIL"
        err = (
            r["eval"]["stderr"].strip().splitlines()[-1]
            if r["eval"]["stderr"].strip()
            else ""
        )
        lines.append(
            f"- Task {r['task_id']}: {status}"
            f"{(' | ' + err) if err else ''}"
        )
    return "\n".join(lines)


def propose_patch(
    dna: AgentDNA,
    results: list,
    editable_fields: list,
    llm_call,
) -> dict:
    summary = build_batch_summary(results)

    user_prompt = (
        f"Current DNA:\n{json.dumps(dna.to_dict(), indent=2)}\n\n"
        f"Editable fields ONLY: {editable_fields}\n\n"
        f"Visible batch results this generation:\n{summary}\n\n"
        "Propose one minimal patch. If there is no justified improvement, "
        "you may return field_to_change=null."
    )

    raw = llm_call(
        system_prompt=REFLECTOR_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.3,
    )

    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.replace("json\n", "", 1)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "diagnosis": "reflector returned invalid JSON",
            "field_to_change": None,
            "new_value": None,
            "reasoning": "",
        }
