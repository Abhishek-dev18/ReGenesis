"""
Looks at a batch of (task, code, eval_result) outcomes for the current
generation and asks the LLM: what should change about the agent's DNA to do
better next time? Returns a structured patch proposal (JSON), never raw code
to execute directly -- patcher.py is the enforcement point that validates it.
"""
import json
from agent.dna import AgentDNA

REFLECTOR_SYSTEM_PROMPT = """You are analyzing the performance of a coding agent \
across a batch of benchmark tasks. You will propose ONE small change to the \
agent's configuration (its "DNA") that is likely to improve its pass rate. \
Respond ONLY with valid JSON in this exact schema, no other text:
{
  "diagnosis": "<one sentence on what's failing>",
  "field_to_change": "<one of the editable field names given>",
  "new_value": <new value, correct type for that field>,
  "reasoning": "<one sentence why this should help>"
}
"""


def build_batch_summary(results: list) -> str:
    lines = []
    for r in results:
        status = "PASS" if r["eval"]["passed"] else "FAIL"
        err = r["eval"]["stderr"].strip().splitlines()[-1] if r["eval"]["stderr"].strip() else ""
        lines.append(f"- Task {r['task_id']}: {status} {('| ' + err) if err else ''}")
    return "\n".join(lines)


def propose_patch(dna: AgentDNA, results: list, editable_fields: list, llm_call) -> dict:
    summary = build_batch_summary(results)
    user_prompt = (
        f"Current DNA:\n{json.dumps(dna.to_dict(), indent=2)}\n\n"
        f"Editable fields ONLY: {editable_fields}\n\n"
        f"Batch results this generation:\n{summary}\n\n"
        "Propose one patch."
    )
    raw = llm_call(system_prompt=REFLECTOR_SYSTEM_PROMPT, user_prompt=user_prompt, temperature=0.3)
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.replace("json\n", "", 1)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"diagnosis": "reflector returned invalid JSON", "field_to_change": None,
                "new_value": None, "reasoning": ""}
