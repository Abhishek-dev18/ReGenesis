"""
The 'agent' that attempts benchmark tasks using its current DNA.
Provider-agnostic: llm_call is injected from app.py so this module has no
hard dependency on a specific SDK.
"""
import re
from agent.dna import AgentDNA

_STRATEGY_HINTS = {
    "direct": "Write the solution directly.",
    "step_by_step": "First think step by step in comments, then write the solution.",
    "test_first": "First write 2-3 example test cases as comments, then write the solution.",
}


def _build_prompt(dna: AgentDNA, task: dict) -> str:
    strategy_hint = _STRATEGY_HINTS.get(dna.decomposition_strategy, _STRATEGY_HINTS["direct"])
    critique_hint = "\nBefore finalizing, briefly re-check your solution for edge cases." \
        if dna.self_critique_enabled else ""

    return (
        f"Strategy: {strategy_hint}{critique_hint}\n\n"
        f"Problem:\n{task['prompt']}\n\n"
        f"Function name to implement: {task['entry_point']}\n"
        f"Return your answer as a single ```python code block containing only the function."
    )


def _extract_code(raw_text: str) -> str:
    match = re.search(r"```(?:python)?\s*(.*?)```", raw_text, re.DOTALL)
    return match.group(1).strip() if match else raw_text.strip()


def solve(dna: AgentDNA, task: dict, llm_call) -> str:
    """
    llm_call: a function (system_prompt: str, user_prompt: str, temperature: float) -> str
    """
    prompt = _build_prompt(dna, task)
    raw = llm_call(system_prompt=dna.system_prompt, user_prompt=prompt, temperature=dna.temperature)
    return _extract_code(raw)
