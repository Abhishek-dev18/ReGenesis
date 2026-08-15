"""
The benchmark-solving agent.

The solver never receives hidden tests. The hidden-test evaluator is kept
outside this module so hidden performance cannot accidentally leak into the
self-reflection/evolution prompt.
"""
import re
from agent.dna import AgentDNA

_STRATEGY_HINTS = {
    "direct": "Write the solution directly.",
    "step_by_step": "First reason through the problem step by step, then write the solution.",
    "test_first": "First mentally construct a few representative examples and edge cases, then write the solution.",
}


def _build_prompt(dna: AgentDNA, task: dict, feedback: str = "") -> str:
    strategy_hint = _STRATEGY_HINTS.get(
        dna.decomposition_strategy,
        _STRATEGY_HINTS["direct"],
    )
    critique_hint = (
        "\nBefore finalizing, re-check the implementation for edge cases and "
        "the exact specification."
        if dna.self_critique_enabled
        else ""
    )

    feedback_hint = ""
    if feedback:
        feedback_hint = (
            "\n\nThe previous implementation failed the visible evaluator. "
            "Use the following evaluator feedback to correct it:\n"
            f"{feedback}\n"
            "Return a corrected implementation."
        )

    return (
        f"Strategy: {strategy_hint}{critique_hint}\n\n"
        f"Problem:\n{task['prompt']}\n\n"
        f"Function name to implement: {task['entry_point']}\n"
        f"Return your answer as a single ```python code block containing only the function."
        f"{feedback_hint}"
    )


def _extract_code(raw_text: str) -> str:
    match = re.search(r"```(?:python)?\s*(.*?)```", raw_text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else raw_text.strip()


def solve(dna: AgentDNA, task: dict, llm_call, feedback: str = "") -> str:
    """
    llm_call: (system_prompt, user_prompt, temperature) -> str
    feedback is visible-test evaluator feedback from a prior failed attempt.
    Hidden-test feedback is never passed here.
    """
    prompt = _build_prompt(dna, task, feedback=feedback)
    raw = llm_call(
        system_prompt=dna.system_prompt,
        user_prompt=prompt,
        temperature=dna.temperature,
    )
    return _extract_code(raw)
