"""
Main RSCF loop.

Visible tests are the optimization/selection signal.
Hidden tests are evaluated for measurement but are never passed to the
reflector or used to accept/reject a DNA patch.
"""
import hashlib
import json
import random
import statistics
import time
from difflib import SequenceMatcher

import config
from agent.dna import AgentDNA
from agent.solver import solve
from database.db import get_conn, log_generation
from drift.tracker import similarity_to_origin
from evaluation.evaluator import evaluate, evaluate_hidden
from reflection.patcher import apply_patch
from reflection.reflector import propose_patch

MAX_RETRIES = 4


def _with_retry(fn):
    """Retry transient API/network failures with exponential backoff."""
    for attempt in range(MAX_RETRIES):
        try:
            return fn()
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            wait = 2 ** attempt
            print(
                f"  [retry] {type(e).__name__}: {e} -- retrying in {wait}s "
                f"(attempt {attempt + 1}/{MAX_RETRIES})"
            )
            time.sleep(wait)


def make_llm_call(repeat_seed=None):
    """Return llm_call(system_prompt, user_prompt, temperature) -> str."""
    if config.LLM_PROVIDER == "anthropic":
        import anthropic
        client = anthropic.Anthropic()

        def llm_call(system_prompt, user_prompt, temperature):
            def _call():
                kwargs = {
                    "model": config.SOLVER_MODEL,
                    "max_tokens": 1024,
                    "temperature": temperature,
                    "system": system_prompt,
                    "messages": [{"role": "user", "content": user_prompt}],
                }
                if repeat_seed is not None:
                    kwargs["extra_headers"] = {"X-Repeat-Seed": str(repeat_seed)}
                resp = client.messages.create(**kwargs)
                return resp.content[0].text
            return _with_retry(_call)

        return llm_call

    if config.LLM_PROVIDER == "openai":
        from openai import OpenAI
        client = OpenAI()

        def llm_call(system_prompt, user_prompt, temperature):
            def _call():
                kwargs = {
                    "model": config.SOLVER_MODEL,
                    "temperature": temperature,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                }
                if repeat_seed is not None:
                    kwargs["seed"] = repeat_seed
                resp = client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content
            return _with_retry(_call)

        return llm_call

    raise ValueError(f"Unknown LLM_PROVIDER: {config.LLM_PROVIDER}")


def load_tasks():
    return json.loads(config.BENCHMARK_PATH.read_text(encoding="utf-8"))


def run_generation(dna, tasks, llm_call, evaluate_hidden_tests=True):
    """
    Run one generation.

    Hidden tests are evaluated here for measurement only. The returned hidden
    score is deliberately not consumed by the reflector or candidate
    acceptance logic.
    """
    results = []
    total_solver_attempts = 0
    retry_events = 0

    for task in tasks:
        t0 = time.time()
        code = ""
        attempts = 1

        try:
            code = solve(dna, task, llm_call)
            eval_result = evaluate(code, task)

            # This is an agent-DNA retry policy, not the API/network retry
            # wrapper above. The constrained arm starts with "single" and is
            # not allowed to evolve this field. The unconstrained arm may.
            if (
                not eval_result["passed"]
                and dna.retry_policy == "retry_on_fail"
            ):
                feedback = eval_result.get("stderr", "")
                code = solve(dna, task, llm_call, feedback=feedback)
                eval_result = evaluate(code, task)
                attempts = 2
                retry_events += 1

            hidden_result = None
            if evaluate_hidden_tests:
                hidden_result = evaluate_hidden(code, task)

        except Exception as e:
            eval_result = {
                "passed": False,
                "stdout": "",
                "stderr": str(e),
            }
            hidden_result = (
                {
                    "passed": False,
                    "stdout": "",
                    "stderr": str(e),
                }
                if evaluate_hidden_tests
                else None
            )

        total_solver_attempts += attempts
        results.append({
            "task_id": task["task_id"],
            "code": code,
            "eval": eval_result,
            "hidden_eval": hidden_result,
            "runtime_s": time.time() - t0,
        })

    if not results:
        return results, 0.0, None, 0.0, 0, 0

    pass_rate = sum(r["eval"]["passed"] for r in results) / len(results)

    hidden_pass_rate = None
    if evaluate_hidden_tests:
        total_hidden = sum(
            r["hidden_eval"].get("total_cases", 0)
            for r in results
        )
        passed_hidden = sum(
            r["hidden_eval"].get("passed_cases", 0)
            for r in results
        )
        hidden_pass_rate = (
            passed_hidden / total_hidden
            if total_hidden
            else 0.0
        )

    avg_runtime = statistics.mean(r["runtime_s"] for r in results)
    return results, pass_rate, hidden_pass_rate, avg_runtime, total_solver_attempts, retry_events



def solution_drift_from_origin(origin_results, current_results) -> float:
    """Return mean 0..1 code drift from generation-0 solutions."""
    origin = {r["task_id"]: r.get("code", "") for r in origin_results}
    current = {r["task_id"]: r.get("code", "") for r in current_results}

    scores = []
    for task_id, origin_code in origin.items():
        current_code = current.get(task_id, "")
        if not origin_code and not current_code:
            scores.append(0.0)
            continue
        similarity = SequenceMatcher(None, origin_code, current_code).ratio()
        scores.append(1.0 - similarity)

    return statistics.mean(scores) if scores else 0.0


def main(arm=None, repeat_id=0, llm_call=None, conn=None, repeat_seed=None, task_order_seed=None, ordered_tasks=None):
    owns_conn = conn is None
    arm = arm or config.ARM_MODE
    repeat_seed = repeat_seed if repeat_seed is not None else config.BASE_SEED + repeat_id
    task_order_seed = task_order_seed if task_order_seed is not None else repeat_seed
    llm_call = llm_call or make_llm_call(repeat_seed=repeat_seed)
    conn = conn or get_conn(config.DB_PATH)
    tasks = ordered_tasks if ordered_tasks is not None else load_tasks()

    editable_fields = (
        config.UNCONSTRAINED_EDITABLE_FIELDS
        if arm == "unconstrained"
        else config.CONSTRAINED_EDITABLE_FIELDS
    )

    dna = AgentDNA.initial()
    dna_dir = config.DNA_DIR / arm / f"repeat_{repeat_id}"
    dna_dir.mkdir(parents=True, exist_ok=True)
    dna.save(dna_dir)
    origin_dna = AgentDNA.initial()

    print(
        f"=== RSCF run | arm={arm} | repeat={repeat_id} "
        f"| repeat_seed={repeat_seed} | task_order_seed={task_order_seed} "
        f"| generations={config.NUM_GENERATIONS} "
        f"| tasks/gen={config.TASKS_PER_GENERATION} ==="
    )

    best_dna, best_pass_rate = dna, -1.0
    origin_results = None
    previous_pass_rate = None
    regression_events = 0

    for gen in range(config.NUM_GENERATIONS):
        subset = tasks[:config.TASKS_PER_GENERATION]

        results, pass_rate, hidden_pass_rate, avg_runtime, total_solver_attempts, retry_events = run_generation(
            dna,
            subset,
            llm_call,
            evaluate_hidden_tests=True,
        )

        drift = similarity_to_origin(origin_dna, dna)

        if origin_results is None:
            origin_results = results
        solution_drift = solution_drift_from_origin(origin_results, results)

        regression = (
            previous_pass_rate is not None
            and pass_rate < previous_pass_rate
        )
        if regression:
            regression_events += 1
        previous_pass_rate = pass_rate

        hidden_display = (
            f" hidden={hidden_pass_rate:.2f}"
            if hidden_pass_rate is not None
            else ""
        )

        print(
            f"  [gen {gen}] pass_rate={pass_rate:.2f}"
            f"{hidden_display} drift={drift:.3f} "
            f"strategy={dna.decomposition_strategy} "
            f"critique={dna.self_critique_enabled} "
            f"retry={dna.retry_policy} "
            f"total_solver_attempts={total_solver_attempts} "
            f"retry_events={retry_events} "
            f"repeat={repeat_id} repeat_seed={repeat_seed} "
            f"task_order_seed={task_order_seed}"
        )

        accepted_patch = None
        patch_status = "skipped: last generation"
        new_dna = None

        if gen < config.NUM_GENERATIONS - 1:
            # IMPORTANT: reflector sees only visible evaluator outcomes.
            # hidden_eval is intentionally ignored by build_batch_summary().
            patch = propose_patch(dna, results, editable_fields, llm_call)
            candidate_dna, candidate_status = apply_patch(
                dna,
                patch,
                editable_fields,
            )

            if candidate_dna is not None:
                # Candidate acceptance is based ONLY on visible tests.
                _, cand_pass_rate, _, _, _, _ = run_generation(
                    candidate_dna,
                    subset,
                    llm_call,
                    evaluate_hidden_tests=False,
                )

                if cand_pass_rate >= pass_rate:
                    new_dna = candidate_dna
                    accepted_patch = candidate_status
                    patch_status = (
                        f"accepted (candidate {cand_pass_rate:.2f} "
                        f">= current {pass_rate:.2f})"
                    )
                else:
                    patch_status = (
                        f"rejected: candidate scored lower "
                        f"({cand_pass_rate:.2f} < {pass_rate:.2f})"
                    )
            else:
                patch_status = candidate_status

        log_generation(
            conn,
            arm,
            repeat_id,
            gen,
            dna,
            pass_rate,
            hidden_pass_rate,
            len(subset),
            accepted_patch,
            patch_status,
            drift,
            avg_runtime,
            total_tokens=0,
            solution_drift=solution_drift,
            regression=int(regression),
            total_solver_attempts=total_solver_attempts,
            retry_events=retry_events,
            base_seed=config.BASE_SEED,
            repeat_seed=repeat_seed,
            task_order_seed=task_order_seed,
        )

        if pass_rate > best_pass_rate:
            best_dna, best_pass_rate = dna, pass_rate

        dna = new_dna if new_dna is not None else dna
        dna.save(dna_dir)

    print(
        f"  best pass_rate={best_pass_rate:.2f} "
        f"at generation {best_dna.generation} "
        f"regressions={regression_events}"
    )

    if owns_conn:
        conn.close()

    return best_pass_rate


if __name__ == "__main__":
    main()
