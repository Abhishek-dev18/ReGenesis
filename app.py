"""
Main entry point: runs the RSCF loop for NUM_GENERATIONS generations, for
whichever ARM_MODE is set (env var RSCF_ARM, default "constrained").

Usage:
    python app.py
    RSCF_ARM=unconstrained python app.py          (Mac/Linux)
    $env:RSCF_ARM="unconstrained"; python app.py  (Windows PowerShell)
"""
import json
import statistics
import time

import config
from agent.dna import AgentDNA
from agent.solver import solve
from database.db import get_conn, log_generation
from drift.tracker import similarity_to_origin
from evaluation.evaluator import evaluate
from reflection.patcher import apply_patch
from reflection.reflector import propose_patch


def make_llm_call():
    """Returns a llm_call(system_prompt, user_prompt, temperature) -> str function."""
    if config.LLM_PROVIDER == "anthropic":
        import anthropic
        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

        def llm_call(system_prompt, user_prompt, temperature):
            resp = client.messages.create(
                model=config.SOLVER_MODEL,
                max_tokens=1024,
                temperature=temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return resp.content[0].text
        return llm_call

    elif config.LLM_PROVIDER == "openai":
        from openai import OpenAI
        client = OpenAI()  # reads OPENAI_API_KEY from env

        def llm_call(system_prompt, user_prompt, temperature):
            resp = client.chat.completions.create(
                model=config.SOLVER_MODEL,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return resp.choices[0].message.content
        return llm_call

    raise ValueError(f"Unknown LLM_PROVIDER: {config.LLM_PROVIDER}")


def load_tasks():
    return json.loads(config.BENCHMARK_PATH.read_text())


def run_generation(dna, tasks, llm_call):
    results = []
    for task in tasks:
        t0 = time.time()
        try:
            code = solve(dna, task, llm_call)
            eval_result = evaluate(code, task)
        except Exception as e:
            code = ""
            eval_result = {"passed": False, "stdout": "", "stderr": str(e)}
        results.append({
            "task_id": task["task_id"],
            "code": code,
            "eval": eval_result,
            "runtime_s": time.time() - t0,
        })
    pass_rate = sum(r["eval"]["passed"] for r in results) / len(results)
    avg_runtime = statistics.mean(r["runtime_s"] for r in results)
    return results, pass_rate, avg_runtime


def main():
    llm_call = make_llm_call()
    tasks = load_tasks()
    arm = config.ARM_MODE
    editable_fields = (
        config.UNCONSTRAINED_EDITABLE_FIELDS if arm == "unconstrained"
        else config.CONSTRAINED_EDITABLE_FIELDS
    )

    conn = get_conn(config.DB_PATH)
    dna = AgentDNA.initial()
    dna.save(config.DNA_DIR)
    origin_dna = dna

    print(f"=== RSCF run | arm={arm} | generations={config.NUM_GENERATIONS} "
          f"| tasks/gen={config.TASKS_PER_GENERATION} ===")

    best_dna, best_pass_rate = dna, -1.0

    for gen in range(config.NUM_GENERATIONS):
        subset = tasks[:config.TASKS_PER_GENERATION]
        results, pass_rate, avg_runtime = run_generation(dna, subset, llm_call)
        drift = similarity_to_origin(origin_dna, dna)

        print(f"[gen {gen}] pass_rate={pass_rate:.2f} drift={drift:.3f} "
              f"strategy={dna.decomposition_strategy} critique={dna.self_critique_enabled}")

        accepted_patch, patch_status, new_dna = None, "skipped: last generation", None

        if gen < config.NUM_GENERATIONS - 1:
            patch = propose_patch(dna, results, editable_fields, llm_call)
            candidate_dna, patch_status = apply_patch(dna, patch, editable_fields)

            if candidate_dna is not None:
                _, cand_pass_rate, _ = run_generation(candidate_dna, subset, llm_call)
                if cand_pass_rate >= pass_rate:
                    new_dna = candidate_dna
                    accepted_patch = patch_status
                    patch_status = f"accepted (candidate {cand_pass_rate:.2f} >= current {pass_rate:.2f})"
                else:
                    patch_status = f"rejected: candidate scored lower ({cand_pass_rate:.2f} < {pass_rate:.2f})"

        log_generation(
            conn, arm, gen, dna, pass_rate, len(subset),
            accepted_patch, patch_status, drift, avg_runtime, total_tokens=0
        )

        if pass_rate > best_pass_rate:
            best_dna, best_pass_rate = dna, pass_rate

        dna = new_dna if new_dna is not None else dna
        dna.save(config.DNA_DIR)

    print(f"\nBest pass_rate={best_pass_rate:.2f} at generation {best_dna.generation}")
    conn.close()


if __name__ == "__main__":
    main()
