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

MAX_RETRIES = 4


def _with_retry(fn):
    """Wraps a zero-arg callable with exponential backoff on transient API errors.
    A multi-hour, multi-repeat run should not die because of one rate limit or
    network hiccup -- that's the difference between finishing overnight and
    finding a stack trace in the morning."""
    for attempt in range(MAX_RETRIES):
        try:
            return fn()
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            wait = 2 ** attempt
            print(f"  [retry] {type(e).__name__}: {e} -- retrying in {wait}s "
                  f"(attempt {attempt + 1}/{MAX_RETRIES})")
            time.sleep(wait)


def make_llm_call():
    """Returns a llm_call(system_prompt, user_prompt, temperature) -> str function."""
    if config.LLM_PROVIDER == "anthropic":
        import anthropic
        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

        def llm_call(system_prompt, user_prompt, temperature):
            def _call():
                resp = client.messages.create(
                    model=config.SOLVER_MODEL,
                    max_tokens=1024,
                    temperature=temperature,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                return resp.content[0].text
            return _with_retry(_call)
        return llm_call

    elif config.LLM_PROVIDER == "openai":
        from openai import OpenAI
        client = OpenAI()  # reads OPENAI_API_KEY from env

        def llm_call(system_prompt, user_prompt, temperature):
            def _call():
                resp = client.chat.completions.create(
                    model=config.SOLVER_MODEL,
                    temperature=temperature,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                return resp.choices[0].message.content
            return _with_retry(_call)
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


def main(arm=None, repeat_id=0, llm_call=None, conn=None):
    """Runs one full lineage (NUM_GENERATIONS generations) for one arm, one repeat_id.

    arm/llm_call/conn are optional so run_all.py can drive many calls to main()
    efficiently (reusing one API client and one DB connection across repeats)
    while `python app.py` on its own still works unchanged, reading arm from
    config.ARM_MODE and opening its own client/connection.
    """
    owns_conn = conn is None
    arm = arm or config.ARM_MODE
    llm_call = llm_call or make_llm_call()
    conn = conn or get_conn(config.DB_PATH)
    tasks = load_tasks()
    editable_fields = (
        config.UNCONSTRAINED_EDITABLE_FIELDS if arm == "unconstrained"
        else config.CONSTRAINED_EDITABLE_FIELDS
    )

    dna = AgentDNA.initial()
    dna_dir = config.DNA_DIR / arm / f"repeat_{repeat_id}"
    dna_dir.mkdir(parents=True, exist_ok=True)
    dna.save(dna_dir)
    origin_dna = dna

    print(f"=== RSCF run | arm={arm} | repeat={repeat_id} | generations={config.NUM_GENERATIONS} "
          f"| tasks/gen={config.TASKS_PER_GENERATION} ===")

    best_dna, best_pass_rate = dna, -1.0

    for gen in range(config.NUM_GENERATIONS):
        subset = tasks[:config.TASKS_PER_GENERATION]
        results, pass_rate, avg_runtime = run_generation(dna, subset, llm_call)
        drift = similarity_to_origin(origin_dna, dna)

        print(f"  [gen {gen}] pass_rate={pass_rate:.2f} drift={drift:.3f} "
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
            conn, arm, repeat_id, gen, dna, pass_rate, len(subset),
            accepted_patch, patch_status, drift, avg_runtime, total_tokens=0
        )

        if pass_rate > best_pass_rate:
            best_dna, best_pass_rate = dna, pass_rate

        dna = new_dna if new_dna is not None else dna
        dna.save(dna_dir)

    print(f"  best pass_rate={best_pass_rate:.2f} at generation {best_dna.generation}")
    if owns_conn:
        conn.close()
    return best_pass_rate


if __name__ == "__main__":
    main()
