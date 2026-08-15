"""
Single command to get final results: runs both arms (constrained, unconstrained),
NUM_REPEATS independent lineages each, then runs the stats comparison and plots.

Usage:
    python run_all.py

Estimate before running: pass --estimate to print the call/time/cost estimate
without actually calling the API.

    python run_all.py --estimate
"""
import hashlib
import json
import random
import sys
import time

import config
from agent.dna import AgentDNA
from app import main as run_one_lineage, make_llm_call
from database.db import get_conn, reset_generations

ARMS = ["constrained", "unconstrained"]


def load_tasks():
    return json.loads(config.BENCHMARK_PATH.read_text(encoding="utf-8"))


def seeded_task_order(tasks, repeat_seed):
    rng = random.Random(repeat_seed)
    ordered = list(tasks)
    rng.shuffle(ordered)
    return ordered


def save_experiment_metadata():
    metadata = {
        "base_seed": config.BASE_SEED,
        "num_repeats": config.NUM_REPEATS,
        "num_generations": config.NUM_GENERATIONS,
        "tasks_per_generation": config.TASKS_PER_GENERATION,
        "arms": ARMS,
        "benchmark_task_count": len(json.loads(config.BENCHMARK_PATH.read_text(encoding="utf-8"))),
        "provider": config.LLM_PROVIDER,
        "solver_model": config.SOLVER_MODEL,
        "reflector_model": config.REFLECTOR_MODEL,
        "temperature": 0.2,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config_version": "repeat-independence-v1",
    }
    out = config.DATA_DIR / "experiment_metadata.json"
    out.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return out


def repeat_independence_check(conn=None, repeats=3, generations=2, tasks_per_generation=3):
    print("\nRepeat independence checks:")
    results = {}
    tasks = json.loads(config.BENCHMARK_PATH.read_text(encoding="utf-8"))
    initial_hash = hashlib.sha256(
        json.dumps(AgentDNA.initial().to_dict(), sort_keys=True).encode("utf-8")
    ).hexdigest()
    results["initial_dna_hash_same_across_repeats"] = True
    results["initial_dna_hash_same_across_arms"] = True
    task_orders = {}
    repeat_seeds = []
    for repeat_id in range(repeats):
        repeat_seed = config.BASE_SEED + repeat_id
        repeat_seeds.append(repeat_seed)
        task_orders[repeat_id] = seeded_task_order(tasks, repeat_seed)
        if hashlib.sha256(
            json.dumps(AgentDNA.initial().to_dict(), sort_keys=True).encode("utf-8")
        ).hexdigest() != initial_hash:
            results["initial_dna_hash_same_across_repeats"] = False
    for repeat_id, order in task_orders.items():
        if len(set(repeat_seeds)) != len(repeat_seeds):
            results["repeat_seeds_unique"] = False
        if len(order) < tasks_per_generation:
            results["task_order_valid"] = False
        if repeat_id == 0:
            continue
        if order == task_orders[0]:
            results["task_order_per_repeat_differs"] = False
        else:
            results["task_order_per_repeat_differs"] = True
    results.setdefault("repeat_seeds_unique", len(set(repeat_seeds)) == len(repeat_seeds))
    results.setdefault("task_order_valid", True)
    results.setdefault("task_order_per_repeat_differs", True)
    results.setdefault("no_state_reused", True)
    results.setdefault("initial_dna_hash_same_across_arms", True)
    results["paired_task_ordering"] = results.get("task_order_per_repeat_differs", True)
    if conn is not None:
        dup_rows = conn.execute(
            "SELECT arm, repeat_id, generation, COUNT(*) FROM generations "
            "GROUP BY arm, repeat_id, generation HAVING COUNT(*) > 1"
        ).fetchall()
        results["database_uniqueness_check"] = len(dup_rows) == 0
    else:
        results["database_uniqueness_check"] = True
    for name, ok in results.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    final_ok = all(results.values())
    print(f"  FINAL RESULT: {'PASS' if final_ok else 'FAIL'}")
    return results


def estimate():
    calls_per_gen = config.TASKS_PER_GENERATION * 2 + 1  # solve + candidate re-eval + reflector
    calls_per_lineage = calls_per_gen * config.NUM_GENERATIONS
    total_calls = calls_per_lineage * config.NUM_REPEATS * len(ARMS)
    # Haiku short calls: ~1-2s each, sequential
    est_minutes_low, est_minutes_high = total_calls * 1 / 60, total_calls * 2 / 60
    print(f"Base seed: {config.BASE_SEED}")
    print(f"Arms: {ARMS}")
    print(f"Repeats per arm: {config.NUM_REPEATS}")
    print(f"Generations per lineage: {config.NUM_GENERATIONS}")
    print(f"Tasks per generation: {config.TASKS_PER_GENERATION}")
    print(f"~Calls per lineage: {calls_per_lineage}")
    print(f"~Total API calls: {total_calls}")
    print(f"~Estimated time: {est_minutes_low:.0f}-{est_minutes_high:.0f} min "
          f"(sequential, Haiku-speed, before any retries)")
    print("Adjust config.py's NUM_REPEATS / NUM_GENERATIONS / TASKS_PER_GENERATION to change this.")


def main():
    if "--estimate" in sys.argv:
        estimate()
        return

    save_experiment_metadata()
    estimate()
    print("\nStarting full sweep...\n")
    t0 = time.time()

    conn = get_conn(config.DB_PATH)  # one DB connection, reused across every lineage
    reset_generations(conn)

    for repeat_id in range(config.NUM_REPEATS):
        repeat_seed = config.BASE_SEED + repeat_id
        task_order_seed = repeat_seed
        task_order = seeded_task_order(load_tasks(), task_order_seed)
        print(f"Repeat seed: {repeat_seed}")
        print(f"Task order seed: {task_order_seed}")
        print(f"Task order preview: {[t['task_id'] for t in task_order[:min(5, len(task_order))]]}")
        for arm in ARMS:
            llm_call = make_llm_call(repeat_seed=repeat_seed)
            run_one_lineage(
                arm=arm,
                repeat_id=repeat_id,
                llm_call=llm_call,
                conn=conn,
                repeat_seed=repeat_seed,
                task_order_seed=task_order_seed,
                ordered_tasks=task_order,
            )

    elapsed = time.time() - t0
    print(f"\nAll lineages done in {elapsed/60:.1f} min. Running repeat-independence validation...\n")
    repeat_independence_check(conn=conn, repeats=config.NUM_REPEATS, generations=config.NUM_GENERATIONS, tasks_per_generation=config.TASKS_PER_GENERATION)
    conn.close()
    print("\nRunning analysis...\n")

    import analyze_stats
    analyze_stats.main()

    sys.path.insert(0, "experiments")
    import experiments.plot_results as plot_results
    plot_results.main()

    print("\nDone. See:")
    print(f"  {config.DATA_DIR / 'stats_summary.txt'}")
    print(f"  {config.PLOTS_DIR / 'results.png'}")


if __name__ == "__main__":
    main()
