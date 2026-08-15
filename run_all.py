"""
Single command to get final results: runs both arms (constrained, unconstrained),
NUM_REPEATS independent lineages each, then runs the stats comparison and plots.

Usage:
    python run_all.py

Estimate before running: pass --estimate to print the call/time/cost estimate
without actually calling the API.

    python run_all.py --estimate
"""
import sys
import time

import config
from app import main as run_one_lineage, make_llm_call
from database.db import get_conn

ARMS = ["constrained", "unconstrained"]


def estimate():
    calls_per_gen = config.TASKS_PER_GENERATION * 2 + 1  # solve + candidate re-eval + reflector
    calls_per_lineage = calls_per_gen * config.NUM_GENERATIONS
    total_calls = calls_per_lineage * config.NUM_REPEATS * len(ARMS)
    # Haiku short calls: ~1-2s each, sequential
    est_minutes_low, est_minutes_high = total_calls * 1 / 60, total_calls * 2 / 60
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

    estimate()
    print("\nStarting full sweep...\n")
    t0 = time.time()

    llm_call = make_llm_call()   # one client, reused across every lineage
    conn = get_conn(config.DB_PATH)  # one DB connection, reused across every lineage

    for arm in ARMS:
        for repeat_id in range(config.NUM_REPEATS):
            run_one_lineage(arm=arm, repeat_id=repeat_id, llm_call=llm_call, conn=conn)

    conn.close()
    elapsed = time.time() - t0
    print(f"\nAll lineages done in {elapsed/60:.1f} min. Running analysis...\n")

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
