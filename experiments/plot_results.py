"""
Reads runs.db and produces the two core plots:
  1. pass_rate vs generation, one line per arm
  2. drift_score vs generation, one line per arm

Run this after you've done at least one full run for each arm you want to
compare (constrained and unconstrained).

Usage:
    python experiments/plot_results.py
"""
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_runs():
    """Returns {arm: {generation: {'pass_rate': [...one per repeat...], 'drift': [...]}}}"""
    conn = sqlite3.connect(str(config.DB_PATH))
    cur = conn.execute(
        "SELECT arm, repeat_id, generation, pass_rate, drift_score FROM generations "
        "ORDER BY arm, repeat_id, generation"
    )
    rows = cur.fetchall()
    conn.close()

    data = defaultdict(lambda: defaultdict(lambda: {"pass_rate": [], "drift": []}))
    for arm, repeat_id, gen, pass_rate, drift in rows:
        data[arm][gen]["pass_rate"].append(pass_rate)
        data[arm][gen]["drift"].append(drift)
    return data


def _mean_ci(values):
    values = np.array(values)
    mean = values.mean()
    if len(values) > 1:
        sem = stats.sem(values)
        ci = sem * stats.t.ppf(0.975, len(values) - 1)
    else:
        ci = 0.0
    return mean, ci


def plot(data):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for arm, by_gen in data.items():
        gens = sorted(by_gen.keys())
        means, cis = zip(*[_mean_ci(by_gen[g]["pass_rate"]) for g in gens])
        axes[0].errorbar(gens, means, yerr=cis, marker="o", capsize=3, label=arm)

    axes[0].set_title("Pass rate vs generation (mean \u00b1 95% CI across repeats)")
    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel("Pass rate")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    for arm, by_gen in data.items():
        gens = sorted(by_gen.keys())
        means, cis = zip(*[_mean_ci(by_gen[g]["drift"]) for g in gens])
        axes[1].errorbar(gens, means, yerr=cis, marker="o", capsize=3, label=arm)

    axes[1].set_title("DNA drift from generation 0 (mean \u00b1 95% CI)")
    axes[1].set_xlabel("Generation")
    axes[1].set_ylabel("Cosine similarity to origin")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.PLOTS_DIR / "results.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


def main():
    data = load_runs()
    if not data:
        print("No data in runs.db yet -- run run_all.py (or app.py) first.")
    else:
        plot(data)


if __name__ == "__main__":
    main()
