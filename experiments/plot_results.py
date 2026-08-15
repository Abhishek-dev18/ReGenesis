"""
Plot generation-level performance, drift, solution drift, and regression rate.
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
    conn = sqlite3.connect(str(config.DB_PATH))
    rows = conn.execute(
        """SELECT arm, repeat_id, generation,
                  pass_rate, hidden_pass_rate, drift_score,
                  solution_drift, regression
           FROM generations
           ORDER BY arm, repeat_id, generation"""
    ).fetchall()
    conn.close()

    data = defaultdict(
        lambda: defaultdict(
            lambda: {
                "pass_rate": [],
                "hidden_pass_rate": [],
                "drift": [],
                "solution_drift": [],
                "regression": [],
            }
        )
    )

    for (
        arm,
        repeat_id,
        gen,
        pass_rate,
        hidden_pass_rate,
        drift,
        solution_drift,
        regression,
    ) in rows:
        data[arm][gen]["pass_rate"].append(pass_rate)
        if hidden_pass_rate is not None:
            data[arm][gen]["hidden_pass_rate"].append(hidden_pass_rate)
        data[arm][gen]["drift"].append(drift)
        data[arm][gen]["solution_drift"].append(solution_drift or 0.0)
        data[arm][gen]["regression"].append(float(regression or 0))

    return data


def _mean_ci(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("nan"), 0.0

    mean = values.mean()

    if len(values) > 1:
        sem = stats.sem(values)
        ci = sem * stats.t.ppf(0.975, len(values) - 1)
    else:
        ci = 0.0

    return mean, ci


def _plot_metric(ax, data, key, title, ylabel, ylim=None):
    for arm, by_gen in data.items():
        gens = sorted(by_gen.keys())
        valid = [g for g in gens if by_gen[g][key]]

        if not valid:
            continue

        means, cis = zip(*[
            _mean_ci(by_gen[g][key])
            for g in valid
        ])

        ax.errorbar(
            valid,
            means,
            yerr=cis,
            marker="o",
            capsize=3,
            label=arm,
        )

    ax.set_title(title)
    ax.set_xlabel("Generation")
    ax.set_ylabel(ylabel)

    if ylim:
        ax.set_ylim(*ylim)

    ax.legend()
    ax.grid(alpha=0.3)


def plot(data):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    _plot_metric(
        axes[0, 0],
        data,
        "pass_rate",
        "Visible pass rate",
        "Pass rate",
        (0.0, 1.05),
    )

    _plot_metric(
        axes[0, 1],
        data,
        "hidden_pass_rate",
        "Hidden case pass rate",
        "Hidden pass rate",
        (0.0, 1.05),
    )

    _plot_metric(
        axes[1, 0],
        data,
        "drift",
        "DNA similarity to origin",
        "DNA similarity",
        (0.0, 1.05),
    )

    _plot_metric(
        axes[1, 1],
        data,
        "solution_drift",
        "Solution drift from generation 0",
        "Code drift",
        (0.0, 1.05),
    )

    plt.tight_layout()
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.PLOTS_DIR / "results.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {out_path}")


def main():
    data = load_runs()

    if not data:
        print("No data in runs.db yet -- run run_all.py first.")
    else:
        plot(data)


if __name__ == "__main__":
    main()
