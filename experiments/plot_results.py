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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
import matplotlib.pyplot as plt


def load_runs():
    conn = sqlite3.connect(str(config.DB_PATH))
    cur = conn.execute(
        "SELECT arm, generation, pass_rate, drift_score FROM generations ORDER BY arm, generation"
    )
    rows = cur.fetchall()
    conn.close()
    data = {}
    for arm, gen, pass_rate, drift in rows:
        data.setdefault(arm, {"gen": [], "pass_rate": [], "drift": []})
        data[arm]["gen"].append(gen)
        data[arm]["pass_rate"].append(pass_rate)
        data[arm]["drift"].append(drift)
    return data


def plot(data):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for arm, d in data.items():
        axes[0].plot(d["gen"], d["pass_rate"], marker="o", label=arm)
    axes[0].set_title("Pass rate vs generation")
    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel("Pass rate")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    for arm, d in data.items():
        axes[1].plot(d["gen"], d["drift"], marker="o", label=arm)
    axes[1].set_title("DNA drift from generation 0")
    axes[1].set_xlabel("Generation")
    axes[1].set_ylabel("Cosine similarity to origin")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out_path = config.PLOTS_DIR / "results.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    data = load_runs()
    if not data:
        print("No data in runs.db yet -- run app.py first.")
    else:
        plot(data)
