"""
Reads runs.db and statistically compares the two arms (constrained vs
unconstrained) on:
  - final-generation pass_rate
  - final-generation drift_score (semantic distance of DNA from generation 0)
  - pass_rate improvement (final - generation 0)

Each arm's sample is one value per repeat_id (i.e. one independent lineage),
so this requires NUM_REPEATS >= 2 per arm to be meaningful -- ideally 5+.

Usage:
    python analyze_stats.py
"""
import sqlite3
from collections import defaultdict

import numpy as np
from scipy import stats

import config


def load_final_and_initial_per_lineage():
    """Returns {arm: {'final_pass_rate': [...], 'final_drift': [...], 'improvement': [...]}}"""
    conn = sqlite3.connect(str(config.DB_PATH))
    rows = conn.execute(
        "SELECT arm, repeat_id, generation, pass_rate, drift_score FROM generations "
        "ORDER BY arm, repeat_id, generation"
    ).fetchall()
    conn.close()

    by_lineage = defaultdict(list)  # (arm, repeat_id) -> [(gen, pass_rate, drift), ...]
    for arm, repeat_id, gen, pass_rate, drift in rows:
        by_lineage[(arm, repeat_id)].append((gen, pass_rate, drift))

    data = defaultdict(lambda: {"final_pass_rate": [], "final_drift": [], "improvement": []})
    for (arm, repeat_id), gens in by_lineage.items():
        gens.sort(key=lambda x: x[0])
        initial_pass_rate = gens[0][1]
        final_gen, final_pass_rate, final_drift = gens[-1]
        data[arm]["final_pass_rate"].append(final_pass_rate)
        data[arm]["final_drift"].append(final_drift)
        data[arm]["improvement"].append(final_pass_rate - initial_pass_rate)

    return data


def cohens_d(a, b):
    a, b = np.array(a), np.array(b)
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return float("nan")
    pooled_std = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (a.mean() - b.mean()) / pooled_std


def compare(data, metric, arm_a="constrained", arm_b="unconstrained"):
    a = data[arm_a][metric]
    b = data[arm_b][metric]
    if len(a) < 2 or len(b) < 2:
        return {
            "metric": metric, "n_a": len(a), "n_b": len(b),
            "mean_a": np.mean(a) if a else float("nan"),
            "mean_b": np.mean(b) if b else float("nan"),
            "t_stat": float("nan"), "p_value": float("nan"), "cohens_d": float("nan"),
            "note": "insufficient repeats for a t-test (need >=2 per arm, ideally >=5)",
        }
    t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)  # Welch's t-test
    d = cohens_d(b, a)
    return {
        "metric": metric, "n_a": len(a), "n_b": len(b),
        "mean_a": float(np.mean(a)), "mean_b": float(np.mean(b)),
        "t_stat": float(t_stat), "p_value": float(p_val), "cohens_d": float(d),
        "note": "",
    }


def format_result(res, arm_a="constrained", arm_b="unconstrained"):
    lines = [
        f"{res['metric']}:",
        f"  {arm_a} mean = {res['mean_a']:.4f}  (n={res['n_a']} lineages)",
        f"  {arm_b} mean = {res['mean_b']:.4f}  (n={res['n_b']} lineages)",
    ]
    if res["note"]:
        lines.append(f"  NOTE: {res['note']}")
    else:
        lines.append(
            f"  Welch's t = {res['t_stat']:.3f}, p = {res['p_value']:.4f}, "
            f"Cohen's d = {res['cohens_d']:.3f}"
        )
    return "\n".join(lines) + "\n"


def main():
    data = load_final_and_initial_per_lineage()

    if "constrained" not in data or "unconstrained" not in data:
        print("Need runs for both 'constrained' and 'unconstrained' arms in runs.db "
              "before comparison. Run run_all.py first (or app.py for each arm).")
        return

    results = [
        compare(data, "final_pass_rate"),
        compare(data, "improvement"),
        compare(data, "final_drift"),
    ]

    summary = "ReGenesis: Constrained vs Unconstrained Arm Comparison\n"
    summary += "=" * 60 + "\n\n"
    for res in results:
        summary += format_result(res) + "\n"

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.DATA_DIR / "stats_summary.txt"
    out_path.write_text(summary)
    print(summary)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
