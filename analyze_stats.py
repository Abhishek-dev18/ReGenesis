"""
Statistical comparison of constrained vs unconstrained lineages.

Metrics:
  - final visible pass rate
  - final hidden case pass rate
  - visible improvement
  - hidden improvement
  - final DNA similarity to origin
  - final solution drift from generation-0 code
  - regression events per lineage

This script treats zero-variance samples explicitly instead of reporting
misleading infinite/NaN t-statistics as significant evidence.
"""
import sqlite3
from collections import defaultdict

import numpy as np
from scipy import stats

import config


def load_lineages():
    conn = sqlite3.connect(str(config.DB_PATH))
    rows = conn.execute(
        """SELECT arm, repeat_id, generation, pass_rate,
                  hidden_pass_rate, drift_score,
                  solution_drift, regression
           FROM generations
           ORDER BY arm, repeat_id, generation"""
    ).fetchall()
    conn.close()

    by_lineage = defaultdict(list)
    for row in rows:
        by_lineage[(row[0], row[1])].append(row[2:])

    return by_lineage


def build_data():
    by_lineage = load_lineages()

    data = defaultdict(
        lambda: {
            "final_pass_rate": [],
            "final_hidden_pass_rate": [],
            "final_drift": [],
            "final_solution_drift": [],
            "improvement": [],
            "hidden_improvement": [],
            "regression_events": [],
        }
    )

    for (arm, repeat_id), gens in by_lineage.items():
        gens.sort(key=lambda x: x[0])

        (
            initial_gen,
            initial_pass,
            initial_hidden,
            initial_drift,
            initial_solution_drift,
            initial_regression,
        ) = gens[0]

        (
            final_gen,
            final_pass,
            final_hidden,
            final_drift,
            final_solution_drift,
            final_regression,
        ) = gens[-1]

        data[arm]["final_pass_rate"].append(final_pass)
        data[arm]["final_hidden_pass_rate"].append(final_hidden)
        data[arm]["final_drift"].append(final_drift)
        data[arm]["final_solution_drift"].append(final_solution_drift)
        data[arm]["improvement"].append(final_pass - initial_pass)

        if initial_hidden is not None and final_hidden is not None:
            data[arm]["hidden_improvement"].append(
                final_hidden - initial_hidden
            )

        data[arm]["regression_events"].append(
            sum(int(g[5] or 0) for g in gens)
        )

    return data


def cohens_d(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if len(a) < 2 or len(b) < 2:
        return float("nan")

    pooled_var = (
        ((len(a) - 1) * a.var(ddof=1))
        + ((len(b) - 1) * b.var(ddof=1))
    ) / (len(a) + len(b) - 2)

    pooled_std = np.sqrt(pooled_var)

    if np.isclose(pooled_std, 0.0):
        return 0.0 if np.isclose(a.mean(), b.mean()) else float("nan")

    return float((a.mean() - b.mean()) / pooled_std)


def compare(data, metric, arm_a="constrained", arm_b="unconstrained"):
    a = data[arm_a][metric]
    b = data[arm_b][metric]

    mean_a = float(np.mean(a)) if a else float("nan")
    mean_b = float(np.mean(b)) if b else float("nan")

    result = {
        "metric": metric,
        "n_a": len(a),
        "n_b": len(b),
        "mean_a": mean_a,
        "mean_b": mean_b,
        "t_stat": float("nan"),
        "p_value": float("nan"),
        "cohens_d": cohens_d(a, b),
        "note": "",
    }

    if len(a) < 2 or len(b) < 2:
        result["note"] = (
            "insufficient repeats; use >=5 per arm for the final experiment"
        )
        return result

    a_var = np.var(a, ddof=1)
    b_var = np.var(b, ddof=1)

    # Avoid misleading Welch results when both groups have effectively no
    # within-group variance.
    if np.isclose(a_var, 0.0) and np.isclose(b_var, 0.0):
        if np.isclose(mean_a, mean_b):
            result["t_stat"] = 0.0
            result["p_value"] = 1.0
            result["note"] = "both groups are identical; no statistical difference"
        else:
            result["note"] = (
                "both groups have zero variance; Welch's t-test is undefined"
            )
        return result

    t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)

    if not np.isfinite(t_stat) or not np.isfinite(p_val):
        result["note"] = (
            "Welch's t-test was numerically unstable because of near-zero variance"
        )
        return result

    result["t_stat"] = float(t_stat)
    result["p_value"] = float(p_val)
    return result


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
            f"  Welch's t = {res['t_stat']:.3f}, "
            f"p = {res['p_value']:.4f}, "
            f"Cohen's d = {res['cohens_d']:.3f}"
        )

    return "\n".join(lines) + "\n"


def main():
    data = build_data()

    if "constrained" not in data or "unconstrained" not in data:
        print(
            "Need runs for both 'constrained' and 'unconstrained' arms "
            "in runs.db before comparison."
        )
        return

    metrics = [
        "final_pass_rate",
        "final_hidden_pass_rate",
        "improvement",
        "hidden_improvement",
        "final_drift",
        "final_solution_drift",
        "regression_events",
    ]

    summary = "ReGenesis: Constrained vs Unconstrained Arm Comparison\n"
    summary += "=" * 60 + "\n\n"

    for metric in metrics:
        summary += format_result(compare(data, metric)) + "\n"

    summary += (
        "Interpretation note:\n"
        "  Treat p-values as exploratory until the final experiment has "
        "at least 5 independent repeats per arm.\n"
        "  Zero-variance metrics are reported as undefined rather than "
        "as artificial infinite t-statistics.\n"
    )

    out_path = config.DATA_DIR / "stats_summary.txt"
    out_path.write_text(summary, encoding="utf-8")

    print(summary)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
