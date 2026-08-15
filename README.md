# Recursive Self-Creation Framework (RSCF)

A research framework for experimentally studying recursive self-improvement in LLM-based autonomous agents.

This project accompanies the research:

"The Theory of Self-Creation: A Philosophical and Technical Exploration of Recursive Artificial Intelligence."

---

## Objectives

- Recursive self-reflection
- Controlled self-modification
- Benchmark-driven evolution
- Semantic drift analysis
- Safe experimentation

---

## Status

✅ Phase 1 — implemented and runnable end-to-end (Aug 2026)

---

## Roadmap

- [x] Baseline Agent (`agent/solver.py`, `agent/dna.py`)
- [x] Reflection Engine (`reflection/reflector.py`)
- [x] DNA Model (`agent/dna.py`)
- [x] Patch Generator / enforcement (`reflection/patcher.py`)
- [x] Evaluation Pipeline (`evaluation/evaluator.py`, sandboxed, never patchable)
- [x] Drift tracking (`drift/tracker.py`)
- [x] Multi-repeat runner + stats comparison (`run_all.py`, `analyze_stats.py`)
- [ ] Evolution Engine (Phase 2+ — population-level, see docs roadmap)
- [ ] Experiment Dashboard (currently: static plots via `experiments/plot_results.py`)

---

## How to run (get final results)

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...

# check scale/time/cost before committing to a full run
python run_all.py --estimate

# runs both arms (constrained, unconstrained) x NUM_REPEATS lineages each,
# then automatically runs the stats comparison and saves the plot
python run_all.py
```

Output:
- `data/runs.db` — every generation, every lineage, every arm (raw data)
- `data/stats_summary.txt` — Welch's t-test + Cohen's d, constrained vs unconstrained,
  on final pass rate, pass-rate improvement, and drift from generation 0
- `data/plots/results.png` — pass rate and drift vs. generation, mean ± 95% CI per arm

Adjust scale in `config.py`: `NUM_REPEATS` (independent lineages per arm — need ≥5 for
a meaningful t-test), `NUM_GENERATIONS`, `TASKS_PER_GENERATION`.

To run a single arm/repeat manually instead (e.g. for debugging):
```bash
python -c "from app import main; main(arm='constrained', repeat_id=0)"
```

---

## What this tests, mapped to the paper

- **§2.2 (Von Neumann constructor)** — each generation's DNA (the "description") is
  produced by a patch proposed and validated against the previous generation, mirroring
  the constructor/description separation.
- **§5.1 (value drift across generations)** — `drift/tracker.py` measures this directly:
  semantic distance of each generation's DNA from generation 0.
- **§5.2 (control problem / capability control)** — the `constrained` vs `unconstrained`
  arm ablation directly tests whether loosening what a self-modifying agent is allowed to
  touch changes its drift or performance trajectory.

## Citation

Coming soon.