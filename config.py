import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ---- LLM settings ----
LLM_PROVIDER = os.environ.get("RSCF_PROVIDER", "anthropic")
SOLVER_MODEL = os.environ.get("RSCF_SOLVER_MODEL", "claude-haiku-4-5-20251001")
REFLECTOR_MODEL = os.environ.get("RSCF_REFLECTOR_MODEL", "claude-haiku-4-5-20251001")

# ---- Experiment settings ----
BASE_SEED = int(os.environ.get("RSCF_BASE_SEED", 20260815))
NUM_GENERATIONS = int(os.environ.get("RSCF_GENERATIONS", 5))
TASKS_PER_GENERATION = int(os.environ.get("RSCF_TASKS_PER_GEN", 10))
NUM_REPEATS = int(os.environ.get("RSCF_REPEATS", 5))
ARM_MODE = os.environ.get("RSCF_ARM", "constrained")

# Constrained can change strategy-oriented fields only.
CONSTRAINED_EDITABLE_FIELDS = [
    "system_prompt",
    "decomposition_strategy",
    "temperature",
    "self_critique_enabled",
]

# Unconstrained additionally gets a real execution-affecting capability.
UNCONSTRAINED_EDITABLE_FIELDS = CONSTRAINED_EDITABLE_FIELDS + [
    "retry_policy",
]

# ---- Paths ----
DATA_DIR = BASE_DIR / "data"
DNA_DIR = DATA_DIR / "dna_versions"
DB_PATH = DATA_DIR / "runs.db"
BENCHMARK_PATH = BASE_DIR / "benchmarks" / "tasks.json"
PLOTS_DIR = DATA_DIR / "plots"

for d in [DATA_DIR, DNA_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
