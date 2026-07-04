import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ---- LLM settings ----
# Set your API key as an environment variable before running, e.g.:
#   export ANTHROPIC_API_KEY="sk-ant-..."      (Mac/Linux/Colab)
#   set ANTHROPIC_API_KEY=sk-ant-...           (Windows cmd)
#   $env:ANTHROPIC_API_KEY="sk-ant-..."        (Windows PowerShell)
LLM_PROVIDER = os.environ.get("RSCF_PROVIDER", "anthropic")  # "anthropic" or "openai"
SOLVER_MODEL = os.environ.get("RSCF_SOLVER_MODEL", "claude-haiku-4-5-20251001")
REFLECTOR_MODEL = os.environ.get("RSCF_REFLECTOR_MODEL", "claude-haiku-4-5-20251001")

# ---- Experiment settings ----
NUM_GENERATIONS = int(os.environ.get("RSCF_GENERATIONS", 10))
TASKS_PER_GENERATION = int(os.environ.get("RSCF_TASKS_PER_GEN", 8))  # subset of benchmark evaluated per generation

# Arm mode controls which DNA fields the patcher is allowed to touch.
# "constrained":   strategy-only fields (safe boundary)
# "unconstrained": strategy + evaluation-adjacent fields (the risk axis for the ablation)
ARM_MODE = os.environ.get("RSCF_ARM", "constrained")  # "constrained" | "unconstrained"

CONSTRAINED_EDITABLE_FIELDS = ["system_prompt", "decomposition_strategy", "temperature", "self_critique_enabled"]
UNCONSTRAINED_EDITABLE_FIELDS = CONSTRAINED_EDITABLE_FIELDS + ["confidence_threshold", "retry_policy"]

# ---- Paths ----
DATA_DIR = BASE_DIR / "data"
DNA_DIR = DATA_DIR / "dna_versions"
DB_PATH = DATA_DIR / "runs.db"
BENCHMARK_PATH = BASE_DIR / "benchmarks" / "tasks.json"
PLOTS_DIR = DATA_DIR / "plots"

for d in [DATA_DIR, DNA_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
