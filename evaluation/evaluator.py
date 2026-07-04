"""
Runs generated code against a task's tests in an isolated subprocess with a
timeout. This is the ONE component the self-improvement loop is never
allowed to touch -- it has no path into either CONSTRAINED_EDITABLE_FIELDS
or UNCONSTRAINED_EDITABLE_FIELDS in config.py, by design. That separation is
the safety boundary the whole ablation is testing.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

TIMEOUT_SECONDS = 8


def evaluate(code: str, task: dict) -> dict:
    # NOTE: deliberately NOT using textwrap.dedent here -- `code` and
    # task['test'] are multi-line strings with their own internal
    # indentation, so wrapping them in an indented template and dedenting
    # afterwards corrupts the indentation of every line but the first.
    # Plain concatenation at column 0 keeps each block's own indentation intact.
    test_code = (
        f"{code}\n\n"
        f"{task['test']}\n\n"
        f"check({task['entry_point']})\n"
        f"print(\"__RSCF_PASS__\")\n"
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(test_code)
        path = f.name

    try:
        result = subprocess.run(
            [sys.executable, path],
            capture_output=True, text=True, timeout=TIMEOUT_SECONDS
        )
        passed = "__RSCF_PASS__" in result.stdout
        return {"passed": passed, "stdout": result.stdout[-500:], "stderr": result.stderr[-500:]}
    except subprocess.TimeoutExpired:
        return {"passed": False, "stdout": "", "stderr": "TIMEOUT"}
    finally:
        Path(path).unlink(missing_ok=True)
