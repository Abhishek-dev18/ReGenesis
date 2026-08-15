"""
Sandboxed benchmark evaluator.

Visible tests are the optimization signal.
Hidden tests are separate case-level checks used only for measurement.
The solver/reflector never receive hidden cases.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

TIMEOUT_SECONDS = 8


def _run_script(script: str) -> dict:
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
        encoding="utf-8",
    ) as f:
        f.write(script)
        path = f.name

    try:
        result = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
        )
        return {
            "passed": "__RSCF_PASS__" in result.stdout,
            "stdout": result.stdout[-2000:],
            "stderr": result.stderr[-2000:],
        }
    except subprocess.TimeoutExpired:
        return {"passed": False, "stdout": "", "stderr": "TIMEOUT"}
    finally:
        Path(path).unlink(missing_ok=True)


def _run_check_function(code: str, test: str, entry_point: str) -> dict:
    script = (
        f"{code}\n\n"
        f"{test}\n\n"
        f"check({entry_point})\n"
        f"print('__RSCF_PASS__')\n"
    )
    return _run_script(script)


def evaluate(code: str, task: dict) -> dict:
    return _run_check_function(code, task["test"], task["entry_point"])


def evaluate_hidden(code: str, task: dict) -> dict:
    cases = task.get("hidden_tests")

    if not isinstance(cases, list) or not cases:
        # Backward compatibility with the previous hidden_test string.
        hidden_test = task.get("hidden_test")
        if hidden_test:
            result = _run_check_function(
                code,
                hidden_test,
                task["entry_point"],
            )
            return {
                "passed": result["passed"],
                "passed_cases": int(result["passed"]),
                "total_cases": 1,
                "stdout": result["stdout"],
                "stderr": result["stderr"],
            }

        return {
            "passed": False,
            "passed_cases": 0,
            "total_cases": 0,
            "stdout": "",
            "stderr": "No hidden test cases configured",
        }

    # Run all hidden cases for a task in ONE subprocess. This gives us
    # case-level resolution without creating 10 subprocesses per task.
    case_lines = [
        "import json",
        "_hidden_results = []",
        f"candidate = {task['entry_point']}",
    ]

    for assertion in cases:
        case_lines.extend([
            "try:",
            f"    assert {assertion}",
            "    _hidden_results.append(True)",
            "except Exception:",
            "    _hidden_results.append(False)",
        ])

    case_lines.append("print('__RSCF_HIDDEN__' + json.dumps(_hidden_results))")

    script = f"{code}\n\n" + "\n".join(case_lines) + "\n"

    result = _run_script(script)

    marker = "__RSCF_HIDDEN__"
    raw = result["stdout"]

    if marker not in raw:
        return {
            "passed": False,
            "passed_cases": 0,
            "total_cases": len(cases),
            "stdout": raw,
            "stderr": result["stderr"] or "Hidden evaluator produced no result",
        }

    payload = raw.split(marker, 1)[1].strip()

    try:
        case_results = json.loads(payload)
    except json.JSONDecodeError:
        return {
            "passed": False,
            "passed_cases": 0,
            "total_cases": len(cases),
            "stdout": raw,
            "stderr": "Could not parse hidden evaluator result",
        }

    passed_cases = sum(bool(x) for x in case_results)
    total_cases = len(case_results)

    return {
        "passed": passed_cases == total_cases,
        "passed_cases": passed_cases,
        "total_cases": total_cases,
        "stdout": "",
        "stderr": "",
    }
