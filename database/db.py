import json
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS generations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    arm TEXT,
    repeat_id INTEGER DEFAULT 0,
    generation INTEGER,
    dna_json TEXT,
    pass_rate REAL,
    hidden_pass_rate REAL,
    num_tasks INTEGER,
    accepted_patch TEXT,
    patch_status TEXT,
    drift_score REAL,
    solution_drift REAL DEFAULT 0.0,
    regression INTEGER DEFAULT 0,
    total_solver_attempts INTEGER DEFAULT 0,
    retry_events INTEGER DEFAULT 0,
    avg_runtime_s REAL,
    total_tokens INTEGER,
    base_seed INTEGER DEFAULT 20260815,
    repeat_seed INTEGER DEFAULT 0,
    task_order_seed INTEGER DEFAULT 0,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


def reset_generations(conn):
    conn.execute("DELETE FROM generations")
    conn.execute("DELETE FROM sqlite_sequence WHERE name='generations'")
    conn.commit()


def get_conn(db_path: Path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute(SCHEMA)

    cols = [row[1] for row in conn.execute("PRAGMA table_info(generations)")]

    migrations = {
        "repeat_id": "ALTER TABLE generations ADD COLUMN repeat_id INTEGER DEFAULT 0",
        "hidden_pass_rate": "ALTER TABLE generations ADD COLUMN hidden_pass_rate REAL",
        "solution_drift": "ALTER TABLE generations ADD COLUMN solution_drift REAL DEFAULT 0.0",
        "regression": "ALTER TABLE generations ADD COLUMN regression INTEGER DEFAULT 0",
        "total_solver_attempts": "ALTER TABLE generations ADD COLUMN total_solver_attempts INTEGER DEFAULT 0",
        "retry_events": "ALTER TABLE generations ADD COLUMN retry_events INTEGER DEFAULT 0",
        "base_seed": "ALTER TABLE generations ADD COLUMN base_seed INTEGER DEFAULT 20260815",
        "repeat_seed": "ALTER TABLE generations ADD COLUMN repeat_seed INTEGER DEFAULT 0",
        "task_order_seed": "ALTER TABLE generations ADD COLUMN task_order_seed INTEGER DEFAULT 0",
    }

    for col, statement in migrations.items():
        if col not in cols:
            conn.execute(statement)

    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_generations_arm_repeat_generation "
        "ON generations(arm, repeat_id, generation)"
    )
    conn.commit()
    return conn


def log_generation(
    conn,
    arm,
    repeat_id,
    generation,
    dna,
    pass_rate,
    hidden_pass_rate,
    num_tasks,
    accepted_patch,
    patch_status,
    drift_score,
    avg_runtime_s,
    total_tokens,
    solution_drift=0.0,
    regression=0,
    total_solver_attempts=0,
    retry_events=0,
    base_seed=20260815,
    repeat_seed=0,
    task_order_seed=0,
):
    conn.execute(
        """INSERT INTO generations
           (arm, repeat_id, generation, dna_json, pass_rate, hidden_pass_rate,
            num_tasks, accepted_patch, patch_status, drift_score,
            solution_drift, regression, total_solver_attempts, retry_events,
            avg_runtime_s, total_tokens, base_seed, repeat_seed, task_order_seed)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            arm,
            repeat_id,
            generation,
            json.dumps(dna.to_dict()),
            pass_rate,
            hidden_pass_rate,
            num_tasks,
            accepted_patch,
            patch_status,
            drift_score,
            solution_drift,
            regression,
            total_solver_attempts,
            retry_events,
            avg_runtime_s,
            total_tokens,
            base_seed,
            repeat_seed,
            task_order_seed,
        ),
    )
    conn.commit()
