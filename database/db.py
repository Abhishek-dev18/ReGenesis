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
    num_tasks INTEGER,
    accepted_patch TEXT,
    patch_status TEXT,
    drift_score REAL,
    avg_runtime_s REAL,
    total_tokens INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


def get_conn(db_path: Path):
    conn = sqlite3.connect(str(db_path))
    conn.execute(SCHEMA)
    # Backward-compat: add repeat_id if an old runs.db from before this column existed is reused.
    cols = [row[1] for row in conn.execute("PRAGMA table_info(generations)")]
    if "repeat_id" not in cols:
        conn.execute("ALTER TABLE generations ADD COLUMN repeat_id INTEGER DEFAULT 0")
        conn.commit()
    return conn


def log_generation(conn, arm, repeat_id, generation, dna, pass_rate, num_tasks,
                    accepted_patch, patch_status, drift_score, avg_runtime_s, total_tokens):
    conn.execute(
        """INSERT INTO generations
           (arm, repeat_id, generation, dna_json, pass_rate, num_tasks, accepted_patch,
            patch_status, drift_score, avg_runtime_s, total_tokens)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (arm, repeat_id, generation, json.dumps(dna.to_dict()), pass_rate, num_tasks,
         accepted_patch, patch_status, drift_score, avg_runtime_s, total_tokens)
    )
    conn.commit()
