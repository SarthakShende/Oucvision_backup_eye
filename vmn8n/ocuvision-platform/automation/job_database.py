import sqlite3
import json
import time
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, List

DB_PATH = Path(__file__).resolve().parent.parent / "automation_data" / "jobs.db"

class JobDB:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    job_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    started_at REAL,
                    completed_at REAL,
                    progress INTEGER DEFAULT 0,
                    result TEXT,
                    error TEXT,
                    configuration TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def create_job(self, job_type: str, configuration: Dict[str, Any]) -> str:
        job_id = str(uuid.uuid4())
        now = time.time()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO jobs (job_id, job_type, status, created_at, progress, result, error, configuration)
                VALUES (?, ?, 'queued', ?, 0, NULL, NULL, ?)
                """,
                (job_id, job_type, now, json.dumps(configuration or {})),
            )
            conn.commit()
        return job_id

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
            row = cur.fetchone()
        if not row:
            return None
        return self._row_to_dict(row)

    def list_jobs(self, limit: int = 50) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,))
            rows = cur.fetchall()
        return [self._row_to_dict(r) for r in rows]

    def update_job(self, job_id: str, **fields) -> bool:
        allowed = {"status", "started_at", "completed_at", "progress", "result", "error"}
        updates = []
        params = []
        for k, v in fields.items():
            if k in allowed:
                updates.append(f"{k} = ?")
                if k in ("result", "configuration"):
                    params.append(json.dumps(v) if v is not None else None)
                elif k == "progress":
                    params.append(max(0, min(100, int(v))))
                elif k == "error" and v is not None:
                    params.append(str(v)[:4000])
                else:
                    params.append(v)
        if not updates:
            return False
        params.append(job_id)
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(f"UPDATE jobs SET {', '.join(updates)} WHERE job_id = ?", params)
            conn.commit()
            return cur.rowcount > 0

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        if d.get("configuration"):
            try:
                d["configuration"] = json.loads(d["configuration"])
            except Exception:
                d["configuration"] = {}
        if d.get("result"):
            try:
                d["result"] = json.loads(d["result"])
            except Exception:
                d["result"] = None
        return d

    def count_running(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("SELECT COUNT(*) FROM jobs WHERE status = 'running'")
            return cur.fetchone()[0]

    def get_queued(self) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute("SELECT * FROM jobs WHERE status = 'queued' ORDER BY created_at ASC")
            rows = cur.fetchall()
        return [self._row_to_dict(r) for r in rows]

    def recover_interrupted(self):
        now = time.time()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE jobs SET status='failed', completed_at=?, error='Interrupted on restart' WHERE status='running'",
                (now,),
            )
            conn.commit()
