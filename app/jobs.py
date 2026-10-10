import json
import logging
import sqlite3
import threading
import uuid
from datetime import datetime, timedelta, timezone

from .config import JOB_RETENTION_HOURS

logger = logging.getLogger("jobs")

ACTIVE_STATUSES = ("queued", "running")
INTERRUPTED_MESSAGE = "Service restarted while the job was running."


def utc_now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def expiry_iso():
    return (datetime.now(timezone.utc) + timedelta(hours=JOB_RETENTION_HOURS)).strftime("%Y-%m-%dT%H:%M:%SZ")


def is_job_expired(job):
    return job["expires_at"] is not None and job["expires_at"] <= utc_now_iso()


def build_status_body(job):
    body = {
        "job_id": job["job_id"],
        "status": job["status"],
        "total": job["total"],
        "processed": job["processed"],
        "created_at": job["created_at"],
        "started_at": job["started_at"],
        "finished_at": job["finished_at"],
        "expires_at": job["expires_at"],
        "model_version": job["model_version"],
        "error": None,
    }
    if job["status"] == "failed":
        body["error"] = {"code": job["error_code"], "message": job["error_message"]}
    return body


class JobStore:
    def __init__(self, database_path):
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.connection = sqlite3.connect(str(database_path), check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.create_table()

    def run_write(self, query, parameters=()):
        with self.lock:
            cursor = self.connection.execute(query, parameters)
            self.connection.commit()
            return cursor.rowcount

    def fetch_one(self, query, parameters=()):
        with self.lock:
            return self.connection.execute(query, parameters).fetchone()

    def create_table(self):
        self.run_write(
            """CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                total INTEGER NOT NULL,
                processed INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT,
                expires_at TEXT,
                model_version TEXT NOT NULL,
                error_code TEXT,
                error_message TEXT,
                idempotency_key TEXT UNIQUE,
                tickets_json TEXT,
                results_json TEXT
            )"""
        )

    def mark_interrupted_jobs(self):
        self.run_write(
            """UPDATE jobs SET status = 'failed', finished_at = ?, expires_at = ?,
               error_code = 'interrupted', error_message = ?, tickets_json = NULL
               WHERE status IN ('queued', 'running')""",
            (utc_now_iso(), expiry_iso(), INTERRUPTED_MESSAGE),
        )

    def submit_job(self, tickets, idempotency_key, model_version, max_active_jobs):
        with self.lock:
            if idempotency_key:
                existing_job = self.connection.execute(
                    "SELECT * FROM jobs WHERE idempotency_key = ?", (idempotency_key,)
                ).fetchone()
                if existing_job:
                    return "existing", existing_job
            active_count = self.connection.execute(
                "SELECT COUNT(*) FROM jobs WHERE status IN ('queued', 'running')"
            ).fetchone()[0]
            if active_count >= max_active_jobs:
                return "full", None
            job_id = str(uuid.uuid4())
            self.connection.execute(
                """INSERT INTO jobs (job_id, status, total, processed, created_at, model_version,
                   idempotency_key, tickets_json) VALUES (?, 'queued', ?, 0, ?, ?, ?, ?)""",
                (job_id, len(tickets), utc_now_iso(), model_version, idempotency_key,
                 json.dumps(tickets, ensure_ascii=False)),
            )
            self.connection.commit()
            return "created", self.connection.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()

    def get_job(self, job_id):
        return self.fetch_one("SELECT * FROM jobs WHERE job_id = ?", (job_id,))

    def job_exists(self, job_id):
        return self.fetch_one("SELECT 1 FROM jobs WHERE job_id = ?", (job_id,)) is not None

    def claim_next_job(self):
        with self.lock:
            next_job = self.connection.execute(
                "SELECT job_id FROM jobs WHERE status = 'queued' ORDER BY created_at, rowid LIMIT 1"
            ).fetchone()
            if next_job is None:
                return None
            self.connection.execute(
                "UPDATE jobs SET status = 'running', started_at = ? WHERE job_id = ?",
                (utc_now_iso(), next_job["job_id"]),
            )
            self.connection.commit()
            return next_job["job_id"]

    def load_tickets(self, job_id):
        row = self.fetch_one("SELECT tickets_json FROM jobs WHERE job_id = ?", (job_id,))
        if row is None or row["tickets_json"] is None:
            return None
        return json.loads(row["tickets_json"])

    def load_results(self, job_id):
        row = self.fetch_one("SELECT results_json FROM jobs WHERE job_id = ?", (job_id,))
        if row is None or row["results_json"] is None:
            return []
        return json.loads(row["results_json"])

    def update_progress(self, job_id, processed_count):
        self.run_write(
            "UPDATE jobs SET processed = ? WHERE job_id = ? AND status = 'running'", (processed_count, job_id)
        )

    def mark_succeeded(self, job_id, predictions):
        self.run_write(
            """UPDATE jobs SET status = 'succeeded', processed = ?, finished_at = ?, expires_at = ?,
               results_json = ?, tickets_json = NULL WHERE job_id = ? AND status = 'running'""",
            (len(predictions), utc_now_iso(), expiry_iso(), json.dumps(predictions, ensure_ascii=False), job_id),
        )

    def mark_failed(self, job_id, error_code, error_message):
        self.run_write(
            """UPDATE jobs SET status = 'failed', finished_at = ?, expires_at = ?, error_code = ?,
               error_message = ?, tickets_json = NULL WHERE job_id = ? AND status = 'running'""",
            (utc_now_iso(), expiry_iso(), error_code, error_message, job_id),
        )

    def delete_job(self, job_id):
        return self.run_write("DELETE FROM jobs WHERE job_id = ?", (job_id,)) > 0

    def clear_expired_payloads(self):
        self.run_write(
            """UPDATE jobs SET tickets_json = NULL, results_json = NULL
               WHERE expires_at IS NOT NULL AND expires_at <= ? AND results_json IS NOT NULL""",
            (utc_now_iso(),),
        )


class JobWorker:
    def __init__(self, job_store, get_predictor, chunk_size):
        self.job_store = job_store
        self.get_predictor = get_predictor
        self.chunk_size = chunk_size
        self.wake_event = threading.Event()

    def start(self):
        threading.Thread(target=self.run_forever, daemon=True, name="job-worker").start()

    def wake(self):
        self.wake_event.set()

    def wait_for_work(self):
        self.wake_event.wait(timeout=2)
        self.wake_event.clear()

    def run_forever(self):
        while True:
            predictor = self.get_predictor()
            job_id = self.job_store.claim_next_job() if predictor else None
            if job_id is None:
                self.job_store.clear_expired_payloads()
                self.wait_for_work()
                continue
            self.process_job(job_id, predictor)

    def process_job(self, job_id, predictor):
        try:
            tickets = self.job_store.load_tickets(job_id)
            if tickets is None:
                return
            predictions = []
            for start in range(0, len(tickets), self.chunk_size):
                if not self.job_store.job_exists(job_id):
                    return
                predictions.extend(predictor.predict_tickets(tickets[start:start + self.chunk_size]))
                self.job_store.update_progress(job_id, len(predictions))
            self.job_store.mark_succeeded(job_id, predictions)
        except Exception:
            logger.exception("job %s failed", job_id)
            self.job_store.mark_failed(job_id, "prediction_error", "The job could not be completed.")
