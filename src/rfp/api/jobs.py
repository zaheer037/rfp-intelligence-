"""Small durable job registry for one-process deployments with bounded active work."""

import asyncio
import json
import sqlite3
import time
import uuid

from rfp.core.observability import ACTIVE_JOBS, ERRORS, event


class JobManager:
    """Persist statuses/results in SQLite and reject work beyond configured capacity."""

    def __init__(self, settings):
        self.settings = settings
        settings.storage_dir.mkdir(parents=True, exist_ok=True)
        self.path = settings.storage_dir / "jobs.sqlite"
        self.tasks = {}
        with self._db() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute(
                "CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, kind TEXT, status TEXT, "
                "created REAL, updated REAL, result TEXT, error TEXT)"
            )
            db.execute(
                "UPDATE jobs SET status='failed', error='Process restarted before completion', updated=? "
                "WHERE status='running'",
                (time.time(),),
            )

    def _db(self):
        """Open short-lived connections; SQLite serializes these small status writes."""
        return sqlite3.connect(self.path, timeout=10)

    def submit(self, kind: str, operation) -> str:
        """Reserve a slot before scheduling work; no unbounded in-memory queue."""
        if len(self.tasks) >= self.settings.max_active_jobs:
            raise OverflowError("Job capacity reached; retry later")
        identifier = uuid.uuid4().hex
        now = time.time()
        with self._db() as db:
            db.execute(
                "INSERT INTO jobs VALUES (?, ?, 'running', ?, ?, NULL, NULL)", (identifier, kind, now, now)
            )
        task = asyncio.create_task(self._run(identifier, kind, operation))
        self.tasks[identifier] = task
        task.add_done_callback(lambda _: self.tasks.pop(identifier, None))
        return identifier

    async def _run(self, identifier, kind, operation):
        """Save successful outputs and sanitized errors, including interrupted shutdowns."""
        ACTIVE_JOBS.inc()
        try:
            if kind == "index":
                # Index writes run to completion; cancelling a thread would not stop its disk mutations.
                result = await operation(identifier)
            else:
                async with asyncio.timeout(self.settings.task_timeout_seconds):
                    result = await operation(identifier)
            if hasattr(result, "model_dump"):
                result = result.model_dump()
            status = (
                "partial" if isinstance(result, dict) and result.get("status") == "partial" else "completed"
            )
            with self._db() as db:
                db.execute(
                    "UPDATE jobs SET status=?, result=?, updated=? WHERE id=?",
                    (status, json.dumps(result), time.time(), identifier),
                )
        except BaseException as exc:
            with self._db() as db:
                db.execute(
                    "UPDATE jobs SET status='failed', error=?, updated=? WHERE id=?",
                    (f"{type(exc).__name__}: inspect server logs/trace", time.time(), identifier),
                )
            ERRORS.labels("job").inc()
            event("job_failed", job_id=identifier, kind=kind, error_type=type(exc).__name__)
            if isinstance(exc, asyncio.CancelledError):
                raise
        finally:
            ACTIVE_JOBS.dec()

    def get(self, identifier: str) -> dict | None:
        """Return a durable job record and its structured result."""
        with self._db() as db:
            db.row_factory = sqlite3.Row
            row = db.execute("SELECT * FROM jobs WHERE id=?", (identifier,)).fetchone()
        if not row:
            return None
        record = dict(row)
        record["result"] = json.loads(record["result"]) if record["result"] else None
        return record

    def recent(self, limit: int = 100) -> list[dict]:
        """Read newest durable jobs for workspace views without exposing SQL to routes."""
        if not 1 <= limit <= 100:
            raise ValueError("Job history limit must be between 1 and 100")
        with self._db() as db:
            db.row_factory = sqlite3.Row
            rows = db.execute("SELECT * FROM jobs ORDER BY created DESC LIMIT ?", (limit,)).fetchall()
        records = []
        for row in rows:
            record = dict(row)
            record["result"] = json.loads(record["result"]) if record["result"] else None
            records.append(record)
        return records

    async def close(self):
        """Drain accepted jobs during graceful shutdown instead of abandoning writes."""
        if self.tasks:
            await asyncio.gather(*list(self.tasks.values()), return_exceptions=True)
