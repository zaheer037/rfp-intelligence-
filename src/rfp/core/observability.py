"""JSON logs, per-run traces and bounded-cardinality Prometheus metrics."""

import contextvars
import json
import logging
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from prometheus_client import Counter, Gauge, Histogram

REQUEST_ID = contextvars.ContextVar("request_id", default="")
HTTP_REQUESTS = Counter("rfp_http_requests_total", "HTTP requests", ["method", "route", "status"])
HTTP_LATENCY = Histogram("rfp_http_duration_seconds", "HTTP latency", ["method", "route"])
STAGE_LATENCY = Histogram("rfp_stage_duration_seconds", "Pipeline stage latency", ["stage"])
ERRORS = Counter("rfp_errors_total", "Failures by bounded operation name", ["operation"])
TOKENS = Counter("rfp_llm_tokens_total", "Actual provider token counts", ["direction"])
LLM_CALLS = Counter("rfp_llm_calls_total", "LLM calls", ["agent", "status"])
ACTIVE_JOBS = Gauge("rfp_active_jobs", "Running background jobs")
INDEX_CHUNKS = Gauge("rfp_index_chunks", "Loaded chunk count")
SEARCHES = Counter("rfp_searches_total", "Search requests", ["mode"])
SEARCH_HITS = Histogram("rfp_search_hits", "Returned results", buckets=(0, 1, 3, 5, 10, 20, 50))
VALIDATION = Counter("rfp_validation_fields_total", "Field validation outcomes", ["outcome"])
RETRIES = Counter("rfp_agent_retries_total", "Retried agent calls", ["agent"])


class JsonFormatter(logging.Formatter):
    def format(self, record):
        """Emit one JSON event per line; never format model/client objects or secrets."""
        body = {
            "time": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": REQUEST_ID.get(),
        }
        body.update(getattr(record, "event_data", {}))
        if record.exc_info:
            body["exception_type"] = record.exc_info[0].__name__
        return json.dumps(body, default=str)


def configure_logging(level: str) -> None:
    """Configure application logging and suppress verbose third-party request bodies."""
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=level, handlers=[handler], force=True)
    for name in ("httpx", "httpcore", "LiteLLM", "google.adk", "uvicorn.access"):
        logging.getLogger(name).setLevel(logging.WARNING)


def event(message: str, **data: Any) -> None:
    logging.getLogger("rfp").info(message, extra={"event_data": data})


@contextmanager
def stage(name: str):
    """Measure operation latency, including failed operations."""
    start = time.perf_counter()
    try:
        yield
    except Exception:
        ERRORS.labels(name).inc()
        raise
    finally:
        STAGE_LATENCY.labels(name).observe(time.perf_counter() - start)


class Trace:
    """Write restricted per-run JSONL; content capture is explicit opt-in."""

    def __init__(self, directory: Path, run_id: str, capture_content: bool):
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"{run_id}.jsonl"
        self.run_id = run_id
        self.capture_content = capture_content

    def write(self, step: str, **data: Any) -> None:
        """Record metadata always, retaining potentially sensitive payloads only by opt-in."""
        if not self.capture_content:
            for key in ("input", "output", "tool_arguments", "tool_result"):
                if key in data:
                    data[key] = {"redacted": True, "characters": len(str(data[key]))}
        record = {"time": datetime.now(timezone.utc).isoformat(), "run_id": self.run_id, "step": step, **data}
        with self.path.open("a", encoding="utf-8") as handle:
            self.path.chmod(0o600)
            handle.write(json.dumps(record, default=str) + "\n")
        event(
            "agent_step",
            run_id=self.run_id,
            step=step,
            **{
                k: v for k, v in data.items() if k not in ("input", "output", "tool_arguments", "tool_result")
            },
        )
