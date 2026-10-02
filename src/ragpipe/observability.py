"""Structured logging + per-query tracing. No external deps.

Every query produces a Trace: per-stage latency, hit counts, final citations.
That's how you actually find out *where* a RAG pipeline is slow instead of guessing.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

_LOG_CONFIGURED = False
_LOCK = threading.Lock()


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": round(record.created, 4),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "extra_fields", None)
        if extra:
            payload.update(extra)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", json_logs: bool = False) -> None:
    global _LOG_CONFIGURED
    with _LOCK:
        if _LOG_CONFIGURED:
            logging.getLogger().setLevel(level.upper())
            return
        handler = logging.StreamHandler(sys.stderr)
        if json_logs:
            handler.setFormatter(_JsonFormatter())
        else:
            handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S")
            )
        root = logging.getLogger()
        root.handlers = [handler]
        root.setLevel(level.upper())
        _LOG_CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


@dataclass
class Span:
    name: str
    started: float = field(default_factory=time.perf_counter)
    duration_ms: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    def finish(self) -> Span:
        self.duration_ms = (time.perf_counter() - self.started) * 1000.0
        return self


@dataclass
class Trace:
    trace_id: str
    query: str
    spans: list[Span] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    started: float = field(default_factory=time.perf_counter)

    @contextmanager
    def span(self, name: str, **meta: Any) -> Iterator[Span]:
        s = Span(name=name, meta=dict(meta))
        self.spans.append(s)
        try:
            yield s
        finally:
            s.finish()

    @property
    def total_ms(self) -> float:
        return (time.perf_counter() - self.started) * 1000.0

    def stage_ms(self, name: str) -> float:
        return sum(s.duration_ms for s in self.spans if s.name == name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "query": self.query,
            "total_ms": round(self.total_ms, 2),
            "spans": [
                {"name": s.name, "ms": round(s.duration_ms, 2), **s.meta} for s in self.spans
            ],
            **self.meta,
        }


class Tracer:
    """Creates traces and optionally persists them as JSONL for offline analysis."""

    def __init__(self, sink: str = "stdout", path: str = "data/traces.jsonl") -> None:
        self.sink = sink
        self.path = path
        self._log = get_logger("ragpipe.trace")
        if sink == "file":
            from pathlib import Path

            Path(path).parent.mkdir(parents=True, exist_ok=True)

    def start(self, query: str) -> Trace:
        return Trace(trace_id=uuid.uuid4().hex[:12], query=query)

    def emit(self, trace: Trace) -> None:
        if self.sink == "none":
            return
        record = trace.to_dict()
        if self.sink == "stdout":
            self._log.info("trace %s", json.dumps(record, default=str))
        elif self.sink == "file":
            try:
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record, default=str) + "\n")
            except OSError as exc:  # never let telemetry break the request path
                self._log.warning("failed to write trace: %s", exc)
