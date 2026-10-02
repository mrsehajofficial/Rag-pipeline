"""WSGI adapter for RAGPipeline -- stdlib only, no framework dependency.

This module exposes a WSGI-compatible application that wraps the RAGPipeline
in a standard ``__call__(environ, start_response)`` interface, making it
deployable under any WSGI server (gunicorn, uWSGI, mod_wsgi, etc.).

    POST /query          {"question": "...", "top_k": 8}
    POST /ingest         {"path": "./docs"}
    GET  /health         (unauthenticated)
    GET  /stats
    GET  /metrics        (Prometheus text format)

Security:
  * API key auth: set RAG_API_KEY in env. If set, every request must include
    the header ``Authorization: Bearer <key>`` or the query param
    ``?api_key=<key>``.  If RAG_API_KEY is unset the server accepts
    unauthenticated requests -- suitable for local dev, not for internet-
    facing deployments.

  * Path-injection guard: /ingest only allows paths inside a configurable
    root directory (RAG_INGEST_ROOT, defaults to the current working
    directory).  Traversal attempts are rejected with a 403 before touching
    the filesystem.

  * Query-length cap: RAG_MAX_QUERY_CHARS (default 2000).  Longer queries
    are rejected before the expensive embed+retrieve+generate path runs.

  * Body-size cap: MAX_BODY_BYTES (1 MB).  An unbounded read is a free DoS.

Usage:
    # As a module (e.g. gunicorn):
    #   gunicorn ragpipe.wsgi:app
    #
    # Programmatically:
    #   from ragpipe.wsgi import create_app
    #   app = create_app(my_pipeline)
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .observability import get_logger
from .pipeline import RAGPipeline

log = get_logger("ragpipe.wsgi")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_BODY_BYTES = 1_000_000  # refuse oversized bodies; an unbounded read is a free DoS
_MAX_QUERY_CHARS = int(os.environ.get("RAG_MAX_QUERY_CHARS", "2000"))

# ---------------------------------------------------------------------------
# API key auth
# ---------------------------------------------------------------------------

_API_KEY: str | None = (os.environ.get("RAG_API_KEY") or "").strip() or None

if _API_KEY is None:
    log.warning(
        "RAG_API_KEY is not set. The server will accept unauthenticated requests. "
        "Set RAG_API_KEY=<secret> for any network-accessible deployment."
    )


def _check_auth(environ: dict, qs: dict) -> bool:
    """Return True if the request carries a valid API key, or if no key is configured."""
    if _API_KEY is None:
        return True
    # Check Authorization: Bearer <key> header
    auth_header = environ.get("HTTP_AUTHORIZATION", "").strip()
    if auth_header.startswith("Bearer "):
        return auth_header[7:] == _API_KEY
    # Also accept ?api_key=... for simple curl / browser testing
    param = (qs.get("api_key") or [""])[0]
    return param == _API_KEY


# ---------------------------------------------------------------------------
# Path-injection guard
# ---------------------------------------------------------------------------

def _get_ingest_root() -> Path:
    """Read the ingest root at call time so CLI overrides (which set the env var
    after import) are honoured."""
    return Path(os.environ.get("RAG_INGEST_ROOT", ".")).resolve()


def _safe_ingest_path(raw_path: str) -> Path:
    """Resolve and validate that the requested ingest path sits inside the ingest root.

    Returns the resolved Path on success, raises ValueError on traversal attempts.
    """
    root = _get_ingest_root()
    resolved = (root / raw_path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        raise ValueError(
            f"Path '{raw_path}' escapes the allowed ingest root ({root}). "
            f"Set RAG_INGEST_ROOT to explicitly widen the allowed directory."
        )
    if not resolved.exists():
        raise FileNotFoundError(f"No such file or directory: {resolved}")
    return resolved


# ---------------------------------------------------------------------------
# WSGI application factory
# ---------------------------------------------------------------------------

def create_app(pipeline: RAGPipeline) -> Callable:
    """Create a WSGI application wrapping the given RAGPipeline.

    This factory is the primary entry point for programmatic use.  It returns
    a callable conforming to the WSGI spec (PEP 3333).

    Args:
        pipeline: An initialised RAGPipeline instance.

    Returns:
        A WSGI application callable.
    """

    def _read_body(environ: dict) -> bytes:
        """Read the request body from wsgi.input, capped at MAX_BODY_BYTES."""
        try:
            content_length = int(environ.get("CONTENT_LENGTH") or 0)
        except (TypeError, ValueError):
            content_length = 0
        if content_length > MAX_BODY_BYTES:
            raise ValueError(f"request body too large ({content_length} bytes)")
        if content_length == 0:
            return b""
        body = environ["wsgi.input"].read(content_length)
        return body

    def _parse_json(body: bytes) -> dict:
        """Parse a JSON body, raising ValueError on bad input."""
        if not body:
            return {}
        return json.loads(body.decode("utf-8"))

    def _send_json(
        start_response: Callable,
        status: int,
        payload: Any,
        content_type: str = "application/json",
    ) -> list[bytes]:
        """Send a JSON response with proper Content-Type and Content-Length headers."""
        body = json.dumps(payload, default=str).encode("utf-8")
        status_line = f"{status} {_status_reason(status)}"
        headers = [
            ("Content-Type", content_type),
            ("Content-Length", str(len(body))),
        ]
        start_response(status_line, headers)
        return [body]

    def _send_text(
        start_response: Callable,
        status: int,
        text: str,
        content_type: str = "text/plain; version=0.0.4; charset=utf-8",
    ) -> list[bytes]:
        """Send a plain-text response (used for Prometheus metrics)."""
        body = text.encode("utf-8")
        status_line = f"{status} {_status_reason(status)}"
        headers = [
            ("Content-Type", content_type),
            ("Content-Length", str(len(body))),
        ]
        start_response(status_line, headers)
        return [body]

    def _status_reason(code: int) -> str:
        """Return the standard HTTP reason phrase for a status code."""
        reasons = {
            200: "OK",
            400: "Bad Request",
            401: "Unauthorized",
            403: "Forbidden",
            404: "Not Found",
            405: "Method Not Allowed",
            413: "Payload Too Large",
            500: "Internal Server Error",
            503: "Service Unavailable",
        }
        return reasons.get(code, "Unknown")

    def _get_qs(environ: dict) -> dict:
        """Parse the query string from the WSGI environ."""
        query_string = environ.get("QUERY_STRING", "")
        return parse_qs(query_string)

    def _handle_health(environ: dict, start_response: Callable) -> list[bytes]:
        """Active health check: tests the embedding provider with a throwaway vector.

        Returns 200 if everything is functional, 503 if the provider is unreachable.
        A load-balancer that only checks HTTP status will pull the instance if the
        provider key rotates or the endpoint goes down -- not just on process crash.
        """
        try:
            pipeline.embedder.embed_one("health check")
            return _send_json(start_response, 200, {"status": "ok", "chunks": len(pipeline.store)})
        except Exception as exc:  # noqa: BLE001
            log.warning("health check: embedding provider unavailable: %s", exc)
            return _send_json(start_response, 503, {"status": "degraded", "error": str(exc)})

    def _handle_stats(environ: dict, start_response: Callable) -> list[bytes]:
        """Return pipeline statistics as JSON."""
        return _send_json(start_response, 200, pipeline.stats)

    def _handle_metrics(environ: dict, start_response: Callable) -> list[bytes]:
        """Return Prometheus text-format metrics."""
        stats = pipeline.stats
        lines = [
            "# HELP ragpipe_chunks_total Number of chunks in the vector store.",
            "# TYPE ragpipe_chunks_total gauge",
            f"ragpipe_chunks_total {stats.get('chunks', 0)}",
            "",
            "# HELP ragpipe_vocabulary_size Number of unique terms in the BM25 index.",
            "# TYPE ragpipe_vocabulary_size gauge",
            f"ragpipe_vocabulary_size {stats.get('vocab', 0)}",
            "",
            "# HELP ragpipe_embedding_dimensions Dimensionality of embedding vectors.",
            "# TYPE ragpipe_embedding_dimensions gauge",
            f"ragpipe_embedding_dimensions {stats.get('dimensions', 0)}",
            "",
            "# HELP ragpipe_up Whether the pipeline is up (1) or down (0).",
            "# TYPE ragpipe_up gauge",
            "ragpipe_up 1",
            "",
        ]
        return _send_text(start_response, 200, "\n".join(lines))

    def _handle_query(environ: dict, start_response: Callable, body: bytes) -> list[bytes]:
        """Handle POST /query."""
        try:
            payload = _parse_json(body)
        except (ValueError, json.JSONDecodeError) as exc:
            return _send_json(start_response, 400, {"error": f"bad request: {exc}"})

        question = (payload.get("question") or "").strip()
        if not question:
            return _send_json(start_response, 400, {"error": "question is required"})
        if len(question) > _MAX_QUERY_CHARS:
            return _send_json(start_response, 400, {
                "error": f"question too long: {len(question)} chars (max {_MAX_QUERY_CHARS}). "
                         "Set RAG_MAX_QUERY_CHARS to raise the limit."
            })

        started = time.perf_counter()
        try:
            response = pipeline.query(
                question,
                top_k=payload.get("top_k"),
                filters=payload.get("filters"),
            )
            result = response.to_dict()
            result["server_ms"] = round((time.perf_counter() - started) * 1000, 2)
            return _send_json(start_response, 200, result)
        except Exception as exc:
            log.exception("query failed")
            return _send_json(start_response, 500, {"error": str(exc)})

    def _handle_ingest(environ: dict, start_response: Callable, body: bytes) -> list[bytes]:
        """Handle POST /ingest."""
        try:
            payload = _parse_json(body)
        except (ValueError, json.JSONDecodeError) as exc:
            return _send_json(start_response, 400, {"error": f"bad request: {exc}"})

        raw_path = payload.get("path")
        if not raw_path:
            return _send_json(start_response, 400, {"error": "path is required"})

        try:
            safe = _safe_ingest_path(raw_path)
        except (ValueError, FileNotFoundError) as exc:
            return _send_json(start_response, 403, {"error": str(exc)})

        try:
            stats = pipeline.index_path(safe, recursive=payload.get("recursive", True))
            pipeline.save()
            return _send_json(start_response, 200, stats)
        except Exception as exc:
            log.exception("ingest failed")
            return _send_json(start_response, 500, {"error": str(exc)})

    # -----------------------------------------------------------------------
    # The WSGI callable
    # -----------------------------------------------------------------------

    def application(environ: dict, start_response: Callable) -> Iterable[bytes]:
        """WSGI entry point. Dispatches on (method, path)."""
        method = environ.get("REQUEST_METHOD", "GET").upper()
        path = urlparse(environ.get("PATH_INFO", "/")).path
        qs = _get_qs(environ)

        # Health endpoint must be unauthenticated: Docker HEALTHCHECK, Kubernetes
        # probes, and load balancer health checks all send no Authorization header.
        # If /health required auth, setting RAG_API_KEY would make the container
        # permanently unhealthy.
        if path != "/health" and not _check_auth(environ, qs):
            return _send_json(start_response, 401, {"error": "unauthorized: provide Authorization: Bearer <key>"})

        # Read body for POST requests
        body = b""
        if method == "POST":
            try:
                body = _read_body(environ)
            except ValueError as exc:
                return _send_json(start_response, 413, {"error": str(exc)})

        # Route dispatch
        try:
            if method == "GET":
                if path == "/health":
                    return _handle_health(environ, start_response)
                elif path == "/stats":
                    return _handle_stats(environ, start_response)
                elif path == "/metrics":
                    return _handle_metrics(environ, start_response)
                else:
                    return _send_json(start_response, 404, {"error": "not found"})
            elif method == "POST":
                if path == "/query":
                    return _handle_query(environ, start_response, body)
                elif path == "/ingest":
                    return _handle_ingest(environ, start_response, body)
                else:
                    return _send_json(start_response, 404, {"error": "not found"})
            else:
                return _send_json(start_response, 405, {"error": "method not allowed"})
        except Exception as exc:
            log.exception("unhandled error: %s %s", method, path)
            return _send_json(start_response, 500, {"error": str(exc)})

    return application


# ---------------------------------------------------------------------------
# Module-level default application
# ---------------------------------------------------------------------------

def _build_default_app() -> Callable:
    """Build the default WSGI application with a fresh RAGPipeline.

    Auto-loads the persisted index on startup if meta.json exists.
    """
    pipeline = RAGPipeline()

    # Auto-load the persisted index before serving. Without this,
    # a container restart (OOM, deploy, node rebalancer) leaves the service
    # empty until someone manually runs `rag load`.
    persist_path = Path(pipeline.settings.store.persist_path)
    if (persist_path / "meta.json").exists():
        try:
            count = pipeline.load(persist_path)
            log.info("auto-loaded %d chunks from %s", count, persist_path)
        except Exception as exc:  # noqa: BLE001 - log and continue with empty index
            log.warning("failed to auto-load index from %s: %s", persist_path, exc)

    return create_app(pipeline)


# Module-level app: gunicorn ragpipe.wsgi:app
app = _build_default_app()
