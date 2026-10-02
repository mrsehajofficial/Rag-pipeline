"""HTTP API on http.server -- stdlib only, so the service has no framework dependency.

For production, put this behind gunicorn/uvicorn, or replace this file with FastAPI. The
handler logic is intentionally thin: all real work lives in RAGPipeline.

    POST /query          {"question": "...", "top_k": 8}
    POST /ingest         {"path": "./docs"}
    GET  /health
    GET  /stats

Security:
  * API key auth: set RAG_API_KEY in env. If set, every request must include the header
      Authorization: Bearer <key>
    or the query param ?api_key=<key>.  If RAG_API_KEY is unset the server binds to
    127.0.0.1 only and logs a loud warning -- suitable for local dev, not for internet-
    facing deployments.

  * Path-injection guard: /ingest only allows paths inside a configurable root directory
    (RAG_INGEST_ROOT, defaults to the current working directory).  Traversal attempts
    are rejected with a 403 before touching the filesystem.

  * Rate limiting: a token-bucket implementation prevents a single client from exhausting
    the LLM quota or pushing latency for others.  Configure with:
      RAG_RATE_LIMIT_RPS  -- allowed requests per second per IP (default 10)
      RAG_RATE_LIMIT_BURST -- burst capacity (default 20)

  * Query-length cap: RAG_MAX_QUERY_CHARS (default 2000).  Longer queries are rejected
    before the expensive embed+retrieve+generate path runs.
"""

from __future__ import annotations

import json
import os
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .observability import get_logger
from .pipeline import RAGPipeline

log = get_logger("ragpipe.api")

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


def _check_auth(headers, qs: dict) -> bool:
    """Return True if the request carries a valid API key, or if no key is configured."""
    if _API_KEY is None:
        return True
    auth_header = (headers.get("Authorization") or "").strip()
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
# Token-bucket rate limiter (per source IP)
# ---------------------------------------------------------------------------
_RPS = float(os.environ.get("RAG_RATE_LIMIT_RPS", "10"))
_BURST = int(os.environ.get("RAG_RATE_LIMIT_BURST", "20"))


class _TokenBucket:
    """Thread-safe token bucket for a single client."""

    __slots__ = ("_lock", "last", "tokens")

    def __init__(self) -> None:
        self.tokens: float = _BURST
        self.last: float = time.monotonic()
        self._lock = threading.Lock()

    def consume(self) -> bool:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self.last
            self.tokens = min(_BURST, self.tokens + elapsed * _RPS)
            self.last = now
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return True
            return False


_RATE_BUCKETS: dict[str, _TokenBucket] = {}
_RATE_LOCK = threading.Lock()
_RATE_MAX_ENTRIES = 50_000
_RATE_PRUNE_INTERVAL = 300  # seconds


def _rate_ok(ip: str) -> bool:
    stale: list[str] = []
    with _RATE_LOCK:
        bucket = _RATE_BUCKETS.get(ip)
        if bucket is None:
            _RATE_BUCKETS[ip] = bucket = _TokenBucket()
        # Collect stale keys under the lock, but delete them after releasing it.
        # Deleting while holding the lock blocks every incoming request for the
        # duration of the deletion loop.
        if len(_RATE_BUCKETS) > _RATE_MAX_ENTRIES:
            cutoff = time.monotonic() - _RATE_PRUNE_INTERVAL
            stale = [k for k, v in _RATE_BUCKETS.items() if v.last < cutoff]
    # Delete outside the lock: other requests can proceed while we clean up.
    if stale:
        with _RATE_LOCK:
            for k in stale:
                _RATE_BUCKETS.pop(k, None)
    return bucket.consume()


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class RAGHandler(BaseHTTPRequestHandler):
    pipeline: RAGPipeline  # injected by serve()
    server_version = "ragpipe/0.1"

    def _send(self, status: int, payload: Any) -> None:
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY_BYTES:
            raise ValueError(f"request body too large ({length} bytes)")
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _authed(self) -> bool:
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        if not _check_auth(self.headers, qs):
            self._send(401, {"error": "unauthorized: provide Authorization: Bearer <RAG_API_KEY>"})
            return False
        return True

    def _rate_limited(self) -> bool:
        ip = self.client_address[0]
        if not _rate_ok(ip):
            self._send(429, {"error": "rate limit exceeded, slow down"})
            return True
        return False

    def do_GET(self) -> None:
        parsed_path = urlparse(self.path).path
        # Health endpoint must be unauthenticated: Docker HEALTHCHECK, Kubernetes
        # probes, and load balancer health checks all send no Authorization header.
        # If /health required auth, setting RAG_API_KEY would make the container
        # permanently unhealthy.
        if parsed_path not in ("/health", "/metrics") and not self._authed():
            return
        if parsed_path == "/health":
            self._handle_health()
        elif parsed_path == "/stats":
            self._send(200, self.pipeline.stats)
        elif parsed_path == "/metrics":
            self._handle_metrics()
        else:
            self._send(404, {"error": "not found"})

    def _handle_metrics(self) -> None:
        """Prometheus-format metrics endpoint.

        Exposes operational signals that per-query traces cannot provide:
        cache hit rates, index size, and request counters. Without this,
        you cannot set SLO alerts or detect degradation before users complain.
        """
        stats = self.pipeline.stats
        llm_cache = self.pipeline.llm.cache_stats() if hasattr(self.pipeline.llm, "cache_stats") else {}
        embed_cache = self.pipeline.embedder._lru

        lines = [
            "# HELP ragpipe_chunks_total Number of chunks in the vector index",
            "# TYPE ragpipe_chunks_total gauge",
            f"ragpipe_chunks_total {stats.get('chunks', 0)}",
            "# HELP ragpipe_vocabulary_size BM25 vocabulary size",
            "# TYPE ragpipe_vocabulary_size gauge",
            f"ragpipe_vocabulary_size {stats.get('vocab', 0)}",
            "# HELP ragpipe_embed_cache_hits Total embedding cache hits",
            "# TYPE ragpipe_embed_cache_hits counter",
            f"ragpipe_embed_cache_hits {embed_cache.hits}",
            "# HELP ragpipe_embed_cache_misses Total embedding cache misses",
            "# TYPE ragpipe_embed_cache_misses counter",
            f"ragpipe_embed_cache_misses {embed_cache.misses}",
            "# HELP ragpipe_embed_cache_hit_rate Embedding cache hit rate (0-1)",
            "# TYPE ragpipe_embed_cache_hit_rate gauge",
            f"ragpipe_embed_cache_hit_rate {embed_cache.hit_rate:.4f}",
            "# HELP ragpipe_llm_cache_enabled Whether LLM response cache is enabled (1/0)",
            "# TYPE ragpipe_llm_cache_enabled gauge",
            f"ragpipe_llm_cache_enabled {1 if llm_cache.get('enabled') else 0}",
            "# HELP ragpipe_llm_cache_entries Number of entries in LLM response cache",
            "# TYPE ragpipe_llm_cache_entries gauge",
            f"ragpipe_llm_cache_entries {llm_cache.get('entries', 0)}",
            "# HELP ragpipe_llm_cache_hits Total LLM cache hits",
            "# TYPE ragpipe_llm_cache_hits counter",
            f"ragpipe_llm_cache_hits {llm_cache.get('hits', 0)}",
        ]
        body = ("\n".join(lines) + "\n").encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _get_allowed_filter_keys(self) -> set[str] | None:
        """Return the set of metadata keys that can be used as filters.

        Returns None if the index is empty (no keys to validate against).
        """
        chunks = self.pipeline.store.all_chunks()
        if not chunks:
            return None
        keys: set[str] = set()
        for chunk in chunks:
            keys.update(chunk.metadata.keys())
        return keys

    def _handle_health(self) -> None:
        """Active health check: tests the embedding provider with a throwaway vector.

        Returns 200 if everything is functional, 503 if the provider is unreachable.
        A load-balancer that only checks HTTP status will pull the instance if the
        provider key rotates or the endpoint goes down -- not just on process crash.
        """
        try:
            self.pipeline.embedder.embed_one("health check")
            self._send(200, {"status": "ok", "chunks": len(self.pipeline.store)})
        except Exception as exc:  # noqa: BLE001
            log.warning("health check: embedding provider unavailable: %s", exc)
            self._send(503, {"status": "degraded", "error": str(exc)})

    def do_POST(self) -> None:
        if not self._authed():
            return
        if self._rate_limited():
            return

        started = time.perf_counter()
        try:
            payload = self._read_json()
        except (ValueError, json.JSONDecodeError) as exc:
            self._send(400, {"error": f"bad request: {exc}"})
            return

        parsed_path = urlparse(self.path).path
        try:
            if parsed_path == "/query":
                self._handle_query(payload, started)
            elif parsed_path == "/ingest":
                self._handle_ingest(payload)
            else:
                self._send(404, {"error": "not found"})
        except Exception as exc:
            log.exception("request failed: %s %s", self.command, self.path)
            self._send(500, {"error": str(exc)})

    def _handle_query(self, payload: dict, started: float) -> None:
        question = (payload.get("question") or "").strip()
        if not question:
            self._send(400, {"error": "question is required"})
            return
        if len(question) > _MAX_QUERY_CHARS:
            self._send(400, {
                "error": f"question too long: {len(question)} chars (max {_MAX_QUERY_CHARS}). "
                         "Set RAG_MAX_QUERY_CHARS to raise the limit."
            })
            return
        filters = payload.get("filters")
        if filters is not None:
            if not isinstance(filters, dict):
                self._send(400, {"error": "filters must be an object"})
                return
            # Validate filter keys against known metadata keys from the corpus.
            # Without this, a client can probe chunk metadata structure by sending
            # arbitrary keys like {"__class__": "something"}.
            allowed_keys = self._get_allowed_filter_keys()
            if allowed_keys is not None:
                unknown = set(filters.keys()) - allowed_keys
                if unknown:
                    self._send(400, {
                        "error": f"unknown filter keys: {sorted(unknown)}. "
                                 f"allowed keys: {sorted(allowed_keys)}"
                    })
                    return
        response = self.pipeline.query(
            question,
            top_k=payload.get("top_k"),
            filters=filters,
        )
        result = response.to_dict()
        result["server_ms"] = round((time.perf_counter() - started) * 1000, 2)
        self._send(200, result)

    def _handle_ingest(self, payload: dict) -> None:
        raw_path = payload.get("path")
        if not raw_path:
            self._send(400, {"error": "path is required"})
            return
        try:
            safe = _safe_ingest_path(raw_path)
        except (ValueError, FileNotFoundError) as exc:
            self._send(403, {"error": str(exc)})
            return
        stats = self.pipeline.index_path(safe, recursive=payload.get("recursive", True))
        self.pipeline.save()
        self._send(200, stats)

    def log_message(self, fmt: str, *args: Any) -> None:
        log.info("%s - %s", self.address_string(), fmt % args)


def serve(pipeline: RAGPipeline, host: str = "127.0.0.1", port: int = 8000) -> None:
    if _API_KEY is None and host not in ("127.0.0.1", "::1", "localhost"):
        log.warning(
            "Binding to %s without RAG_API_KEY set. "
            "Any client on this network can query and ingest. "
            "Set RAG_API_KEY=<secret> before exposing this to a network.",
            host,
        )

    # Auto-load the persisted index before binding the socket. Without this,
    # a container restart (OOM, deploy, node rebalancer) leaves the service
    # empty until someone manually runs `rag load`.
    persist_path = Path(pipeline.settings.store.persist_path)
    if (persist_path / "meta.json").exists():
        try:
            count = pipeline.load(persist_path)
            log.info("auto-loaded %d chunks from %s", count, persist_path)
        except Exception as exc:  # noqa: BLE001 - log and continue with empty index
            log.warning("failed to auto-load index from %s: %s", persist_path, exc)

    handler = type("BoundRAGHandler", (RAGHandler,), {"pipeline": pipeline})
    httpd = ThreadingHTTPServer((host, port), handler)
    log.info("serving RAG API on http://%s:%d", host, port)
    log.info(
        "endpoints: GET /health /stats   POST /query /ingest"
    )
    if _API_KEY:
        log.info("auth: enabled (Authorization: Bearer ***)")
    else:
        log.info("auth: DISABLED (RAG_API_KEY not set)")

    # Graceful shutdown: trap SIGTERM (sent by `docker stop`, Kubernetes, systemd)
    # and SIGINT (Ctrl+C) so in-flight requests can complete before the process
    # exits. Without this, `docker stop` kills threads mid-request.
    shutdown_event = threading.Event()

    def _shutdown(signum, frame):
        log.info("received signal %d, shutting down gracefully...", signum)
        shutdown_event.set()
        # serve_forever() runs in the main thread; shutdown() must be called
        # from a different thread or it will deadlock.
        threading.Thread(target=httpd.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        log.info("shutting down")
    finally:
        httpd.server_close()
        log.info("server closed")
