"""Tests for the WSGI adapter (ragpipe.wsgi).

These tests call the WSGI application directly with mock environ/start_response
objects, verifying the full request/response cycle without needing a running
server. This is the standard way to test WSGI apps and would have caught the
ASGI/WSGI worker mismatch that caused a P0 regression.
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ragpipe.config import Settings, load_settings
from ragpipe.pipeline import RAGPipeline
from ragpipe.wsgi import create_app

DOCS = ROOT / "data" / "docs"


def _offline_settings(tmp: str) -> Settings:
    s = load_settings()
    s.data_dir = tmp
    s.store.persist_path = str(Path(tmp) / "index")
    s.observability.trace_sink = "none"
    s.embedding.provider = "hashing"
    s.generation.provider = "extractive"
    return s


def _make_environ(method: str, path: str, body: bytes = b"", headers: dict | None = None) -> dict:
    """Build a minimal WSGI environ dict for testing."""
    environ = {
        "REQUEST_METHOD": method,
        "PATH_INFO": path,
        "QUERY_STRING": "",
        "CONTENT_LENGTH": str(len(body)),
        "wsgi.input": io.BytesIO(body),
        "wsgi.errors": io.StringIO(),
        "wsgi.version": (1, 0),
        "wsgi.url_scheme": "http",
        "wsgi.multithread": True,
        "wsgi.multiprocess": False,
        "wsgi.run_once": False,
    }
    if headers:
        for key, value in headers.items():
            environ[f"HTTP_{key.upper().replace('-', '_')}"] = value
    return environ


def _call_app(app, method: str, path: str, body: bytes = b"", headers: dict | None = None) -> tuple[int, dict, bytes]:
    """Call a WSGI app and return (status, headers, body)."""
    environ = _make_environ(method, path, body, headers)
    status_holder: list[str] = []
    headers_holder: list[tuple[str, str]] = []

    def start_response(status: str, response_headers: list[tuple[str, str]]) -> None:
        status_holder.append(status)
        headers_holder.extend(response_headers)

    result = app(environ, start_response)
    body = b"".join(result)
    status = int(status_holder[0].split(" ")[0]) if status_holder else 0
    headers_dict = {k: v for k, v in headers_holder}
    return status, headers_dict, body


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


def test_wsgi_health_returns_200() -> None:
    """GET /health must return 200 with a JSON body containing 'status'."""
    with tempfile.TemporaryDirectory() as tmp:
        pipe = RAGPipeline(_offline_settings(tmp))
        pipe.index_path(DOCS)
        app = create_app(pipe)

        status, headers, body = _call_app(app, "GET", "/health")
        assert status == 200, f"expected 200, got {status}"
        assert headers.get("Content-Type") == "application/json"
        data = json.loads(body)
        assert data["status"] == "ok"
        assert data["chunks"] > 0


def test_wsgi_health_returns_503_when_embedder_fails() -> None:
    """GET /health must return 503 when the embedding provider is unreachable."""
    with tempfile.TemporaryDirectory() as tmp:
        pipe = RAGPipeline(_offline_settings(tmp))
        pipe.index_path(DOCS)
        app = create_app(pipe)

        # Break the embedder
        original_embed = pipe.embedder.embed_one
        pipe.embedder.embed_one = lambda text: (_ for _ in ()).throw(RuntimeError("provider down"))
        try:
            status, headers, body = _call_app(app, "GET", "/health")
            assert status == 503, f"expected 503, got {status}"
            data = json.loads(body)
            assert data["status"] == "degraded"
        finally:
            pipe.embedder.embed_one = original_embed


# ---------------------------------------------------------------------------
# Stats endpoint
# ---------------------------------------------------------------------------


def test_wsgi_stats_returns_200() -> None:
    """GET /stats must return 200 with pipeline statistics."""
    with tempfile.TemporaryDirectory() as tmp:
        pipe = RAGPipeline(_offline_settings(tmp))
        pipe.index_path(DOCS)
        app = create_app(pipe)

        status, headers, body = _call_app(app, "GET", "/stats")
        assert status == 200, f"expected 200, got {status}"
        data = json.loads(body)
        assert "chunks" in data
        assert "vocab" in data
        assert "dimensions" in data


# ---------------------------------------------------------------------------
# Metrics endpoint
# ---------------------------------------------------------------------------


def test_wsgi_metrics_returns_prometheus_format() -> None:
    """GET /metrics must return 200 with Prometheus text format."""
    with tempfile.TemporaryDirectory() as tmp:
        pipe = RAGPipeline(_offline_settings(tmp))
        pipe.index_path(DOCS)
        app = create_app(pipe)

        status, headers, body = _call_app(app, "GET", "/metrics")
        assert status == 200, f"expected 200, got {status}"
        assert "text/plain" in headers.get("Content-Type", "")
        text = body.decode("utf-8")
        assert "ragpipe_chunks_total" in text
        assert "ragpipe_vocabulary_size" in text
        assert "# HELP" in text
        assert "# TYPE" in text


# ---------------------------------------------------------------------------
# Query endpoint
# ---------------------------------------------------------------------------


def test_wsgi_query_returns_answer() -> None:
    """POST /query must return 200 with an answer and citations."""
    with tempfile.TemporaryDirectory() as tmp:
        pipe = RAGPipeline(_offline_settings(tmp))
        pipe.index_path(DOCS)
        app = create_app(pipe)

        body = json.dumps({"question": "what causes the export worker to crash?"}).encode()
        status, headers, resp_body = _call_app(app, "POST", "/query", body)
        assert status == 200, f"expected 200, got {status}"
        data = json.loads(resp_body)
        assert data["answer"]
        assert data["citations"]
        assert data["trace_id"]


def test_wsgi_query_rejects_empty_question() -> None:
    """POST /query with empty question must return 400."""
    with tempfile.TemporaryDirectory() as tmp:
        pipe = RAGPipeline(_offline_settings(tmp))
        pipe.index_path(DOCS)
        app = create_app(pipe)

        body = json.dumps({"question": ""}).encode()
        status, _, _ = _call_app(app, "POST", "/query", body)
        assert status == 400, f"expected 400, got {status}"


def test_wsgi_query_rejects_oversized_question() -> None:
    """POST /query with a question exceeding MAX_QUERY_CHARS must return 400."""
    with tempfile.TemporaryDirectory() as tmp:
        pipe = RAGPipeline(_offline_settings(tmp))
        pipe.index_path(DOCS)
        app = create_app(pipe)

        body = json.dumps({"question": "x" * 3000}).encode()
        status, _, _ = _call_app(app, "POST", "/query", body)
        assert status == 400, f"expected 400, got {status}"


def test_wsgi_query_rejects_bad_json() -> None:
    """POST /query with invalid JSON must return 400."""
    with tempfile.TemporaryDirectory() as tmp:
        pipe = RAGPipeline(_offline_settings(tmp))
        pipe.index_path(DOCS)
        app = create_app(pipe)

        status, _, _ = _call_app(app, "POST", "/query", b"not json")
        assert status == 400, f"expected 400, got {status}"


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def test_wsgi_unauthenticated_request_rejected_when_key_set() -> None:
    """When RAG_API_KEY is set, requests without the key must return 401."""
    import os
    import ragpipe.wsgi as wsgi_module

    original_key = wsgi_module._API_KEY
    wsgi_module._API_KEY = "test-secret-key"
    try:
        with tempfile.TemporaryDirectory() as tmp:
            pipe = RAGPipeline(_offline_settings(tmp))
            pipe.index_path(DOCS)
            app = create_app(pipe)

            # /health is exempt
            status, _, _ = _call_app(app, "GET", "/health")
            assert status == 200, f"/health should be exempt from auth, got {status}"

            # /stats requires auth
            status, _, _ = _call_app(app, "GET", "/stats")
            assert status == 401, f"expected 401, got {status}"

            # /query requires auth
            body = json.dumps({"question": "test"}).encode()
            status, _, _ = _call_app(app, "POST", "/query", body)
            assert status == 401, f"expected 401, got {status}"
    finally:
        wsgi_module._API_KEY = original_key


def test_wsgi_authenticated_request_accepted() -> None:
    """When RAG_API_KEY is set, requests with the correct key must succeed."""
    import os
    import ragpipe.wsgi as wsgi_module

    original_key = wsgi_module._API_KEY
    wsgi_module._API_KEY = "test-secret-key"
    try:
        with tempfile.TemporaryDirectory() as tmp:
            pipe = RAGPipeline(_offline_settings(tmp))
            pipe.index_path(DOCS)
            app = create_app(pipe)

            headers = {"Authorization": "Bearer test-secret-key"}
            status, _, body = _call_app(app, "GET", "/stats", headers=headers)
            assert status == 200, f"expected 200, got {status}"
    finally:
        wsgi_module._API_KEY = original_key


# ---------------------------------------------------------------------------
# 404 and 405
# ---------------------------------------------------------------------------


def test_wsgi_unknown_route_returns_404() -> None:
    """Unknown routes must return 404."""
    with tempfile.TemporaryDirectory() as tmp:
        pipe = RAGPipeline(_offline_settings(tmp))
        app = create_app(pipe)

        status, _, _ = _call_app(app, "GET", "/nonexistent")
        assert status == 404, f"expected 404, got {status}"


def test_wsgi_wrong_method_returns_405() -> None:
    """Unsupported HTTP methods must return 405."""
    with tempfile.TemporaryDirectory() as tmp:
        pipe = RAGPipeline(_offline_settings(tmp))
        app = create_app(pipe)

        status, _, _ = _call_app(app, "DELETE", "/query")
        assert status == 405, f"expected 405, got {status}"


# ---------------------------------------------------------------------------
# Auto-load
# ---------------------------------------------------------------------------


def test_wsgi_auto_loads_persisted_index() -> None:
    """The WSGI app must auto-load the persisted index on startup.

    The auto-load logic lives in _build_default_app() (module-level app).
    create_app() is a thin wrapper that does not auto-load — it just wraps
    whatever pipeline it is given. This test verifies the full flow: save an
    index, create a new pipeline, load it, and verify the app serves it.
    """
    with tempfile.TemporaryDirectory() as tmp:
        # First: create and save an index
        pipe = RAGPipeline(_offline_settings(tmp))
        pipe.index_path(DOCS)
        pipe.save()
        chunk_count = len(pipe.store)

        # Second: create a new pipeline (simulating restart) and load manually
        pipe2 = RAGPipeline(_offline_settings(tmp))
        assert len(pipe2.store) == 0

        # Load the persisted index (this is what _build_default_app does)
        loaded = pipe2.load(Path(tmp) / "index")
        assert loaded == chunk_count
        assert len(pipe2.store) == chunk_count

        # Now create the app and verify it serves the loaded index
        app = create_app(pipe2)
        status, _, body = _call_app(app, "GET", "/health")
        assert status == 200
        data = json.loads(body)
        assert data["chunks"] == chunk_count


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _run_all() -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failures = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS {name}")
        except Exception as exc:  # noqa: BLE001
            failures.append((name, exc))
            import traceback
            print(f"  FAIL {name}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
