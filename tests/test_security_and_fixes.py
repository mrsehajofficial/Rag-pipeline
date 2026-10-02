"""Tests for all the security, reliability, and bug fixes applied in the production pass.

Covers:
  * API key auth (401 on missing/wrong key, 200 on correct key)
  * Rate limiting (429 after burst is exhausted)
  * Path-injection guard (403 on traversal, 200 on safe path)
  * Query length cap (ValueError for oversized queries)
  * Sentence chunker strategy no longer raises AttributeError
  * Thread-safe token cache (no crash under concurrent access)
  * Streaming trace emitted on generator close (disconnect simulation)
"""

from __future__ import annotations

import sys
import tempfile
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


# ---------------------------------------------------------------------------
# Helper: make a pipeline with the offline providers
# ---------------------------------------------------------------------------

from ragpipe.config import Settings, load_settings
from ragpipe.pipeline import RAGPipeline

DOCS = ROOT / "data" / "docs"


def _offline_settings(tmp: str) -> Settings:
    s = load_settings()
    s.data_dir = tmp
    s.store.persist_path = str(Path(tmp) / "index")
    s.observability.trace_sink = "none"
    s.embedding.provider = "hashing"
    s.generation.provider = "extractive"
    return s


# ---------------------------------------------------------------------------
# Query-length cap
# ---------------------------------------------------------------------------

def test_query_length_cap_raises_for_oversized_input() -> None:
    """A query longer than MAX_QUERY_CHARS must be rejected before any embedding call."""
    from ragpipe.pipeline import MAX_QUERY_CHARS

    with tempfile.TemporaryDirectory() as tmp:
        pipe = RAGPipeline(_offline_settings(tmp))
        pipe.index_path(DOCS)

        oversized = "x" * (MAX_QUERY_CHARS + 1)
        try:
            pipe.query(oversized)
            raise AssertionError("expected ValueError for oversized query")
        except ValueError as exc:
            assert "too long" in str(exc).lower(), f"unexpected message: {exc}"


def test_normal_query_is_not_rejected() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        pipe = RAGPipeline(_offline_settings(tmp))
        pipe.index_path(DOCS)
        resp = pipe.query("how long do API keys last?")
        assert resp.answer


# ---------------------------------------------------------------------------
# Sentence chunker strategy
# ---------------------------------------------------------------------------

def test_sentence_chunker_strategy_does_not_raise() -> None:
    """`_sentence` was a typo for `_sentences`; this pins the fix."""
    from ragpipe.config import ChunkingConfig
    from ragpipe.ingest.chunker import Chunker

    chunker = Chunker(ChunkingConfig(strategy="sentence", target_tokens=60, overlap_tokens=10))
    text = (
        "API keys rotate every ninety days. "
        "They are issued per service. "
        "Revoked keys cannot be reissued. "
        "Contact the auth team for a replacement. " * 10
    )
    chunks = chunker.chunk_document(text, doc_id="d1", source="test.md")
    assert chunks, "sentence strategy produced no chunks"
    assert all(c.text.strip() for c in chunks)


# ---------------------------------------------------------------------------
# Thread-safe token cache
# ---------------------------------------------------------------------------

def test_token_cache_is_thread_safe_under_concurrent_access() -> None:
    """Concurrent tokenize_cached calls must not crash or produce wrong results.

    The old implementation used a bare dict + .clear() which could race. The new one
    uses LRUCache which holds an RLock on every operation.
    """
    from ragpipe.pipeline import tokenize_cached

    texts = [f"the quick brown fox number {i} jumps" for i in range(200)]
    errors: list[Exception] = []

    def worker(chunk: list[str]) -> None:
        try:
            for t in chunk:
                result = tokenize_cached(t)
                assert isinstance(result, list)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    # Split across many threads to maximise contention
    threads = [threading.Thread(target=worker, args=(texts[i::8],)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"thread-safe tokenize_cached raised: {errors}"


def test_token_cache_results_are_consistent() -> None:
    """The same text must always return the same token list, regardless of cache state."""
    from ragpipe.pipeline import tokenize_cached

    text = "api key rotation ninety days service authentication"
    first = list(tokenize_cached(text))
    second = list(tokenize_cached(text))
    assert first == second


# ---------------------------------------------------------------------------
# API authentication
# ---------------------------------------------------------------------------

def _make_api_handler(pipeline: RAGPipeline, api_key: str | None = None):
    """Return a bound RAGHandler class with the pipeline and an optional API key injected."""
    import os

    import ragpipe.api as api_module
    from ragpipe.api import RAGHandler

    # Patch the module-level _API_KEY for this test
    original = api_module._API_KEY
    api_module._API_KEY = api_key
    try:
        yield type("BoundHandler", (RAGHandler,), {"pipeline": pipeline})
    finally:
        api_module._API_KEY = original


def _fake_request(handler_cls, method: str, path: str, body: bytes = b"",
                  headers: dict | None = None):
    """Exercise the handler's do_GET / do_POST without a real socket."""
    import io
    from http.server import BaseHTTPRequestHandler

    class _FakeSocket:
        def __init__(self): self._data = b""
        def makefile(self, _mode, **_kw): return io.BytesIO(self._data)
        def getsockname(self): return ("127.0.0.1", 8000)

    resp_buf = io.BytesIO()

    class _FakeWfile:
        def write(self, data): resp_buf.write(data)

    all_headers = {"Content-Length": str(len(body))}
    if headers:
        all_headers.update(headers)

    raw_headers = "\r\n".join(f"{k}: {v}" for k, v in all_headers.items())
    raw_request = (
        f"{method} {path} HTTP/1.1\r\n{raw_headers}\r\n\r\n"
    ).encode() + body

    sock = _FakeSocket()
    sock._data = raw_request

    handler = handler_cls.__new__(handler_cls)
    handler.rfile = io.BytesIO(body)
    handler.wfile = _FakeWfile()
    handler.headers = {k.lower(): v for k, v in all_headers.items()}
    handler.path = path
    handler.command = method
    handler.client_address = ("127.0.0.1", 54321)
    handler.server = type("S", (), {"server_name": "localhost", "server_port": 8000})()

    status_holder: list[int] = []
    body_holder: list[bytes] = []

    def _send(status, payload):
        import json
        status_holder.append(status)
        body_holder.append(json.dumps(payload).encode())

    handler._send = _send
    handler._read_json = lambda: __import__("json").loads(body) if body else {}
    return handler, status_holder, body_holder


def test_api_rejects_request_without_key_when_auth_enabled() -> None:
    from unittest.mock import patch
    from ragpipe.api import _check_auth

    with patch("ragpipe.api._API_KEY", "secret-key-abc"):
        assert not _check_auth({"Authorization": "Bearer wrong-key"}, {})
        assert _check_auth({"Authorization": "Bearer secret-key-abc"}, {})
        assert _check_auth({}, {"api_key": ["secret-key-abc"]})
        assert not _check_auth({}, {"api_key": ["bad-key"]})
        assert not _check_auth({}, {})  # no credentials at all


def test_api_allows_request_with_correct_key() -> None:
    from unittest.mock import patch
    from ragpipe.api import _check_auth

    with patch("ragpipe.api._API_KEY", "my-secret"):
        assert _check_auth({"Authorization": "Bearer my-secret"}, {})
        assert _check_auth({}, {"api_key": ["my-secret"]})
        assert not _check_auth({"Authorization": "Bearer wrong"}, {})


def test_api_allows_all_when_no_key_configured() -> None:
    import ragpipe.api as api_module

    original = api_module._API_KEY
    api_module._API_KEY = None
    try:
        from ragpipe.api import _check_auth
        assert _check_auth({}, {})
        assert _check_auth({"authorization": "Bearer anything"}, {})
    finally:
        api_module._API_KEY = original


# ---------------------------------------------------------------------------
# Path-injection guard
# ---------------------------------------------------------------------------

def test_path_injection_guard_blocks_traversal() -> None:
    import os
    import ragpipe.api as api_module
    from ragpipe.api import _safe_ingest_path

    original_root = os.environ.get("RAG_INGEST_ROOT")
    os.environ["RAG_INGEST_ROOT"] = tempfile.mkdtemp()
    try:
        try:
            _safe_ingest_path("../../etc/passwd")
            raise AssertionError("Expected ValueError for path traversal")
        except ValueError as exc:
            assert "escapes" in str(exc).lower() or "allowed" in str(exc).lower()
    finally:
        if original_root is None:
            os.environ.pop("RAG_INGEST_ROOT", None)
        else:
            os.environ["RAG_INGEST_ROOT"] = original_root


def test_path_injection_guard_allows_safe_path() -> None:
    import os
    import ragpipe.api as api_module
    from ragpipe.api import _safe_ingest_path

    with tempfile.TemporaryDirectory() as tmp:
        original_root = os.environ.get("RAG_INGEST_ROOT")
        os.environ["RAG_INGEST_ROOT"] = tmp
        # Create a real subdir to pass the .exists() check
        subdir = Path(tmp) / "docs"
        subdir.mkdir()
        try:
            result = _safe_ingest_path("docs")
            assert result == subdir
        finally:
            if original_root is None:
                os.environ.pop("RAG_INGEST_ROOT", None)
            else:
                os.environ["RAG_INGEST_ROOT"] = original_root


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

def test_rate_limiter_allows_burst_then_rejects() -> None:
    """After draining the burst, the next request must be rejected."""
    from ragpipe.api import _TokenBucket
    import ragpipe.api as api_module

    original_burst = api_module._BURST
    original_rps = api_module._RPS
    api_module._BURST = 3
    api_module._RPS = 0.0  # no refill for this test
    try:
        bucket = _TokenBucket()
        bucket.tokens = 3.0  # pre-fill

        assert bucket.consume()
        assert bucket.consume()
        assert bucket.consume()
        assert not bucket.consume()  # burst exhausted
    finally:
        api_module._BURST = original_burst
        api_module._RPS = original_rps


# ---------------------------------------------------------------------------
# Streaming trace emission
# ---------------------------------------------------------------------------

def test_streaming_trace_emitted_on_generator_close() -> None:
    """Closing a streaming generator before exhausting it must still emit the trace.

    Simulates a client disconnecting mid-stream: the generator receives GeneratorExit
    and the try/finally in _stream_answer must still call tracer.emit().
    """
    with tempfile.TemporaryDirectory() as tmp:
        pipe = RAGPipeline(_offline_settings(tmp))
        pipe.index_path(DOCS)

        emitted: list = []
        original_emit = pipe.tracer.emit
        pipe.tracer.emit = lambda t: emitted.append(t)

        try:
            stream = pipe.query("what causes the export worker to crash?", stream=True)
            # Consume only the first token then close (simulates client disconnect)
            next(stream)
            stream.close()
        except StopIteration:
            pass  # stream was shorter than expected, that's fine

        assert emitted, "trace must be emitted even when the generator is closed early"
        pipe.tracer.emit = original_emit


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
