"""Smoke test: boot gunicorn and verify the production server actually works.

The WSGI unit tests (test_wsgi.py) call the app directly, which validates the
app contract but would NOT catch a wrong worker_class in the gunicorn config
(that's config-level, not app-level). This test boots a real gunicorn server
and curls /health to verify the full production path.

This test is skipped if gunicorn is not installed (e.g. in a minimal dev env).
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DOCS = ROOT / "data" / "docs"


def _free_port() -> int:
    """Find a free TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _wait_for_port(port: int, timeout: float = 15.0) -> bool:
    """Wait until a TCP port accepts connections."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return True
        except (ConnectionRefusedError, OSError):
            time.sleep(0.2)
    return False


def _http_get(port: int, path: str) -> tuple[int, dict]:
    """Make a simple HTTP GET request and return (status, body_dict)."""
    import urllib.request
    import urllib.error

    url = f"http://127.0.0.1:{port}{path}"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_gunicorn_health_endpoint() -> None:
    """Boot gunicorn with the production CMD and verify /health returns 200.

    This is the test that would have caught the ASGI/WSGI worker mismatch.
    """
    try:
        import gunicorn  # noqa: F401
    except ImportError:
        print("  SKIP test_gunicorn_health_endpoint (gunicorn not installed)")
        return

    port = _free_port()
    with tempfile.TemporaryDirectory() as tmp:
        env = os.environ.copy()
        env["RAG_DATA_DIR"] = tmp
        env["RAG_LOG_LEVEL"] = "WARNING"
        env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")

        # Build an index first so the server has data to serve
        from ragpipe.config import load_settings
        from ragpipe.pipeline import RAGPipeline

        settings = load_settings()
        settings.data_dir = tmp
        settings.store.persist_path = str(Path(tmp) / "index")
        settings.observability.trace_sink = "none"
        settings.embedding.provider = "hashing"
        settings.generation.provider = "extractive"
        pipe = RAGPipeline(settings)
        pipe.index_path(DOCS)
        pipe.save()

        # Boot gunicorn with the exact production CMD (minus the bind address)
        cmd = [
            sys.executable, "-m", "gunicorn",
            "-w", "2",
            "--bind", f"127.0.0.1:{port}",
            "--timeout", "30",
            "ragpipe.wsgi:app",
        ]
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(ROOT),
        )
        try:
            assert _wait_for_port(port), f"gunicorn did not start within 15s on port {port}"

            status, body = _http_get(port, "/health")
            assert status == 200, f"expected 200, got {status}: {body}"
            assert body["status"] == "ok"
            assert body["chunks"] > 0
        finally:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


def test_gunicorn_query_endpoint() -> None:
    """Boot gunicorn and verify POST /query returns an answer."""
    try:
        import gunicorn  # noqa: F401
    except ImportError:
        print("  SKIP test_gunicorn_query_endpoint (gunicorn not installed)")
        return

    port = _free_port()
    with tempfile.TemporaryDirectory() as tmp:
        env = os.environ.copy()
        env["RAG_DATA_DIR"] = tmp
        env["RAG_LOG_LEVEL"] = "WARNING"
        env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")

        from ragpipe.config import load_settings
        from ragpipe.pipeline import RAGPipeline

        settings = load_settings()
        settings.data_dir = tmp
        settings.store.persist_path = str(Path(tmp) / "index")
        settings.observability.trace_sink = "none"
        settings.embedding.provider = "hashing"
        settings.generation.provider = "extractive"
        pipe = RAGPipeline(settings)
        pipe.index_path(DOCS)
        pipe.save()

        cmd = [
            sys.executable, "-m", "gunicorn",
            "-w", "2",
            "--bind", f"127.0.0.1:{port}",
            "--timeout", "30",
            "ragpipe.wsgi:app",
        ]
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(ROOT),
        )
        try:
            assert _wait_for_port(port), f"gunicorn did not start within 15s on port {port}"

            import urllib.request
            import urllib.error

            url = f"http://127.0.0.1:{port}/query"
            data = json.dumps({"question": "what causes the export worker to crash?"}).encode()
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=10) as resp:
                    body = json.loads(resp.read())
                    assert resp.status == 200
                    assert body["answer"]
                    assert body["citations"]
            except urllib.error.HTTPError as e:
                raise AssertionError(f"POST /query returned {e.code}: {e.read().decode()}")
        finally:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


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
