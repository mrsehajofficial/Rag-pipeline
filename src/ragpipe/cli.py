"""Command-line interface. argparse only -- no click dependency.

    rag ingest ./docs            # index a directory
    rag ask "how do I deploy?"   # one-off question
    rag serve                    # HTTP API
    rag eval cases.jsonl         # measure retrieval quality
    rag stats                    # inspect the index
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_settings
from .evaluation import PipelineRetriever, RetrievalEvaluator, load_cases
from .pipeline import RAGPipeline


def _build(args: argparse.Namespace) -> RAGPipeline:
    settings = load_settings()
    if getattr(args, "index_path", None):
        settings.store.persist_path = args.index_path
    pipeline = RAGPipeline(settings)
    if getattr(args, "no_load", False):
        return pipeline
    target = Path(pipeline.settings.store.persist_path)
    if (target / "meta.json").exists():
        try:
            count = pipeline.load(target)
            print(f"loaded {count} chunks from {target}", file=sys.stderr)
        except ValueError as exc:
            # Stale index (e.g. dim mismatch from a previous embedding model).
            # Log a warning and continue with an empty index rather than crashing.
            print(f"warning: could not load index from {target}: {exc}", file=sys.stderr)
            print("  the index will be empty. Re-run `rag ingest` to rebuild it.", file=sys.stderr)
    return pipeline


def cmd_ingest(args: argparse.Namespace) -> int:
    pipeline = _build(args)
    result = pipeline.index_path(args.path, recursive=not args.no_recursive)
    pipeline.save()
    print(json.dumps({**result, "stats": pipeline.stats}, indent=2))
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    pipeline = _build(args)
    response = pipeline.query(args.question, top_k=args.top_k)
    if args.json:
        print(json.dumps(response.to_dict(), indent=2))
        return 0

    print(f"\nQ: {response.query}\n")
    print(f"A: {response.answer}\n")
    if response.citations:
        print("Sources:")
        for c in response.citations:
            via = "+".join(c.retrieved_by)
            snippet = c.text[:120].replace("\n", " ")
            print(f"  [{c.index}] {c.source} (score {c.score:.3f}, via {via})")
            print(f"      {snippet}...")
    print(f"\n[{response.total_ms:.1f}ms total | trace {response.trace_id}]", file=sys.stderr)
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import os

    if getattr(args, "ingest_root", None):
        os.environ["RAG_INGEST_ROOT"] = str(args.ingest_root)

    # Prefer the production WSGI server (gunicorn) when available.
    # Falls back to the stdlib dev server for zero-dependency local use.
    try:
        import gunicorn  # noqa: F401
        from .wsgi import create_app

        pipeline = _build(args)
        app = create_app(pipeline)
        from gunicorn.app.base import BaseApplication

        class RagpipeApplication(BaseApplication):
            def __init__(self, app, options=None):
                self.options = options or {}
                self.application = app
                super().__init__()

            def load_config(self):
                for key, value in self.options.items():
                    if key in self.cfg.settings and value is not None:
                        self.cfg.set(key.lower(), value)

            def load(self):
                return self.application

        options = {
            "bind": f"{args.host}:{args.port}",
            "workers": int(os.environ.get("RAG_WORKERS", "4")),
            # Default sync worker is correct for WSGI. Do NOT set
            # "worker_class": "uvicorn.workers.UvicornWorker" — that's ASGI and
            # will call the WSGI app with (scope, receive, send), causing 500s.
            "timeout": 120,
            "graceful_timeout": 30,
        }
        RagpipeApplication(app, options).run()
        return 0
    except ImportError:
        # gunicorn not installed -- fall back to stdlib server
        from .api import serve
        serve(_build(args), host=args.host, port=args.port)
        return 0


def cmd_eval(args: argparse.Namespace) -> int:
    pipeline = _build(args)
    cases = load_cases(args.cases)
    evaluator = RetrievalEvaluator(PipelineRetriever(pipeline, top_k=args.top_k))
    result = evaluator.run(cases, top_k=args.top_k, name=Path(args.cases).stem)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(evaluator.report(result))
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    pipeline = _build(args)
    print(json.dumps(pipeline.stats, indent=2))
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    """Check that the configured providers are actually usable, without spending a
    real query. Almost every 'my base URL doesn't work' is a model-name or a missing
    /v1 suffix, and both show up here."""
    from .openai_client import DEFAULT_BASE_URL, resolve_credentials

    print("ragpipe doctor\n")

    try:
        import openai

        print(f"  openai sdk    {openai.__version__}")
    except ImportError:
        print("  openai sdk    NOT INSTALLED   -> pip install openai")
        openai = None

    try:
        import numpy

        print(f"  numpy         {numpy.__version__}  (dense search: BLAS fast path)")
    except ImportError:
        print("  numpy         not installed   (dense search: pure python)")

    settings = load_settings()
    print(f"\n  embed         provider={settings.embedding.provider} model={settings.embedding.model} dim={settings.embedding.dimensions}")
    if settings.embedding.provider == "local":
        print(f"                 local model={settings.embedding.local_model}")
    print(f"  llm           provider={settings.generation.provider} model={settings.generation.model}")

    # `local` embeddings need no credentials at all -- only the LLM might.
    llm_needs_key = settings.generation.provider == "openai"
    if not llm_needs_key:
        print("\n  generation is offline (extractive); no API key needed")
        if settings.embedding.provider != "openai":
            print("  embeddings need no API key either -- ready to index")
            return 0

    try:
        api_key, base_url = resolve_credentials(
            settings.generation.api_key_env, settings.generation.base_url_env
        )
    except RuntimeError as exc:
        print(f"\n  CREDENTIALS   {exc}")
        return 1

    shown = f"{api_key[:7]}...{api_key[-4:]}" if len(api_key) > 12 else "***"
    print(f"\n  api key       {shown}  (len {len(api_key)})")
    print(f"  base url      {base_url or DEFAULT_BASE_URL + '  (default)'}")

    if openai is None:
        return 1

    from .openai_client import probe_models

    try:
        models = probe_models(
            settings.generation.api_key_env, settings.generation.base_url_env, limit=100
        )
    except Exception as exc:  # noqa: BLE001 - report, don't crash a diagnostic command
        print(f"  connectivity  FAILED: {type(exc).__name__}: {exc}")
        return 1

    print(f"  connectivity  OK ({len(models)} models advertised)")

    # Only check models this endpoint would actually have to serve.
    checks = []
    if settings.embedding.provider == "openai":
        checks.append((settings.embedding.model, "embed"))
    if settings.generation.provider == "openai":
        checks.append((settings.generation.model, "llm  "))

    missing = []
    for wanted, label in checks:
        found = wanted in models
        if not found:
            missing.append(wanted)
        print(f"    {label} {wanted:<34} {'found ' if found else 'MISSING'}")

    if missing:
        print("\n  Those model names are not advertised by this endpoint. Pick one from the")
        print("  list above -- a wrong model name is the most common cause of a base URL")
        print("  that 'connects but always fails'.")
        if settings.embedding.provider == "openai" and any("embed" in m for m in missing):
            print("\n  If this endpoint serves chat models only (no embeddings), use:")
            print("    export RAG_EMBED_PROVIDER=local")
            print("  ...which embeds on CPU via ONNX, free and offline.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag", description="Production RAG pipeline")
    parser.add_argument("--index-path", default=None, help="where the index lives (default: data/index)")
    parser.add_argument("--no-load", action="store_true", help="start with an empty index")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="index a file or directory")
    p_ingest.add_argument("path")
    p_ingest.add_argument("--no-recursive", action="store_true")
    p_ingest.set_defaults(func=cmd_ingest)

    p_ask = sub.add_parser("ask", help="ask a question")
    p_ask.add_argument("question")
    p_ask.add_argument("--top-k", type=int, default=None)
    p_ask.add_argument("--json", action="store_true", help="machine-readable output")
    p_ask.set_defaults(func=cmd_ask)

    p_serve = sub.add_parser("serve", help="run the HTTP API")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument(
        "--ingest-root",
        default=None,
        metavar="DIR",
        help="directory that /ingest is allowed to read (default: cwd). "
             "Overrides RAG_INGEST_ROOT env var.",
    )
    p_serve.set_defaults(func=cmd_serve)

    p_eval = sub.add_parser("eval", help="evaluate retrieval against labelled queries")
    p_eval.add_argument("cases", help="JSONL of {query, relevant[]}")
    p_eval.add_argument("--top-k", type=int, default=8)
    p_eval.add_argument("--json", action="store_true")
    p_eval.set_defaults(func=cmd_eval)

    p_stats = sub.add_parser("stats", help="show index statistics")
    p_stats.set_defaults(func=cmd_stats)

    p_doctor = sub.add_parser("doctor", help="verify provider credentials and connectivity")
    p_doctor.set_defaults(func=cmd_doctor)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001 - CLI boundary should print, not traceback
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
