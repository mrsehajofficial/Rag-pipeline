"""OpenAI client construction, in one place.

Two clients (embeddings and chat) need identical credential handling. Duplicating that
logic is how you end up with the embedder honouring a custom base URL and the LLM
silently ignoring it -- so both call `build_openai_client`.

Env vars honoured (all optional but the key):
    OPENAI_API_KEY    required
    OPENAI_BASE_URL   optional; unset means the official OpenAI endpoint
"""

from __future__ import annotations

import os
from typing import Any

from .observability import get_logger

log = get_logger("ragpipe.openai")

DEFAULT_BASE_URL = "https://api.openai.com/v1"


def resolve_credentials(api_key_env: str = "OPENAI_API_KEY", base_url_env: str = "OPENAI_BASE_URL") -> tuple[str, str | None]:
    """Return (api_key, base_url). Raises a clear error instead of letting the SDK
    raise something cryptic about a missing key."""
    api_key = (os.environ.get(api_key_env) or "").strip()
    if not api_key:
        raise RuntimeError(
            f"{api_key_env} is not set.\n"
            f"  export {api_key_env}=sk-...\n"
            f"  ...or run with a default provider (RAG_EMBED_PROVIDER=hashing) to stay offline."
        )

    base_url = (os.environ.get(base_url_env) or "").strip() or None
    if base_url:
        # Normalize: the SDK wants a root ending in /v1. Users routinely paste the
        # dashboard URL ("https://oai.hf.co") or add a trailing slash; both 404 later
        # with an error that does not point at the real cause.
        base_url = base_url.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"
        log.info("using custom OpenAI-compatible endpoint: %s", base_url)
    return api_key, base_url


def build_openai_client(api_key_env: str = "OPENAI_API_KEY", base_url_env: str = "OPENAI_BASE_URL", **kwargs: Any):
    """Construct an OpenAI SDK client with credentials + base URL resolved from env."""
    from openai import OpenAI  # lazy: keeps the optional dependency truly optional

    api_key, base_url = resolve_credentials(api_key_env, base_url_env)
    return OpenAI(api_key=api_key, base_url=base_url, **kwargs)


def has_credentials(api_key_env: str = "OPENAI_API_KEY") -> bool:
    return bool((os.environ.get(api_key_env) or "").strip())


def require_openai() -> Any:
    """Import the SDK or raise an actionable error. The package is an optional
    dependency, so a missing install is an expected state rather than a crash -- and
    'No module named openai' tells the reader nothing about what to do next."""
    try:
        import openai

        return openai
    except ImportError as exc:
        raise RuntimeError(
            "The 'openai' package is required for the openai provider.\n"
            "  pip install openai\n"
            "  ...or stay offline: RAG_EMBED_PROVIDER=hashing RAG_LLM_PROVIDER=extractive"
        ) from exc


def probe_models(api_key_env: str = "OPENAI_API_KEY", base_url_env: str = "OPENAI_BASE_URL", limit: int = 40) -> list[str]:
    """List model ids the endpoint advertises.

    Worth running before you configure anything: a proxy or self-hosted gateway usually
    serves a different model catalogue than OpenAI, and guessing a model name is the
    most common reason a new base URL 'does not work'.
    """
    client = build_openai_client(api_key_env, base_url_env)
    page = client.models.list()
    ids = [m.id for m in page.data]
    return sorted(ids)[:limit]
