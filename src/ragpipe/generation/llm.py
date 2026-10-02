"""LLM clients: OpenAI (production) and a deterministic extractive fallback.

The extractive provider is not a toy. It answers by selecting and stitching the actual
retrieved sentences, which means the whole pipeline is runnable and testable end to end
with zero API spend and zero nondeterminism in CI. It also gives a sane degradation path
when the API key is missing or the API is down.
"""

from __future__ import annotations

import re
import sqlite3
import threading
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from pathlib import Path

from ..cache import LRUCache, stable_hash
from ..config import CacheConfig, GenerationConfig
from ..embedding import tokenize
from ..observability import get_logger

log = get_logger("ragpipe.generate")

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


class LLMClient(ABC):
    @abstractmethod
    def complete(self, messages: Sequence[dict[str, str]], **kwargs) -> str: ...

    def stream(self, messages: Sequence[dict[str, str]], **kwargs) -> Iterator[str]:
        """Default: no real streaming, yield the answer in one piece."""
        yield self.complete(messages, **kwargs)


class OpenAIClient(LLMClient):
    def __init__(self, config: GenerationConfig) -> None:
        self.config = config
        self._client = None
        self._lock = threading.Lock()

    @property
    def client(self):
        if self._client is None:
            with self._lock:
                if self._client is None:
                    from ..openai_client import build_openai_client

                    self._client = build_openai_client(
                        self.config.api_key_env, self.config.base_url_env, max_retries=0
                    )
        return self._client

    def complete(self, messages: Sequence[dict[str, str]], **kwargs) -> str:
        import time

        from ..embedding import describe_http_error, is_transient
        from ..openai_client import require_openai

        require_openai()
        last_err: Exception | None = None
        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model=kwargs.get("model", self.config.model),
                    messages=list(messages),
                    temperature=kwargs.get("temperature", self.config.temperature),
                    max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                )
                return resp.choices[0].message.content or ""
            except Exception as exc:
                last_err = exc
                if not is_transient(exc):
                    # A 404 or 401 will not fix itself; retrying just delays the message.
                    raise RuntimeError(
                        f"LLM request failed permanently:\n{describe_http_error(exc)}"
                    ) from exc
                if attempt == 2:
                    break
                time.sleep(min(2**attempt, 4))
        raise RuntimeError(f"LLM call failed after 3 attempts (last error): {last_err}")

    def stream(self, messages: Sequence[dict[str, str]], **kwargs) -> Iterator[str]:
        from ..openai_client import require_openai

        require_openai()
        try:
            stream = self.client.chat.completions.create(
                model=kwargs.get("model", self.config.model),
                messages=list(messages),
                temperature=kwargs.get("temperature", self.config.temperature),
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                stream=True,
            )
            for part in stream:
                delta = part.choices[0].delta
                if delta and delta.content:
                    yield delta.content
        except Exception as exc:  # noqa: BLE001
            log.warning("streaming failed (%s), falling back to non-streaming", exc)
            yield self.complete(messages, **kwargs)


class ExtractiveClient(LLMClient):
    """Offline answerer: ranks context sentences by query-term overlap and stitches the
    best ones, appending the citation of the chunk each came from.

    It will not paraphrase or reason -- but it is grounded, fast, free, and reproducible,
    which is exactly what you want from a fallback.
    """

    def __init__(self, config: GenerationConfig | None = None) -> None:
        self.config = config or GenerationConfig()

    def complete(self, messages: Sequence[dict[str, str]], **kwargs) -> str:
        user_prompt = messages[-1]["content"]
        query, context = _split_prompt(user_prompt)
        q_tokens = set(tokenize(query))
        if not q_tokens:
            return "I don't have that in the indexed sources."

        best_sentences: list[tuple[float, str, str]] = []  # (score, sentence, source)
        for source, text in context:
            for sentence in _SENT_SPLIT.split(text):
                sentence = sentence.strip()
                if len(sentence) < 15:
                    continue
                s_tokens = tokenize(sentence)
                if not s_tokens:
                    continue
                hits = q_tokens.intersection(s_tokens)
                if not hits:
                    continue
                coverage = len(hits) / len(q_tokens)
                density = len(hits) / len(s_tokens)
                best_sentences.append((0.75 * coverage + 0.25 * min(1.0, density * 8), sentence, source))

        if not best_sentences:
            return "I don't have that in the indexed sources."

        best_sentences.sort(key=lambda t: -t[0])
        top = best_sentences[:4]
        source_ids = {s: i + 1 for i, s in enumerate(dict.fromkeys(src for _, _, src in top))}
        return " ".join(
            sentence.rstrip(".") + f" [{source_ids[src]}]." for _, sentence, src in top
        )


def _split_prompt(user_prompt: str) -> tuple[str, list[tuple[str, str]]]:
    """Parse the prompt back into (query, [(source, text)]) for the extractive client."""
    sources: list[tuple[str, str]] = []
    lines = user_prompt.splitlines()
    current_source: str | None = None
    current_text: list[str] = []

    for line in lines:
        if line.startswith("[Context"):
            if current_source is not None:
                sources.append((current_source, " ".join(current_text).strip()))
            current_source, current_text = None, []
        elif line.startswith("Source: "):
            current_source = line[len("Source: ") :].strip()
        elif current_source and not line.startswith("Question:"):
            current_text.append(line)
    if current_source is not None:
        sources.append((current_source, " ".join(current_text).strip()))

    query = ""
    for i, line in enumerate(lines):
        if line.startswith("Question:"):
            query = line[len("Question:") :].strip()
            break
    return query, sources


class CachedLLM(LLMClient):
    """Caches LLM responses on disk, keyed by the full request.

    Why disk rather than a plain dict: a CLI invocation is a fresh process, so an
    in-memory cache starts empty every time and can never produce a hit -- it silently
    did nothing for anyone using `rag ask`. SQLite survives restarts and is shared
    across processes, which is also what makes it work behind multiple workers.

    Against a slow gateway this is the single biggest lever available: a repeat query
    drops from seconds to milliseconds.
    """

    def __init__(self, inner: LLMClient, config: CacheConfig, cache_path: str | Path | None = None) -> None:
        self.inner = inner
        self.config = config
        self._cache: LRUCache[str, str] = LRUCache(max_size=config.max_entries)
        self._path = Path(cache_path) if cache_path else None
        self._local = threading.local()
        self._enabled = False
        if self.config.enabled and self._path is not None:
            try:
                self._init_db()
                self._enabled = True
            except Exception as exc:  # noqa: BLE001 - a broken cache must not break queries
                log.warning("LLM cache unavailable (%s); continuing without it", exc)

    # -- sqlite plumbing ----------------------------------------------------

    def _conn(self):
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._path, timeout=10.0, check_same_thread=False)
            self._local.conn = conn
        return conn

    def _init_db(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as con:
            con.execute("PRAGMA journal_mode=WAL")
            con.execute("PRAGMA synchronous=NORMAL")
            con.execute(
                "CREATE TABLE IF NOT EXISTS llm_cache ("
                " key TEXT PRIMARY KEY, answer TEXT NOT NULL, hits INTEGER DEFAULT 0)"
            )

    def _read(self, key: str) -> str | None:
        row = self._conn().execute(
            "SELECT answer FROM llm_cache WHERE key=?", (key,)
        ).fetchone()
        if row:
            try:  # hit accounting is best-effort; never fail a query over it
                with self._conn() as con:
                    con.execute("UPDATE llm_cache SET hits = hits + 1 WHERE key=?", (key,))
            except sqlite3.Error:
                pass
            return row[0]
        return None

    def _write(self, key: str, answer: str) -> None:
        with self._conn() as con:
            con.execute(
                "INSERT OR REPLACE INTO llm_cache (key, answer) VALUES (?,?)", (key, answer)
            )
            # Bound the table: unbounded growth in a long-lived service is a slow OOM.
            con.execute(
                "DELETE FROM llm_cache WHERE key IN ("
                " SELECT key FROM llm_cache ORDER BY hits ASC, rowid ASC"
                " LIMIT MAX(0, (SELECT COUNT(*) FROM llm_cache) - ?))",
                (self.config.max_entries,),
            )

    def _key(self, messages: Sequence[dict[str, str]], kwargs: dict) -> str:
        return stable_hash(
            kwargs.get("model", getattr(self.inner.config, "model", "")),
            kwargs.get("temperature", getattr(self.inner.config, "temperature", 0.0)),
            [dict(m) for m in messages],
            length=32,
        )

    def complete(self, messages: Sequence[dict[str, str]], **kwargs) -> str:
        if not self._enabled:
            return self.inner.complete(messages, **kwargs)
        key = self._key(messages, kwargs)
        hit = self._read(key)
        if hit is not None:
            log.debug("LLM cache hit")
            return hit
        answer = self.inner.complete(messages, **kwargs)
        self._write(key, answer)
        return answer

    def cache_stats(self) -> dict:
        if not self._enabled:
            return {"enabled": False}
        row = self._conn().execute("SELECT COUNT(*), COALESCE(SUM(hits),0) FROM llm_cache").fetchone()
        return {"enabled": True, "entries": row[0], "hits": row[1], "path": str(self._path)}

    def stream(self, messages: Sequence[dict[str, str]], **kwargs) -> Iterator[str]:
        # Stream from cache in word-sized pieces so the client sees a uniform interface.
        if not self.config.enabled:
            yield from self.inner.stream(messages, **kwargs)
            return
        answer = self.complete(messages, **kwargs)
        for i in range(0, len(answer), 24):
            yield answer[i : i + 24]


def build_llm(config: GenerationConfig, cache: CacheConfig | None = None, cache_path: str | Path | None = None) -> LLMClient:
    if config.provider == "openai":
        from ..openai_client import has_credentials

        if not has_credentials(config.api_key_env):
            log.warning(
                "%s not set, falling back to extractive client. "
                "Answers will be stitched from raw sentences, not generated.",
                config.api_key_env,
            )
            client: LLMClient = ExtractiveClient(config)
        else:
            client = OpenAIClient(config)
    else:
        client = ExtractiveClient(config)
    if cache and cache.enabled:
        return CachedLLM(client, cache, cache_path)
    return client
