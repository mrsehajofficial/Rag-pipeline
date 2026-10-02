"""Embedding layer: provider abstraction, disk-backed cache, batching, parallelism.

Latency strategy, in order of impact:
  1. Cache aggressively (content-addressed, survives restarts, shared across workers).
  2. Batch (one API round-trip per 256 texts, not per text).
  3. Parallelize *across* batches with threads -- these are network-bound, so the
     GIL is irrelevant and threads give near-linear speedup.
  4. Normalize once at index time so query-time scoring is a pure dot product.
"""

from __future__ import annotations

import math
import re
import sqlite3
import struct
import threading
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .cache import LRUCache, stable_hash
from .config import EmbeddingConfig
from .observability import get_logger

log = get_logger("ragpipe.embed")

Vector = tuple[float, ...]

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    ["a", "an", "the", "and", "or", "but", "if", "then", "else", "of", "to", "in", "on", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "from", "up", "down", "out", "off", "over", "under", "again", "further", "once", "here", "there", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "can", "will", "just", "should", "now", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "this", "that", "these", "those", "it", "its", "as", "i", "you", "he", "she", "they", "we", "what", "which", "who", "whom"]
)


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens, stopwords removed. Used by hashing embeddings,
    the BM25 index, and the extractive answerer -- one tokenizer, one behaviour."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1]


class EmbeddingProvider(ABC):
    @property
    @abstractmethod
    def dimensions(self) -> int: ...

    @abstractmethod
    def embed_batch(self, texts: Sequence[str]) -> list[Vector]: ...

    def embed(self, text: str) -> Vector:
        return self.embed_batch([text])[0]

    def embed_many(self, texts: Sequence[str], batch_size: int = 256) -> list[Vector]:
        out: list[Vector] = []
        for i in range(0, len(texts), batch_size):
            out.extend(self.embed_batch(list(texts[i : i + batch_size])))
        return out


class HashingEmbedder(EmbeddingProvider):
    """Feature-hashing embeddings: token + character n-grams -> fixed dim, L2-normalized.

    Not a neural model, so it will never match a real one on semantic similarity. It is
    fast, fully offline, and deterministic across processes -- which makes it genuinely
    useful for tests, smoke tests, and as the graceful fallback when no API key exists.
    Set RAG_EMBED_PROVIDER=openai for anything user-facing.
    """

    def __init__(self, dimensions: int = 256, use_char_ngrams: bool = True) -> None:
        self._dimensions = dimensions
        self.use_char_ngrams = use_char_ngrams

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def _features(self, text: str) -> Iterable[tuple[str, float]]:
        tokens = tokenize(text)
        for tok in tokens:
            yield tok, 1.0
        # Character trigrams give robustness to typos and morphological variants
        # ("deploying" -> "deploy") that pure word tokens miss.
        if self.use_char_ngrams:
            for tok in tokens:
                padded = f"^{tok}$"
                for i in range(len(padded) - 2):
                    yield padded[i : i + 3], 0.35
            # Bigrams preserve word order, which catches "not approved" vs "approved".
            from itertools import pairwise
            for a, b in pairwise(tokens):
                yield f"{a}_{b}", 0.6

    def embed_batch(self, texts: Sequence[str]) -> list[Vector]:
        dim = self._dimensions
        results: list[Vector] = []
        for text in texts:
            acc = [0.0] * dim
            for feature, weight in self._features(text):
                idx = int(stable_hash(feature, length=12), 16) % dim
                sign = 1.0 if (int(stable_hash(feature, length=13), 16) & 1) else -1.0
                acc[idx] += sign * weight
            norm = math.sqrt(sum(v * v for v in acc))
            if norm > 0:
                inv = 1.0 / norm
                acc = [v * inv for v in acc]
            results.append(tuple(acc))
        return results


def is_transient(exc: BaseException) -> bool:
    """Should this error be retried?

    Retrying a permanent error is worse than useless: it turns a one-second 404 into a
    ten-second 404 and buries the real cause. Only rate limits, timeouts, and 5xx are
    worth another attempt.
    """
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    if status is None:
        # No status: connection reset, DNS failure, timeout. Worth one more try.
        return True
    if status in (408, 409, 429):  # timeout, conflict, rate limited
        return True
    return 500 <= status < 600


def describe_http_error(exc: BaseException) -> str:
    """Turn an SDK exception into something that names the actual fix."""
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    if status == 404:
        return (
            "HTTP 404 from the embeddings endpoint.\n"
            "  This endpoint does not serve embeddings. Chat-only gateways (most hosted\n"
            "  Gemini, Claude, and Llama proxies) expose /chat/completions but not\n"
            "  /embeddings, and no model name will change that.\n"
            "  Use local embeddings instead -- free, offline, no key:\n"
            "    pip install onnxruntime tokenizers huggingface_hub\n"
            "    export RAG_EMBED_PROVIDER=local\n"
            "  ...then re-run `rag ingest` to rebuild the index at 384 dims."
        )
    if status == 401:
        return (
            "HTTP 401 (unauthorized). The API key is missing, wrong, or not valid for "
            "this base URL -- keys are per-endpoint and are not interchangeable."
        )
    if status == 400:
        return (
            "HTTP 400 (bad request). Often a dimension mismatch: RAG_EMBED_DIM must match "
            "the model. For RAG_EMBED_PROVIDER=local, ignore RAG_EMBED_DIM entirely -- "
            "the real width is read from the model."
        )
    if status == 404 or status is None:
        return str(exc)
    return f"HTTP {status}: {exc}"


def _backoff(attempt: int) -> float:
    return min(2**attempt * 0.5, 8.0)


class OpenAIEmbedder(EmbeddingProvider):
    """Production embedder. Batches + retries + token-budget safety."""

    def __init__(
        self,
        model: str,
        dimensions: int,
        batch_size: int = 256,
        api_key_env: str = "OPENAI_API_KEY",
        base_url_env: str = "OPENAI_BASE_URL",
    ) -> None:
        self.model = model
        self._dimensions = dimensions
        self.batch_size = batch_size
        self.api_key_env = api_key_env
        self.base_url_env = base_url_env
        self._client = None
        self._lock = threading.Lock()

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def client(self):  # lazy: importing/constructing the client is ~100ms
        if self._client is None:
            with self._lock:
                if self._client is None:
                    from .openai_client import build_openai_client

                    self._client = build_openai_client(
                        self.api_key_env, self.base_url_env, max_retries=0
                    )
        return self._client

    def embed_batch(self, texts: Sequence[str]) -> list[Vector]:
        from .openai_client import require_openai

        require_openai()  # actionable error if the optional SDK is missing
        import time

        last_err: Exception | None = None
        for attempt in range(4):
            try:
                resp = self.client.embeddings.create(
                    model=self.model, input=list(texts), encoding_format="float"
                )
                data = sorted(resp.data, key=lambda d: d.index)
                return [tuple(d.embedding) for d in data]
            except Exception as exc:
                last_err = exc
                if not is_transient(exc):
                    # 404/401/400 will never succeed. Say why, once, and stop.
                    raise RuntimeError(
                        f"embedding request failed permanently:\n{describe_http_error(exc)}"
                    ) from exc
                if attempt == 3:
                    break
                sleep_s = _backoff(attempt)
                log.warning(
                    "embedding batch failed (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1, 4, sleep_s, exc,
                )
                time.sleep(sleep_s)
        raise RuntimeError(
            f"embedding failed after 4 attempts (last error): {last_err}"
        )


class DiskVectorCache:
    """SQLite-backed vector cache keyed by (model, text).

    Why SQLite and not a dict: embeddings are deterministic, so recomputing them on
    restart is pure waste, and a dict cannot be shared across worker processes.
    Read path caches a page via the OS -- near-instant, and writes batch up.
    """

    def __init__(self, path: str | Path, dimensions: int) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.dimensions = dimensions
        self._local = threading.local()
        with self._conn() as con:
            con.execute("PRAGMA journal_mode=WAL")  # concurrent readers, one writer
            con.execute("PRAGMA synchronous=NORMAL")
            con.execute(
                "CREATE TABLE IF NOT EXISTS vectors ("
                " model TEXT NOT NULL, key TEXT NOT NULL, dim INTEGER NOT NULL,"
                " vec BLOB NOT NULL, PRIMARY KEY (model, key))"
            )
            con.execute("CREATE INDEX IF NOT EXISTS idx_vectors_model ON vectors(model)")

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=30.0, check_same_thread=False)
            self._local.conn = conn
        return conn

    @staticmethod
    def _pack(vec: Sequence[float]) -> bytes:
        return struct.pack(f"{len(vec)}f", *vec)

    @staticmethod
    def _unpack(blob: bytes) -> Vector:
        return struct.unpack(f"{len(blob) // 4}f", blob)

    def get_many(self, model: str, keys: Sequence[str]) -> dict[str, Vector]:
        if not keys:
            return {}
        found: dict[str, Vector] = {}
        con = self._conn()
        # Chunk the IN clause: SQLite caps bound parameters (~999 by default).
        for i in range(0, len(keys), 500):
            batch = list(keys[i : i + 500])
            placeholders = ",".join("?" * len(batch))
            rows = con.execute(
                f"SELECT key, vec FROM vectors WHERE model=? AND key IN ({placeholders})",
                [model, *batch],
            ).fetchall()
            for key, blob in rows:
                found[key] = self._unpack(blob)
        return found

    def put_many(self, model: str, items: dict[str, Sequence[float]]) -> None:
        if not items:
            return
        con = self._conn()
        rows = [(model, key, len(vec), self._pack(vec)) for key, vec in items.items()]
        with con:  # single transaction: one fsync instead of N
            con.executemany(
                "INSERT OR REPLACE INTO vectors (model, key, dim, vec) VALUES (?,?,?,?)", rows
            )

    def count(self, model: str) -> int:
        row = self._conn().execute("SELECT COUNT(*) FROM vectors WHERE model=?", (model,)).fetchone()
        return row[0] if row else 0

    def clear(self) -> None:
        with self._conn() as con:
            con.execute("DELETE FROM vectors")


class Embedder:
    """Provider + disk cache + in-process LRU + batch parallelism. The public entry point."""

    def __init__(self, config: EmbeddingConfig, cache_dir: str | Path = "data") -> None:
        self.config = config
        self._cache_dir = Path(cache_dir)
        self.provider: EmbeddingProvider = self._build_provider(config)
        self._lru: LRUCache[str, Vector] = LRUCache(max_size=config.cache_size)
        self._disk = DiskVectorCache(Path(cache_dir) / "embeddings.db", self.provider.dimensions)
        # Disk cache is meaningless for hashing (embedding is ~microseconds), and it
        # costs a sqlite round-trip per text. Skip it there.
        self._use_disk = config.provider != "hashing"
        # Cache keys must identify the *embedding function*, not just the text. The
        # model name is the discriminator: the same sentence hashed by all-MiniLM-L6-v2
        # and by text-embedding-3-small produces different vectors, and reusing one for
        # the other silently corrupts the index.
        self._model_key = self._cache_model_key()

    def _cache_model_key(self) -> str:
        if self.config.provider == "local":
            from .embedding_local import DEFAULT_MODEL

            return f"local:{self.config.local_model or DEFAULT_MODEL}"
        return self.config.model or self.config.provider

    @staticmethod
    def _build_provider(config: EmbeddingConfig) -> EmbeddingProvider:
        if config.provider == "openai":
            return OpenAIEmbedder(
                config.model,
                config.dimensions,
                config.batch_size,
                api_key_env=config.api_key_env,
                base_url_env=config.base_url_env,
            )
        if config.provider == "local":
            # Real semantic embeddings, no API and no key. Chosen when the configured
            # gateway serves chat models but no embedding model.
            from .embedding_local import LocalOnnxEmbedder

            return LocalOnnxEmbedder(
                model=config.local_model,
                cache_dir=config.local_cache_dir or None,
                batch_size=min(config.batch_size, 32),
            )
        if config.provider == "hashing":
            return HashingEmbedder(config.dimensions)
        raise ValueError(
            f"Unknown embedding provider: {config.provider!r} "
            "(expected one of: local, openai, hashing)"
        )

    @property
    def dimensions(self) -> int:
        return self.provider.dimensions

    def warmup(self) -> int | None:
        """Pre-load a heavy provider so the first real request isn't slow.

        No-op for providers that are already cheap to construct. Returns the ready
        dimension count, or None if the provider has no warmup step.
        """
        warm = getattr(self.provider, "warmup", None)
        if warm is None:
            return None
        try:
            dims = warm()
        except Exception as exc:  # noqa: BLE001 - never block startup on this
            log.warning("embedder warmup failed (continuing): %s", exc)
            return None
        # The real dim is only known after inference; refresh the now-stale cache.
        self._disk = DiskVectorCache(Path(self._cache_dir) / "embeddings.db", dims)
        log.info("embedder warmed up: %s, %d dims", self.config.provider, dims)
        return dims

    def embed_one(self, text: str) -> Vector:
        key = stable_hash(text, length=32)
        cached = self._lru.get(key)
        if cached is not None:
            return cached

        if self._use_disk:
            hit = self._disk.get_many(self._model_key, [key]).get(key)
            if hit is not None:
                self._lru.put(key, hit)
                return hit

        vector = self.provider.embed(text)
        self._lru.put(key, vector)
        if self._use_disk:
            self._disk.put_many(self._model_key, {key: vector})
        return vector

    def embed(self, text: str) -> Vector:
        """Alias matching EmbeddingProvider.embed, so a provider and an Embedder are
        interchangeable in tests and in caller code."""
        return self.embed_one(text)

    def embed_many(self, texts: Sequence[str], workers: int = 8) -> list[Vector]:
        """Returns vectors in the same order as `texts`.

        Flow: LRU hit -> disk hit -> compute. Only the misses hit the provider, and
        they go out as parallel batches (network-bound, so threads scale).
        """
        if not texts:
            return []

        keys = [stable_hash(t, length=32) for t in texts]
        out: list[Vector | None] = [None] * len(texts)

        # Pass 1: in-memory LRU
        pending_idx: list[int] = []
        for i, key in enumerate(keys):
            hit = self._lru.get(key)
            if hit is not None:
                out[i] = hit
            else:
                pending_idx.append(i)

        # Pass 2: disk cache
        if self._use_disk and pending_idx:
            wanted = {keys[i]: i for i in pending_idx}
            found = self._disk.get_many(self._model_key, list(wanted))
            for key, vec in found.items():
                out[wanted[key]] = vec
                self._lru.put(key, vec)
            pending_idx = [i for i in pending_idx if out[i] is None]

        # Pass 3: compute whatever is still missing, batch by batch
        if pending_idx:
            pending_texts = [texts[i] for i in pending_idx]
            self._prefetch(pending_texts, workers)  # parallel warm; fills the LRU

            for chunk_start in range(0, len(pending_texts), self.config.batch_size):
                chunk = pending_texts[chunk_start : chunk_start + self.config.batch_size]
                vecs = self.provider.embed_batch(chunk)
                fresh: dict[str, Vector] = {}
                for offset, vec in enumerate(vecs):
                    pos = pending_idx[chunk_start + offset]
                    out[pos] = vec
                    key = keys[pos]
                    self._lru.put(key, vec)
                    fresh[key] = vec
                if self._use_disk:
                    self._disk.put_many(self._model_key, fresh)

        assert all(v is not None for v in out), "embedding pass left gaps"
        return out  # type: ignore[return-value]

    def _prefetch(self, texts: Sequence[str], workers: int) -> None:
        """Warm the cache in parallel before the sequential fill pass reads it."""
        if workers <= 1 or len(texts) < 64:
            return
        bs = self.config.batch_size

        def run(i: int) -> None:
            chunk = texts[i : i + bs]
            for t, vec in zip(chunk, self.provider.embed_batch(chunk)):
                self._lru.put(stable_hash(t, length=32), vec)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(run, range(0, len(texts), bs)))
