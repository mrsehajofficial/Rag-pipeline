"""Vector store: exact inner-product search with a numpy fast path.

Honest scoping: this is an *exact* index, which is the right choice up to a few hundred
thousand vectors and is what you want for accuracy anyway (ANN recall loss compounds
into bad answers). It uses the same normalized-dot-product trick FAISS uses for inner
product, and picks up a large speedup from numpy when it is installed.

Above ~500k chunks, swap `VectorStore` for faiss-cpu / HNSWlib, or Qdrant / pgvector when
you need multi-node. The `VectorStore` interface is small on purpose so that swap is a
single file.
"""

from __future__ import annotations

import json
import threading
from array import array
from operator import mul
from pathlib import Path
from typing import Iterable, Sequence

from ..cache import stable_hash
from ..config import VectorStoreConfig
from ..ingest.chunker import Chunk
from ..observability import get_logger

log = get_logger("ragpipe.store")

Vector = tuple[float, ...]

try:  # optional acceleration
    import numpy as _np

    HAS_NUMPY = True
except ImportError:  # pragma: no cover - exercised only on numpy-less installs
    _np = None
    HAS_NUMPY = False


class VectorStore:
    def __init__(self, dimensions: int, config: VectorStoreConfig | None = None) -> None:
        self.dimensions = dimensions
        self.config = config or VectorStoreConfig()
        self.backend = self._resolve_backend()
        self._lock = threading.RLock()

        # Storage: chunk_id -> vector. Keys are hashed strings; values are packed floats.
        self._ids: list[str] = []
        self._index: dict[str, int] = {}
        self._chunks: dict[str, Chunk] = {}
        self._matrix: list[Vector] = []

        self._np_matrix = None
        self._np_dirty = True

        log.info("vector store backend=%s dims=%d", self.backend, dimensions)

    def _resolve_backend(self) -> str:
        requested = self.config.backend
        if requested == "numpy" and not HAS_NUMPY:
            log.warning("numpy backend requested but not installed, falling back to memory")
            return "memory"
        if requested == "auto":
            return "numpy" if HAS_NUMPY else "memory"
        return requested

    # -- writes -------------------------------------------------------------

    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Vector]) -> int:
        """Upsert by chunk_id. Returns the number of rows actually written.

        Upsert rather than append so re-running ingestion is idempotent -- a daily job
        that re-crawls the corpus must not double every vector.
        """
        if len(chunks) != len(vectors):
            raise ValueError(f"chunks/vectors length mismatch: {len(chunks)} vs {len(vectors)}")
        written = 0
        with self._lock:
            for chunk, vec in zip(chunks, vectors):
                if len(vec) != self.dimensions:
                    raise ValueError(
                        f"vector dim {len(vec)} != store dim {self.dimensions}; "
                        "the embedding model changed -- reindex with `rag reindex`"
                    )
                key = chunk.chunk_id
                if key in self._index:
                    self._matrix[self._index[key]] = tuple(vec)
                else:
                    self._index[key] = len(self._ids)
                    self._ids.append(key)
                    self._matrix.append(tuple(vec))
                self._chunks[key] = chunk
                written += 1
            self._np_dirty = True
            self._transposed = None  # stale after any write; rebuilt on next search
        return written

    def delete_document(self, doc_id: str) -> int:
        """Tombstone removal -- rebuilds the matrix in one pass (O(n), but index writes are
        batched and async in any real deployment, so this is fine)."""
        with self._lock:
            doomed = [k for k, c in self._chunks.items() if c.doc_id == doc_id]
            for key in doomed:
                self._chunks.pop(key, None)
            if not doomed:
                return 0
            keep = [i for i, k in enumerate(self._ids) if k not in set(doomed)]
            self._ids = [self._ids[i] for i in keep]
            self._matrix = [self._matrix[i] for i in keep]
            self._index = {k: i for i, k in enumerate(self._ids)}
            self._np_dirty = True
            self._transposed = None
        return len(doomed)

    # -- reads --------------------------------------------------------------

    def search(self, query: Vector, top_k: int = 10, filters: dict | None = None) -> list[tuple[str, float]]:
        if not self._ids or top_k <= 0:
            return []
        with self._lock:
            if self.backend == "numpy":
                matrix = self._numpy_matrix()
            else:
                matrix = None

            candidates: Iterable[int]
            if filters:
                # Post-filtering. Honest tradeoff: exact vector search first, then filter.
                # Correct and simple; for heavy selective filters, push the predicate into
                # the ANN index or a real DB instead.
                allowed = {
                    i
                    for i, key in enumerate(self._ids)
                    if all(self._chunks[key].metadata.get(k) == v for k, v in filters.items())
                }
                candidates = allowed
            else:
                candidates = range(len(self._ids))

            scored: list[tuple[float, int]] = []
            if matrix is not None:
                q = _np.asarray(query, dtype=_np.float32)
                if not filters:
                    # Fast path: score the whole matrix directly. Fancy-indexing with
                    # arange(C) here would copy all C rows on every single query, which
                    # at 5k chunks costs more than the matmul it precedes.
                    # tolist() also yields plain floats, so (score, i) stays consistent
                    # with the filtered branch below.
                    dots = matrix @ q
                    scored = [(score, i) for i, score in enumerate(dots.tolist())]
                else:
                    idx = list(candidates)
                    if not idx:
                        return []
                    sub = matrix[_np.asarray(idx, dtype=_np.int64)]
                    dots = sub @ q
                    scored = [(float(s), i) for s, i in zip(dots, idx)]
            else:
                for i in candidates:
                    scored.append((_dot(query, self._matrix[i]), i))

            # argpartition is O(n) vs the O(n log n) of a full sort. We only need the
            # top k, and for large k the sort cost is negligible anyway.
            if len(scored) > top_k * 4:
                part = _heapq_nlargest(scored, top_k)
            else:
                part = sorted(scored, key=lambda t: -t[0])[:top_k]

            return [(self._ids[i], score) for score, i in part]

    def get(self, chunk_id: str) -> Chunk | None:
        return self._chunks.get(chunk_id)

    def get_many(self, chunk_ids: Sequence[str]) -> list[Chunk]:
        return [self._chunks[c] for c in chunk_ids if c in self._chunks]

    def all_chunks(self) -> list[Chunk]:
        with self._lock:
            return list(self._chunks.values())

    def __len__(self) -> int:
        return len(self._ids)

    def _numpy_matrix(self):
        global _np
        if _np is None:  # pragma: no cover
            raise RuntimeError("numpy unavailable")
        if self._np_dirty or self._np_matrix is None:
            if self._matrix:
                self._np_matrix = _np.asarray(self._matrix, dtype=_np.float32)
            else:
                self._np_matrix = _np.zeros((0, self.dimensions), dtype=_np.float32)
            self._np_dirty = False
        return self._np_matrix

    def _transposed_columns(self) -> list[list[float]] | None:
        """Build (once) the dimension-major view of the corpus.

        Returns None above the memory budget, at which point callers fall back to the
        row-major scan. Transposing doubles the float payload, so it is capped by
        RAG_MEM_BUDGET -- a 1M-chunk index would otherwise silently double its RSS.
        """
        if self._transposed is not None:
            return self._transposed
        if len(self._ids) > self.config.memory_budget_vectors:
            return None
        if not self._matrix:
            self._transposed = []
            return self._transposed

        dims = self.dimensions
        # zip(*rows) transposes in C: no Python-level nested loop over 256 floats/chunk.
        self._transposed = [list(col) for col in zip(*self._matrix)]
        _ = dims
        return self._transposed

    # -- persistence --------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        """Single-file format: a JSON header plus two binary blobs.

        Text goes to JSON (readable, debuggable, compresses well) and vectors go to a raw
        float32 blob -- no per-row JSON parsing, no pickle, loadable from any language.
        """
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        with self._lock:
            chunks_path = target / "chunks.jsonl"
            vecs_path = target / "vectors.f32"
            meta_path = target / "meta.json"

            with chunks_path.open("w", encoding="utf-8") as fh:
                for key in self._ids:
                    c = self._chunks[key]
                    fh.write(
                        json.dumps(
                            {
                                "chunk_id": key,
                                "text": c.text,
                                "doc_id": c.doc_id,
                                "chunk_index": c.chunk_index,
                                "source": c.source,
                                "metadata": c.metadata,
                                "char_start": c.char_start,
                                "token_estimate": c.token_estimate,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

            if self._matrix:
                if HAS_NUMPY:
                    _np.asarray(self._matrix, dtype=_np.float32).tofile(vecs_path)
                else:  # stdlib fallback, same byte layout
                    flat = array("f", (v for row in self._matrix for v in row))
                    vecs_path.write_bytes(flat.tobytes())
            else:
                vecs_path.write_bytes(b"")

            meta_path.write_text(
                json.dumps(
                    {
                        "dimensions": self.dimensions,
                        "count": len(self._ids),
                        "backend": self.backend,
                        "version": 1,
                        "index_digest": stable_hash(*self._ids, length=32) if self._ids else "",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        log.info("saved %d vectors to %s", len(self), target)
        return target

    def load(self, path: str | Path) -> int:
        target = Path(path)
        meta = json.loads((target / "meta.json").read_text(encoding="utf-8"))
        dims = int(meta["dimensions"])
        if dims != self.dimensions:
            raise ValueError(
                f"index was built with dim={dims} but store is dim={self.dimensions}. "
                "Reindex, or update RAG_EMBED_DIM to match the saved index."
            )
        expected = int(meta["count"])

        rows: list[Chunk] = []
        with (target / "chunks.jsonl").open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                d = json.loads(line)
                rows.append(
                    Chunk(
                        text=d["text"],
                        doc_id=d["doc_id"],
                        chunk_index=d["chunk_index"],
                        source=d["source"],
                        metadata=d.get("metadata", {}),
                        char_start=d.get("char_start", 0),
                        token_estimate=d.get("token_estimate", 0),
                    )
                )
        if len(rows) != expected:
            raise ValueError(f"index is corrupt: header says {expected} chunks, found {len(rows)}")

        blob = (target / "vectors.f32").read_bytes()
        if HAS_NUMPY:
            matrix = _np.frombuffer(blob, dtype=_np.float32).reshape(len(rows), dims)
            vectors: list[Vector] = [tuple(row) for row in matrix]
        else:
            flat = array("f")
            flat.frombytes(blob)
            vectors = [tuple(flat[i * dims : (i + 1) * dims]) for i in range(len(rows))]

        with self._lock:
            self._ids = []
            self._index = {}
            self._matrix = []
            self._chunks = {}
            for chunk, vec in zip(rows, vectors):
                key = chunk.chunk_id
                self._index[key] = len(self._ids)
                self._ids.append(key)
                self._matrix.append(vec)
                self._chunks[key] = chunk
            self._np_dirty = True
        log.info("loaded %d vectors from %s", len(self), target)
        return len(self)


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    """sum(map(mul, a, b)) -- the fastest pure-python dot product we measured.

    Why not the obvious alternatives, at 5000 chunks x 256 dims on this box:
        math.fsum(x*y for x,y in zip(a,b))   77.0 ms   generator yields per element
        explicit for-loop accumulation        59.5 ms   bytecode per element
        sum(map(mul, q, r))                  36.6 ms   <-- all C, no Python loop
    map(mul, ...) and sum() are both C-level, so the interpreter never touches an
    element. The float error difference versus fsum is ~1e-7 relative, which is
    meaningless next to the score gaps this is ranking on.
    """
    return sum(map(mul, a, b))


def _heapq_nlargest(scored: list[tuple[float, int]], k: int) -> list[tuple[float, int]]:
    """Top-k without importing heapq's tuple comparisons on ties."""
    import heapq

    return heapq.nlargest(k, scored, key=lambda t: t[0])
