"""FAISS-backed vector store for approximate nearest neighbor search.

Use this backend when the corpus grows past ~500k chunks, where exact search
becomes too slow. FAISS IndexFlatIP provides exact inner-product search with
optimized C++ routines; for true ANN at billion-scale, swap in IndexHNSW or
IndexIVFFlat.

The save/load format is compatible with the existing VectorStore (meta.json +
chunks.jsonl+vectors.f32), so indexes are interchangeable.

Usage:
    from ragpipe.store.faiss_store import FaissStore
    store = FaissStore(dimensions=256, config=VectorStoreConfig(backend="faiss"))
"""

from __future__ import annotations

import json
import threading
from collections.abc import Sequence
from pathlib import Path

from ..cache import stable_hash
from ..config import VectorStoreConfig
from ..ingest.chunker import Chunk
from ..observability import get_logger

log = get_logger("ragpipe.store.faiss")

Vector = tuple[float, ...]

try:
    import faiss
    import numpy as _np

    HAS_FAISS = True
except ImportError:
    _np = None
    faiss = None
    HAS_FAISS = False


class FaissStore:
    """FAISS-backed vector store with the same interface as VectorStore.

    Uses IndexFlatIP (exact inner product) since vectors are L2-normalized,
    making inner product equivalent to cosine similarity. For approximate
    search at scale, replace with IndexHNSW or IndexIVFFlat.
    """

    def __init__(self, dimensions: int, config: VectorStoreConfig | None = None) -> None:
        if not HAS_FAISS:
            raise ImportError(
                "FAISS backend requires faiss-cpu. Install it with:\n"
                "  pip install faiss-cpu\n"
                "Or use the default backend (exact numpy search):\n"
                "  RAG_STORE_BACKEND=auto"
            )
        self.dimensions = dimensions
        self.config = config or VectorStoreConfig()
        self._lock = threading.RLock()

        # Storage: chunk_id -> vector. Keys are hashed strings; values are packed floats.
        self._ids: list[str] = []
        self._index: dict[str, int] = {}
        self._chunks: dict[str, Chunk] = {}
        self._matrix: list[Vector] = []

        self._faiss_index = None
        self._faiss_dirty = True

        log.info("faiss store initialized: dims=%d", dimensions)

    def _rebuild_faiss(self) -> None:
        """Rebuild the FAISS index from the current matrix."""
        if not self._matrix:
            self._faiss_index = None
            self._faiss_dirty = False
            return
        matrix = _np.asarray(self._matrix, dtype=_np.float32)
        self._faiss_index = faiss.IndexFlatIP(self.dimensions)
        self._faiss_index.add(matrix)
        self._faiss_dirty = False

    # -- writes -------------------------------------------------------------

    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Vector]) -> int:
        """Upsert by chunk_id. Returns the number of rows actually written."""
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
            self._faiss_dirty = True
        return written

    def delete_document(self, doc_id: str) -> int:
        """Remove all chunks belonging to a document."""
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
            self._faiss_dirty = True
        return len(doomed)

    # -- reads --------------------------------------------------------------

    def search(self, query: Vector, top_k: int = 10, filters: dict | None = None) -> list[tuple[str, float]]:
        if not self._ids or top_k <= 0:
            return []
        with self._lock:
            if self._faiss_dirty or self._faiss_index is None:
                self._rebuild_faiss()
            if self._faiss_index is None or self._faiss_index.ntotal == 0:
                return []

            q = _np.asarray(query, dtype=_np.float32).reshape(1, -1)
            # Request more than top_k when filtering, so we have enough after filtering
            search_k = top_k * 4 if filters else top_k
            search_k = min(search_k, self._faiss_index.ntotal)
            scores, indices = self._faiss_index.search(q, search_k)

            results: list[tuple[str, float]] = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0:
                    continue
                key = self._ids[idx]
                if filters:
                    chunk = self._chunks.get(key)
                    if chunk is None or not all(chunk.metadata.get(k) == v for k, v in filters.items()):
                        continue
                results.append((key, float(score)))
                if len(results) >= top_k:
                    break
            return results

    def get(self, chunk_id: str) -> Chunk | None:
        with self._lock:
            return self._chunks.get(chunk_id)

    def get_many(self, chunk_ids: Sequence[str]) -> list[Chunk]:
        with self._lock:
            return [self._chunks[c] for c in chunk_ids if c in self._chunks]

    def all_chunks(self) -> list[Chunk]:
        with self._lock:
            return list(self._chunks.values())

    def __len__(self) -> int:
        with self._lock:
            return len(self._ids)

    # -- persistence --------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        """Single-file format: a JSON header plus two binary blobs.

        Compatible with VectorStore.save() so indexes are interchangeable.
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
                _np.asarray(self._matrix, dtype=_np.float32).tofile(vecs_path)
            else:
                vecs_path.write_bytes(b"")

            meta_path.write_text(
                json.dumps(
                    {
                        "dimensions": self.dimensions,
                        "count": len(self._ids),
                        "backend": "faiss",
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
        matrix = _np.frombuffer(blob, dtype=_np.float32).reshape(len(rows), dims)
        vectors: list[Vector] = [tuple(row) for row in matrix]

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
            self._faiss_dirty = True
        log.info("loaded %d vectors from %s", len(self), target)
        return len(self)
