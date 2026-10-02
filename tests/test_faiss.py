"""Tests for the FAISS-backed vector store.

These tests are skipped when faiss-cpu is not installed (it's an optional
dependency). They verify that FaissStore implements the same interface as
VectorStore and produces compatible results.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

faiss = pytest.importorskip("faiss", reason="faiss-cpu not installed")

from ragpipe.config import VectorStoreConfig
from ragpipe.embedding import HashingEmbedder
from ragpipe.ingest.chunker import Chunker
from ragpipe.store.faiss_store import FaissStore


def _make_chunks_and_vectors(n: int = 10, dims: int = 64):
    """Create n chunks and their vectors for testing."""
    chunker = Chunker()
    embedder = HashingEmbedder(dimensions=dims)
    chunks = []
    texts = []
    for i in range(n):
        text = f"document {i} discusses topic {i} in some detail with unique words"
        chunks.extend(chunker.chunk_document(text, doc_id=f"d{i}", source=f"f{i}.md"))
        texts.append(text)
    vectors = embedder.embed_many(texts)
    return chunks, vectors, embedder


def test_faiss_store_add_and_search() -> None:
    """FaissStore.add() + search() must return results in descending score order."""
    chunks, vectors, embedder = _make_chunks_and_vectors(n=10, dims=64)
    store = FaissStore(dimensions=64, config=VectorStoreConfig(backend="faiss"))
    store.add(chunks, vectors)

    query = embedder.embed("document 5 discusses topic 5")
    hits = store.search(query, top_k=3)

    assert len(hits) == 3
    # Scores must be in descending order
    for i in range(len(hits) - 1):
        assert hits[i][1] >= hits[i + 1][1]


def test_faiss_store_save_load_roundtrip() -> None:
    """FaissStore save/load must preserve search results."""
    chunks, vectors, embedder = _make_chunks_and_vectors(n=10, dims=64)
    store = FaissStore(dimensions=64, config=VectorStoreConfig(backend="faiss"))
    store.add(chunks, vectors)

    query = embedder.embed("document 3 discusses topic 3")
    before = store.search(query, top_k=5)

    with tempfile.TemporaryDirectory() as tmp:
        store.save(Path(tmp) / "index")
        loaded = FaissStore(dimensions=64, config=VectorStoreConfig(backend="faiss"))
        loaded.load(Path(tmp) / "index")
        after = loaded.search(query, top_k=5)

    assert [cid for cid, _ in before] == [cid for cid, _ in after]
    for (_, a), (_, b) in zip(before, after):
        assert abs(a - b) < 1e-5


def test_faiss_store_delete_document() -> None:
    """FaissStore.delete_document() must remove all chunks for a doc_id."""
    chunks, vectors, _embedder = _make_chunks_and_vectors(n=5, dims=64)
    store = FaissStore(dimensions=64, config=VectorStoreConfig(backend="faiss"))
    store.add(chunks, vectors)

    removed = store.delete_document("d0")
    assert removed > 0
    assert all(c.doc_id != "d0" for c in store.all_chunks())


def test_faiss_store_len() -> None:
    """FaissStore.__len__() must return the number of indexed chunks."""
    chunks, vectors, _ = _make_chunks_and_vectors(n=7, dims=64)
    store = FaissStore(dimensions=64, config=VectorStoreConfig(backend="faiss"))
    assert len(store) == 0
    store.add(chunks, vectors)
    assert len(store) == len(chunks)


def test_faiss_store_get() -> None:
    """FaissStore.get() must return the chunk for a given chunk_id."""
    chunks, vectors, _ = _make_chunks_and_vectors(n=5, dims=64)
    store = FaissStore(dimensions=64, config=VectorStoreConfig(backend="faiss"))
    store.add(chunks, vectors)

    chunk = store.get(chunks[0].chunk_id)
    assert chunk is not None
    assert chunk.chunk_id == chunks[0].chunk_id


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
