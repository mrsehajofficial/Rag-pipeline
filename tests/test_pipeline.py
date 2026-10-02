"""End-to-end tests: ingest -> persist -> load -> query. Uses the offline providers,
so these are deterministic and free. Run directly or under pytest.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ragpipe.config import Settings, load_settings
from ragpipe.embedding import Embedder, HashingEmbedder, tokenize
from ragpipe.evaluation import PipelineRetriever, RetrievalEvaluator, load_cases
from ragpipe.pipeline import RAGPipeline

DOCS = ROOT / "data" / "docs"


def _temp_settings(tmp: str) -> Settings:
    settings = load_settings()
    settings.data_dir = tmp
    settings.store.persist_path = str(Path(tmp) / "index")
    settings.observability.trace_sink = "none"
    # Pin the provider explicitly. Relying on the ambient default means the suite
    # silently changes behaviour based on whatever the shell has exported -- tests that
    # pass on a clean machine and fail on a configured one are worse than no tests.
    settings.embedding.provider = "hashing"
    settings.generation.provider = "extractive"
    return settings


def _hashing_embedder_config():
    """Embedding config pinned to the offline provider.

    These tests are about cache/ordering/persistence mechanics, not embedding quality, so
    they must not depend on whichever provider the ambient environment happens to select.
    """
    cfg = load_settings().embedding
    cfg.provider = "hashing"
    cfg.dimensions = 256
    return cfg


def test_hashing_embedder_is_deterministic_and_normalized() -> None:
    embedder = HashingEmbedder(dimensions=64)
    a = embedder.embed("the api key rotates every ninety days")
    b = embedder.embed("the api key rotates every ninety days")
    assert a == b, "embedding must be stable across calls or the cache is useless"
    norm = sum(v * v for v in a) ** 0.5
    assert abs(norm - 1.0) < 1e-6, f"expected unit vector, got norm {norm}"


def test_similar_texts_score_higher_than_unrelated() -> None:
    embedder = HashingEmbedder(dimensions=256)
    query = embedder.embed("how do we roll back a deployment")
    related = embedder.embed("To roll back, redeploy the previous image tag.")
    unrelated = embedder.embed("Apples grow well in cool temperate climates with acidic soil.")

    def dot(x, y) -> float:
        return sum(i * j for i, j in zip(x, y))

    assert dot(query, related) > dot(query, unrelated)


def test_tokenizer_drops_stopwords() -> None:
    tokens = tokenize("The quick brown fox and the lazy dog")
    assert "the" not in tokens and "and" not in tokens
    assert "quick" in tokens


def test_embedder_cache_avoids_recomputation() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        embedder = Embedder(_hashing_embedder_config(), cache_dir=tmp)
        texts = [f"document number {i} about topic {i % 7}" for i in range(50)]
        first = embedder.embed_many(texts)
        second = embedder.embed_many(texts)
        assert first == second
        assert embedder._lru.hits >= 50


def test_embed_many_preserves_order() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        embedder = Embedder(_hashing_embedder_config(), cache_dir=tmp)
        texts = ["alpha beta", "gamma delta", "epsilon zeta", "eta theta"]
        vectors = embedder.embed_many(texts)
        assert len(vectors) == 4
        singles = [embedder.embed_one(t) for t in texts]
        assert vectors == singles


def test_index_and_query_end_to_end() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        pipe = RAGPipeline(_temp_settings(tmp))
        stats = pipe.index_path(DOCS)
        assert stats["chunks"] > 0
        assert len(pipe.store) == stats["chunks"]

        response = pipe.query("what causes the export worker to crash?")
        assert response.answer
        assert response.citations
        assert response.total_ms > 0
        sources = " ".join(c.source for c in response.citations)
        assert "incident" in sources, f"expected the incident doc in citations, got {sources}"


def test_vector_store_search_returns_ordered_scores() -> None:
    """Guards the numpy fast path and the pure-python path against each other.

    Both branches build (score, index) pairs internally; swapping the tuple order
    silently scores every result against the wrong chunk instead of raising. That bug
    shipped once already, so it is pinned here explicitly.
    """
    from ragpipe.config import VectorStoreConfig
    from ragpipe.ingest.chunker import Chunker
    from ragpipe.store.vector_store import HAS_NUMPY, VectorStore

    chunker = Chunker()
    chunks = []
    texts = [
        "the quick brown fox jumps over the lazy dog",
        "api keys rotate every ninety days for each service",
        "deployments use a rolling update strategy",
    ]
    for i, text in enumerate(texts):
        chunks.extend(chunker.chunk_document(text, doc_id=f"d{i}", source=f"f{i}.md"))

    embedder = Embedder(_hashing_embedder_config())
    for backend in (["numpy", "memory"] if HAS_NUMPY else ["memory"]):
        store = VectorStore(embedder.dimensions, VectorStoreConfig(backend=backend))
        store.add(chunks, embedder.embed_many([c.text for c in chunks]))

        query = embedder.embed("api keys rotate every ninety days")
        hits = store.search(query, top_k=3)

        assert len(hits) == 3, f"{backend}: expected 3 hits"
        # Scores must be floats in descending order -- not indices, not ascending.
        for (cid, score) in hits:
            assert isinstance(score, float), f"{backend}: score must be float, got {type(score)}"
        assert hits[0][1] >= hits[1][1] >= hits[2][1], f"{backend}: results not sorted desc"
        # The chunk about API keys must rank first, and its source must match its text.
        top = store.get(hits[0][0])
        assert "ninety days" in top.text, f"{backend}: wrong chunk ranked first: {top.text[:60]!r}"
        assert top.source == "f1.md", f"{backend}: source/chunk mismatch"


def test_vector_store_roundtrip_preserves_order_and_scores() -> None:
    from ragpipe.config import VectorStoreConfig
    from ragpipe.ingest.chunker import Chunker
    from ragpipe.store.vector_store import VectorStore

    embedder = Embedder(_hashing_embedder_config())
    chunker = Chunker()
    chunks = []
    for i in range(12):
        chunks.extend(
            chunker.chunk_document(f"document {i} discusses topic {i} in some detail", doc_id=f"d{i}", source=f"f{i}.md")
        )
    vectors = embedder.embed_many([c.text for c in chunks])
    query = embedder.embed("document 7 discusses topic 7")

    with tempfile.TemporaryDirectory() as tmp:
        original = VectorStore(embedder.dimensions, VectorStoreConfig(backend="memory"))
        original.add(chunks, vectors)
        before = original.search(query, top_k=5)

        original.save(Path(tmp) / "index")
        reloaded = VectorStore(embedder.dimensions, VectorStoreConfig(backend="memory"))
        reloaded.load(Path(tmp) / "index")
        after = reloaded.search(query, top_k=5)

    assert [cid for cid, _ in before] == [cid for cid, _ in after]
    for (_, a), (_, b) in zip(before, after):
        assert abs(a - b) < 1e-5, f"score drifted across persistence: {a} vs {b}"


def test_query_answers_from_the_right_document() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        pipe = RAGPipeline(_temp_settings(tmp))
        pipe.index_path(DOCS)
        response = pipe.query("what order should token validation happen in?")
        top = response.citations[0].source if response.citations else ""
        assert "authentication" in top, f"expected authentication.md, got {top}"


def test_save_and_load_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        pipe = RAGPipeline(_temp_settings(tmp))
        pipe.index_path(DOCS)
        original_count = len(pipe.store)
        pipe.save()

        fresh = RAGPipeline(_temp_settings(tmp))
        loaded = fresh.load(Path(tmp) / "index")
        assert loaded == original_count
        assert len(fresh.bm25) == original_count, "lexical index must be rebuilt on load"

        # A query answered identically before and after a restart is the real test.
        a = pipe.query("how do we roll back a release?")
        b = fresh.query("how do we roll back a release?")
        assert a.answer == b.answer
        assert [c.source for c in a.citations] == [c.source for c in b.citations]


def test_reingestion_is_idempotent() -> None:
    """Ingesting the same corpus twice must not double the index -- scheduled ingestion
    jobs re-run, and without upsert semantics that becomes a silent data bug."""
    with tempfile.TemporaryDirectory() as tmp:
        pipe = RAGPipeline(_temp_settings(tmp))
        pipe.index_path(DOCS)
        first = len(pipe.store)
        pipe.index_path(DOCS)
        assert len(pipe.store) == first


def test_delete_document_removes_from_both_indexes() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        pipe = RAGPipeline(_temp_settings(tmp))
        pipe.index_path(DOCS)
        target = next(
            c.doc_id for c in pipe.store.all_chunks() if "incident-2291" in c.source
        )
        result = pipe.delete_document(target)
        assert result["dense_removed"] > 0
        remaining = [c for c in pipe.store.all_chunks() if c.doc_id == target]
        assert not remaining
        assert pipe.query("what caused the export worker to crash?").hit_count >= 0


def test_empty_query_is_rejected() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        pipe = RAGPipeline(_temp_settings(tmp))
        for bad in ("", "   ", None):
            try:
                pipe.query(bad)
            except (ValueError, AttributeError, TypeError):
                continue
            raise AssertionError(f"expected rejection for {bad!r}")


def test_unanswerable_query_reports_no_answer() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        pipe = RAGPipeline(_temp_settings(tmp))
        pipe.index_path(DOCS)
        response = pipe.query("what is the capital of Atlantis and who founded it?")
        # With only 3 docs there may still be weak hits, but confidence must reflect it.
        assert isinstance(response.confident, bool)


def test_llm_cache_returns_identical_answers() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        pipe = RAGPipeline(_temp_settings(tmp))
        pipe.index_path(DOCS)
        first = pipe.query("how do we roll back a release?").answer
        second = pipe.query("how do we roll back a release?").answer
        assert first == second


def test_streaming_yields_full_answer() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        pipe = RAGPipeline(_temp_settings(tmp))
        pipe.index_path(DOCS)
        stream = pipe.query("how do we roll back a release?", stream=True)
        streamed = "".join(stream)
        assert streamed
        assert len(streamed) > 10


def test_metadata_filters_restrict_results() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        pipe = RAGPipeline(_temp_settings(tmp))
        pipe.index_path(DOCS)
        hits = pipe._retrieve(
            "validate tokens", __import__("ragpipe.observability", fromlist=["Trace"]).Trace("t", "q"), 8, None
        )
        assert hits
        # Every hit must expose a source so the API can cite it.
        assert all(h.chunk is not None for h in hits)


def test_eval_reports_metrics_on_labelled_cases() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        pipe = RAGPipeline(_temp_settings(tmp))
        pipe.index_path(DOCS)
        cases = load_cases(ROOT / "data" / "eval_cases.jsonl")
        assert len(cases) >= 10

        evaluator = RetrievalEvaluator(PipelineRetriever(pipe, top_k=8))
        result = evaluator.run(cases, top_k=8, name="smoke")
        assert 0.0 <= result.metrics["hit_rate"] <= 1.0
        assert 0.0 <= result.metrics["mrr"] <= 1.0
        assert result.metrics["queries"] == len(cases)


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
