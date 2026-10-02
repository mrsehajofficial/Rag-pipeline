"""Tests for the pure-logic layers: cache, chunker, BM25, fusion, metrics.

These run with zero third-party packages. If pytest isn't installed, run them with:
    python tests/test_core.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragpipe.cache import LRUCache, stable_hash
from ragpipe.config import ChunkingConfig, RetrievalConfig
from ragpipe.evaluation import ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank
from ragpipe.ingest.chunker import Chunker, estimate_tokens
from ragpipe.ingest.loaders import Document, Loader
from ragpipe.retrieval.bm25 import BM25Index
from ragpipe.retrieval.fusion import Reranker, SearchHit, reciprocal_rank_fusion


def test_lru_evicts_least_recently_used() -> None:
    cache: LRUCache[str, int] = LRUCache(max_size=2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a")  # 'a' is now the most recent, so 'b' should be evicted
    cache.put("c", 3)
    assert cache.get("b") is None
    assert cache.get("a") == 1
    assert cache.get("c") == 3
    assert len(cache) == 2


def test_lru_get_or_compute_runs_once() -> None:
    cache: LRUCache[str, int] = LRUCache(max_size=4)
    calls = []

    def factory() -> int:
        calls.append(1)
        return 42

    assert cache.get_or_compute("k", factory) == 42
    assert cache.get_or_compute("k", factory) == 42
    assert len(calls) == 1


def test_stable_hash_is_deterministic_and_collision_resistant() -> None:
    assert stable_hash("hello") == stable_hash("hello")
    # Different split points must not collide ("ab"+"c" vs "a"+"bc").
    assert stable_hash("ab", "c") != stable_hash("a", "bc")
    assert stable_hash("a") != stable_hash("b")


def test_token_estimate_scales_with_length() -> None:
    assert estimate_tokens("") == 1
    assert estimate_tokens("a" * 400) > estimate_tokens("a" * 40)


def test_chunker_respects_target_size() -> None:
    chunker = Chunker(ChunkingConfig(target_tokens=50, overlap_tokens=10, hard_max_tokens=100))
    chunks = chunker.chunk_document("word " * 600, doc_id="d1", source="test.txt")
    assert len(chunks) > 1
    assert all(c.token_estimate <= 100 for c in chunks)


def test_chunker_drops_empty_input() -> None:
    chunker = Chunker()
    assert chunker.chunk_document("   \n\n  ", doc_id="d1", source="s") == []


def test_chunker_produces_stable_ids() -> None:
    chunker = Chunker()
    text = "The quick brown fox jumps over the lazy dog. " * 40
    a = chunker.chunk_document(text, doc_id="d1", source="s.txt")
    b = chunker.chunk_document(text, doc_id="d1", source="s.txt")
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]


def test_chunker_folds_tiny_fragments() -> None:
    """A 2-token tail should merge into the previous chunk, not ship as an orphan."""
    chunker = Chunker(ChunkingConfig(target_tokens=40, min_chunk_tokens=10, overlap_tokens=0))
    text = ("A" * 200 + ". ") * 20 + "tiny."
    chunks = chunker.chunk_document(text, doc_id="d1", source="s")
    assert chunks
    assert all(c.token_estimate >= 10 or len(chunks) == 1 for c in chunks)


def test_loader_reads_markdown_and_strips_front_matter() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "doc.md"
        path.write_text("---\ntitle: Test\nowner: me\n---\n\n# Heading\n\nBody text here.\n", encoding="utf-8")
        docs = Loader().load(path)
        assert len(docs) == 1
        assert "Body text here." in docs[0].text
        assert "---" not in docs[0].text
        assert docs[0].metadata["title"] == "Test"


def test_loader_csv_folds_headers_into_rows() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "data.csv"
        path.write_text("name,role\nAlice,engineer\nBob,designer\n", encoding="utf-8")
        docs = Loader().load(path)
        assert len(docs) == 2
        assert "role: engineer" in docs[0].text


def test_loader_skips_bad_jsonl_lines() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "data.jsonl"
        path.write_text('{"text": "ok"}\nnot json\n{"text": "also ok"}\n', encoding="utf-8")
        assert len(Loader().load(path)) == 2


def test_bm25_finds_the_right_document() -> None:
    index = BM25Index()
    chunker = Chunker(ChunkingConfig(target_tokens=200))
    chunks = []
    for i, (src, body) in enumerate(
        [
            ("api-keys.md", "API keys are issued per service and rotate every ninety days."),
            ("tokens.md", "Validate the signature, then the issuer, then the audience."),
            ("deploy.md", "We deploy with a rolling update, never a recreate strategy."),
        ]
    ):
        chunks.extend(chunker.chunk_document(body, doc_id=f"d{i}", source=src))
    index.add(chunks)

    hits = index.search("how long do api keys last", top_k=3)
    assert hits, "expected at least one BM25 hit"
    top_source = index.get(hits[0][0]).source
    assert top_source == "api-keys.md", f"expected api-keys.md first, got {top_source}"


def test_bm25_remove_document() -> None:
    index = BM25Index()
    chunker = Chunker()
    index.add(chunker.chunk_document("api keys rotate every ninety days", doc_id="d1", source="a.md"))
    index.add(chunker.chunk_document("deployments use a rolling update", doc_id="d2", source="b.md"))
    assert len(index) == 2

    removed = index.remove_document("d1")
    assert removed == 1
    assert len(index) == 1
    assert index.search("api keys rotate", top_k=5) == []


def test_bm25_returns_nothing_for_unknown_terms() -> None:
    index = BM25Index()
    chunker = Chunker()
    index.add(chunker.chunk_document("apples and oranges", doc_id="d", source="s.md"))
    assert index.search("quantum entanglement zzzz", top_k=3) == []


def test_rrf_favours_items_ranked_by_both_lists() -> None:
    dense = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    sparse = [("b", 12.0), ("a", 9.0), ("d", 5.0)]
    fused = reciprocal_rank_fusion([dense, sparse])
    scores = dict(fused)
    # 'a' and 'b' appear in both lists and must outrank 'c' (dense only) and 'd' (sparse only).
    assert scores["a"] > scores["c"]
    assert scores["b"] > scores["d"]
    assert scores["a"] == scores["b"]  # symmetric ranks in both lists


def test_rrf_handles_empty_input() -> None:
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[]]) == []


def test_reranker_prefers_chunk_containing_all_query_terms() -> None:
    cfg = RetrievalConfig(top_k=5)
    reranker = Reranker(cfg)
    candidates = [
        ("c1", "Unrelated discussion about weather patterns.", 0.9),
        ("c2", "The API key is issued per service and rotates every ninety days.", 0.5),
    ]
    ranked = reranker.rerank("how long do API keys last", candidates)
    assert ranked[0][0] == "c2"


def test_eval_metrics_match_hand_computed_values() -> None:
    flags = [True, False, True, False, True]
    assert recall_at_k(flags, 3) == 2 / 3
    assert precision_at_k(flags, 5) == 3 / 5
    assert reciprocal_rank(flags) == 1.0  # first hit is relevant
    # nDCG@k compares the observed ordering against the best ordering of the same set.
    # These two docs are ordered wrongly (relevant at 1,3,5 rather than 1,2,3), so the
    # score is high but capped below 1.0 no matter how large k is.
    assert abs(ndcg_at_k(flags, 3) - 0.7039180890341347) < 1e-9
    assert abs(ndcg_at_k(flags, 5) - 0.8854598815714874) < 1e-9
    # Same relevance set, correct ordering -> perfect score at any k that covers it.
    assert ndcg_at_k([True, True, True, False, False], 5) == 1.0


def test_eval_metrics_degrade_on_bad_ranking() -> None:
    good = [True, False, False]
    bad = [False, False, True]
    assert reciprocal_rank(bad) < reciprocal_rank(good)
    assert ndcg_at_k(bad, 3) < ndcg_at_k(good, 3)
    assert recall_at_k([False, False], 3) == 0.0


def test_base_url_normalization_and_credentials() -> None:
    """Base URLs get pasted in every wrong shape imaginable, and each wrong shape
    produces a 404 whose error message does not point at the real cause. Normalizing
    here is cheaper than debugging it later."""
    import os

    from ragpipe.openai_client import resolve_credentials

    saved = {k: os.environ.get(k) for k in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "MY_KEY", "MY_URL")}

    def restore() -> None:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    try:
        # Missing key must raise something actionable, not a cryptic SDK error.
        os.environ.pop("OPENAI_API_KEY", None)
        try:
            resolve_credentials()
            raise AssertionError("expected a RuntimeError with no API key set")
        except RuntimeError as exc:
            assert "OPENAI_API_KEY" in str(exc)

        os.environ["OPENAI_API_KEY"] = "sk-test-1234567890"

        # Unset base URL means the official endpoint (SDK default), not "".
        os.environ.pop("OPENAI_BASE_URL", None)
        assert resolve_credentials() == ("sk-test-1234567890", None)

        # Whitespace-only must be treated as unset rather than sent as a blank string.
        os.environ["OPENAI_BASE_URL"] = "   "
        assert resolve_credentials()[1] is None

        # Every shape a user realistically pastes, normalized.
        cases = {
            "https://proxy.internal:8080": "https://proxy.internal:8080/v1",
            "https://proxy.internal:8080/": "https://proxy.internal:8080/v1",
            "https://oai.hf.co/v1": "https://oai.hf.co/v1",       # already correct
            "https://oai.hf.co/v1/": "https://oai.hf.co/v1",      # trailing slash
            "https://api.example.com/openai/v1": "https://api.example.com/openai/v1",
        }
        for given, expected in cases.items():
            os.environ["OPENAI_BASE_URL"] = given
            _, resolved = resolve_credentials()
            assert resolved == expected, f"{given} -> {resolved}, expected {expected}"
            assert not resolved.endswith("/v1/v1"), "appended /v1 twice"

        # A custom env var pair works for third-party gateways.
        os.environ.pop("OPENAI_API_KEY", None)
        os.environ["MY_KEY"] = "gateway-key-9999"
        assert resolve_credentials("MY_KEY", "MY_URL")[0] == "gateway-key-9999"
    finally:
        restore()


def test_permanent_errors_are_not_retried() -> None:
    """404/401/400 must fail immediately.

    The original bug retried a 404 four times with backoff, turning a one-second failure
    into a ten-second one while hiding the cause. Only rate limits, timeouts and 5xx are
    worth another attempt.
    """
    from ragpipe.embedding import describe_http_error, is_transient

    class FakeAPIError(Exception):
        def __init__(self, status):
            super().__init__(f"HTTP {status}")
            self.status_code = status

    # Transient: worth retrying.
    for status in (408, 409, 429, 500, 502, 503, 504):
        assert is_transient(FakeAPIError(status)), f"{status} should be retried"

    # Permanent: retrying cannot help.
    for status in (400, 401, 403, 404, 422):
        assert not is_transient(FakeAPIError(status)), f"{status} must not be retried"

    # No status at all (connection reset, DNS, timeout) -> one more try.
    assert is_transient(ConnectionResetError("boom"))

    # The 404 message must name the actual fix, not just restate the status code.
    msg = describe_http_error(FakeAPIError(404))
    assert "RAG_EMBED_PROVIDER=local" in msg, "404 guidance must point at the fix"
    assert "chat" in msg.lower(), "404 guidance should explain why it happened"
    assert "401" in describe_http_error(FakeAPIError(401))


def test_onnx_filename_preference_is_arch_aware() -> None:
    """Quantized ONNX filenames differ per repo AND per architecture.

    An earlier version hardcoded `model_..._quantized.onnx`, which does not exist in the
    all-MiniLM-L6-v2 repo. The fallback worked, but silently downloaded the 90MB fp32
    model instead of the 23MB quantized one. The order matters: smallest correct file
    first, never an avx512 file on a machine that may lack it.
    """
    import platform

    from ragpipe.embedding_local import _onnx_candidates

    machine = platform.machine().lower()
    cands = _onnx_candidates()
    assert cands, "must offer at least one candidate"
    # fp32 is always the last resort, never the first choice.
    assert cands[-1] == "onnx/model.onnx", f"fp32 must be last, got {cands}"
    # No duplicate /v1-style double-appending.
    assert len(set(cands)) == len(cands)

    if machine in ("x86_64", "amd64"):
        assert cands[0] == "onnx/model_quint8_avx2.onnx"
        # avx512 may be absent on older CPUs, so it must rank below avx2.
        if len(cands) > 1 and "avx512" in cands[1]:
            assert "avx2" in cands[0]
    if machine in ("arm64", "aarch64"):
        assert "arm64" in cands[0]


def test_llm_cache_persists_across_processes() -> None:
    """The cache must survive process boundaries.

    An earlier version kept it in-process only, which meant it could never produce a
    single hit for CLI usage -- `rag ask` is a fresh process every time. That is exactly
    the case where a repeat query matters most against a slow gateway.
    """
    from ragpipe.cache import LRUCache  # noqa: F401  (documents the replaced design)
    from ragpipe.config import CacheConfig, GenerationConfig
    from ragpipe.generation.llm import CachedLLM, LLMClient

    calls = []

    class Counting(LLMClient):
        def __init__(self):
            self.config = GenerationConfig(model="test-model")

        def complete(self, messages, **kwargs):
            calls.append(1)
            return "cached answer"

    messages = [{"role": "user", "content": "what is up"}]

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "llm_cache.db"

        # Simulate two separate processes sharing one cache file.
        first = CachedLLM(Counting(), CacheConfig(enabled=True), path)
        assert first.complete(messages) == "cached answer"
        assert len(calls) == 1

        second = CachedLLM(Counting(), CacheConfig(enabled=True), path)
        assert second.complete(messages) == "cached answer"
        assert len(calls) == 1, "second instance must hit the cache, not call the inner client again"

        # A different prompt must miss and call through.
        assert second.complete([{"role": "user", "content": "different"}]) == "cached answer"
        assert len(calls) == 2

        stats = second.cache_stats()
        assert stats["enabled"] and stats["entries"] == 2


def test_llm_cache_bounded_by_max_entries() -> None:
    """An unbounded response cache is a slow OOM in a long-running service."""
    from ragpipe.config import CacheConfig, GenerationConfig
    from ragpipe.generation.llm import CachedLLM, LLMClient

    class Counter(LLMClient):
        def __init__(self):
            self.config = GenerationConfig(model="m")

        def complete(self, messages, **kwargs):
            return f"answer-{messages[0]['content']}"

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "c.db"
        client = CachedLLM(Counter(), CacheConfig(enabled=True, max_entries=5), path)
        for i in range(25):
            client.complete([{"role": "user", "content": f"q{i}"}])

        count = client._conn().execute("SELECT COUNT(*) FROM llm_cache").fetchone()[0]
        assert count <= 5, f"cache grew to {count}, expected <= 5"


def _run_all() -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failures = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS {name}")
        except Exception as exc:  # noqa: BLE001
            failures.append((name, exc))
            print(f"  FAIL {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
