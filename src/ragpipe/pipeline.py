"""The orchestrator: the actual RAG pipeline.

    query -> [rewrite?] -> parallel(dense, sparse) -> RRF fusion -> rerank
          -> MMR diversify -> score floor -> prompt -> LLM -> cited answer

Every stage is timed into the trace, so when latency regresses you can see whether it was
embedding, retrieval, or generation -- instead of profiling the whole thing.

Production notes:
  * _TOKEN_CACHE is backed by the thread-safe LRUCache so concurrent .clear()/.get()
    calls can never race -- the plain-dict version used dict.clear() without a lock.
  * _stream_answer emits the trace in a try/finally so a client disconnect cannot
    silently swallow telemetry.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from .cache import LRUCache
from .config import Settings, load_settings
from .embedding import Embedder
from .generation.llm import LLMClient, build_llm
from .generation.prompts import PromptBuilder
from .ingest.chunker import Chunk, Chunker
from .ingest.loaders import Document, Loader
from .observability import Trace, Tracer, configure_logging, get_logger
from .retrieval.bm25 import BM25Index
from .retrieval.fusion import (
    Reranker,
    SearchHit,
    apply_score_floor,
    maximal_marginal_relevance,
    reciprocal_rank_fusion,
)
from .store.vector_store import VectorStore

log = get_logger("ragpipe.pipeline")

NO_ANSWER = "I don't have that in the indexed sources."
MAX_QUERY_CHARS = int(os.environ.get("RAG_MAX_QUERY_CHARS", "2000"))


@dataclass(slots=True)
class Citation:
    index: int
    chunk_id: str
    source: str
    text: str
    score: float
    retrieved_by: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RAGResponse:
    query: str
    answer: str
    citations: list[Citation]
    trace_id: str
    total_ms: float
    stage_ms: dict[str, float] = field(default_factory=dict)
    hit_count: int = 0
    confident: bool = True
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "answer": self.answer,
            "citations": [
                {
                    "index": c.index,
                    "source": c.source,
                    "score": round(c.score, 4),
                    "retrieved_by": c.retrieved_by,
                    "text": c.text,
                }
                for c in self.citations
            ],
            "trace_id": self.trace_id,
            "total_ms": round(self.total_ms, 2),
            "stage_ms": {k: round(v, 2) for k, v in self.stage_ms.items()},
            "hit_count": self.hit_count,
            "confident": self.confident,
            "meta": self.meta,
        }


class RAGPipeline:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or load_settings()
        cfg = self.settings
        configure_logging(cfg.observability.log_level, cfg.observability.log_json)

        self.embedder = Embedder(cfg.embedding, cache_dir=cfg.data_path)
        if cfg.store.backend == "faiss":
            from .store.faiss_store import FaissStore
            self.store = FaissStore(self.embedder.dimensions, cfg.store)
        else:
            self.store = VectorStore(self.embedder.dimensions, cfg.store)
        self.bm25 = BM25Index()
        self.reranker = Reranker(cfg.retrieval)
        self.prompts = PromptBuilder()
        self.tracer = Tracer(cfg.observability.trace_sink, cfg.observability.trace_path)
        self.llm: LLMClient = build_llm(
            cfg.generation, cfg.cache, cache_path=str(cfg.data_path / "llm_cache.db")
        )
        self.chunker = Chunker(cfg.chunking)
        self.loader = Loader()
        self._index_lock = threading.RLock()

        # Pay heavy provider startup costs here, not on the first user request. For the
        # local ONNX embedder this is a ~2s model load; for API providers it's a no-op.
        self.embedder.warmup()

    # -- indexing -----------------------------------------------------------

    def _preflight_embedding(self) -> None:
        """Prove the embedding provider works before ingesting a single chunk.

        Otherwise a chat-only endpoint fails on chunk 1 with a raw 404, after loading and
        chunking the whole corpus. One throwaway embedding costs ~200ms and turns a late,
        confusing failure into an immediate, actionable one.
        """
        provider = self.settings.embedding.provider
        if provider == "hashing":
            return
        try:
            self.embedder.embed_one("preflight embedding check")
        except RuntimeError as exc:
            message = str(exc)
            if "404" in message or "embeddings endpoint" in message:
                raise RuntimeError(
                    f"\nEmbedding preflight failed before indexing anything.\n{message}\n"
                    f"\nCurrent setting: RAG_EMBED_PROVIDER={provider}"
                    + (
                        f"  RAG_EMBED_MODEL={self.settings.embedding.model}\n"
                        if provider == "openai"
                        else "\n"
                    )
                ) from None
            raise

    def index_documents(self, documents: Sequence[Document], show_progress: bool = False) -> dict:
        self._preflight_embedding()
        chunks: list[Chunk] = []
        for doc in documents:
            chunks.extend(self.chunker.chunk_document(doc.text, doc.doc_id, doc.source, doc.metadata))

        if not chunks:
            return {"documents": len(documents), "chunks": 0, "embedded": 0, "skipped": 0}

        # Content-hash dedup: the same paragraph appearing in two files is one chunk.
        # Without this, near-duplicate files flood the top-k with the same text.
        unique: dict[str, Chunk] = {}
        for c in chunks:
            unique[c.chunk_id] = c
        chunk_list = list(unique.values())

        with self._index_lock:
            log.info("embedding %d chunks (%d raw, %d dupes)", len(chunk_list), len(chunks), len(chunks) - len(chunk_list))
            vectors = self.embedder.embed_many([c.text for c in chunk_list], workers=8)
            written = self.store.add(chunk_list, vectors)
            self.bm25.add(chunk_list)

        return {
            "documents": len(documents),
            "chunks": written,
            "embedded": len(chunk_list),
            "skipped": len(chunks) - len(chunk_list),
        }

    def index_path(self, path: str | Path, recursive: bool = True) -> dict:
        documents = self.loader.load_dir(path, recursive=recursive) if Path(path).is_dir() else self.loader.load(path)
        return self.index_documents(documents)

    def delete_document(self, doc_id: str) -> dict:
        """Remove a document from both indexes. Takes the lock so a concurrent query
        never sees a half-deleted document."""
        with self._index_lock:
            dense = self.store.delete_document(doc_id)
            sparse = self.bm25.remove_document(doc_id)
        return {"dense_removed": dense, "sparse_removed": sparse}

    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path else Path(self.settings.store.persist_path)
        with self._index_lock:
            return self.store.save(target)

    def load(self, path: str | Path | None = None) -> int:
        target = Path(path) if path else Path(self.settings.store.persist_path)
        with self._index_lock:
            count = self.store.load(target)
            # Rebuild the lexical index from the stored chunks. Deliberate: persisting
            # postings separately is a second source of truth that can drift from the
            # chunk list, and rebuild cost is milliseconds.
            self.bm25 = BM25Index()
            self.bm25.add(self.store.all_chunks())
        return count

    @property
    def stats(self) -> dict:
        # Report the model that actually produced the vectors. For the local provider,
        # RAG_EMBED_MODEL is ignored entirely -- printing "text-embedding-3-small" next to
        # embed_provider=local is actively misleading when you are debugging a setup.
        if self.settings.embedding.provider == "local":
            embed_model = f"local:{self.settings.embedding.local_model}"
        else:
            embed_model = self.settings.embedding.model
        return {
            "chunks": len(self.store),
            "vocab": self.bm25.vocabulary_size,
            "dimensions": self.embedder.dimensions,
            "embed_provider": self.settings.embedding.provider,
            "embed_model": embed_model,
            "llm_provider": self.settings.generation.provider,
            "llm_model": self.settings.generation.model,
            "store_backend": self.store.backend,
        }

    # -- query --------------------------------------------------------------

    def query(
        self,
        question: str,
        top_k: int | None = None,
        filters: dict | None = None,
        stream: bool = False,
    ) -> RAGResponse | Iterator[str]:
        question = (question or "").strip()
        if not question:
            raise ValueError("question must be a non-empty string")
        if len(question) > MAX_QUERY_CHARS:
            raise ValueError(
                f"question too long: {len(question)} chars (max {MAX_QUERY_CHARS}). "
                "Raise RAG_MAX_QUERY_CHARS to allow longer inputs."
            )

        top_k = top_k or self.settings.retrieval.top_k
        trace = self.tracer.start(question)
        with trace.span("retrieve"):
            hits = self._retrieve(question, trace, top_k, filters)

        chunks = [h.chunk for h in hits if h.chunk is not None]

        with trace.span("generate"):
            messages = self.prompts.build_messages(question, chunks)
            if stream:
                return self._stream_answer(question, hits, trace, messages)
            answer = self.llm.complete(messages)

        confident = bool(chunks) and answer.strip() != NO_ANSWER
        trace.meta.update({"hits": len(hits), "confident": confident})
        self.tracer.emit(trace)

        citations = [
            Citation(
                index=i,
                chunk_id=h.chunk_id,
                source=h.chunk.source if h.chunk else "unknown",
                text=h.chunk.text if h.chunk else "",
                score=h.score,
                retrieved_by=h.sources,
            )
            for i, h in enumerate(hits, start=1)
        ]

        return RAGResponse(
            query=question,
            answer=answer,
            citations=citations,
            trace_id=trace.trace_id,
            total_ms=trace.total_ms,
            stage_ms={
                "retrieve": trace.stage_ms("retrieve"),
                "generate": trace.stage_ms("generate"),
            },
            hit_count=len(hits),
            confident=confident,
            meta={"top_k": top_k},
        )

    def _retrieve(self, question: str, trace: Trace, top_k: int, filters: dict | None) -> list[SearchHit]:
        cfg = self.settings.retrieval
        candidate_k = max(cfg.candidate_k, top_k)

        # Dense and sparse legs are independent -> run them concurrently. The embedding
        # call is the long pole, so the BM25 work hides entirely behind it.
        with ThreadPoolExecutor(max_workers=2) as pool:
            dense_future = pool.submit(self._dense_search, question, candidate_k, filters, trace)
            sparse_future = pool.submit(self.bm25.search, question, candidate_k, filters)
            dense_hits = dense_future.result()
            sparse_hits = sparse_future.result()

        with trace.span("fusion", dense=len(dense_hits), sparse=len(sparse_hits)):
            fused = reciprocal_rank_fusion(
                [dense_hits, sparse_hits],
                weights=[cfg.dense_weight, cfg.sparse_weight],
                k=cfg.rrf_k,
            )
            if not fused:
                return []

            hit_by_id: dict[str, SearchHit] = {}
            for rank, (cid, _s) in enumerate(dense_hits, start=1):
                hit_by_id[cid] = SearchHit(
                    chunk_id=cid,
                    score=0.0,
                    dense_rank=rank,
                    dense_score=dict(dense_hits)[cid],
                    chunk=self.store.get(cid) or self.bm25.get(cid),
                )
            for rank, (cid, s) in enumerate(sparse_hits, start=1):
                if cid in hit_by_id:
                    hit_by_id[cid].sparse_rank = rank
                    hit_by_id[cid].sparse_score = s
                else:
                    hit_by_id[cid] = SearchHit(
                        chunk_id=cid,
                        score=0.0,
                        sparse_rank=rank,
                        sparse_score=s,
                        chunk=self.store.get(cid) or self.bm25.get(cid),
                    )
            fused_scores = dict(fused)
            for cid, hit in hit_by_id.items():
                hit.score = fused_scores.get(cid, 0.0)

        candidates = [(h.chunk_id, h.chunk.text if h.chunk else "", h.score) for h in hit_by_id.values()]
        for hit in hit_by_id.values():
            if hit.chunk is not None:
                hit.tokens = frozenset(tokenize_cached(hit.chunk.text))

        if cfg.rerank and candidates:
            with trace.span("rerank", candidates=len(candidates)):
                reranked = self.reranker.rerank(question, candidates, top_k=top_k)
                order = {cid: i for i, (cid, _, _) in enumerate(reranked)}
                by_id = {h.chunk_id: h for h in hit_by_id.values()}
                hits = []
                for cid, _text, score in reranked:
                    h = by_id[cid]
                    h.score = score
                    hits.append(h)
                hits.sort(key=lambda h: order.get(h.chunk_id, 1e9))
        else:
            hits = sorted(hit_by_id.values(), key=lambda h: -h.score)[:top_k]

        if cfg.mmr_lambda < 1.0:
            with trace.span("mmr"):
                hits = maximal_marginal_relevance(hits, lambda_=cfg.mmr_lambda, top_k=top_k)

        hits = apply_score_floor(hits, cfg.score_floor_ratio)
        return hits[:top_k]

    def _dense_search(self, question: str, k: int, filters: dict | None, trace: Trace) -> list[tuple[str, float]]:
        with trace.span("embed_query"):
            vector = self.embedder.embed_one(question)
        with trace.span("vector_search", candidates=k):
            return self.store.search(vector, top_k=k, filters=filters)

    def _stream_answer(self, question: str, hits: list[SearchHit], trace: Trace, messages) -> Iterator[str]:
        # try/finally ensures the trace is emitted even when the client disconnects
        # mid-stream (which causes GeneratorExit, bypassing normal flow).
        try:
            with trace.span("generate"):
                yield from self.llm.stream(messages)
        finally:
            confident = bool(hits)
            trace.meta.update({"hits": len(hits), "confident": confident})
            self.tracer.emit(trace)


# Thread-safe LRU: the old plain-dict + .clear() pattern could race when one thread
# called .clear() while another was inside a .get() iteration.  LRUCache holds an
# RLock around every operation, so this is safe under ThreadPoolExecutor.
_TOKEN_CACHE: LRUCache[str, list[str]] = LRUCache(max_size=20_000)


def tokenize_cached(text: str) -> list[str]:
    """Tokenize once per unique text. MMR compares every pair of hits, so without this
    the same chunk gets tokenized O(n) times."""
    from .embedding import tokenize

    cached = _TOKEN_CACHE.get(text)
    if cached is not None:
        return cached
    tokens = tokenize(text)
    _TOKEN_CACHE.put(text, tokens)
    return tokens
