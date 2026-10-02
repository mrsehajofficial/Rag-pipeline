"""Score fusion and reranking.

Reciprocal Rank Fusion is the reason this pipeline works on mixed query types:
  * Dense retrieval nails paraphrases ("how do I stop it crashing" -> "unhandled exception").
  * BM25 nails exact tokens (error codes, SKUs, people's names) that embeddings blur away.
RRF merges them on *rank*, not raw score, so a BM25 score of 12.4 and a cosine of 0.83
are commensurable. Normalizing scores instead is fragile: the two distributions have
different shapes, and any min-max normalization is thrown off by a single outlier.

That merge-then-rerank order is the load-bearing decision here. Fusing first and reranking
after gives reranker #2 veto power over the fused order. Rerank-then-merge gives the lexical
leg the final say, undoing the semantic leg's best work.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..config import RetrievalConfig
from ..embedding import tokenize


@dataclass(slots=True)
class SearchHit:
    chunk_id: str
    score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None
    dense_score: float | None = None
    sparse_score: float | None = None
    chunk: object | None = None
    # Token set for MMR's similarity function. Declared as a real field rather than
    # setattr'd: @dataclass(slots=True) has no __dict__, so an undeclared attribute
    # raises. Computing it once here also avoids re-tokenizing inside MMR's O(n^2) loop.
    tokens: frozenset[str] = frozenset()

    @property
    def sources(self) -> list[str]:
        src = []
        if self.dense_rank is not None:
            src.append("dense")
        if self.sparse_rank is not None:
            src.append("sparse")
        return src


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[tuple[str, float]]],
    weights: Sequence[float] | None = None,
    k: int = 60,
) -> list[tuple[str, float]]:
    """rank_lists: list of [(id, score), ...] each sorted best-first. Returns [(id, rrf), ...]."""
    if not ranked_lists:
        return []
    weights = list(weights) if weights else [1.0] * len(ranked_lists)
    fused: dict[str, float] = {}
    for weight, ranked in zip(weights, ranked_lists):
        for rank, (doc_id, _score) in enumerate(ranked, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + weight / (k + rank)
    return sorted(fused.items(), key=lambda t: -t[1])


class Reranker:
    """Cross-encoder-shaped reranking without a model: lexical-overlap + proximity +
    coverage scoring, tuned to approximate what a cross-encoder learns.

    The proximity term is what a bag-of-words overlap score cannot express: two chunks
    that contain the query's terms *adjacent* beat two that scatter them across 500 words.
    """

    def __init__(self, config: RetrievalConfig | None = None) -> None:
        self.config = config or RetrievalConfig()

    def rerank(
        self, query: str, candidates: Sequence[tuple[str, str, float]], top_k: int | None = None
    ) -> list[tuple[str, str, float]]:
        """candidates: [(chunk_id, text, base_score)]. Returns the same shape, reordered."""
        if not candidates:
            return []
        top_k = top_k or self.config.top_k
        q_tokens = tokenize(query)
        if not q_tokens:
            return candidates[:top_k]

        q_set = set(q_tokens)
        q_pos = {t: i for i, t in enumerate(q_tokens)}
        scored: list[tuple[str, str, float]] = []

        for chunk_id, text, base in candidates:
            doc_tokens = tokenize(text)
            if not doc_tokens:
                scored.append((chunk_id, text, base * 0.5))
                continue

            overlap = q_set.intersection(doc_tokens)
            coverage = len(overlap) / len(q_set)  # what fraction of the query we hit
            density = len(overlap) / len(doc_tokens)  # chunk saturation

            positions = [q_pos[t] for t in overlap]
            span = max(positions) - min(positions) + 1 if positions else 1
            proximity = len(overlap) / span  # 1.0 = all terms adjacent

            # Coverage dominates; density and proximity break ties among equally-covered
            # chunks. base_score only nudges, so fusion order still carries signal.
            rerank_score = (
                0.55 * coverage
                + 0.15 * min(1.0, density * 12)  # density*12 ~= saturating around 8 terms
                + 0.20 * min(1.0, proximity * 2)
                + 0.10 * base
            )
            scored.append((chunk_id, text, rerank_score))

        scored.sort(key=lambda t: -t[2])
        return scored[:top_k]


def maximal_marginal_relevance(
    hits: Sequence, lambda_: float = 0.7, top_k: int | None = None
) -> list:
    """MMR: pick results that are relevant but not near-duplicates of what we picked.

    Pure relevance ranking fills the context window with 6 chunks saying the same thing,
    which is the classic RAG failure where one fact gets answered and the others are
    wasted. lambda=1.0 disables diversity, 0.0 maximizes it.
    """
    top_k = top_k or len(hits)
    if len(hits) <= 1:
        return list(hits)

    def similarity(a, b) -> float:
        # Jaccard over the precomputed token sets. Chunks are already deduped by id at
        # this point, so an identical-id check is the only exact-duplicate case left.
        if a.chunk_id == b.chunk_id:
            return 1.0
        ta, tb = getattr(a, "tokens", None), getattr(b, "tokens", None)
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)

    selected = [max(hits, key=lambda h: h.score)]
    while len(selected) < min(top_k, len(hits)):
        remaining = [h for h in hits if h not in selected]
        if not remaining:
            break
        best = max(
            remaining,
            key=lambda h: lambda_ * h.score - (1.0 - lambda_) * max(
                similarity(h, s) for s in selected
            ),
        )
        selected.append(best)
    return selected


def apply_score_floor(hits: Sequence, ratio: float = 0.35) -> list:
    """Drop weak tails. If the best hit scores 1.0, a 0.3 hit is noise and burns context."""
    if not hits:
        return []
    best = max(h.score for h in hits)
    if best <= 0:
        return list(hits)
    floor = best * ratio
    return [h for h in hits if h.score >= floor]
