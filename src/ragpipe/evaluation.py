"""Retrieval evaluation. You cannot tune a RAG pipeline without this.

Metrics implemented: Recall@k, Precision@k, MRR, nDCG@k, plus per-stage diagnostics.
Works off a labelled set of (query -> relevant source/doc_ids) and reports where the
pipeline loses the relevant chunk: not retrieved, or retrieved but ranked too low.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from .observability import get_logger

log = get_logger("ragpipe.eval")


@dataclass(slots=True)
class EvalCase:
    query: str
    relevant: list[str]  # source paths, doc_ids, or substrings -- matched by relevance_fn
    name: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass(slots=True)
class EvalResult:
    name: str
    metrics: dict[str, float]
    per_query: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"name": self.name, "metrics": self.metrics, "per_query": self.per_query}


def _relevance(source: str, relevant: Sequence[str]) -> bool:
    """A hit counts as relevant if its source matches any label, by exact path, by
    basename, or by substring. Loose on purpose: labels come from humans and are rarely
    byte-exact, and a strict matcher just makes the eval set look worse than it is."""
    if not source:
        return False
    s = source.lower()
    for label in relevant:
        if not label:
            continue
        label_l = label.lower()
        if s == label_l or s.endswith("/" + label_l) or label_l in s:
            return True
    return False


def recall_at_k(ranked_relevant: Sequence[bool], k: int) -> float:
    """Fraction of the relevant documents that appear in the top k.

    Denominator is the number of *labels*, not the number of hits: recall answers "of the
    things I knew were good, how many did I surface?", so it can never exceed 1.0 and a
    retriever that returns nothing scores 0.
    """
    if not ranked_relevant:
        return 0.0
    total_relevant = sum(1 for r in ranked_relevant if r)
    if total_relevant == 0:
        return 0.0
    return sum(1 for r in ranked_relevant[:k] if r) / total_relevant


def precision_at_k(ranked_relevant: Sequence[bool], k: int) -> float:
    if k <= 0:
        return 0.0
    return sum(1 for r in ranked_relevant[:k] if r) / k


def reciprocal_rank(ranked_relevant: Sequence[bool]) -> float:
    for i, rel in enumerate(ranked_relevant, start=1):
        if rel:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked_relevant: Sequence[bool], k: int) -> float:
    dcg = sum((1.0 / math.log2(i + 2)) for i, rel in enumerate(ranked_relevant[:k]) if rel)
    ideal = sum((1.0 / math.log2(i + 2)) for i, rel in enumerate(sorted(ranked_relevant, reverse=True)[:k]) if rel)
    return dcg / ideal if ideal else 0.0


class Retriever:
    """Callable protocol so eval works against the pipeline or a bare retriever."""

    def search(self, query: str, top_k: int) -> Sequence: ...


class PipelineRetriever(Retriever):
    def __init__(self, pipeline, top_k: int = 8) -> None:
        self.pipeline = pipeline
        self.top_k = top_k

    def search(self, query: str, top_k: int | None = None):
        # Reach into the private retriever so eval measures *retrieval*, not generation.
        from .observability import Trace

        trace = Trace(trace_id="eval", query=query)
        return self.pipeline._retrieve(query, trace, top_k or self.top_k, None)


class RetrievalEvaluator:
    def __init__(self, retriever: Retriever) -> None:
        self.retriever = retriever

    def run(self, cases: Sequence[EvalCase], top_k: int = 8, name: str = "retrieval") -> EvalResult:
        per_query: list[dict] = []
        for case in cases:
            hits = self.retriever.search(case.query, top_k)
            sources = [
                (h.chunk.source if h.chunk else getattr(h, "source", "")) for h in hits
            ]
            flags = [_relevance(s, case.relevant) for s in sources]

            row = {
                "query": case.query,
                "name": case.name or case.query[:60],
                "top_sources": sources[:3],
                "recall_at_k": recall_at_k(flags, top_k),
                "precision_at_k": precision_at_k(flags, top_k),
                "reciprocal_rank": reciprocal_rank(flags),
                "ndcg_at_k": ndcg_at_k(flags, top_k),
                "found": any(flags),
            }
            per_query.append(row)
            if not row["found"]:
                log.info("MISS: %r -> %s", case.query, sources[:3])

        n = max(1, len(per_query))
        metrics = {
            "recall_at_k": sum(r["recall_at_k"] for r in per_query) / n,
            "precision_at_k": sum(r["precision_at_k"] for r in per_query) / n,
            "mrr": sum(r["reciprocal_rank"] for r in per_query) / n,
            "ndcg_at_k": sum(r["ndcg_at_k"] for r in per_query) / n,
            "hit_rate": sum(1 for r in per_query if r["found"]) / n,
            "queries": float(len(per_query)),
        }
        return EvalResult(name=name, metrics=metrics, per_query=per_query)

    def report(self, result: EvalResult) -> str:
        m = result.metrics
        lines = [
            f"=== {result.name} ({int(m['queries'])} queries) ===",
            f"  hit_rate      {m['hit_rate']:.3f}",
            f"  recall@k      {m['recall_at_k']:.3f}",
            f"  precision@k   {m['precision_at_k']:.3f}",
            f"  MRR           {m['mrr']:.3f}",
            f"  nDCG@k        {m['ndcg_at_k']:.3f}",
        ]
        misses = [r for r in result.per_query if not r["found"]]
        if misses:
            lines.append(f"  misses ({len(misses)}):")
            for miss in misses[:10]:
                lines.append(f"    - {miss['query']}")
        return "\n".join(lines)


def load_cases(path: str | Path) -> list[EvalCase]:
    """JSONL, one case per line: {"query": "...", "relevant": ["a.md"]}."""
    cases: list[EvalCase] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            cases.append(
                EvalCase(
                    query=d["query"],
                    relevant=d.get("relevant", []),
                    name=d.get("name", ""),
                    metadata=d.get("metadata", {}),
                )
            )
    return cases
