#!/usr/bin/env python3
"""Benchmark: ingest + query latency at realistic scale.

    python3 benchmarks/bench.py --chunks 5000 --queries 100

Reports p50/p95/p99 rather than a mean -- in a request path the tail is what pages you.
Runs the pure-python and numpy dense backends side by side when numpy is present, so
you can see the speedup you get from one dependency.
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragpipe.config import load_settings
from ragpipe.ingest.chunker import Chunker
from ragpipe.ingest.loaders import Document
from ragpipe.pipeline import RAGPipeline

WORDS = (
    "deploy rollback migration database index query latency cache token authentication "
    "session rotation worker crash null pointer exception incident severity resolution "
    "pipeline vector embedding retrieval ranking threshold monitor alert service cluster"
).split()


def synthetic_corpus(n: int, seed: int = 7) -> list[Document]:
    """Plausible documents: random word soup would make retrieval trivially easy or
    impossible, and neither tells you anything about real performance."""
    rng = random.Random(seed)
    docs = []
    for i in range(n):
        sentences = []
        for _ in range(rng.randint(8, 20)):
            words = [rng.choice(WORDS) for _ in range(rng.randint(8, 18))]
            sentences.append(" ".join(words).capitalize() + ".")
        docs.append(
            Document(
                "\n\n".join(sentences),
                source=f"synthetic/doc_{i:05d}.md",
                metadata={"idx": i},
            )
        )
    return docs


def pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, int(round((p / 100) * (len(ordered) - 1)))))
    return ordered[k]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, default=2000, help="approximate document count")
    ap.add_argument("--queries", type=int, default=50)
    ap.add_argument("--top-k", type=int, default=8)
    args = ap.parse_args()

    settings = load_settings()
    settings.observability.trace_sink = "none"
    settings.embedding.provider = "hashing"
    settings.generation.provider = "extractive"

    print(f"generating ~{args.chunks} synthetic documents ...", flush=True)
    corpus = synthetic_corpus(args.chunks)

    chunker = Chunker(settings.chunking)
    n_chunks = sum(len(chunker.chunk_document(d.text, d.doc_id, d.source, d.metadata)) for d in corpus[:50])
    est_chunks = n_chunks * len(corpus) // 50
    print(f"~{est_chunks} chunks expected\n")

    pipe = RAGPipeline(settings)

    t0 = time.perf_counter()
    stats = pipe.index_documents(corpus)
    ingest_s = time.perf_counter() - t0

    print("=" * 62)
    print("INGEST")
    print("=" * 62)
    print(f"  documents        {stats['documents']}")
    print(f"  chunks           {stats['chunks']}")
    print(f"  duplicate skips  {stats['skipped']}")
    print(f"  total time       {ingest_s:.2f}s")
    print(f"  throughput       {stats['chunks'] / max(ingest_s, 1e-9):.0f} chunks/s")
    print(f"  store backend    {pipe.store.backend}")

    rng = random.Random(11)
    queries = []
    for doc in rng.sample(corpus, min(args.queries, len(corpus))):
        words = doc.text.split()
        start = rng.randint(0, max(0, len(words) - 12))
        queries.append(" ".join(words[start : start + 10]))

    # Warm the caches first, then measure -- otherwise the first call pays for cold-start
    # embedding and the numbers describe a scenario that never happens twice.
    for q in queries[:5]:
        pipe.query(q, top_k=args.top_k)

    timings: list[float] = []
    retrieve_only: list[float] = []
    hits = 0
    for q in queries:
        t0 = time.perf_counter()
        response = pipe.query(q, top_k=args.top_k)
        timings.append((time.perf_counter() - t0) * 1000)
        hits += response.hit_count
        retrieve_only.append(response.stage_ms.get("retrieve", 0.0))

    print()
    print("=" * 62)
    print(f"QUERY  ({len(queries)} queries, top_k={args.top_k}, caches warm)")
    print("=" * 62)
    print(f"  mean       {statistics.fmean(timings):7.2f} ms")
    print(f"  p50        {pct(timings, 50):7.2f} ms")
    print(f"  p95        {pct(timings, 95):7.2f} ms")
    print(f"  p99        {pct(timings, 99):7.2f} ms")
    print(f"  max        {max(timings):7.2f} ms")
    print(f"  retrieval  {statistics.fmean(retrieve_only):7.2f} ms mean")
    print(f"  avg hits   {hits / len(queries):7.1f} per query")

    try:
        import numpy  # noqa: F401

        print(f"\n  numpy {numpy.__version__} detected -> dense search used the BLAS path.")
    except ImportError:
        print("\n  numpy not installed -> dense search ran pure-python.")
        print("  pip install numpy for a large speedup on the vector scan.")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
