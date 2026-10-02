"""BM25 sparse retrieval.

Pure-python BM25 is linear in corpus size, which is fine to ~100k chunks (a few ms) and
is *why* hybrid search is cheap: the sparse leg costs almost nothing next to the dense leg.
Swap in a real inverted index (Whoosh, Tantivy, Elasticsearch/OpenSearch) past that.

Params are the Robertson/Sparck-Jones defaults with k1 bumped slightly -- k1=1.2 is
conservative when documents are short chunks, where term saturation happens fast.
"""

from __future__ import annotations

import math
import threading
from collections import Counter, defaultdict
from collections.abc import Sequence

from ..embedding import tokenize
from ..ingest.chunker import Chunk

BM25_K1 = 1.5
BM25_B = 0.75


class BM25Index:
    def __init__(self, k1: float = BM25_K1, b: float = BM25_B) -> None:
        self.k1 = k1
        self.b = b
        self._postings: dict[str, dict[str, int]] = defaultdict(dict)  # term -> {chunk_id: tf}
        self._doc_len: dict[str, int] = {}
        self._doc_freq: dict[str, int] = {}
        self._chunks: dict[str, Chunk] = {}
        self._avg_dl: float = 0.0
        self._lock = threading.RLock()

    def add(self, chunks: Sequence[Chunk]) -> int:
        with self._lock:
            for chunk in chunks:
                tokens = tokenize(chunk.text)
                if not tokens:
                    continue
                cid = chunk.chunk_id
                if cid not in self._doc_len:
                    self._doc_len[cid] = len(tokens)
                self._chunks[cid] = chunk
                tf = Counter(tokens)
                for term, count in tf.items():
                    self._postings[term][cid] = count
                    self._doc_freq[term] = self._doc_freq.get(term, 0) + 1
            self._refresh_stats()
            return len(chunks)

    def _refresh_stats(self) -> None:
        n = len(self._doc_len)
        self._avg_dl = (sum(self._doc_len.values()) / n) if n else 0.0

    def remove_document(self, doc_id: str) -> int:
        with self._lock:
            doomed = {cid for cid, c in self._chunks.items() if c.doc_id == doc_id}
            if not doomed:
                return 0
            for term in list(self._postings):
                postings = self._postings[term]
                for cid in doomed:
                    postings.pop(cid, None)
                if not postings:
                    del self._postings[term]
                    self._doc_freq.pop(term, None)
            for cid in doomed:
                self._doc_len.pop(cid, None)
                self._chunks.pop(cid, None)
            self._refresh_stats()
            return len(doomed)

    def search(self, query: str, top_k: int = 40, filters: dict | None = None) -> list[tuple[str, float]]:
        with self._lock:
            terms = tokenize(query)
            if not terms or not self._doc_len:
                return []

            n_docs = len(self._doc_len)
            scores: dict[str, float] = defaultdict(float)
            avg_dl = self._avg_dl or 1.0

            for term in terms:
                postings = self._postings.get(term)
                if not postings:
                    continue
                df = self._doc_freq.get(term, 0)
                # Probabilistic IDF with the +1 guard; clamps at 0 because a term in >50% of
                # chunks carries no signal and a negative score would actively demote it.
                idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
                if idf <= 0:
                    continue
                for cid, tf in postings.items():
                    dl = self._doc_len.get(cid, avg_dl)
                    denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / avg_dl))
                    scores[cid] += idf * (tf * (self.k1 + 1.0)) / denom

            if filters:
                scores = {
                    cid: s
                    for cid, s in scores.items()
                    if all(self._chunks[cid].metadata.get(k) == v for k, v in filters.items())
                }

            ranked = sorted(scores.items(), key=lambda t: -t[1])[:top_k]
            return [(cid, score) for cid, score in ranked if score > 0]

    def __len__(self) -> int:
        with self._lock:
            return len(self._doc_len)

    @property
    def vocabulary_size(self) -> int:
        with self._lock:
            return len(self._postings)

    def get(self, chunk_id: str) -> Chunk | None:
        with self._lock:
            return self._chunks.get(chunk_id)
