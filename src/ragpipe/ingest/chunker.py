"""Token-aware chunking.

Design notes that matter in production:
  * Estimate tokens as chars/4 -- close enough for budgeting, and zero-dependency.
    Swap in tiktoken when you care about exact counts for a real tokenizer.
  * Overlap exists so a sentence spanning a boundary is still findable from either side.
  * Chunks are packed to a *target* size and only hard-split at hard_max, so most chunks
    stay semantically whole (a chunk cut mid-sentence retrieves worse).
  * Chunk ids are content hashes: re-ingesting an unchanged file is idempotent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from ..cache import stable_hash
from ..config import ChunkingConfig

# Abbreviations that must not end a sentence.
_ABBREV = r"(?<!\bMr)(?<!\bMrs)(?<!\bDr)(?<!\bProf)(?<!\bvs)(?<!\be\.g)(?<!\bi\.e)(?<!\betc)(?<!\bInc)(?<!\bLtd)(?<!\bCo)(?<!\bJr)(?<!\bSr)"
_SENTENCE_BOUNDARY = re.compile(_ABBREV + r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def estimate_tokens(text: str) -> int:
    """chars/4 heuristic. English averages ~4 chars/token including spaces."""
    return max(1, len(text) // 4)


@dataclass(slots=True)
class Chunk:
    text: str
    doc_id: str
    chunk_index: int
    source: str
    metadata: dict[str, object] = field(default_factory=dict)
    char_start: int = 0
    token_estimate: int = 0

    @property
    def chunk_id(self) -> str:
        return stable_hash(self.doc_id, self.chunk_index, self.text, length=16)


class Chunker:
    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config or ChunkingConfig()

    def chunk_document(self, text: str, doc_id: str, source: str, metadata: dict | None = None) -> list[Chunk]:
        text = text.strip()
        if not text:
            return []
        base_meta = dict(metadata or {})

        if self.config.strategy == "fixed":
            pieces = self._fixed(text)
        elif self.config.strategy == "sentence":
            pieces = self._sentences(text)
        else:
            pieces = self._recursive(text)

        chunks: list[Chunk] = []
        for i, (piece, offset) in enumerate(pieces):
            token_est = estimate_tokens(piece)
            if token_est < self.config.min_chunk_tokens and chunks:
                # Too small to retrieve on its own -> fold into the previous chunk
                # rather than emitting an orphan fragment.
                prev = chunks[-1]
                merged = prev.text.rstrip() + " " + piece.lstrip()
                if estimate_tokens(merged) <= self.config.hard_max_tokens:
                    prev.text = merged
                    prev.token_estimate = estimate_tokens(merged)
                    continue
            if not piece.strip():
                continue
            chunks.append(
                Chunk(
                    text=piece.strip(),
                    doc_id=doc_id,
                    chunk_index=i,
                    source=source,
                    metadata=dict(base_meta),
                    char_start=offset,
                    token_estimate=token_est,
                )
            )
        return chunks

    # -- strategies ---------------------------------------------------------

    def _fixed(self, text: str) -> list[tuple[str, int]]:
        chars = self.config.target_tokens * 4
        step = max(1, chars - self.config.overlap_tokens * 4)
        out = []
        for start in range(0, len(text), step):
            piece = text[start : start + chars]
            if piece.strip():
                out.append((piece, start))
            if start + chars >= len(text):
                break
        return out

    def _sentences(self, text: str) -> list[tuple[str, int]]:
        """Sentence-aware, greedy-packed. Prefers not to cut a sentence at all."""
        sentences: list[tuple[str, int]] = []
        pos = 0
        for m in _SENTENCE_BOUNDARY.finditer(text):
            sentences.append((text[pos : m.start()], pos))
            pos = m.end()
        if pos < len(text):
            sentences.append((text[pos:], pos))

        out: list[tuple[str, int]] = []
        buf, buf_start = "", 0
        for sent, offset in sentences:
            if not buf:
                buf, buf_start = sent, offset
            elif estimate_tokens(buf) + estimate_tokens(sent) <= self.config.target_tokens:
                buf += " " + sent
            else:
                out.append((buf, buf_start))
                buf, buf_start = sent, offset
            if estimate_tokens(buf) >= self.config.hard_max_tokens:
                out.append((buf, buf_start))
                buf, buf_start = "", 0
        if buf.strip():
            out.append((buf, buf_start))
        return out

    def _recursive(self, text: str) -> list[tuple[str, int]]:
        """Split on the most semantic separator available, recursing into oversized parts.

        Separator order matters: paragraph > line > sentence > clause > word. Splitting on
        \n\n first keeps topic boundaries intact far more often than naive fixed windows.
        """
        separators = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
        target_chars = self.config.target_tokens * 4
        overlap_chars = self.config.overlap_tokens * 4
        hard_chars = self.config.hard_max_tokens * 4

        def split(chunk: str, start: int, depth: int) -> list[tuple[str, int]]:
            if len(chunk) <= target_chars:
                return [(chunk, start)]

            sep = separators[min(depth, len(separators) - 1)]
            if sep == "":
                # Hard character wrap -- last resort.
                out = []
                for i in range(0, len(chunk), target_chars):
                    out.append((chunk[i : i + target_chars], start + i))
                return out

            pieces = chunk.split(sep)
            results: list[tuple[str, int]] = []
            buf = ""
            buf_start = start
            cursor = start
            for piece in pieces:
                piece_with_sep = piece + sep
                if not buf:
                    buf, buf_start = piece_with_sep, cursor
                elif len(buf) + len(piece_with_sep) <= target_chars:
                    buf += piece_with_sep
                else:
                    results.extend(split(buf, buf_start, depth + 1))
                    buf, buf_start = piece_with_sep, cursor
                cursor += len(piece_with_sep)

            if buf.strip():
                results.extend(split(buf, buf_start, depth + 1))

            # Add overlap so a boundary-spanning idea is retrievable from either chunk.
            if overlap_chars > 0 and results:
                merged: list[tuple[str, int]] = [results[0]]
                for i in range(1, len(results)):
                    prev_text, prev_start = results[i - 1]
                    tail = prev_text[-overlap_chars:]
                    text_i, start_i = results[i]
                    merged.append((tail + text_i, start_i - len(tail)))
                results = merged

            return [(t[:hard_chars], s) for t, s in results if t.strip()]

        return [(t, s) for t, s in split(text, 0, 0) if t.strip()]


def chunk_documents(docs: Iterable, config: ChunkingConfig | None = None) -> list[Chunk]:
    chunker = Chunker(config)
    out: list[Chunk] = []
    for doc in docs:
        out.extend(chunker.chunk_document(doc.text, doc.doc_id, doc.source, doc.metadata))
    return out
