"""Prompt construction. Kept separate from the LLM client so prompts are reviewable,
diffable, and testable without an API key.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

SYSTEM_PROMPT = """You are a precise retrieval-augmented assistant.

Rules:
- Answer ONLY from the provided context. It is the sole source of truth.
- If the context does not contain the answer, say exactly: "I don't have that in the
  indexed sources." Do not use prior knowledge, and do not speculate.
- Cite every factual claim with its bracketed source number, e.g. [1].
- If sources disagree, say so explicitly and cite both.
- Be concise and direct. No preamble, no restating the question.
"""

CONTEXT_TEMPLATE = """[Context {i}]
Source: {source}
{text}
"""


@dataclass(slots=True)
class PromptBuilder:
    """Assembles the final prompt. Citations are 1-indexed and match the order the
    chunks were handed to the model, so the caller can map [n] back to a chunk id."""

    system_prompt: str = SYSTEM_PROMPT
    max_context_chars: int = 12_000
    context_template: str = CONTEXT_TEMPLATE

    def build(self, query: str, chunks: Sequence, history: Sequence[tuple[str, str]] | None = None) -> tuple[str, list[str]]:
        """Returns (user_prompt, citation_map) where citation_map[i] = chunk_id for [i+1]."""
        blocks: list[str] = []
        citation_map: list[str] = []
        used = 0

        for i, chunk in enumerate(chunks, start=1):
            block = self.context_template.format(
                i=i, source=getattr(chunk, "source", "unknown"), text=getattr(chunk, "text", str(chunk))
            )
            # Hard cap on context: overfill it and you get truncation mid-sentence plus a
            # bill. Drop the tail instead, and let the model know it happened.
            if used + len(block) > self.max_context_chars:
                blocks.append(f"[Context truncated: {len(chunks) - i + 1} further sources omitted]")
                break
            blocks.append(block)
            used += len(block)
            citation_map.append(getattr(chunk, "chunk_id", f"chunk_{i}"))

        sections = ["".join(blocks)]
        if history:
            turns = "\n".join(f"{'User' if r == 'user' else 'Assistant'}: {t}" for r, t in history[-4:])
            sections.insert(0, f"Conversation so far:\n{turns}\n")

        sections.append(f"Question: {query}\n\nAnswer using only the context above, with [n] citations.")
        return "\n".join(sections), citation_map

    def build_messages(
        self, query: str, chunks: Sequence, history: Sequence[tuple[str, str]] | None = None
    ) -> list[dict[str, str]]:
        user, _ = self.build(query, chunks, history)
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user},
        ]
