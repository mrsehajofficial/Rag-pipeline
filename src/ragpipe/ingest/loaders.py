"""Document loaders. Each returns raw text plus whatever metadata it can salvage.

Kept deliberately dumb: loaders do I/O and cleanup only. Chunking, embedding, and
dedup live elsewhere so each piece stays testable in isolation.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from ..observability import get_logger

log = get_logger("ragpipe.ingest")


@dataclass(slots=True)
class Document:
    text: str
    source: str = "unknown"
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def doc_id(self) -> str:
        from ..cache import stable_hash

        return stable_hash(self.source, self.text, length=16)


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    # utf-8 first, then latin-1: latin-1 never raises, so this always returns something.
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


_MD_HEADING = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


def _front_matter(text: str) -> tuple[dict[str, object], str]:
    """Strip a leading --- ... --- YAML-ish block, parsing only flat key: value pairs.

    Pulling YAML in just for front-matter would mean a PyYAML dependency; the subset
    people actually use in front-matter is flat, so parse that and leave the rest.
    """
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    block = text[3:end]
    meta: dict[str, object] = {}
    for line in block.splitlines():
        if ":" in line and not line.strip().startswith("#"):
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip().strip("'\"")
    return meta, text[end + 4 :].lstrip()


class Loader:
    """Dispatch to the right loader by file extension. Returns a list of Documents
    because some formats are inherently multi-document (JSONL, notebooks)."""

    EXTENSIONS: ClassVar[set[str]] = {".txt", ".md", ".markdown", ".rst", ".html", ".htm", ".json", ".jsonl", ".csv", ".py"}

    def load(self, path: str | Path) -> list[Document]:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"No such file: {p}")
        if not p.is_file():
            raise ValueError(f"Expected a file, got a directory: {p}")

        ext = p.suffix.lower()
        if ext in {".md", ".markdown"}:
            return [self._markdown(p)]
        if ext in {".txt", ".rst"}:
            return [Document(_read_text(p), source=str(p), metadata={"ext": ext})]
        if ext in {".html", ".htm"}:
            return [self._html(p)]
        if ext == ".jsonl":
            return self._jsonl(p)
        if ext == ".json":
            return self._json(p)
        if ext == ".csv":
            return self._csv(p)
        if ext == ".py":
            return [Document(_read_text(p), source=str(p), metadata={"ext": ext})]

        log.warning("unknown extension %s, treating %s as plain text", ext, p.name)
        return [Document(_read_text(p), source=str(p), metadata={"ext": ext or "none"})]

    def load_dir(self, directory: str | Path, recursive: bool = True, limit: int | None = None) -> list[Document]:
        root = Path(directory)
        pattern = "**/*" if recursive else "*"
        docs: list[Document] = []
        for path in sorted(root.glob(pattern)):
            if len(docs) >= (limit or float("inf")):
                break
            if path.is_file() and path.suffix.lower() in self.EXTENSIONS:
                if any(part.startswith(".") for part in path.parts):
                    continue  # skip .git, .venv, .mypy_cache
                try:
                    docs.extend(self.load(path))
                except (OSError, UnicodeDecodeError, ValueError) as exc:
                    log.warning("skipping %s: %s", path, exc)
        return docs

    # -- per-format ---------------------------------------------------------

    def _markdown(self, p: Path) -> Document:
        text = _read_text(p)
        meta, body = _front_matter(text)
        headings = [m.group(2).strip() for m in _MD_HEADING.finditer(body)]
        # Inferred values fill gaps only -- an explicit `title:` in front matter is an
        # author-set decision and must not be overwritten by the first H1 we scrape.
        meta.setdefault("headings", headings)
        meta.setdefault("title", headings[0] if headings else p.stem)
        meta["ext"] = ".md"
        return Document(body, source=str(p), metadata=meta)

    def _html(self, p: Path) -> Document:
        html = _read_text(p)
        title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        title = re.sub(r"\s+", " ", title_match.group(1)).strip() if title_match else p.stem
        body = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.IGNORECASE | re.DOTALL)
        body = re.sub(r"<[^>]+>", " ", body)
        body = re.sub(r"&nbsp;?", " ", body)
        body = re.sub(r"&amp;", "&", body)
        body = re.sub(r"\s+", " ", body).strip()
        return Document(body, source=str(p), metadata={"ext": ".html", "title": title})

    def _jsonl(self, p: Path) -> list[Document]:
        docs: list[Document] = []
        for i, line in enumerate(_read_text(p).splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                log.warning("%s line %d: invalid JSON, skipped", p.name, i + 1)
                continue
            docs.append(Document(self._obj_to_text(obj), source=f"{p}::{i}", metadata={"row": i}))
        return docs

    def _json(self, p: Path) -> list[Document]:
        try:
            obj = json.loads(_read_text(p))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{p.name} is not valid JSON: {exc}") from exc
        if isinstance(obj, list):
            return [Document(self._obj_to_text(o), source=f"{p}::{i}", metadata={"row": i}) for i, o in enumerate(obj)]
        return [Document(self._obj_to_text(obj), source=str(p), metadata={"ext": ".json"})]

    def _csv(self, p: Path) -> list[Document]:
        import csv
        import io

        docs: list[Document] = []
        reader = csv.DictReader(io.StringIO(_read_text(p)))
        for i, row in enumerate(reader):
            # Fold each row into "col: value" lines so the embedding sees the labels;
            # a bare "Alice,Engineering" tells the retriever almost nothing.
            body = "\n".join(f"{k}: {v}" for k, v in row.items() if k and v)
            if body.strip():
                docs.append(Document(body, source=f"{p}::row{i}", metadata={"row": i}))
        return docs

    @staticmethod
    def _obj_to_text(obj: object) -> str:
        if isinstance(obj, str):
            return obj
        if isinstance(obj, dict):
            return "\n".join(f"{k}: {json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v}" for k, v in obj.items())
        return json.dumps(obj, ensure_ascii=False, indent=2)
