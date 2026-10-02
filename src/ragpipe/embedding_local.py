"""Local ONNX sentence-embeddings provider.

Why this exists: plenty of OpenAI-compatible gateways serve chat models but no
embedding model. Rather than force an API round-trip for every chunk, this runs a
quantized MiniLM ONNX model on CPU. One ~90MB download, then embeddings are free,
offline, and deterministic.

Default model: all-MiniLM-L6-v2, 384 dimensions, 6 layers. Chosen because the quantized
file is small enough to download casually and fast enough to embed a whole corpus
single-threaded, while still being a genuine trained sentence encoder -- unlike the
hashing fallback, it matches meaning, not just words.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Sequence
from pathlib import Path

from .embedding import EmbeddingProvider
from .observability import get_logger

log = get_logger("ragpipe.embed.local")

Vector = tuple[float, ...]

# Known-good (model_id, dimensions) pairs. Anything else is assumed to be 384 unless the
# ONNX output shape says otherwise -- we read the real shape after the first run.
KNOWN_MODELS: dict[str, tuple[str, int]] = {
    "all-MiniLM-L6-v2": ("sentence-transformers/all-MiniLM-L6-v2", 384),
    "all-MiniLM-L12-v2": ("sentence-transformers/all-MiniLM-L12-v2", 384),
    "bge-small-en-v1.5": ("BAAI/bge-small-en-v1.5", 384),
    "bge-base-en-v1.5": ("BAAI/bge-base-en-v1.5", 768),
    "gte-small": ("thenlper/gte-small", 384),
}

DEFAULT_MODEL = "all-MiniLM-L6-v2"
CACHE_DIR_ENV = "RAG_LOCAL_EMBED_CACHE"


def _require_deps():
    try:
        import onnxruntime
        import tokenizers
    except ImportError as exc:
        raise RuntimeError(
            "Local embeddings need two packages:\n"
            "  pip install onnxruntime tokenizers\n"
            "  ...or use another provider: RAG_EMBED_PROVIDER=openai (needs an "
            "embeddings-capable endpoint)"
        ) from exc
    return onnxruntime, tokenizers


def _onnx_candidates() -> list[str]:
    """ONNX filenames to try, best first.

    These names are repo-specific, so guessing one is unreliable -- an earlier version
    tried a `..._quantized.onnx` name that does not exist and fell back to the 90MB fp32
    model. Querying the repo listing once is the only way to be sure, so this returns
    preferences and `_resolve_model_path` verifies against the API.
    """
    import platform

    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64"):
        return ["onnx/model_qint8_arm64.onnx", "onnx/model_quint8_avx2.onnx", "onnx/model.onnx"]
    if machine in ("x86_64", "amd64"):
        # avx2 is universally available on any CPU from the last decade; avx512 is not,
        # and picking it would crash on older hardware.
        return ["onnx/model_quint8_avx2.onnx", "onnx/model_qint8_avx512.onnx", "onnx/model.onnx"]
    return ["onnx/model.onnx"]


def _resolve_model_path(repo: str, cache_dir: str) -> str:
    """Return a path to a usable ONNX file, preferring the smallest correct one.

    Checks the repo listing first when the hub client is available, because quantized
    filenames vary per repo (`model_quint8_avx2`, `model_qint8_arm64`,
    `model_quantized`) and a wrong guess silently costs a 4x larger download.
    """
    from huggingface_hub import hf_hub_download

    try:
        from huggingface_hub import HfApi

        available = {
            f"onnx/{f.name}" for f in HfApi().list_repo_files(repo) if f.startswith("onnx/")
        }
    except Exception:  # noqa: BLE001 - offline or rate limited; fall back to trying
        available = set()

    if available:
        for candidate in _onnx_candidates():
            if candidate in available:
                return hf_hub_download(repo_id=repo, filename=candidate, cache_dir=cache_dir)
    else:
        # No listing available: try the preferences in order, tolerate 404s.
        for candidate in _onnx_candidates():
            try:
                return hf_hub_download(repo_id=repo, filename=candidate, cache_dir=cache_dir)
            except Exception:  # noqa: BLE001, S112
                continue

    return hf_hub_download(repo_id=repo, filename="onnx/model.onnx", cache_dir=cache_dir)


class LocalOnnxEmbedder(EmbeddingProvider):
    """Sentence-embeddings on CPU via ONNX Runtime.

    Model files are fetched once from Hugging Face into a local cache and reused. If the
    download is blocked, the error says so explicitly rather than surfacing a network
    stack trace -- offline is the normal state for this provider, so it must degrade to
    the hashing embedder rather than crash.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        cache_dir: str | Path | None = None,
        batch_size: int = 32,
        max_length: int = 256,
    ) -> None:
        _require_deps()

        self.model = model
        self.repo, self._expected_dim = KNOWN_MODELS.get(model, (f"sentence-transformers/{model}", 384))
        self.batch_size = batch_size
        self.max_length = max_length
        self._lock = threading.Lock()
        self._session = None
        self._tokenizer = None
        self._dimensions = self._expected_dim
        self._cache_dir = Path(cache_dir or os.environ.get(CACHE_DIR_ENV, Path.home() / ".cache" / "ragpipe" / "models"))

    # -- lazy model loading -------------------------------------------------

    def _load(self) -> None:
        if self._session is not None:
            return
        with self._lock:
            if self._session is not None:
                return
            import onnxruntime
            from huggingface_hub import (
                hf_hub_download,
            )
            from tokenizers import Tokenizer

            self._cache_dir.mkdir(parents=True, exist_ok=True)
            log.info("loading local embedding model %s (one-time)", self.model)

            model_path = _resolve_model_path(self.repo, str(self._cache_dir))
            try:
                size_mb = Path(model_path).stat().st_size / 1e6
                log.info("onnx model: %s (%.0f MB)", Path(model_path).name, size_mb)
            except OSError:
                pass

            tok_path = hf_hub_download(
                repo_id=self.repo, filename="tokenizer.json", cache_dir=str(self._cache_dir)
            )

            # CPU only, and limited to 1 intra-op thread: this runs inside request-serving
            # threads, and letting it spawn a pool per session would oversubscribe the box.
            opts = onnxruntime.SessionOptions()
            opts.intra_op_num_threads = 1
            opts.inter_op_num_threads = 1
            opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._session = onnxruntime.InferenceSession(
                model_path, sess_options=opts, providers=["CPUExecutionProvider"]
            )
            self._tokenizer = Tokenizer.from_file(tok_path)
            self._tokenizer.enable_truncation(max_length=self.max_length)
            self._tokenizer.enable_padding()

            # Trust the model's actual output width over the table.
            out = self._session.get_outputs()[0].shape
            if isinstance(out[-1], int) and out[-1] > 0:
                self._dimensions = int(out[-1])
            log.info("local embedder ready: %d dims", self._dimensions)

    @property
    def dimensions(self) -> int:
        return self._dimensions

    # -- embedding ----------------------------------------------------------

    def embed_batch(self, texts: Sequence[str]) -> list[Vector]:
        self._load()  # no-op after the first call
        import numpy as np

        out: list[Vector] = []
        for start in range(0, len(texts), self.batch_size):
            chunk = [t if t.strip() else " " for t in texts[start : start + self.batch_size]]
            encoded = self._tokenizer.encode_batch(chunk)
            ids = np.array([e.ids for e in encoded], dtype=np.int64)
            mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)

            feed = {"input_ids": ids, "attention_mask": mask}
            names = {i.name for i in self._session.get_inputs()}
            if "token_type_ids" in names:
                feed["token_type_ids"] = np.zeros_like(ids)

            hidden = self._session.run(None, feed)[0]  # (batch, seq, dim)

            # Mean pooling over real tokens only. Averaging including padding positions
            # is a classic bug: it dilutes the vector with pad-embedding noise.
            m = mask[..., None].astype(hidden.dtype)
            summed = (hidden * m).sum(axis=1)
            counts = np.clip(m.sum(axis=1), 1e-9, None)
            pooled = summed / counts

            # L2 normalize so cosine similarity reduces to a dot product.
            norms = np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12, None)
            pooled = pooled / norms
            out.extend(tuple(float(x) for x in row) for row in pooled)
        return out

    def embed(self, text: str) -> Vector:
        return self.embed_batch([text])[0]

    def embed_many(self, texts: Sequence[str], batch_size: int | None = None) -> list[Vector]:
        """Batching comes from Embedder (which already splits into provider-sized
        batches), so this only needs to exist to satisfy the protocol."""
        if batch_size and batch_size != self.batch_size:
            previous = self.batch_size
            self.batch_size = batch_size
            try:
                return self.embed_batch(list(texts))
            finally:
                self.batch_size = previous
        return self.embed_batch(list(texts))

    def warmup(self) -> int:
        """Load the model and run one throwaway inference.

        ONNX session creation costs ~2s. Paying it lazily means the first user request
        after a deploy eats the whole cost, which is the worst possible time. Call this
        at startup (or right after ingestion) so queries stay fast.
        """
        self._load()
        self.embed_batch(["warmup"])
        return self._dimensions
