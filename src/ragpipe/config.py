"""Central configuration. Every tunable lives here, loaded from env with sane defaults.

Design rule: never read os.environ deep inside the code. Thread a Settings object
through instead. That makes tests deterministic and swapping prod/dev configs trivial.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class EmbeddingConfig:
    """Embedding provider selection.

    provider="hashing"  -> zero deps, deterministic, offline, ~ok for dev/smoke tests.
    provider="openai"   -> production quality, needs OPENAI_API_KEY.
    """

    provider: str = field(default_factory=lambda: os.environ.get("RAG_EMBED_PROVIDER", "hashing"))
    model: str = field(default_factory=lambda: os.environ.get("RAG_EMBED_MODEL", "text-embedding-3-small"))
    dimensions: int = field(default_factory=lambda: _env_int("RAG_EMBED_DIM", 256))
    batch_size: int = field(default_factory=lambda: _env_int("RAG_EMBED_BATCH", 256))
    cache_size: int = field(default_factory=lambda: _env_int("RAG_EMBED_CACHE", 50_000))
    # Redact nothing by default; text is hashed for the cache key.
    normalize: bool = True
    # Which env var holds the key, and which optional base URL to route through.
    # Kept per-provider so the embedder and the LLM can point at different endpoints
    # (e.g. OpenAI embeddings, a self-hosted LLM) without fighting over one variable.
    api_key_env: str = "OPENAI_API_KEY"
    base_url_env: str = "OPENAI_BASE_URL"
    # --- provider="local" (ONNX, offline, no key) ---
    # Sentence-transformers ONNX model. Real semantic embeddings, which the hashing
    # provider cannot do; useful when your gateway serves chat but no embeddings.
    local_model: str = field(
        default_factory=lambda: os.environ.get("RAG_LOCAL_EMBED_MODEL", "all-MiniLM-L6-v2")
    )
    local_cache_dir: str = field(
        default_factory=lambda: os.environ.get("RAG_LOCAL_EMBED_CACHE", "")
    )


@dataclass(slots=True)
class ChunkingConfig:
    strategy: str = field(default_factory=lambda: os.environ.get("RAG_CHUNK_STRATEGY", "recursive"))
    target_tokens: int = field(default_factory=lambda: _env_int("RAG_CHUNK_TARGET", 320))
    overlap_tokens: int = field(default_factory=lambda: _env_int("RAG_CHUNK_OVERLAP", 64))
    hard_max_tokens: int = field(default_factory=lambda: _env_int("RAG_CHUNK_MAX", 512))
    min_chunk_tokens: int = field(default_factory=lambda: _env_int("RAG_CHUNK_MIN", 24))


@dataclass(slots=True)
class VectorStoreConfig:
    backend: str = "auto"  # auto | numpy | memory
    persist_path: str = "data/index"
    # Cap how many vectors we keep resident (LRU over the persisted store).
    # Keeps memory flat as the corpus grows.
    memory_budget_vectors: int = field(default_factory=lambda: _env_int("RAG_MEM_BUDGET", 200_000))


@dataclass(slots=True)
class RetrievalConfig:
    top_k: int = field(default_factory=lambda: _env_int("RAG_TOP_K", 8))
    # Candidates pulled from each retriever before fusion.
    candidate_k: int = field(default_factory=lambda: _env_int("RAG_CANDIDATE_K", 40))
    # Reciprocal Rank Fusion constant. 60 is the value from the original RRF paper.
    rrf_k: int = field(default_factory=lambda: _env_int("RAG_RRF_K", 60))
    # Relative leg weights. sparse_weight is derived so the pair always sums to 1 --
    # letting them drift apart makes fused scores uninterpretable.
    dense_weight: float = field(default_factory=lambda: _env_float("RAG_DENSE_WEIGHT", 0.6))
    rerank: bool = _env_bool("RAG_RERANK", True)
    # Drop chunks whose fused score is far below the top score (relative floor).
    score_floor_ratio: float = field(default_factory=lambda: _env_float("RAG_SCORE_FLOOR", 0.35))
    mmr_lambda: float = field(default_factory=lambda: _env_float("RAG_MMR_LAMBDA", 0.7))
    # 1.0 = pure relevance, 0.0 = max diversity

    @property
    def sparse_weight(self) -> float:
        return 1.0 - self.dense_weight


@dataclass(slots=True)
class GenerationConfig:
    provider: str = field(default_factory=lambda: os.environ.get("RAG_LLM_PROVIDER", "extractive"))
    model: str = field(default_factory=lambda: os.environ.get("RAG_LLM_MODEL", "gpt-4o-mini"))
    temperature: float = 0.0
    # Grounded answers are short. 700 is both slower and worse: it invites padding, and
    # on slow gateways the tail latency is dominated by tokens you will never read.
    max_tokens: int = field(default_factory=lambda: _env_int("RAG_LLM_MAX_TOKENS", 300))
    # Refuse to answer from context alone; better to say "not in context".
    require_citations: bool = True
    api_key_env: str = "OPENAI_API_KEY"
    base_url_env: str = "OPENAI_BASE_URL"


@dataclass(slots=True)
class CacheConfig:
    # LLM response cache: identical (prompt, temp, model) -> same answer. Huge in prod.
    enabled: bool = _env_bool("RAG_LLM_CACHE", True)
    max_entries: int = field(default_factory=lambda: _env_int("RAG_LLM_CACHE_SIZE", 2_000))
    semantic_answers: bool = True


@dataclass(slots=True)
class ObservabilityConfig:
    log_level: str = field(default_factory=lambda: os.environ.get("RAG_LOG_LEVEL", "INFO"))
    log_json: bool = _env_bool("RAG_LOG_JSON", False)
    trace_sink: str = "stdout"  # stdout | file | none
    trace_path: str = "data/traces.jsonl"


@dataclass(slots=True)
class Settings:
    app_name: str = "rag-pipeline"
    data_dir: str = field(default_factory=lambda: os.environ.get("RAG_DATA_DIR", "data"))
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)

    @property
    def data_path(self) -> Path:
        p = Path(self.data_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def __post_init__(self) -> None:
        """Validate settings at construction time so misconfigurations fail fast
        rather than at query time with an opaque error."""
        if self.embedding.dimensions <= 0:
            raise ValueError(f"RAG_EMBED_DIM must be positive, got {self.embedding.dimensions}")
        if self.embedding.batch_size <= 0:
            raise ValueError(f"RAG_EMBED_BATCH must be positive, got {self.embedding.batch_size}")
        if self.embedding.cache_size <= 0:
            raise ValueError(f"RAG_EMBED_CACHE must be positive, got {self.embedding.cache_size}")
        if self.retrieval.top_k <= 0:
            raise ValueError(f"RAG_TOP_K must be positive, got {self.retrieval.top_k}")
        if self.retrieval.candidate_k <= 0:
            raise ValueError(f"RAG_CANDIDATE_K must be positive, got {self.retrieval.candidate_k}")
        if not 0.0 <= self.retrieval.dense_weight <= 1.0:
            raise ValueError(
                f"RAG_DENSE_WEIGHT must be in [0, 1], got {self.retrieval.dense_weight}"
            )
        if not 0.0 <= self.retrieval.mmr_lambda <= 1.0:
            raise ValueError(
                f"RAG_MMR_LAMBDA must be in [0, 1], got {self.retrieval.mmr_lambda}"
            )
        if not 0.0 <= self.retrieval.score_floor_ratio <= 1.0:
            raise ValueError(
                f"RAG_SCORE_FLOOR must be in [0, 1], got {self.retrieval.score_floor_ratio}"
            )
        if self.generation.max_tokens <= 0:
            raise ValueError(f"RAG_LLM_MAX_TOKENS must be positive, got {self.generation.max_tokens}")
        if self.chunking.target_tokens <= 0:
            raise ValueError(f"RAG_CHUNK_TARGET must be positive, got {self.chunking.target_tokens}")
        if self.chunking.overlap_tokens < 0:
            raise ValueError(
                f"RAG_CHUNK_OVERLAP must be non-negative, got {self.chunking.overlap_tokens}"
            )
        if self.chunking.hard_max_tokens <= 0:
            raise ValueError(f"RAG_CHUNK_MAX must be positive, got {self.chunking.hard_max_tokens}")
        if self.chunking.min_chunk_tokens <= 0:
            raise ValueError(f"RAG_CHUNK_MIN must be positive, got {self.chunking.min_chunk_tokens}")
        if self.chunking.overlap_tokens >= self.chunking.target_tokens:
            raise ValueError(
                f"RAG_CHUNK_OVERLAP ({self.chunking.overlap_tokens}) must be less than "
                f"RAG_CHUNK_TARGET ({self.chunking.target_tokens})"
            )
        if self.store.memory_budget_vectors <= 0:
            raise ValueError(
                f"RAG_MEM_BUDGET must be positive, got {self.store.memory_budget_vectors}"
            )
        if self.store.backend not in ("auto", "numpy", "memory", "faiss"):
            raise ValueError(
                f"RAG_STORE_BACKEND must be one of: auto, numpy, memory, faiss. "
                f"Got {self.store.backend!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_settings(**overrides: Any) -> Settings:
    """Build settings from env, then apply explicit overrides (tests use this)."""
    s = Settings()
    for key, value in overrides.items():
        if not hasattr(s, key):
            raise ValueError(f"Unknown setting: {key}")
        setattr(s, key, value)
    return s
