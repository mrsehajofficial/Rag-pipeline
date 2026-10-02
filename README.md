# RAG Pipeline

A production-shaped retrieval-augmented generation pipeline in Python. Hybrid retrieval,
score fusion, reranking, grounded generation, and a retrieval eval harness.

**Start here: [USER_GUIDE.md](USER_GUIDE.md)** — task-oriented, with copy-pasteable
commands for asking questions, indexing your own docs, running the API, and tuning
quality. This README is the reference: architecture, benchmarks, and design rationale.

**Zero dependencies by default.** The core runs on a bare Python 3.10+ install. `numpy`,
`openai`, and `onnxruntime` are optional and switch on automatically when present.

```bash
cd "Project/Rag pipeline"
./demo.sh
```

## Run it

```bash
cd "Project/Rag pipeline"

python3 run.py ingest data/docs        # index documents
python3 run.py ask "how long do API keys live?"
python3 run.py eval data/eval_cases.jsonl
python3 run.py serve --port 8000
```

`run.py` needs nothing — no `PYTHONPATH`, no install step. It works on a bare Python
3.10+, and numpy/openai are picked up automatically if present.

For real models, use a venv (keeps the packages off your system Python):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-accelerated.txt     # numpy + openai
export OPENAI_API_KEY="sk-..."
python3 run.py doctor                            # verify
```

Or install it properly and use the `rag` command anywhere:

```bash
pip install -e .
rag ingest data/docs
```

---

## Why it's built this way

Three decisions drive most of the design:

**1. Hybrid retrieval, fused on rank.** Dense embeddings catch paraphrases ("how do I
stop it crashing" → "unhandled exception"). BM25 catches exact tokens — error codes,
SKUs, names — that embeddings blur away. Reciprocal Rank Fusion merges them because
BM25 scores (~12.4) and cosine scores (~0.83) are not commensurable, and any
min-max normalization is wrecked by a single outlier. Rank is comparable across both.

**2. Fuse first, rerank after.** Rerank-then-fuse gives the lexical leg the final say and
undoes the semantic leg's best work. Fusing first gives reranker #2 veto power over the
fused order, which is the better failure mode.

**3. Ground or refuse.** The prompt forbids answering from parametric knowledge. An
answer with no supporting chunk is worse than an honest "not in the indexed sources",
because the user cannot tell the difference.

---

## Architecture

```
query
  ├── embed query ──────────┐
  │                        ├──► RRF fusion ──► rerank ──► MMR ──► score floor
  └── BM25 scan ───────────┘                                            │
                                                                        ▼
                                                            prompt + grounded answer
```

| Layer | Module | Responsibility |
|---|---|---|
| Config | `config.py` | Every tunable, env-overridable. Nothing reads `os.environ` downstream. |
| Observability | `observability.py` | Structured logs, per-query traces, per-stage timing. |
| Cache | `cache.py` | Thread-safe bounded LRU + stable cross-process hashing. |
| Embeddings | `embedding.py` | Provider abstraction, disk cache, batching, parallel fetch. |
| Ingest | `ingest/` | Loaders (md/txt/html/json/jsonl/csv/py) + token-aware chunker. |
| Store | `store/vector_store.py` | Exact inner-product search, numpy fast path, binary persistence. |
| Retrieval | `retrieval/` | BM25, RRF, reranker, MMR. |
| Generation | `generation/` | Prompts, OpenAI client, offline extractive client, response cache. |
| Eval | `evaluation.py` | Recall@k, precision@k, MRR, nDCG@k, miss reporting. |
| Interface | `pipeline.py`, `api.py`, `cli.py` | Orchestrator, HTTP service, CLI. |

---

## Usage

```bash
export PYTHONPATH=src

python3 -m ragpipe.cli ingest data/docs
python3 -m ragpipe.cli ask "how do we roll back a release?"
python3 -m ragpipe.cli eval data/eval_cases.jsonl
python3 -m ragpipe.cli serve --port 8000
```

Equivalent forms — `run.py` (no setup) or the installed `rag` command:

```bash
python3 run.py ingest data/docs
rag ingest data/docs
```

```python
from ragpipe import RAGPipeline

pipe = RAGPipeline()
pipe.index_path("./docs")
pipe.save()

response = pipe.query("what caused the export worker to crash?")
print(response.answer)
for c in response.citations:
    print(f"[{c.index}] {c.source} ({c.score:.3f}, {'+'.join(c.retrieved_by)})")
```

### HTTP

```bash
curl -s localhost:8000/query -H 'Content-Type: application/json' \
  -d '{"question": "how long do API keys live?"}'
```

---

## OpenAI setup

The defaults are fully offline (`hashing` embeddings + `extractive` answers) so the
pipeline runs with no key and no spend. To use real models:

```bash
pip install openai
export OPENAI_API_KEY="sk-..."

# Embeddings
export RAG_EMBED_PROVIDER=openai
export RAG_EMBED_MODEL=text-embedding-3-small
export RAG_EMBED_DIM=1536          # MUST match the model (see table below)

# Generation
export RAG_LLM_PROVIDER=openai
export RAG_LLM_MODEL=gpt-4o-mini
```

Verify before indexing anything:

```bash
python3 -m ragpipe.cli doctor
```

`doctor` checks the SDK, resolves credentials, calls `/models`, and tells you whether
your configured model names are actually advertised by the endpoint.

### Base URL / custom endpoints

Set `OPENAI_BASE_URL` for any OpenAI-compatible API (LiteLLM, vLLM, Ollama, OpenRouter,
Groq, Together, Azure gateways, Hugging Face Inference):

```bash
export OPENAI_API_KEY="your-gateway-key"
export OPENAI_BASE_URL="https://api.myprovider.com/v1"
```

**The `/v1` suffix is optional.** It's normalized for you — `https://api.myprovider.com`,
a trailing slash, and an existing `/v1` all resolve to the same correct URL. This is
deliberate, because a missing `/v1` is the single most common cause of a base URL that
"connects but always 404s".

Point the embedder and the LLM at *different* endpoints (common: OpenAI embeddings plus
a self-hosted LLM) by overriding which env var each reads:

```python
from ragpipe.config import EmbeddingConfig, GenerationConfig, load_settings

settings = load_settings()
settings.embedding.api_key_env = "OPENAI_API_KEY"      # embeddings -> OpenAI
settings.embedding.base_url_env = "OPENAI_BASE_URL"
settings.generation.api_key_env = "VLLM_API_KEY"       # LLM -> local vLLM
settings.generation.base_url_env = "VLLM_BASE_URL"
```

### If your gateway has no embedding model

Very common: OpenAI-compatible gateways often serve **chat models only**. `doctor` will
tell you:

```
  connectivity  OK (10 models advertised)
    embed text-embedding-3-small             MISSING
    llm   gemini-3.8-flash                   found
```

A missing embed model is not an error to work around — it means embeddings come from
somewhere else. Use the **local** provider: a real sentence-transformer running on CPU
via ONNX. No API, no key, no cost, works offline forever.

> **The one export people miss.** If you see
> `POST .../v1/embeddings → 404 Not Found`, `RAG_EMBED_PROVIDER` is still `openai`.
> Run `export RAG_EMBED_PROVIDER=local`. Also unset `RAG_EMBED_DIM` — the local provider
> reads the true width from the model (384) and ignores it.

```bash
export RAG_EMBED_PROVIDER=local        # <- this is the one that matters
unset RAG_EMBED_DIM RAG_EMBED_MODEL    # ignored by the local provider

export RAG_LLM_PROVIDER=openai         # chat still goes to your gateway
export RAG_LLM_MODEL=gemini-3.8-flash
export OPENAI_API_KEY="..."
export OPENAI_BASE_URL="https://morrowtest.wasmer.app/v1"

python3 run.py doctor
python3 run.py ingest data/docs
```

Confirming it worked — the log line is the check:

```
vector store backend=numpy dims=384        <- 384 = local. If it says 1536, still openai.
embedding 3 chunks (3 raw, 0 dupes)
```

Or just `./setup_local.sh`, which wires it up and verifies end to end.

Or run **everything** local, with no gateway at all:

```bash
export RAG_EMBED_PROVIDER=local RAG_LLM_PROVIDER=extractive
```

**Local models available** (set `RAG_LOCAL_EMBED_MODEL`):

| Model | Dims | Notes |
|---|---|---|
| `all-MiniLM-L6-v2` | 384 | **Default.** Fast, small, good general quality |
| `bge-small-en-v1.5` | 384 | Stronger on retrieval benchmarks |
| `bge-base-en-v1.5` | 768 | Better still, ~2× slower |
| `all-MiniLM-L12-v2` | 384 | Slightly better, slower |

Measured on the sample corpus, local vs the built-in hashing fallback:

| Query | local | hashing |
|---|---|---|
| rollback paraphrase | **+0.49** | +0.03 |
| crash, indirect wording | **+0.60** | +0.24 |
| token validation, reworded | **+0.55** | +0.46 |
| "credentials" → "API keys" | **+0.52** | +0.11 |

The last row is the one that matters: hashing has no idea credentials and API keys are
the same thing. That is the difference between matching words and matching meaning.

Run `python3 tests/test_semantic_quality.py` to reproduce these numbers.

## Latency: where the time actually goes

Measured on this setup (local embeddings + Gemini via the Wasmer gateway):

| Stage | Time | Note |
|---|---|---|
| Retrieve (embed + search + fuse + rerank) | **~6–25 ms** | the whole RAG part |
| Generate (your gateway) | **4–27 s** | highly variable, entirely upstream |

To confirm the split, a bare `curl` to the gateway with a three-word prompt took 14.8s,
then 4.5s, then 4.0s. Repeated identical queries through the pipeline ranged from 6.6s
to 27.1s. **That variance is the gateway, not this code.**

Three levers, in order of impact:

1. **The response cache (built in, on by default).** Answers are cached in
   `data/llm_cache.db`, keyed by the full request, and shared across processes. Repeat
   queries: **5,689 ms → 10.5 ms**. This is the big one — a CLI invocation is a fresh
   process every time, so an in-memory cache could never help. Disable with
   `RAG_LLM_CACHE=false`.
2. **A faster model.** `gemini-flash-lite` and `gemini-3.5-flash-lite` are on your
   endpoint. For grounded extraction, flash-lite is usually fine.
3. **Streaming** (`stream=True`) so users see the first tokens immediately instead of
   waiting out the full latency.

`RAG_EMBED_DIM` has to match the model, or every search silently returns nonsense. The
default is 256 for the offline hashing provider; the real models are:

| Model | Dimensions | `RAG_EMBED_DIM` |
|---|---|---|
| `text-embedding-3-small` | 1536 | `1536` |
| `text-embedding-3-large` | 3072 | `3072` |
| `text-embedding-3-*` (truncated) | any ≤ model max | e.g. `512` to save space |

Changing dimensions invalidates the index. The store raises a clear error on load rather
than returning wrong results — rebuild it with `rag ingest`.

### Common failures

| Symptom | Cause | Fix |
|---|---|---|
| `No module named openai` | SDK not installed | `pip install openai` |
| 404 on every request | missing `/v1` | already normalized; verify the URL |
| `model_not_found` | wrong model name for the endpoint | `rag doctor` lists what it serves |
| All results are garbage | dimension mismatch | set `RAG_EMBED_DIM`, re-ingest |
| Answers look like stitched sentences | key missing, silently fell back | check `OPENAI_API_KEY`; `rag doctor` |
| 401 | wrong key for that base URL | keys are per-endpoint, not interchangeable |



All tunables are environment variables; every one has a working default.

| Variable | Default | Effect |
|---|---|---|
| `OPENAI_API_KEY` | — | Required for any `openai` provider |
| `OPENAI_BASE_URL` | official | Any OpenAI-compatible endpoint; `/v1` optional |
| `RAG_EMBED_PROVIDER` | `hashing` | `local` (ONNX, offline) · `openai` · `hashing` |
| `RAG_LOCAL_EMBED_MODEL` | `all-MiniLM-L6-v2` | Only for `RAG_EMBED_PROVIDER=local` |
| `RAG_EMBED_MODEL` | `text-embedding-3-small` | Embedding model |
| `RAG_EMBED_DIM` | `256` | Vector width |
| `RAG_TOP_K` | `8` | Chunks handed to the LLM |
| `RAG_CANDIDATE_K` | `40` | Candidates per retriever before fusion |
| `RAG_RERANK` | `true` | Enable the reranker |
| `RAG_LLM_PROVIDER` | `extractive` | `extractive` (offline) or `openai` |
| `RAG_LLM_CACHE` | `true` | Cache identical LLM calls |
| `RAG_LOG_LEVEL` | `INFO` | Log verbosity |
| `RAG_LOG_JSON` | `false` | JSON logs for log shippers |

For real quality, set `RAG_EMBED_PROVIDER=openai` and `RAG_LLM_PROVIDER=openai` with
`OPENAI_API_KEY`. The defaults are deliberately offline so the pipeline runs and is
testable with no key and no spend.

---

## Performance

Measured on this machine, 4,994 chunks, 60 queries, `top_k=8`, caches warm
(`benchmarks/bench.py`):

| Backend | mean | p50 | p95 | p99 |
|---|---|---|---|---|
| pure Python | 61.9 ms | 61.9 ms | 63.6 ms | 64.3 ms |
| + numpy | 23.6 ms | 23.4 ms | 24.4 ms | 27.7 ms |

**4.1× from one dependency**, and 4.1× again from profiling. Two changes did the work,
both found by measurement rather than guesswork:

- `math.fsum(gen)` → `sum(map(mul, q, r))` for the dot product: **77 ms → 36.6 ms** at
  5k chunks. `fsum` is a Python-level generator that yields per element; `map`+`sum` are
  both C-level, so the interpreter never touches an element. The ~1e-7 accuracy
  difference is irrelevant next to the score gaps being ranked.
- Skipping the fancy-index copy in the unfiltered numpy path: the old code built
  `arange(C)` and copied every row on every query, which cost more than the matmul it
  preceded.

Things that sound fast but measured *slower* and were not kept: a column-major
(transposed) corpus layout (87 ms vs 36.6 ms), and `array('f')` storage for the scan
(58 ms vs 36.6 ms — indexing a `array` boxes each element back into a float).

What dominates where, at 5k chunks: dense scan ~85%, BM25 ~7%, tokenization ~7%.

Scaling notes: the exact scan is O(n) and comfortable into the low hundreds of thousands
of chunks. Past ~500k, swap `VectorStore` for `faiss-cpu`/HNSWlib, or Qdrant/pgvector when
you need multi-node. The interface is deliberately small so that is a one-file change.
Filtering is currently post-filter (correct, simple); push predicates into the index if
you need selective filters at scale.

---

## Evaluation

You cannot tune what you cannot measure. `data/eval_cases.jsonl` holds 11 labelled
queries; `rag eval` reports:

```
=== eval_cases (11 queries) ===
  hit_rate      1.000
  recall@k      1.000
  precision@k   0.125
  MRR           1.000
  nDCG@k        1.000
```

`precision@k` is 0.125 for the right reason: one relevant document among `k=8` slots,
against a 3-document corpus. It is the expected value, not a defect.

Note this is measured with the **hashing** embedder, not a real model — it validates the
plumbing, not semantic quality. The eval harness is the tool you use to prove a real
embedding model is an improvement.

---

## Tests

```bash
python3 tests/test_core.py       # 19 tests: cache, chunker, loaders, BM25, fusion, metrics
python3 tests/test_pipeline.py   # 18 tests: ingest, persistence, query, eval
```

Both run under pytest too, and both pass with and without numpy. Two of these tests exist
because they caught real bugs during development:

- `test_vector_store_search_returns_ordered_scores` — the numpy fast path returned
  `(index, score)` where the code expected `(score, index)`, silently ranking every
  result against the wrong chunk. It raised on neither backend; only comparing the two
  implementations against each other exposed it.
- `test_loader_reads_markdown_and_strips_front_matter` — an explicit `title:` in front
  matter was being overwritten by the first H1 scraped from the body.

---

## Design decisions worth knowing

**Chunk ids are content hashes.** Re-ingesting an unchanged corpus is idempotent, which
matters because scheduled ingestion jobs re-run. Upsert semantics in the store, not
append.

**Persistence is JSON + a raw float32 blob.** Text stays debuggable, vectors avoid
per-row JSON parsing, and the format loads from any language. No pickle.

**The lexical index is rebuilt on load** rather than persisted alongside the chunks. A
second source of truth that can drift from the chunk list is a data bug waiting to
happen; rebuilding costs milliseconds.

**Every LRU cache is bounded.** An unbounded dict in a long-running service is a slow
OOM. Cache keys use BLAKE2b, not Python's `hash()`, which is salted per-process and
therefore useless for a persistent cache.

**Bounded request bodies** on the API (1 MB). An unbounded read is a free DoS.

---

## Scaling past this

The honest limits, in the order you will hit them:

1. **~500k chunks** — the exact dense scan becomes the bottleneck. Move to faiss/HNSW,
   accepting some recall loss, and re-run `rag eval` to measure exactly how much.
2. **Concurrent writers** — `VectorStore` is process-local. Multi-process indexing needs
   Qdrant/pgvector or a write queue.
3. **Cross-encoder reranking** — the built-in reranker is lexical, not a real
   cross-encoder. Swap in Cohere/Jina/BGE rerank; the interface is one method.
4. **Query rewriting** — the pipeline diagram reserves a `[rewrite?]` slot. Multi-hop
   questions ("compare the rollback policy in A vs B") need it.
