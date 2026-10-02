# User Guide

Task-oriented. Every snippet is copy-pasteable from the project root.

Your setup is already working (local embeddings + Gemini via your Wasmer gateway), so
skip the setup section and start at [Asking questions](#asking-questions).

---

## Contents

- [Asking questions](#asking-questions)
- [Using your own documents](#using-your-own-documents)
- [Running as an API server](#running-as-an-api-server)
- [Using it from Python](#using-it-from-python)
- [Tuning answer quality](#tuning-answer-quality)
- [Checking and debugging](#checking-and-debugging)
- [Command reference](#command-reference)
- [When something goes wrong](#when-something-goes-wrong)
- [Setup (only if you need it)](#setup-only-if-you-need-it)

---

## Asking questions

```bash
source .venv/bin/activate          # every new shell

python3 run.py ask "how do we roll back a release?"
```

Output shows the answer, which document it came from, and how long it took:

```
Q: how do we roll back a release?

A: To roll back, redeploy the previous image tag [1].

Sources:
  [1] data/docs/deployment.md (score 0.658, via dense+sparse)

[32.1ms total | trace cd53cbbab937]
```

| Part | What it means |
|---|---|
| `[1]` | Citation. Ties the claim back to that source. |
| `score` | Relevance, 0–1. |
| `via dense+sparse` | Both retrievers found it — a strong signal. |
| `[32.1ms total]` | If this is single-digit, the answer came from cache. |

**Machine-readable output:**

```bash
python3 run.py ask "how do we roll back?" --json
```

**Fewer or more sources:**

```bash
python3 run.py ask "..." --top-k 3      # tighter, less noise
python3 run.py ask "..." --top-k 15     # broader, slower
```

**Save the setup in a file** so you don't retype exports every session:

```bash
cat > .env <<'EOF'
export RAG_EMBED_PROVIDER=local
export RAG_LLM_PROVIDER=openai
export RAG_LLM_MODEL=gemini-3.8-flash
export OPENAI_API_KEY="your-api-key-here"
export OPENAI_BASE_URL="https://your-gateway.example.com/v1"
EOF

echo 'source .env' >> ~/.zshrc     # or ~/.bashrc
```

---

## Using your own documents

**Supported:** `.md` `.txt` `.rst` `.html` `.json` `.jsonl` `.csv` `.py`

```bash
python3 run.py ingest ~/my-notes          # one folder
python3 run.py ingest ./faq.md             # one file
python3 run.py ingest ./docs --no-recursive
```

Subfolders are included automatically. The index is saved, so you only ingest once.

**Re-indexing is safe** — chunks are content-hashed, so unchanged files are skipped and
changed files are updated. Nothing duplicates.

**Remove a document** (Python — no CLI command yet):

```python
from ragpipe import RAGPipeline
p = RAGPipeline(); p.load("data/index")
p.delete_document("<doc_id>")     # doc_id comes from the chunks
p.save()
```

**Supported document shapes**, so your content retrieves well:

- Markdown with `#` headings (headings become metadata and boundaries)
- YAML front matter (`title:`, `owner:`) — preserved as metadata
- Tables and prose. A chunk of ~320 tokens is the sweet spot.

**Not supported yet:** PDF, DOCX, web pages, Notion exports. Convert to Markdown first
(`pandoc`, `markitdown`, or any HTML→MD converter), then ingest.

---

## Running as an API server

```bash
python3 run.py serve --port 8000
```

In another terminal:

```bash
# Ask a question
curl -s localhost:8000/query -H 'Content-Type: application/json' \
  -d '{"question": "how do we roll back a release?"}'

# Health check
curl -s localhost:8000/health

# Index statistics
curl -s localhost:8000/stats

# Index new documents
curl -s localhost:8000/ingest -H 'Content-Type: application/json' \
  -d '{"path": "./new-docs"}'
```

**Ingest a document over the API:**

```jsonc
// POST /query
{
  "question": "how do we roll back a release?",
  "top_k": 8,              // optional, default 8
  "filters": {"owner": "platform-team"}   // optional metadata filter
}
```

**From Python (no server):**

```python
import requests

resp = requests.post("http://localhost:8000/query", json={
    "question": "how do we roll back a release?"
}).json()

print(resp["answer"])
for c in resp["citations"]:
    print(f"  [{c['index']}] {c['source']} (score {c['score']})")
```

**Production:** the server is stdlib `http.server`, single index held in memory. Fine for
one process. For multiple workers you need a shared vector store (Qdrant/pgvector) —
see [Scaling](#scaling-past-this) in the README.

---

## Using it from Python

```python
from ragpipe import RAGPipeline

pipe = RAGPipeline()
pipe.load("data/index")                    # reuse the built index

response = pipe.query("what caused the crash?")

print(response.answer)
print(response.confident)                  # False if the answer isn't supported
print(response.total_ms)

for c in response.citations:
    print(f"[{c.index}] {c.source}  score={c.score:.3f}  via={'+'.join(c.retrieved_by)}")
```

**Streaming** (first tokens appear immediately — helps most against a slow gateway):

```python
for piece in pipe.query("what caused the crash?", stream=True):
    print(piece, end="", flush=True)
```

**Build an index in code:**

```python
from pathlib import Path
from ragpipe import RAGPipeline

pipe = RAGPipeline()
result = pipe.index_path("./docs")
print(result)                 # {'documents': 3, 'chunks': 5, 'embedded': 5, ...}
pipe.save()
```

**Custom settings:**

```python
from ragpipe import load_settings

s = load_settings()
s.retrieval.top_k = 12
s.retrieval.rerank = True
s.observability.log_level = "DEBUG"

pipe = RAGPipeline(s)
```

---

## Tuning answer quality

| Symptom | Setting | Value |
|---|---|---|
| Misses relevant docs | more candidates | `export RAG_CANDIDATE_K=80` |
| Too many irrelevant sources | fewer | `export RAG_TOP_K=4` |
| All sources say the same thing | more diversity | `export RAG_MMR_LAMBDA=0.5` |
| Needs exact words/names (BM25 side) | weight the sparse leg | `export RAG_DENSE_WEIGHT=0.4` |
| Needs paraphrases (dense side) | weight the dense leg | `export RAG_DENSE_WEIGHT=0.8` |
| Too many weak sources | stricter floor | `export RAG_SCORE_FLOOR=0.5` |
| Chunks too big/small | adjust chunking | `export RAG_CHUNK_TARGET=200` |
| Faster answers | cheaper model | `export RAG_LLM_MODEL=gemini-flash-lite` |
| Refusing to answer | more context | `export RAG_TOP_K=15` |

> Settings not wired to env vars are editable in `src/ragpipe/config.py`.

**How to know if a change helped** — always measure, never guess:

```bash
python3 run.py eval data/eval_cases.jsonl
```

Add your own test cases as you go (`data/eval_cases.jsonl`, one JSON object per line):

```jsonl
{"query": "how do I reset my password?", "relevant": ["account.md"], "name": "password reset"}
```

Aim for `hit_rate` above 0.9. Below 0.7 means retrieval is missing documents — fix that
before touching the prompt or the LLM.

---

## Checking and debugging

```bash
python3 run.py doctor     # setup problems: key, model, connectivity
python3 run.py stats      # what is indexed
```

**What `stats` tells you:**

```json
{
  "chunks": 3,                        "dimensions": 384,        "embed_provider": "local",
  "vocab": 257,                       "embed_model": "local:all-MiniLM-L6-v2",
  "llm_provider": "openai",           "llm_model": "gemini-3.8-flash",
  "store_backend": "numpy"
}
```

`dimensions: 384` = local embeddings working. `1536` means the provider is still `openai`.

**See where time goes** on any query:

```bash
export RAG_LOG_LEVEL=INFO
python3 run.py ask "how do we roll back?"
```

```
trace {"spans": [
  {"name": "retrieve", "ms": 6.0}, {"name": "embed_query", "ms": 1.8},
  {"name": "vector_search", "ms": 0.4}, {"name": "fusion", "ms": 0.1},
  {"name": "rerank", "ms": 0.9}, {"name": "mmr", "ms": 0.3},
  {"name": "generate", "ms": 7111.7}
]}
```

`generate` dominating means it's your gateway, not retrieval.

---

## Command reference

| Command | What it does |
|---|---|
| `ingest <path>` | Index a file or folder |
| `ask "<question>"` | Ask a question |
| `serve` | Start the HTTP API |
| `eval <cases.jsonl>` | Measure retrieval quality |
| `stats` | Show index info |
| `doctor` | Check keys, models, connectivity |

**Useful flags:**

| Flag | Where | Effect |
|---|---|---|
| `--top-k N` | `ask`, `eval` | Number of sources |
| `--json` | `ask`, `eval` | Machine-readable output |
| `--port N` | `serve` | Port (default 8000) |
| `--index-path P` | all | Where the index lives |
| `--no-load` | all | Start empty (use after `ingest`) |

---

## When something goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| `POST /v1/embeddings → 404` | Gateway has no embedding model | `export RAG_EMBED_PROVIDER=local` |
| `dims=1536` in logs | Embeddings still going to the gateway | Same as above |
| `ModuleNotFoundError: ragpipe` | Ran `python3 -m ragpipe.cli` | Use `python3 run.py ...` |
| Answers are stitched sentences | No API key; silently fell back | `python3 run.py doctor` |
| Answers cite the wrong document | Stale index after editing docs | `python3 run.py ingest ./docs` |
| "I don't have that in the indexed sources" | Genuinely not retrieved | Raise `RAG_TOP_K` / `RAG_CANDIDATE_K` |
| `dimension mismatch` on load | Changed embedding provider | `rm -rf data/index` then re-ingest |
| Every query takes 5+ seconds | Slow gateway | Reuse the same question (cache), or a faster model |
| `pip: command not found` | No pip on system Python | `python3 -m venv .venv && source .venv/bin/activate` |

**The answer is wrong but the source is right** — the retrieval worked, generation didn't.
Try a faster/better model, or raise `RAG_TOP_K` so the model sees more context.

---

## Setup (only if you need it)

Already done? Check with `python3 run.py doctor`.

```bash
cd "Project/Rag pipeline"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt        # local embeddings + gateway + numpy

export RAG_EMBED_PROVIDER=local        # no gateway needed for embeddings
export RAG_LLM_PROVIDER=openai
export RAG_LLM_MODEL=gemini-3.8-flash
export OPENAI_API_KEY="your-api-key-here"
export OPENAI_BASE_URL="https://your-gateway.example.com/v1"

python3 run.py doctor
python3 run.py ingest data/docs
```

Or run everything offline, no gateway at all:

```bash
export RAG_EMBED_PROVIDER=local RAG_LLM_PROVIDER=extractive
```

---

## Where things live

```
run.py                 entry point — use this, no PYTHONPATH needed
data/docs/             sample documents
data/index/            built index (delete it to force a rebuild)
data/eval_cases.jsonl  your test cases
data/llm_cache.db      cached answers
setup_local.sh         one-shot setup + verification
demo.sh                full tour

src/ragpipe/
  pipeline.py          the orchestrator — read this first
  embedding.py         provider layer (local / openai / hashing)
  embedding_local.py   ONNX local embeddings
  retrieval/           BM25, RRF fusion, reranker, MMR
  store/               vector store
  generation/          prompts, LLM clients, disk cache
  config.py            every tunable
```

The README has architecture rationale, benchmarks, and scaling limits.
