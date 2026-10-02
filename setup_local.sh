#!/usr/bin/env bash
# Local embeddings + a chat-only gateway (Gemini on Wasmer, LiteLLM, Ollama, ...).
#
# Why this exists: many OpenAI-compatible endpoints serve chat models but no embedding
# model. `RAG_EMBED_PROVIDER=local` sidesteps that entirely -- embeddings run on CPU via
# ONNX, no API, no key, no cost. The gateway is still used for generation.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
export PYTHONPATH="src:${PYTHONPATH:-}"

# --- embeddings: local, free, offline ---
export RAG_EMBED_PROVIDER="${RAG_EMBED_PROVIDER:-local}"
export RAG_LOCAL_EMBED_MODEL="${RAG_LOCAL_EMBED_MODEL:-all-MiniLM-L6-v2}"

# --- generation: your gateway (optional; omit RAG_LLM_PROVIDER to stay offline) ---
if [[ -n "${OPENAI_API_KEY:-}" ]]; then
  export RAG_LLM_PROVIDER="${RAG_LLM_PROVIDER:-openai}"
  export RAG_LLM_MODEL="${RAG_LLM_MODEL:-gemini-3.8-flash}"
fi

echo "== setup =="
$PY -m ragpipe.cli doctor || true

echo
echo "== indexing (model downloads once, ~23MB quantized) =="
$PY -m ragpipe.cli --no-load ingest data/docs

echo
echo "== retrieval quality =="
$PY -m ragpipe.cli eval data/eval_cases.jsonl --top-k 8

echo
echo "== asking =="
$PY -m ragpipe.cli ask "what caused the export worker to crash?"

cat <<'EOF'

Note: the LLM answer is cached on disk (data/llm_cache.db), so re-running the same
question returns in ~10ms instead of waiting on the gateway again.
EOF
