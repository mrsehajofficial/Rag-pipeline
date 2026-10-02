#!/usr/bin/env bash
# One-shot demo: index the sample docs, ask a few questions, and print retrieval metrics.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
export PYTHONPATH="src:${PYTHONPATH:-}"

echo "== indexing sample docs =="
$PY -m ragpipe.cli --no-load ingest data/docs

echo
echo "== asking =="
$PY -m ragpipe.cli ask "what order should token validation happen in?"
echo
$PY -m ragpipe.cli ask "what caused the export worker to crash?"

echo
echo "== retrieval quality =="
# No --no-load here: eval must run against the index we just built, or every metric is 0.
$PY -m ragpipe.cli eval data/eval_cases.jsonl --top-k 8

echo
echo "== stats =="
$PY -m ragpipe.cli stats
