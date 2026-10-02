# syntax=docker/dockerfile:1
# --------------------------------------------------------------------------
# Stage 1: build dependencies in a slim base (no heavy build tools in prod)
# --------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools needed for any C-extension wheels (numpy, onnxruntime)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-accelerated.txt pyproject.toml ./

# Install all runtime deps into a prefix we can copy across
# No `|| true` here: a broken requirements.txt must fail the build, not produce
# a silently broken image that crashes on first import.
RUN pip install --prefix=/deps --no-cache-dir -r requirements.txt
# Accelerated extras (numpy, onnxruntime) are optional; install if they resolve
RUN pip install --prefix=/deps --no-cache-dir -r requirements-accelerated.txt 2>/dev/null || true
# Production WSGI server
RUN pip install --prefix=/deps --no-cache-dir gunicorn


# --------------------------------------------------------------------------
# Stage 2: lean production image
# --------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# Non-root user for principle of least privilege
RUN useradd --create-home --shell /bin/bash ragpipe

# Copy the installed packages from the builder stage
COPY --from=builder /deps /usr/local

WORKDIR /app

# Copy project source
COPY --chown=ragpipe:ragpipe src/ ./src/
COPY --chown=ragpipe:ragpipe pyproject.toml ./

# Install the ragpipe package itself (editable-style, no pip reinstall of deps)
RUN pip install --no-cache-dir --no-deps -e .

# Persistent data directory: mount a volume here in production
RUN mkdir -p /data && chown ragpipe:ragpipe /data

USER ragpipe

# --------------------------------------------------------------------------
# Runtime configuration (all overridable via environment variables or
# docker-compose env_file)
# --------------------------------------------------------------------------
ENV RAG_DATA_DIR=/data \
    RAG_EMBED_PROVIDER=hashing \
    RAG_LLM_PROVIDER=extractive \
    RAG_LOG_JSON=true \
    RAG_LOG_LEVEL=INFO \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" \
    || exit 1

# Production: gunicorn with sync workers (WSGI-compatible).
# The WSGI adapter (ragpipe.wsgi:app) uses the (environ, start_response) protocol.
# Do NOT use uvicorn.workers.UvicornWorker — that's an ASGI worker that calls
# app(scope, receive, send), which is incompatible with WSGI and returns 500 on
# every request.
CMD ["gunicorn", "-w", "4", "--bind", "0.0.0.0:8000", "ragpipe.wsgi:app"]
