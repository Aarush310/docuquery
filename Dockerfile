# DocuQuery container image
#
# Base: official Python 3.12 slim (Debian). Red Hat UBI 9 was evaluated first,
# but it ships SQLite 3.34 and ChromaDB requires SQLite >= 3.35.
#
# OpenShift-compatible: OpenShift starts containers with a random UID that is a
# member of group 0, so every path the app writes to is group-0 writable and the
# image never needs root at runtime.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/app \
    HF_HOME=/app/.cache/huggingface \
    TIKTOKEN_CACHE_DIR=/app/.cache/tiktoken \
    ANONYMIZED_TELEMETRY=False \
    DOCUQUERY_CHROMA_DIR=/app/var/chroma_db \
    DOCUQUERY_LOG_DIR=/app/var/logs

WORKDIR /app

# 1. Dependencies first so code changes reuse this cached layer.
#    CPU-only PyTorch keeps the image gigabytes smaller than the CUDA build.
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt

# 2. Bake the embedding model, re-ranker and tokenizer into the image so the
#    container starts without internet access. Names match configs/config.yaml.
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
import tiktoken; \
SentenceTransformer('all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'); \
tiktoken.get_encoding('cl100k_base')"
ENV HF_HUB_OFFLINE=1

# 3. Application code.
COPY . .

# 4. Non-root user 1001 in group 0; group 0 gets the owner's permissions.
RUN useradd --uid 1001 --gid 0 --home-dir /app --no-create-home appuser && \
    mkdir -p /app/var && \
    chmod +x docker-entrypoint.sh && \
    chgrp -R 0 /app && \
    chmod -R g=u /app
USER 1001

# Vector index and logs live here — mount a volume / PVC to persist them.
VOLUME ["/app/var"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')"

# Builds the index on first start (or empty volume), then serves the API.
#   docker run -p 8000:8000 -e GROQ_API_KEY=... docuquery
#   docker run -p 8000:8000 -e DOCUQUERY_LLM_PROVIDER=anthropic -e ANTHROPIC_API_KEY=... docuquery
ENTRYPOINT ["./docker-entrypoint.sh"]
