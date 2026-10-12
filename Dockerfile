# DocuQuery container image
#
# Base: official Python 3.12 slim image (Debian). Note: Red Hat UBI 9 was tried
# first, but it ships SQLite 3.34 and ChromaDB requires SQLite >= 3.35.
#
# The image runs as a non-root user and is OpenShift-compatible: OpenShift
# starts containers with a random user ID that belongs to group 0, so every
# folder the app writes to is made writable by group 0.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/app \
    HF_HOME=/app/.cache/huggingface \
    TIKTOKEN_CACHE_DIR=/app/.cache/tiktoken \
    ANONYMIZED_TELEMETRY=False

WORKDIR /app

# 1. Dependencies first, so a code change doesn't reinstall everything
#    (Docker reuses this layer from cache). CPU-only PyTorch keeps the image
#    gigabytes smaller than the default CUDA build.
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt

# 2. Download the models and the tokenizer at build time, so the container
#    starts without internet access. Names must match configs/config.yaml.
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
import tiktoken; \
SentenceTransformer('all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'); \
tiktoken.get_encoding('cl100k_base')"
ENV HF_HUB_OFFLINE=1

# 3. Application code.
COPY . .

# 4. Non-root user 1001 in group 0. Group 0 owns /app and has the same
#    permissions as the owner, so a random OpenShift UID (also in group 0)
#    can write chroma_db/, logs/, evals/ and the model cache.
RUN useradd --uid 1001 --gid 0 --home-dir /app --no-create-home appuser && \
    chmod +x docker-entrypoint.sh && \
    chgrp -R 0 /app && \
    chmod -R g=u /app
USER 1001

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')"

# Builds the vector index on first start, then runs the FastAPI app.
# Needs GROQ_API_KEY at runtime for answering questions:
#   docker run -p 8000:8000 -e GROQ_API_KEY=... docuquery
ENTRYPOINT ["./docker-entrypoint.sh"]
