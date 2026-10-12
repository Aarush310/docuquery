#!/bin/sh
set -e

INDEX_DIR="${DOCUQUERY_CHROMA_DIR:-./chroma_db}"

# Build the vector index on first start, or when a freshly mounted volume is empty.
if [ ! -d "$INDEX_DIR" ] || [ -z "$(ls -A "$INDEX_DIR" 2>/dev/null)" ]; then
  echo "No vector index at $INDEX_DIR — ingesting data/ ..."
  python -m src.ingest --source data/
fi

# exec so the app is PID 1 and receives SIGTERM from Docker/Kubernetes directly.
exec python -m src.app
