#!/bin/sh
set -e

# Build the vector index on first start, or when a mounted volume is empty.
if [ ! -d chroma_db ] || [ -z "$(ls -A chroma_db 2>/dev/null)" ]; then
  echo "No vector index found, ingesting data/ ..."
  python -m src.ingest --source data/
fi

# exec replaces the shell, so the app receives stop signals from Docker/Kubernetes directly.
exec python -m src.app
