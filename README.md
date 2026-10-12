# 📚 DocuQuery — Production-Grade RAG System

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://img.shields.io/badge/CI-passing-brightgreen.svg)]()

A production-grade **Retrieval-Augmented Generation** system that answers questions grounded in your documents with verifiable citations. Built with hybrid retrieval (BM25 + vector search), cross-encoder re-ranking, full observability, automated evaluation with CI gating, and a clean web interface.

## ✨ Features

- **Hybrid Retrieval** — BM25 keyword matching + vector semantic search with configurable fusion weights
- **Cross-Encoder Re-ranking** — Re-scores retrieved chunks with a cross-encoder for higher precision
- **Citation Enforcement** — Every claim cites its source chunk; declines when evidence is insufficient
- **Multi-format Ingestion** — PDF, Markdown, plain text, with automatic chunking and overlap
- **Web UI** — Clean FastAPI + HTML interface for interactive querying
- **CLI Interface** — Full-featured command-line tool with rich formatting
- **Observability** — Request tracing, latency tracking (P50/P95), token accounting, per-request logs
- **Automated Evaluation** — Golden Q/A dataset, citation-coverage scoring, and a CI gate that blocks PRs when quality drops
- **Containerised** — Dockerfile that runs as a non-root user (OpenShift-compatible), verified in CI
- **Prompt Versioning** — All prompts and configs tracked in version control
- **Fully Configurable** — Single YAML config controls every parameter

## Architecture

```
                    ┌──────────────┐
                    │   Web UI /   │
                    │   CLI        │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  Orchestrator │  ← Coordinates the full pipeline
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
       ┌──────▼──────┐  ┌─▼──────┐  ┌──▼─────────┐
       │  BM25 Index  │  │ Vector │  │ Cross-Enc.  │
       │  (rank_bm25) │  │  DB    │  │ Re-ranker   │
       └──────┬───────┘  │(Chroma)│  └──────┬──────┘
              │          └───┬────┘         │
              └──────┬───────┘     ┌────────┘
                     │             │
              ┌──────▼─────────────▼──┐
              │   Score Fusion &      │
              │   Re-rank Pipeline    │
              └──────────┬────────────┘
                         │
              ┌──────────▼────────────┐
              │   LLM Generator       │
              │   (cited answers)     │
              └──────────┬────────────┘
                         │
              ┌──────────▼────────────┐
              │   Observability       │
              │   Logger / Tracer     │
              └───────────────────────┘
```

## Quick Start

```bash
# Clone
git clone https://github.com/Aarush310/docuquery.git
cd docuquery

# Setup
python -m venv venv
source venv/bin/activate        # macOS/Linux
pip install -r requirements.txt

# Configure
export GROQ_API_KEY="your-groq-api-key-here"   # free key: https://console.groq.com/keys

# Ingest your documents
python -m src.ingest --source data/

# Query via CLI
python -m src.query "What is hybrid retrieval?" --verbose

# Or launch the web UI
python -m src.app
# → Open http://localhost:8000
```

## Run with Docker

```bash
docker build -t docuquery .
docker run -p 8000:8000 -e GROQ_API_KEY="your-groq-api-key-here" docuquery
# → Open http://localhost:8000   (health check: /healthz)
```

The image is built on Python 3.12 slim, uses CPU-only PyTorch, and bakes the
embedding model, re-ranker and tokenizer into the image so it starts without internet access.
On first start it ingests `data/`; mount a volume at `/app/chroma_db` to keep the index.

It runs as a non-root user and every writable folder belongs to group 0, so it also runs under
OpenShift's random-UID policy. CI checks this by starting the container with `--user 1000870000:0`.

## Project Structure

```
docuquery/
├── README.md
├── LICENSE
├── Dockerfile                   # Non-root, OpenShift-compatible image
├── docker-entrypoint.sh         # Ingest on first start, then run the app
├── requirements.txt
├── .gitignore
├── configs/
│   └── config.yaml              # All tuneable parameters
├── data/                         # Drop your documents here
│   └── sample.md                 # Example document included
├── src/
│   ├── __init__.py
│   ├── config.py                 # Config loader
│   ├── ingest.py                 # Document loading + chunking + embedding
│   ├── retriever.py              # Hybrid retrieval (BM25 + vector)
│   ├── reranker.py               # Cross-encoder re-ranker
│   ├── generator.py              # LLM answer generation with citations
│   ├── orchestrator.py           # End-to-end pipeline coordinator
│   ├── observability.py          # Request tracing & metrics
│   ├── query.py                  # CLI entry point
│   ├── evaluate.py               # Evaluation pipeline + CI gating
│   └── app.py                    # FastAPI web application
├── templates/
│   └── index.html                # Web UI template
├── evals/
│   └── golden_qa.json            # Golden Q/A evaluation dataset
└── tests/
    ├── test_api.py              # Health check, input validation, ingest path safety
    ├── test_chunking.py
    ├── test_citations.py
    └── test_observability.py
```

## Configuration

All parameters are in `configs/config.yaml`:

| Section | Key Parameters |
|---------|---------------|
| **Chunking** | `chunk_size` (600), `chunk_overlap` (100), `min_chunk_size` (50) |
| **Embedding** | `model` (all-MiniLM-L6-v2) |
| **Retrieval** | `top_k` (5), `bm25_weight` (0.3), `vector_weight` (0.7) |
| **Re-ranker** | `model` (cross-encoder/ms-marco-MiniLM-L-6-v2), `top_n` (3) |
| **Generation** | `model` (llama-3.3-70b-versatile via Groq), `temperature` (0.1) |
| **Evaluation** | `faithfulness_threshold` (0.85), `citation_coverage_threshold` (0.90) |

## Evaluation & CI Gating

```bash
# Run evaluation against golden dataset
python -m src.evaluate

# Use as CI gate (exits non-zero if quality drops)
python -m src.evaluate --ci
```

The evaluation runs every golden question through the full pipeline and measures citation coverage, latency percentiles, token usage and errors. In CI mode, the process exits with code 1 if citation coverage falls below the threshold, which blocks the PR merge. Results are saved to `evals/results.json` and uploaded as a CI artifact.

**CI pipeline** (`.github/workflows/ci.yml`):

1. **Unit tests** on Python 3.11, 3.12 and 3.13.
2. **Docker**: builds the image, starts it as a random non-root UID, and waits for `/healthz`.
3. **Evaluation gate** (pull requests): ingests `data/`, runs the golden set, fails the PR below threshold. Needs a `GROQ_API_KEY` repository secret.

Current limitations, and what's next: the golden set is small (10 questions on one sample document); faithfulness is currently approximated by citation coverage, and the next step is an LLM-as-judge that checks each claim against the chunk it cites; retrieval precision needs labelled `relevant_chunk_ids` in the golden set.

## Key Metrics

| Metric | Description | Target | Status |
|--------|-------------|--------|--------|
| Citation Coverage | % of answer sentences backed by a cited chunk | ≥ 90% | Gated in CI |
| Latency P50 / P95 | End-to-end response time | < 3s / < 8s | Reported |
| Faithfulness | % of cited claims actually supported by the source | ≥ 85% | Planned (LLM-as-judge) |
| Retrieval Precision@k | % of top-k chunks that are relevant | ≥ 80% | Planned (needs labels) |

## Tech Stack

- **Python 3.11+** — Core language
- **ChromaDB** — Vector storage and similarity search
- **sentence-transformers** — Embedding and cross-encoder models
- **rank-bm25** — BM25 keyword retrieval
- **Groq API** — LLM generation (Llama 3.3 70B, free tier)
- **FastAPI** — Web API and UI server
- **tiktoken** — Token counting
- **Rich** — Terminal formatting
- **Docker** — Python 3.12 slim image, non-root, OpenShift-compatible

## License

MIT — see [LICENSE](LICENSE).
