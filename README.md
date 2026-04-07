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
- **Automated Evaluation** — Golden Q/A dataset with faithfulness scoring, citation coverage, and CI gating
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
git clone https://github.com/YOUR_USERNAME/docuquery.git
cd docuquery

# Setup
python -m venv venv
source venv/bin/activate        # macOS/Linux
pip install -r requirements.txt

# Configure
export GOOGLE_API_KEY="your-gemini-api-key-here"

# Ingest your documents
python -m src.ingest --source data/

# Query via CLI
python -m src.query "What is hybrid retrieval?" --verbose

# Or launch the web UI
python -m src.app
# → Open http://localhost:8000
```

## Project Structure

```
docuquery/
├── README.md
├── LICENSE
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
    ├── test_chunking.py
    ├── test_retriever.py
    └── test_evaluation.py
```

## Configuration

All parameters are in `configs/config.yaml`:

| Section | Key Parameters |
|---------|---------------|
| **Chunking** | `chunk_size` (600), `chunk_overlap` (100), `min_chunk_size` (50) |
| **Embedding** | `model` (all-MiniLM-L6-v2) |
| **Retrieval** | `top_k` (5), `bm25_weight` (0.3), `vector_weight` (0.7) |
| **Re-ranker** | `model` (cross-encoder/ms-marco-MiniLM-L-6-v2), `top_n` (3) |
| **Generation** | `model` (gemini-2.0-flash), `temperature` (0.1) |
| **Evaluation** | `faithfulness_threshold` (0.85), `citation_coverage_threshold` (0.90) |

## Evaluation & CI Gating

```bash
# Run evaluation against golden dataset
python -m src.evaluate

# Use as CI gate (exits non-zero if quality drops)
python -m src.evaluate --ci
```

The evaluation pipeline measures retrieval precision, citation coverage, faithfulness, and latency percentiles. In CI mode, the process exits with code 1 if any quality threshold is breached, blocking the PR merge.

## Key Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Retrieval Precision@k | % of top-k chunks that are relevant | ≥ 80% |
| Citation Coverage | % of answer claims backed by a cited chunk | ≥ 90% |
| Faithfulness | % of cited claims actually supported by source | ≥ 85% |
| Latency P50 | Median end-to-end response time | < 3s |
| Latency P95 | 95th percentile response time | < 8s |

## Tech Stack

- **Python 3.11+** — Core language
- **ChromaDB** — Vector storage and similarity search
- **sentence-transformers** — Embedding and cross-encoder models
- **rank-bm25** — BM25 keyword retrieval
- **Google Gemini API** — LLM generation (gemini-2.0-flash, free tier)
- **FastAPI** — Web API and UI server
- **tiktoken** — Token counting
- **Rich** — Terminal formatting

## License

MIT — see [LICENSE](LICENSE).
