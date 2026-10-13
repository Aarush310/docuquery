# 📚 DocuQuery — Production-Grade RAG System

[![CI](https://github.com/Aarush310/docuquery/actions/workflows/ci.yml/badge.svg)](https://github.com/Aarush310/docuquery/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A **Retrieval-Augmented Generation** service that answers questions from your documents with verifiable citations. Hybrid retrieval (BM25 + dense vectors), cross-encoder re-ranking, citation enforcement, per-request tracing, Prometheus metrics with quality alerts, an LLM-as-judge evaluation gate in CI, and a container image that runs on Docker, Podman and OpenShift.

## ✨ Features

- **Hybrid retrieval** — BM25 keyword scores + dense vector similarity, normalized and fused with configurable weights (0.3 / 0.7)
- **Cross-encoder re-ranking** — `ms-marco-MiniLM-L-6-v2` re-scores (query, chunk) pairs jointly; top 3 go to the LLM
- **Citation enforcement** — every claim must cite a chunk ID; coverage is measured per answer and citations to chunks that weren't retrieved are flagged
- **Pluggable LLM** — Groq (Llama 3.3 70B, free tier) or Anthropic Claude, switched by config or env var
- **Observability** — per-request traces (JSON lines), per-stage latency, token accounting, throughput, `/metrics` for Prometheus
- **Quality alerting** — answers below the citation-coverage SLO, slower than the latency SLO, or with invalid citations are logged as `QUALITY ALERT` and counted in `docuquery_quality_alerts_total{reason}`
- **Evaluation gate** — golden Q/A set scored on 5 gates; CI fails the PR if any is breached
- **Container-ready** — non-root image with models baked in, liveness/readiness probes, OpenShift manifests

## Architecture

```
                   ┌───────────────┐
                   │ Web UI / CLI  │
                   └───────┬───────┘
                           │  POST /api/query
                   ┌───────▼───────┐
                   │ Orchestrator  │ ─── trace + metrics + quality alerts ──▶ /metrics, logs/traces.jsonl
                   └───────┬───────┘
             ┌─────────────┼──────────────┐
      ┌──────▼──────┐ ┌────▼─────┐        │
      │ BM25 index  │ │ ChromaDB │        │
      │ (rank_bm25) │ │ (cosine) │        │
      └──────┬──────┘ └────┬─────┘        │
             └──── fuse ───┘              │
                   │ top 5                │
          ┌────────▼─────────┐            │
          │ Cross-encoder    │ top 3      │
          │ re-ranker        ├────────────┘
          └────────┬─────────┘
          ┌────────▼─────────┐
          │ LLM (Groq/Claude)│ → cited answer + coverage
          └──────────────────┘
```

**Ingestion:** PDF (PyMuPDF), Markdown and text → paragraph-aware chunks of ≤600 tokens (tiktoken `cl100k_base`) with ~100 tokens of paragraph overlap → `all-MiniLM-L6-v2` embeddings → ChromaDB. Chunk IDs are `sha256(path::index)[:16]`, so re-ingesting is deterministic.

## Quick start

```bash
git clone https://github.com/Aarush310/docuquery.git && cd docuquery
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

export GROQ_API_KEY="..."                      # default provider
# or: export DOCUQUERY_LLM_PROVIDER=anthropic ANTHROPIC_API_KEY="..."

python -m src.ingest --source data/            # build the index
python -m src.query "What is hybrid retrieval?" --verbose
python -m src.app                              # → http://localhost:8000
```

## Run in a container

```bash
docker build -t docuquery .                    # or: podman build -t docuquery .
docker run -p 8000:8000 -e GROQ_API_KEY=... -v docuquery-data:/app/var docuquery
```

On first start the entrypoint ingests `data/` into the volume, then serves the API. The image runs as a non-root user, works with OpenShift's random-UID policy, and needs no internet access for models.

### Deploy to OpenShift

```bash
oc create secret generic docuquery-llm --from-literal=GROQ_API_KEY=...
oc apply -f deploy/openshift/build.yaml        # ImageStream + BuildConfig: builds the image in-cluster from GitHub
oc apply -f deploy/openshift/docuquery.yaml    # PVC, Deployment (startup/liveness/readiness probes), Service, Route
```

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/query` | `{"query": "...", "top_k": 5}` → answer, citations, coverage, chunks, latency, tokens |
| `POST` | `/api/ingest` | `{"source": "data/"}` → re-index (only paths inside `data/` are accepted) |
| `GET` | `/api/metrics` | P50/P95 latency, error rate, tokens, throughput (req/min), alert count |
| `GET` | `/api/traces` | Recent request traces |
| `GET` | `/healthz` | Liveness probe |
| `GET` | `/readyz` | Readiness probe — models loaded and index available |
| `GET` | `/metrics` | Prometheus metrics |

Prometheus metrics: `docuquery_requests_total{status}`, `docuquery_stage_latency_seconds{stage}` (retrieval, reranking, generation, total), `docuquery_tokens_total{type}`, `docuquery_citation_coverage`, `docuquery_quality_alerts_total{reason}`.

## Evaluation & CI gating

```bash
python -m src.evaluate            # full report, writes evals/results.json
python -m src.evaluate --ci       # exit 1 if any gate fails
python -m src.evaluate --no-judge # skip the LLM-judge gates
```

Each golden question runs through the full pipeline; an LLM judge then scores the answer against the retrieved chunks and the reference answer.

| Gate | How it's measured | Threshold |
|---|---|---|
| Citation coverage | Share of answer sentences (5+ words) carrying a chunk citation | ≥ 90% |
| Faithfulness | LLM judge: share of claims supported by the retrieved chunks | ≥ 85% |
| Answer correctness | LLM judge: agreement with the golden reference answer | ≥ 75% |
| Latency P95 | End-to-end pipeline latency | ≤ 8000 ms |
| Error rate | Questions that failed outright | 0% |

GitHub Actions runs four jobs: **unit tests** on Python 3.11/3.12/3.13, a **container** job that builds the image, starts it as a random non-root UID and checks `/healthz`, `/readyz` and `/metrics`, **manifest validation** with kubeconform, and on pull requests the **evaluation gate** (needs a `GROQ_API_KEY` repository secret; skipped with a warning if absent). Results are uploaded as a build artifact.

## Configuration

Everything lives in `configs/config.yaml`; environment variables override it per deployment:

| Variable | Overrides |
|---|---|
| `DOCUQUERY_LLM_PROVIDER` | `generation.provider` (`groq` or `anthropic`; also switches to that provider's default model) |
| `DOCUQUERY_LLM_MODEL` | `generation.model` |
| `DOCUQUERY_CHROMA_DIR` | `vector_store.persist_directory` |
| `DOCUQUERY_LOG_DIR` | `observability.log_dir` |
| `DOCUQUERY_DATA_DIR` | `ingest.allowed_root` |

## Project structure

```
docuquery/
├── configs/config.yaml        # all tuneable parameters and quality thresholds
├── data/                      # documents to ingest
├── evals/golden_qa.json       # golden Q/A set
├── src/
│   ├── ingest.py              # load → chunk → embed → store
│   ├── retriever.py           # BM25 + vector hybrid retrieval
│   ├── reranker.py            # cross-encoder re-ranker
│   ├── generator.py           # LLM providers + cited generation
│   ├── orchestrator.py        # pipeline coordinator with tracing
│   ├── observability.py       # traces, Prometheus metrics, quality alerts
│   ├── evaluate.py            # 5-gate evaluation + CI gating
│   ├── app.py                 # FastAPI app, probes, /metrics
│   └── query.py               # CLI
├── tests/                     # pytest suite (runs without models or API keys)
├── Dockerfile, docker-entrypoint.sh
├── deploy/openshift/          # in-cluster build, PVC, Deployment, Service, Route
└── .github/workflows/ci.yml
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

The suite covers chunking, citation parsing, score fusion and re-ranking, provider clients, config overrides, the orchestrator (with fake components), evaluation gates and judge parsing, observability/alerts, and the HTTP API — with no ML models, vector DB or API keys needed.

## Tech stack

Python 3.11+ · FastAPI · ChromaDB · sentence-transformers · rank-bm25 · Groq / Anthropic Claude · tiktoken · prometheus-client · Docker / OpenShift · GitHub Actions

## License

MIT — see [LICENSE](LICENSE).
