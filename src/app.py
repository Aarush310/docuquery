"""
app.py — FastAPI Web Application

Serves a clean web UI for querying DocuQuery, plus API endpoints
for programmatic access and an observability dashboard.

Usage:
    python -m src.app
    # → http://localhost:8000
"""

import os
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from src.config import load_config
from src.orchestrator import RAGOrchestrator
from src.ingest import ingest_documents

# ── Setup ──
config = load_config()
app_cfg = config.get("app", {})

app = FastAPI(title=app_cfg.get("title", "DocuQuery"), version="1.0.0")

# Templates
BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Orchestrator (lazy-loaded after first ingest)
_orchestrator: RAGOrchestrator | None = None


def get_orchestrator() -> RAGOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = RAGOrchestrator(config)
    return _orchestrator


def reset_orchestrator():
    global _orchestrator
    _orchestrator = None


# ── Routes ──

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main web UI."""
    return templates.TemplateResponse(request, "index.html")


class QueryRequest(BaseModel):
    query: str = Field(..., max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=50)


class IngestRequest(BaseModel):
    source: str = "data/"


# Ingestion is only allowed from inside the project's data/ folder.
DATA_DIR = (BASE_DIR / "data").resolve()


@app.get("/healthz")
async def healthz():
    """Liveness/readiness probe for Docker and Kubernetes."""
    return {"status": "ok"}


# The query and ingest routes are plain `def`, not `async def`: they call
# blocking code (model inference, the Groq HTTP client). FastAPI runs plain
# `def` routes in a thread pool, so one slow request doesn't freeze the
# event loop for every other request.

@app.post("/api/query")
def api_query(body: QueryRequest):
    """
    Query the RAG system.

    Body: {"query": "...", "top_k": 5}
    Returns: {"answer": "...", "citations": [...], "chunks": [...], "metrics": {...}}
    """
    query = body.query.strip()
    top_k = body.top_k

    if not query:
        return JSONResponse({"error": "Query is required"}, status_code=400)

    try:
        orch = get_orchestrator()
        result = orch.query(query, top_k=top_k)

        return JSONResponse({
            "answer": result.answer,
            "citations": result.citations,
            "citation_coverage": result.citation_coverage,
            "chunks": [c.to_dict() for c in result.chunks_used],
            "model": result.model,
            "latency_ms": result.latency_ms,
            "tokens": {
                "prompt": result.prompt_tokens,
                "completion": result.completion_tokens,
                "total": result.total_tokens,
            },
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/ingest")
def api_ingest(body: IngestRequest):
    """
    Trigger document ingestion from a path inside data/.

    Body: {"source": "data/"}
    """
    source_path = (BASE_DIR / body.source).resolve()
    if not source_path.is_relative_to(DATA_DIR):
        return JSONResponse({"error": "source must be inside the data/ folder"}, status_code=400)
    if not source_path.exists():
        return JSONResponse({"error": f"not found: {body.source}"}, status_code=404)
    source = str(source_path)

    try:
        ingest_documents(source, config)
        reset_orchestrator()
        return JSONResponse({"status": "ok", "message": f"Ingested documents from {body.source}"})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/metrics")
async def api_metrics():
    """Return aggregate observability metrics."""
    try:
        orch = get_orchestrator()
        return JSONResponse(orch.get_metrics())
    except Exception:
        return JSONResponse({"total_requests": 0})


@app.get("/api/traces")
async def api_traces(limit: int = 20):
    """Return recent request traces."""
    try:
        orch = get_orchestrator()
        return JSONResponse(orch.get_recent_traces(limit))
    except Exception:
        return JSONResponse([])


def main():
    host = app_cfg.get("host", "0.0.0.0")
    port = app_cfg.get("port", 8000)
    print(f"\n🚀 DocuQuery running at http://localhost:{port}\n")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
