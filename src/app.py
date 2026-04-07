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
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

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


@app.post("/api/query")
async def api_query(request: Request):
    """
    Query the RAG system.

    Body: {"query": "...", "top_k": 5}
    Returns: {"answer": "...", "citations": [...], "chunks": [...], "metrics": {...}}
    """
    body = await request.json()
    query = body.get("query", "").strip()
    top_k = body.get("top_k")

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
async def api_ingest(request: Request):
    """
    Trigger document ingestion.

    Body: {"source": "data/"}
    """
    body = await request.json()
    source = body.get("source", "data/")

    try:
        ingest_documents(source, config)
        reset_orchestrator()
        return JSONResponse({"status": "ok", "message": f"Ingested documents from {source}"})
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
