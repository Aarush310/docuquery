"""
app.py — FastAPI Web Application

Serves the web UI, a JSON API, health probes for Kubernetes/OpenShift, and
Prometheus metrics.

Endpoints:
    GET  /              Web UI
    POST /api/query     {"query": "...", "top_k": 5} -> cited answer + chunks + metrics
    POST /api/ingest    {"source": "data/"} -> re-index documents (restricted to the data dir)
    GET  /api/metrics   Aggregate latency / tokens / throughput / alerts (JSON)
    GET  /api/traces    Recent request traces
    GET  /healthz       Liveness probe  — process is up
    GET  /readyz        Readiness probe — index loaded, pipeline ready to answer
    GET  /metrics       Prometheus exposition format

Endpoints are plain `def` (not `async def`) on purpose: the pipeline does
blocking model inference and HTTP calls, so FastAPI runs them in its
threadpool instead of blocking the event loop.

Usage:
    python -m src.app            # → http://localhost:8000
"""

import threading
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.templating import Jinja2Templates
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from src.config import load_config

# ── Setup ──
config = load_config()
app_cfg = config.get("app", {})

app = FastAPI(title=app_cfg.get("title", "DocuQuery"), version="1.1.0")

BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# The orchestrator loads models and the index, so build it once, lazily, thread-safely.
_orchestrator = None
_orchestrator_lock = threading.Lock()


def get_orchestrator():
    global _orchestrator
    if _orchestrator is None:
        with _orchestrator_lock:
            if _orchestrator is None:
                from src.orchestrator import RAGOrchestrator
                _orchestrator = RAGOrchestrator(config)
    return _orchestrator


def reset_orchestrator():
    global _orchestrator
    with _orchestrator_lock:
        _orchestrator = None


def resolve_ingest_path(source: str) -> Path:
    """Only allow ingesting from inside the configured data directory."""
    root = Path(config["ingest"]["allowed_root"])
    root = (root if root.is_absolute() else BASE_DIR / root).resolve()
    candidate = Path(source)
    candidate = (candidate if candidate.is_absolute() else BASE_DIR / candidate).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"Ingest source must be inside the data directory ({root.name}/)")
    return candidate


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int | None = Field(None, ge=1, le=20)


class IngestRequest(BaseModel):
    source: str = "data/"


# ── Routes ──

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    """Serve the main web UI."""
    return templates.TemplateResponse(request, "index.html")


@app.post("/api/query")
def api_query(body: QueryRequest):
    query = body.query.strip()
    if not query:
        return JSONResponse({"error": "Query is required"}, status_code=400)

    try:
        orch = get_orchestrator()
    except Exception as e:
        return JSONResponse({"error": f"Service not ready: {e}"}, status_code=503)

    try:
        result = orch.query(query, top_k=body.top_k)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    return {
        "answer": result.answer,
        "citations": result.citations,
        "invalid_citations": result.invalid_citations,
        "citation_coverage": result.citation_coverage,
        "chunks": [c.to_dict() for c in result.chunks_used],
        "model": result.model,
        "latency_ms": result.latency_ms,
        "tokens": {
            "prompt": result.prompt_tokens,
            "completion": result.completion_tokens,
            "total": result.total_tokens,
        },
    }


@app.post("/api/ingest")
def api_ingest(body: IngestRequest):
    from src.ingest import IngestError, ingest_documents

    try:
        source = resolve_ingest_path(body.source)
        stats = ingest_documents(str(source), config)
    except (ValueError, IngestError) as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    reset_orchestrator()  # rebuild BM25 + reconnect to the new collection on next query
    return {"status": "ok", "message": f"Ingested documents from {body.source}", **stats}


@app.get("/api/metrics")
def api_metrics():
    if _orchestrator is None:
        return {"total_requests": 0}
    return _orchestrator.get_metrics()


@app.get("/api/traces")
def api_traces(limit: int = 20):
    if _orchestrator is None:
        return []
    return _orchestrator.get_recent_traces(max(1, min(limit, 200)))


@app.get("/healthz")
def healthz():
    """Liveness: the process is running and serving HTTP."""
    return {"status": "ok"}


@app.get("/readyz")
def readyz():
    """Readiness: models loaded and the document index is available."""
    try:
        orch = get_orchestrator()
    except Exception as e:
        return JSONResponse({"status": "not_ready", "reason": str(e)}, status_code=503)
    return {
        "status": "ready",
        "chunks_indexed": orch.retriever.num_chunks,
        "model": f"{config['generation'].get('provider')}/{config['generation']['model']}",
    }


@app.get("/metrics")
def prometheus_metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def main():
    host = app_cfg.get("host", "0.0.0.0")
    port = int(app_cfg.get("port", 8000))
    print(f"\n🚀 DocuQuery running at http://localhost:{port}\n")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
