"""
observability.py — Request tracing, metrics, and structured logging.

Every query through the pipeline produces a RequestTrace that captures:
- Which chunks were retrieved (IDs, scores, sources)
- Re-ranker ordering changes
- Prompt sent to LLM
- Model response and token counts
- End-to-end latency breakdown (retrieval, reranking, generation)

Traces are stored as JSON lines for easy analysis and dashboarding.
"""

import json
import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class LatencyBreakdown:
    """Timing for each pipeline stage in milliseconds."""
    retrieval_ms: float = 0.0
    reranking_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0


@dataclass
class RetrievalTrace:
    """What the retriever returned."""
    chunk_id: str
    score: float
    source: str
    chunk_index: int
    text_preview: str = ""  # First 150 chars


@dataclass
class RequestTrace:
    """Complete trace of a single RAG request."""
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    query: str = ""
    retrieved_chunks: list[RetrievalTrace] = field(default_factory=list)
    reranked_chunks: list[RetrievalTrace] = field(default_factory=list)
    answer: str = ""
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency: LatencyBreakdown = field(default_factory=LatencyBreakdown)
    citation_count: int = 0
    status: str = "success"  # "success" | "no_results" | "error"
    error: str | None = None


class ObservabilityLogger:
    """
    Structured logger that writes request traces to JSON lines files
    and exposes aggregate metrics.
    """

    def __init__(self, config: dict):
        obs_cfg = config.get("observability", {})
        self.log_dir = Path(obs_cfg.get("log_dir", "./logs"))
        self.log_requests = obs_cfg.get("log_requests", True)
        self.log_level = obs_cfg.get("log_level", "INFO")

        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Setup Python logger
        self.logger = logging.getLogger("docuquery")
        self.logger.setLevel(getattr(logging, self.log_level))

        if not self.logger.handlers:
            # Console handler
            ch = logging.StreamHandler()
            ch.setLevel(getattr(logging, self.log_level))
            fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", datefmt="%H:%M:%S")
            ch.setFormatter(fmt)
            self.logger.addHandler(ch)

            # File handler
            fh = logging.FileHandler(self.log_dir / "docuquery.log")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
            self.logger.addHandler(fh)

        # In-memory metrics for the current session
        self._traces: list[RequestTrace] = []

    def log_trace(self, trace: RequestTrace):
        """Write a request trace to the JSON lines log and memory."""
        self._traces.append(trace)

        if self.log_requests:
            trace_file = self.log_dir / "traces.jsonl"
            with open(trace_file, "a") as f:
                f.write(json.dumps(asdict(trace), default=str) + "\n")

        self.logger.info(
            f"[{trace.trace_id}] query=\"{trace.query[:60]}\" "
            f"chunks={len(trace.retrieved_chunks)} "
            f"tokens={trace.total_tokens} "
            f"latency={trace.latency.total_ms:.0f}ms "
            f"status={trace.status}"
        )

    def get_metrics(self) -> dict:
        """Compute aggregate metrics from all traces in this session."""
        if not self._traces:
            return {"total_requests": 0}

        latencies = [t.latency.total_ms for t in self._traces]
        latencies.sort()
        n = len(latencies)

        total_tokens = sum(t.total_tokens for t in self._traces)
        total_prompt = sum(t.prompt_tokens for t in self._traces)
        total_completion = sum(t.completion_tokens for t in self._traces)

        errors = sum(1 for t in self._traces if t.status == "error")

        return {
            "total_requests": n,
            "error_count": errors,
            "error_rate": errors / n if n else 0,
            "latency_p50_ms": latencies[n // 2],
            "latency_p95_ms": latencies[int(n * 0.95)] if n >= 2 else latencies[-1],
            "latency_mean_ms": sum(latencies) / n,
            "total_tokens": total_tokens,
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "avg_tokens_per_request": total_tokens / n,
            "avg_chunks_retrieved": sum(len(t.retrieved_chunks) for t in self._traces) / n,
            "avg_citations_per_answer": sum(t.citation_count for t in self._traces) / n,
        }

    def get_recent_traces(self, limit: int = 20) -> list[dict]:
        """Return the most recent traces as dicts."""
        return [asdict(t) for t in self._traces[-limit:]]


class Timer:
    """Context manager for timing pipeline stages."""

    def __init__(self):
        self.elapsed_ms: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000
