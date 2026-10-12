"""
observability.py — Request tracing, metrics, and structured logging.

Every query through the pipeline produces a RequestTrace that captures:
- Which chunks were retrieved (IDs, scores, sources)
- Re-ranker ordering changes
- Prompt sent to LLM
- Model response and token counts
- End-to-end latency breakdown (retrieval, reranking, generation)

Traces are stored as JSON lines for easy analysis and dashboarding, and every
request also updates Prometheus metrics (served at /metrics) so the service can
be scraped and alerted on in Kubernetes/OpenShift:

  docuquery_requests_total{status}             -> throughput & error rate
  docuquery_stage_latency_seconds{stage}       -> retrieval / rerank / generation / total
  docuquery_tokens_total{type}                 -> prompt / completion token spend
  docuquery_citation_coverage                  -> answer grounding distribution
  docuquery_quality_alerts_total{reason}       -> low coverage, slow, invalid citations

Quality alerts are also logged at WARNING level as "QUALITY ALERT ...".
"""

import json
import logging
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from prometheus_client import Counter, Histogram

# Prometheus metrics are process-wide singletons (registered once at import).
REQUESTS = Counter("docuquery_requests_total", "RAG requests by outcome", ["status"])
STAGE_LATENCY = Histogram(
    "docuquery_stage_latency_seconds", "Latency per pipeline stage", ["stage"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32),
)
TOKENS = Counter("docuquery_tokens_total", "LLM tokens used", ["type"])
CITATION_COVERAGE = Histogram(
    "docuquery_citation_coverage", "Share of answer sentences that cite a chunk",
    buckets=(0.5, 0.7, 0.8, 0.9, 0.95, 1.0),
)
QUALITY_ALERTS = Counter("docuquery_quality_alerts_total", "Answers breaching a quality SLO", ["reason"])


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
    citation_coverage: float | None = None
    invalid_citations: list[str] = field(default_factory=list)
    alerts: list[str] = field(default_factory=list)
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
        self.log_level = str(obs_cfg.get("log_level", "INFO")).upper()
        alert_cfg = obs_cfg.get("alerts", {})
        self.min_coverage = alert_cfg.get("min_citation_coverage", 0.90)
        self.max_latency_ms = alert_cfg.get("max_latency_ms", 8000)

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

        # Recent traces for /api/metrics and /api/traces (bounded so memory can't grow forever).
        self._traces: deque[RequestTrace] = deque(maxlen=obs_cfg.get("max_traces_in_memory", 1000))
        self._lock = threading.Lock()

    def check_quality(self, trace: RequestTrace) -> list[str]:
        """Return the quality SLOs this request breached (and count/log them)."""
        alerts = []
        if trace.status == "success":
            if trace.citation_coverage is not None and trace.citation_coverage < self.min_coverage:
                alerts.append("low_citation_coverage")
            if trace.invalid_citations:
                alerts.append("invalid_citation")
        if trace.latency.total_ms > self.max_latency_ms:
            alerts.append("slow_response")
        for reason in alerts:
            QUALITY_ALERTS.labels(reason=reason).inc()
        if alerts:
            self.logger.warning(
                f"QUALITY ALERT [{trace.trace_id}] {','.join(alerts)} "
                f"coverage={trace.citation_coverage} latency={trace.latency.total_ms:.0f}ms "
                f"query=\"{trace.query[:60]}\""
            )
        return alerts

    def _record_prometheus(self, trace: RequestTrace):
        REQUESTS.labels(status=trace.status).inc()
        lat = trace.latency
        for stage, ms in (("retrieval", lat.retrieval_ms), ("reranking", lat.reranking_ms),
                          ("generation", lat.generation_ms), ("total", lat.total_ms)):
            if ms:
                STAGE_LATENCY.labels(stage=stage).observe(ms / 1000)
        if trace.prompt_tokens:
            TOKENS.labels(type="prompt").inc(trace.prompt_tokens)
        if trace.completion_tokens:
            TOKENS.labels(type="completion").inc(trace.completion_tokens)
        if trace.citation_coverage is not None:
            CITATION_COVERAGE.observe(trace.citation_coverage)

    def log_trace(self, trace: RequestTrace):
        """Record a request: quality checks, Prometheus metrics, JSON-lines log, memory."""
        trace.alerts = self.check_quality(trace)
        self._record_prometheus(trace)

        with self._lock:
            self._traces.append(trace)
            if self.log_requests:
                with open(self.log_dir / "traces.jsonl", "a") as f:
                    f.write(json.dumps(asdict(trace), default=str) + "\n")

        self.logger.info(
            f"[{trace.trace_id}] query=\"{trace.query[:60]}\" "
            f"chunks={len(trace.retrieved_chunks)} "
            f"tokens={trace.total_tokens} "
            f"latency={trace.latency.total_ms:.0f}ms "
            f"status={trace.status}"
        )

    def get_metrics(self) -> dict:
        """Aggregate metrics over the recent traces held in memory."""
        with self._lock:
            traces = list(self._traces)
        if not traces:
            return {"total_requests": 0}

        latencies = [t.latency.total_ms for t in traces]
        latencies.sort()
        n = len(latencies)

        total_tokens = sum(t.total_tokens for t in traces)
        total_prompt = sum(t.prompt_tokens for t in traces)
        total_completion = sum(t.completion_tokens for t in traces)

        errors = sum(1 for t in traces if t.status == "error")

        # Throughput: requests per minute across the window covered by these traces.
        first = datetime.fromisoformat(traces[0].timestamp)
        last = datetime.fromisoformat(traces[-1].timestamp)
        window_s = max((last - first).total_seconds(), 1.0)

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
            "avg_chunks_retrieved": sum(len(t.retrieved_chunks) for t in traces) / n,
            "avg_citations_per_answer": sum(t.citation_count for t in traces) / n,
            "throughput_rpm": n / window_s * 60 if n > 1 else 0.0,
            "quality_alerts": sum(len(t.alerts) for t in traces),
        }

    def get_recent_traces(self, limit: int = 20) -> list[dict]:
        """Return the most recent traces as dicts."""
        with self._lock:
            traces = list(self._traces)[-limit:]
        return [asdict(t) for t in traces]


class Timer:
    """Context manager for timing pipeline stages."""

    def __init__(self):
        self.elapsed_ms: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000
