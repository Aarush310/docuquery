"""
test_observability.py — Tests for observability and tracing.

Run: pytest tests/test_observability.py -v
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.observability import RequestTrace, LatencyBreakdown, ObservabilityLogger


class TestRequestTrace:
    def test_defaults(self):
        t = RequestTrace()
        assert t.query == ""
        assert t.status == "success"
        assert t.error is None
        assert len(t.trace_id) == 12

    def test_custom(self):
        t = RequestTrace(query="test", status="error", error="broke")
        assert t.query == "test"
        assert t.status == "error"


class TestLatencyBreakdown:
    def test_defaults(self):
        lb = LatencyBreakdown()
        assert lb.total_ms == 0.0

    def test_custom(self):
        lb = LatencyBreakdown(retrieval_ms=50, reranking_ms=30, generation_ms=200, total_ms=280)
        assert lb.total_ms == 280


class TestObservabilityLogger:
    def _make_logger(self):
        cfg = {"observability": {"log_dir": "/tmp/dq_test_logs", "log_requests": False, "log_level": "WARNING"}}
        return ObservabilityLogger(cfg)

    def test_empty_metrics(self):
        assert self._make_logger().get_metrics()["total_requests"] == 0

    def test_log_and_metrics(self):
        logger = self._make_logger()
        logger.log_trace(RequestTrace(query="q1", total_tokens=100, prompt_tokens=80, completion_tokens=20,
                                       latency=LatencyBreakdown(total_ms=500), citation_count=3))
        logger.log_trace(RequestTrace(query="q2", total_tokens=200, prompt_tokens=150, completion_tokens=50,
                                       latency=LatencyBreakdown(total_ms=800), citation_count=5))
        m = logger.get_metrics()
        assert m["total_requests"] == 2
        assert m["total_tokens"] == 300
        assert m["error_count"] == 0
        assert m["avg_citations_per_answer"] == 4.0

    def test_recent_traces(self):
        logger = self._make_logger()
        for i in range(5):
            logger.log_trace(RequestTrace(query=f"q{i}", latency=LatencyBreakdown(total_ms=100)))
        assert len(logger.get_recent_traces(limit=3)) == 3

    def test_error_rate(self):
        logger = self._make_logger()
        logger.log_trace(RequestTrace(query="ok", status="success", latency=LatencyBreakdown(total_ms=100)))
        logger.log_trace(RequestTrace(query="fail", status="error", latency=LatencyBreakdown(total_ms=100)))
        assert logger.get_metrics()["error_rate"] == 0.5
