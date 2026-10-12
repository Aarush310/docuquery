"""
test_observability.py — Tracing, metrics, throughput and quality alerts.
Run: pytest tests/test_observability.py -v
"""

import json

from prometheus_client import REGISTRY

from src.observability import LatencyBreakdown, ObservabilityLogger, RequestTrace
from tests.fakes import obs_config


def metric(name, **labels):
    return REGISTRY.get_sample_value(name, labels) or 0.0


class TestRequestTrace:
    def test_defaults(self):
        t = RequestTrace()
        assert t.query == "" and t.status == "success" and t.error is None
        assert len(t.trace_id) == 12

    def test_custom(self):
        t = RequestTrace(query="test", status="error", error="broke")
        assert t.query == "test" and t.status == "error"


class TestLatencyBreakdown:
    def test_defaults(self):
        assert LatencyBreakdown().total_ms == 0.0

    def test_custom(self):
        lb = LatencyBreakdown(retrieval_ms=50, reranking_ms=30, generation_ms=200, total_ms=280)
        assert lb.total_ms == 280


class TestObservabilityLogger:
    def test_empty_metrics(self, tmp_path):
        assert ObservabilityLogger(obs_config(tmp_path)).get_metrics()["total_requests"] == 0

    def test_log_and_metrics(self, tmp_path):
        logger = ObservabilityLogger(obs_config(tmp_path))
        logger.log_trace(RequestTrace(query="q1", total_tokens=100, prompt_tokens=80, completion_tokens=20,
                                      latency=LatencyBreakdown(total_ms=500), citation_count=3))
        logger.log_trace(RequestTrace(query="q2", total_tokens=200, prompt_tokens=150, completion_tokens=50,
                                      latency=LatencyBreakdown(total_ms=800), citation_count=5))
        m = logger.get_metrics()
        assert m["total_requests"] == 2
        assert m["total_tokens"] == 300
        assert m["error_count"] == 0
        assert m["avg_citations_per_answer"] == 4.0
        assert "throughput_rpm" in m

    def test_traces_written_as_jsonl(self, tmp_path):
        logger = ObservabilityLogger(obs_config(tmp_path))
        logger.log_trace(RequestTrace(query="hello", latency=LatencyBreakdown(total_ms=10)))
        lines = (tmp_path / "logs" / "traces.jsonl").read_text().splitlines()
        assert json.loads(lines[-1])["query"] == "hello"

    def test_recent_traces(self, tmp_path):
        logger = ObservabilityLogger(obs_config(tmp_path))
        for i in range(5):
            logger.log_trace(RequestTrace(query=f"q{i}", latency=LatencyBreakdown(total_ms=100)))
        recent = logger.get_recent_traces(limit=3)
        assert [t["query"] for t in recent] == ["q2", "q3", "q4"]

    def test_memory_is_bounded(self, tmp_path):
        cfg = obs_config(tmp_path)
        cfg["observability"]["max_traces_in_memory"] = 3
        cfg["observability"]["log_requests"] = False
        logger = ObservabilityLogger(cfg)
        for i in range(10):
            logger.log_trace(RequestTrace(query=f"q{i}", latency=LatencyBreakdown(total_ms=1)))
        assert logger.get_metrics()["total_requests"] == 3

    def test_error_rate(self, tmp_path):
        logger = ObservabilityLogger(obs_config(tmp_path))
        logger.log_trace(RequestTrace(query="ok", status="success", latency=LatencyBreakdown(total_ms=100)))
        logger.log_trace(RequestTrace(query="fail", status="error", latency=LatencyBreakdown(total_ms=100)))
        assert logger.get_metrics()["error_rate"] == 0.5


class TestQualityAlerts:
    def test_good_answer_raises_no_alert(self, tmp_path):
        logger = ObservabilityLogger(obs_config(tmp_path))
        t = RequestTrace(query="q", citation_coverage=1.0, latency=LatencyBreakdown(total_ms=100))
        logger.log_trace(t)
        assert t.alerts == []

    def test_low_coverage_slow_and_invalid_citation_alerts(self, tmp_path, caplog):
        before = metric("docuquery_quality_alerts_total", reason="low_citation_coverage")
        logger = ObservabilityLogger(obs_config(tmp_path, max_latency_ms=1000))
        t = RequestTrace(query="q", citation_coverage=0.4, invalid_citations=["deadbeef"],
                         latency=LatencyBreakdown(total_ms=5000))
        with caplog.at_level("WARNING", logger="docuquery"):
            logger.log_trace(t)
        assert set(t.alerts) == {"low_citation_coverage", "invalid_citation", "slow_response"}
        assert metric("docuquery_quality_alerts_total", reason="low_citation_coverage") == before + 1
        assert "QUALITY ALERT" in caplog.text

    def test_prometheus_request_and_latency_metrics(self, tmp_path):
        before = metric("docuquery_requests_total", status="success")
        before_count = metric("docuquery_stage_latency_seconds_count", stage="generation")
        logger = ObservabilityLogger(obs_config(tmp_path))
        logger.log_trace(RequestTrace(query="q", citation_coverage=1.0, prompt_tokens=5,
                                      latency=LatencyBreakdown(generation_ms=200, total_ms=250)))
        assert metric("docuquery_requests_total", status="success") == before + 1
        assert metric("docuquery_stage_latency_seconds_count", stage="generation") == before_count + 1
