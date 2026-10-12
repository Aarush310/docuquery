"""
test_orchestrator.py — End-to-end pipeline wiring with fake components.
Run: pytest tests/test_orchestrator.py -v
"""

import pytest

from src.generator import Generator
from src.observability import ObservabilityLogger
from src.orchestrator import RAGOrchestrator
from tests.fakes import FakeLLM, FakeReranker, FakeRetriever, make_chunk, obs_config

CFG = {
    "retrieval": {"reranker": {"enabled": True}},
    "generation": {"provider": "fake", "model": "m", "temperature": 0, "max_tokens": 50, "system_prompt": "S"},
}


def build(tmp_path, chunks, reply="An answer that is cited properly [abc12345]."):
    cfg = {**CFG, **obs_config(tmp_path)}
    return RAGOrchestrator(
        cfg,
        retriever=FakeRetriever(chunks),
        reranker=FakeReranker(),
        generator=Generator(cfg, llm=FakeLLM([reply])),
        obs=ObservabilityLogger(cfg),
    )


def test_full_pipeline_records_trace(tmp_path):
    chunks = [make_chunk(cid="abc12345" + "0" * 8, text="a"), make_chunk(cid="bbbbbbbb" + "0" * 8, text="b"),
              make_chunk(cid="cccccccc" + "0" * 8, text="c")]
    orch = build(tmp_path, chunks)
    result = orch.query("question")

    assert result.citation_coverage == 1.0
    assert [c.text for c in result.chunks_used] == ["c", "b"]  # reranker output reaches the LLM
    trace = orch.get_recent_traces(1)[0]
    assert trace["status"] == "success"
    assert len(trace["retrieved_chunks"]) == 3 and len(trace["reranked_chunks"]) == 2
    assert trace["citation_coverage"] == 1.0
    lat = trace["latency"]
    assert lat["total_ms"] == pytest.approx(lat["retrieval_ms"] + lat["reranking_ms"] + lat["generation_ms"])
    assert result.latency_ms == pytest.approx(lat["total_ms"])


def test_no_results_short_circuits(tmp_path):
    orch = build(tmp_path, [])
    result = orch.query("question")
    assert "No relevant documents" in result.answer
    assert orch.get_recent_traces(1)[0]["status"] == "no_results"


def test_errors_are_traced_and_reraised(tmp_path):
    orch = build(tmp_path, [make_chunk()])

    def boom(*a, **k):
        raise RuntimeError("LLM down")

    orch.generator.generate = boom
    with pytest.raises(RuntimeError):
        orch.query("question")
    trace = orch.get_recent_traces(1)[0]
    assert trace["status"] == "error" and trace["error"] == "LLM down"
