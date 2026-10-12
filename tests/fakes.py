"""Shared test doubles — no models, vector DB or API keys needed."""

from types import SimpleNamespace

from src.generator import Completion, GeneratedAnswer, Generator
from src.retriever import RetrievedChunk


def make_chunk(cid="abc12345deadbeef", text="Some passage.", score=0.5, source="doc.md", idx=0):
    return RetrievedChunk(chunk_id=cid, text=text, score=score, source=source, chunk_index=idx)


class FakeLLM:
    """Returns scripted completions and records each prompt."""

    provider = "fake"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def complete(self, system, user, model, temperature, max_tokens):
        self.calls.append({"system": system, "user": user, "model": model})
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        return Completion(text=reply, prompt_tokens=50, completion_tokens=10)


class FakeRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.num_chunks = len(chunks)

    def search(self, query, top_k=None):
        return list(self.chunks)


class FakeReranker:
    def rerank(self, query, chunks):
        return list(reversed(chunks))[:2]


def obs_config(tmp_path, **alerts):
    return {
        "observability": {
            "log_dir": str(tmp_path / "logs"), "log_requests": True, "log_level": "WARNING",
            "alerts": {"min_citation_coverage": 0.9, "max_latency_ms": 8000, **alerts},
        }
    }


def fake_answer(answer="Grounded claim here [abc12345].", coverage=1.0):
    return GeneratedAnswer(
        query="q", answer=answer, chunks_used=[make_chunk()], model="fake/m",
        latency_ms=12.0, prompt_tokens=50, completion_tokens=10, total_tokens=60,
        citations=Generator.extract_citations(answer), citation_coverage=coverage,
    )


def sdk_ns(**kw):
    return SimpleNamespace(**kw)
