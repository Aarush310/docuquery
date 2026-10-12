"""
orchestrator.py — RAG Pipeline Orchestrator

Coordinates the full pipeline:
  Query → Hybrid Retrieval → Re-ranking → Generation → Tracing

This is the single entry point that the CLI and web UI both call.
Every request produces a complete observability trace.
"""

from src.retriever import Retriever, RetrievedChunk
from src.reranker import Reranker
from src.generator import Generator, GeneratedAnswer
from src.observability import (
    ObservabilityLogger,
    RequestTrace,
    RetrievalTrace,
    LatencyBreakdown,
    Timer,
)


class RAGOrchestrator:
    """
    End-to-end RAG pipeline with observability.

    Usage:
        orch = RAGOrchestrator(config)
        result = orch.query("What is RAG?")
        print(result.answer)
    """

    def __init__(self, config: dict, retriever=None, reranker=None, generator=None, obs=None):
        # Components are injectable so the pipeline can be tested without models or API keys.
        self.config = config
        self.retriever = retriever or Retriever(config)
        self.generator = generator or Generator(config)
        self.obs = obs or ObservabilityLogger(config)

        reranker_cfg = config["retrieval"]["reranker"]
        if reranker is not None:
            self.reranker = reranker
        else:
            self.reranker = Reranker(config) if reranker_cfg.get("enabled", False) else None

    def query(self, question: str, top_k: int | None = None) -> GeneratedAnswer:
        """
        Execute the full RAG pipeline with tracing.

        Returns:
            GeneratedAnswer with the cited response and full metadata.
        """
        trace = RequestTrace(query=question)
        latency = LatencyBreakdown()

        try:
            # ── 1. Hybrid Retrieval ──
            with Timer() as t:
                chunks = self.retriever.search(question, top_k=top_k)
            latency.retrieval_ms = t.elapsed_ms

            trace.retrieved_chunks = [
                RetrievalTrace(
                    chunk_id=c.chunk_id,
                    score=c.score,
                    source=c.source,
                    chunk_index=c.chunk_index,
                    text_preview=c.text[:150],
                )
                for c in chunks
            ]

            if not chunks:
                trace.status = "no_results"
                trace.answer = "No relevant documents found."
                trace.latency = latency
                self.obs.log_trace(trace)
                return GeneratedAnswer(
                    query=question,
                    answer="No relevant documents found. Please ingest documents first.",
                    chunks_used=[],
                    model=self.config["generation"]["model"],
                    latency_ms=latency.retrieval_ms,
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                )

            # ── 2. Re-ranking ──
            if self.reranker:
                with Timer() as t:
                    chunks = self.reranker.rerank(question, chunks)
                latency.reranking_ms = t.elapsed_ms

                trace.reranked_chunks = [
                    RetrievalTrace(
                        chunk_id=c.chunk_id,
                        score=c.score,
                        source=c.source,
                        chunk_index=c.chunk_index,
                        text_preview=c.text[:150],
                    )
                    for c in chunks
                ]

            # ── 3. Generation ──
            with Timer() as t:
                result = self.generator.generate(question, chunks)
            latency.generation_ms = t.elapsed_ms

            latency.total_ms = latency.retrieval_ms + latency.reranking_ms + latency.generation_ms
            # Report end-to-end pipeline latency, not just the LLM call.
            result.latency_ms = latency.total_ms

            # ── 4. Record trace ──
            trace.answer = result.answer
            trace.model = result.model
            trace.prompt_tokens = result.prompt_tokens
            trace.completion_tokens = result.completion_tokens
            trace.total_tokens = result.total_tokens
            trace.citation_count = len(result.citations)
            trace.citation_coverage = result.citation_coverage
            trace.invalid_citations = result.invalid_citations
            trace.latency = latency
            trace.status = "success"

            self.obs.log_trace(trace)
            return result

        except Exception as e:
            trace.status = "error"
            trace.error = str(e)
            latency.total_ms = latency.retrieval_ms + latency.reranking_ms + latency.generation_ms
            trace.latency = latency
            self.obs.log_trace(trace)
            raise

    def get_metrics(self) -> dict:
        """Return aggregate observability metrics."""
        return self.obs.get_metrics()

    def get_recent_traces(self, limit: int = 20) -> list[dict]:
        """Return recent request traces."""
        return self.obs.get_recent_traces(limit)
