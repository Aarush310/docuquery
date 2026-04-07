"""
generator.py — LLM Answer Generation with Citation Enforcement

Produces grounded answers from retrieved context chunks using the
Groq API (free tier, OpenAI-compatible). Every factual claim must cite
its source chunk by ID.
"""

import os
import re
import time
from dataclasses import dataclass, field

from groq import Groq

from src.retriever import RetrievedChunk


@dataclass
class GeneratedAnswer:
    query: str
    answer: str
    chunks_used: list[RetrievedChunk]
    model: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    citations: list[str] = field(default_factory=list)
    citation_coverage: float = 0.0

    def to_dict(self) -> dict:
        return {
            "query": self.query, "answer": self.answer, "model": self.model,
            "latency_ms": self.latency_ms, "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens, "total_tokens": self.total_tokens,
            "citations": self.citations, "citation_coverage": self.citation_coverage,
            "chunks_used": [c.to_dict() for c in self.chunks_used],
        }


class Generator:
    def __init__(self, config: dict):
        self.gen_cfg = config["generation"]
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    def _build_context(self, chunks: list[RetrievedChunk]) -> str:
        parts = []
        for chunk in chunks:
            short_id = chunk.chunk_id[:8]
            parts.append(
                f"--- Chunk [{short_id}] (source: {chunk.source}, "
                f"relevance: {chunk.score:.4f}) ---\n{chunk.text}"
            )
        return "\n\n".join(parts)

    @staticmethod
    def extract_citations(text: str) -> list[str]:
        return re.findall(r"\[([a-f0-9]{8})\]", text)

    @staticmethod
    def compute_citation_coverage(text: str) -> float:
        sentences = [s.strip() for s in re.split(r"[.!?]", text) if len(s.strip().split()) >= 5]
        if not sentences:
            return 1.0
        cited = sum(1 for s in sentences if re.search(r"\[[a-f0-9]{8}\]", s))
        return cited / len(sentences)

    def generate(self, query: str, chunks: list[RetrievedChunk]) -> GeneratedAnswer:
        context = self._build_context(chunks)
        user_message = (
            f"Context:\n\n{context}\n\n---\n\n"
            f"Question: {query}\n\n"
            f"Provide a thorough answer using ONLY the context above. "
            f"Cite every factual claim with the chunk ID in square brackets, e.g. [abc12345]."
        )

        start = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.gen_cfg["model"],
            temperature=self.gen_cfg["temperature"],
            max_tokens=self.gen_cfg["max_tokens"],
            messages=[
                {"role": "system", "content": self.gen_cfg["system_prompt"]},
                {"role": "user", "content": user_message},
            ],
        )
        latency_ms = (time.perf_counter() - start) * 1000

        usage = response.usage
        answer_text = response.choices[0].message.content
        citations = self.extract_citations(answer_text)
        coverage = self.compute_citation_coverage(answer_text)

        return GeneratedAnswer(
            query=query, answer=answer_text, chunks_used=chunks,
            model=self.gen_cfg["model"], latency_ms=latency_ms,
            prompt_tokens=usage.prompt_tokens, completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens, citations=citations, citation_coverage=coverage,
        )
