"""
generator.py — LLM Answer Generation with Citation Enforcement

Produces grounded answers from retrieved context chunks. Every factual claim
must cite its source chunk by ID ([abc12345]); citation coverage is measured on
every answer.

The LLM provider is pluggable (configs/config.yaml -> generation.provider, or
the DOCUQUERY_LLM_PROVIDER env var):
  - "groq"      — Llama 3.3 70B via Groq (free tier)   needs GROQ_API_KEY
  - "anthropic" — Claude via the Anthropic API          needs ANTHROPIC_API_KEY
"""

import os
import re
import time
from dataclasses import dataclass, field

from src.retriever import RetrievedChunk

CITATION_RE = re.compile(r"\[([a-f0-9]{8})\]")

DEFAULT_SYSTEM_PROMPT = (
    "You are a precise research assistant. Answer ONLY from the provided context chunks, "
    "cite every factual claim with its chunk ID in square brackets, e.g. [abc12345], and if "
    "the context is insufficient say: \"I don't have enough information in the provided "
    "documents to answer this question.\""
)


# ─── LLM clients ────────────────────────────────────────────────

@dataclass
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int


class LLMClient:
    """Minimal interface every provider implements."""

    provider = "base"

    def complete(self, system: str, user: str, model: str,
                 temperature: float, max_tokens: int) -> Completion:
        raise NotImplementedError


class GroqClient(LLMClient):
    provider = "groq"

    def __init__(self, client=None):
        if client is None:
            from groq import Groq  # imported lazily so other providers don't need it
            client = Groq(api_key=_require_env("GROQ_API_KEY"))
        self._client = client

    def complete(self, system, user, model, temperature, max_tokens):
        response = self._client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return Completion(
            text=response.choices[0].message.content or "",
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
        )


class AnthropicClient(LLMClient):
    provider = "anthropic"

    def __init__(self, client=None):
        if client is None:
            import anthropic
            client = anthropic.Anthropic(api_key=_require_env("ANTHROPIC_API_KEY"))
        self._client = client

    def complete(self, system, user, model, temperature, max_tokens):
        response = self._client.messages.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(b.text for b in response.content if b.type == "text")
        return Completion(
            text=text,
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
        )


PROVIDERS = {"groq": GroqClient, "anthropic": AnthropicClient}


def make_llm_client(provider: str) -> LLMClient:
    try:
        return PROVIDERS[provider.lower()]()
    except KeyError:
        raise ValueError(f"Unknown LLM provider {provider!r}; expected one of {sorted(PROVIDERS)}")


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not set — export it or add it to your environment/secret.")
    return value


# ─── Generation ─────────────────────────────────────────────────

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
    invalid_citations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "query": self.query, "answer": self.answer, "model": self.model,
            "latency_ms": self.latency_ms, "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens, "total_tokens": self.total_tokens,
            "citations": self.citations, "citation_coverage": self.citation_coverage,
            "invalid_citations": self.invalid_citations,
            "chunks_used": [c.to_dict() for c in self.chunks_used],
        }


class Generator:
    def __init__(self, config: dict, llm: LLMClient | None = None):
        self.gen_cfg = config["generation"]
        self.llm = llm or make_llm_client(self.gen_cfg.get("provider", "groq"))

    @staticmethod
    def build_context(chunks: list[RetrievedChunk]) -> str:
        parts = []
        for chunk in chunks:
            parts.append(
                f"--- Chunk [{chunk.chunk_id[:8]}] (source: {chunk.source}, "
                f"relevance: {chunk.score:.4f}) ---\n{chunk.text}"
            )
        return "\n\n".join(parts)

    @staticmethod
    def extract_citations(text: str) -> list[str]:
        return CITATION_RE.findall(text)

    @staticmethod
    def compute_citation_coverage(text: str) -> float:
        """Share of substantive sentences (5+ words) that carry at least one citation."""
        sentences = [s.strip() for s in re.split(r"[.!?]", text) if len(s.strip().split()) >= 5]
        if not sentences:
            return 1.0
        cited = sum(1 for s in sentences if CITATION_RE.search(s))
        return cited / len(sentences)

    def generate(self, query: str, chunks: list[RetrievedChunk]) -> GeneratedAnswer:
        context = self.build_context(chunks)
        user_message = (
            f"Context:\n\n{context}\n\n---\n\n"
            f"Question: {query}\n\n"
            f"Provide a thorough answer using ONLY the context above. "
            f"Cite every factual claim with the chunk ID in square brackets, e.g. [abc12345]."
        )

        start = time.perf_counter()
        completion = self.llm.complete(
            system=self.gen_cfg.get("system_prompt") or DEFAULT_SYSTEM_PROMPT,
            user=user_message,
            model=self.gen_cfg["model"],
            temperature=self.gen_cfg["temperature"],
            max_tokens=self.gen_cfg["max_tokens"],
        )
        latency_ms = (time.perf_counter() - start) * 1000

        answer_text = completion.text
        citations = self.extract_citations(answer_text)
        valid_ids = {c.chunk_id[:8] for c in chunks}
        invalid = sorted({c for c in citations if c not in valid_ids})

        return GeneratedAnswer(
            query=query, answer=answer_text, chunks_used=chunks,
            model=f"{self.llm.provider}/{self.gen_cfg['model']}", latency_ms=latency_ms,
            prompt_tokens=completion.prompt_tokens,
            completion_tokens=completion.completion_tokens,
            total_tokens=completion.prompt_tokens + completion.completion_tokens,
            citations=citations,
            citation_coverage=self.compute_citation_coverage(answer_text),
            invalid_citations=invalid,
        )
