"""
reranker.py — Cross-Encoder Re-ranker

Re-scores retrieved chunks using a cross-encoder model that processes
(query, document) pairs jointly for higher relevance accuracy.

Cross-encoders are slower than bi-encoders but significantly more accurate,
which is why we apply them only to the top-k candidates from initial retrieval.
"""

from sentence_transformers import CrossEncoder

from src.retriever import RetrievedChunk


class Reranker:
    """Re-ranks chunks using a cross-encoder model."""

    def __init__(self, config: dict):
        reranker_cfg = config["retrieval"]["reranker"]
        self.top_n = reranker_cfg["top_n"]
        self.model = CrossEncoder(reranker_cfg["model"])

    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """
        Re-rank chunks by cross-encoder relevance score.

        Args:
            query: The user's question.
            chunks: Candidate chunks from hybrid retrieval.

        Returns:
            Top-N chunks re-sorted by cross-encoder score.
        """
        if not chunks:
            return []

        pairs = [(query, chunk.text) for chunk in chunks]
        scores = self.model.predict(pairs)

        for chunk, score in zip(chunks, scores):
            chunk.score = float(score)

        chunks.sort(key=lambda c: c.score, reverse=True)
        return chunks[:self.top_n]
