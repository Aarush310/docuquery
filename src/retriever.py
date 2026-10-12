"""
retriever.py — Hybrid Retrieval Engine

Combines BM25 keyword search with vector semantic search using
configurable fusion weights. Supports both hybrid and pure vector modes.

The hybrid approach:
  1. BM25 scores all chunks by keyword relevance (exact terms, names, acronyms)
  2. Vector search scores by semantic similarity (meaning, paraphrases)
  3. Scores are normalized to [0,1] and combined: bm25_weight * bm25 + vector_weight * vector
  4. Top-k results are returned sorted by fused score
"""

import re
from dataclasses import dataclass

_TOKEN_RE = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    """Lower-cased word tokens for BM25 (drops punctuation, so "RAG?" matches "RAG")."""
    return _TOKEN_RE.findall(text.lower())


def normalize_bm25(scores) -> list[float]:
    """Scale BM25 scores to [0, 1] by the best score (all zeros stay zero)."""
    scores = [max(0.0, float(s)) for s in scores]
    best = max(scores, default=0.0)
    return [s / best for s in scores] if best > 0 else [0.0] * len(scores)


def fuse_scores(bm25_norm: list[float], vector_sims: list[float],
                bm25_weight: float, vector_weight: float) -> list[float]:
    """Weighted linear fusion of normalized keyword and semantic scores."""
    return [bm25_weight * b + vector_weight * v for b, v in zip(bm25_norm, vector_sims)]


@dataclass
class RetrievedChunk:
    """A single retrieved chunk with metadata."""
    chunk_id: str
    text: str
    score: float
    source: str
    chunk_index: int

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "score": self.score,
            "source": self.source,
            "chunk_index": self.chunk_index,
        }


class Retriever:
    """
    Hybrid retrieval engine combining BM25 + vector search.
    """

    def __init__(self, config: dict):
        self.config = config
        self.ret_cfg = config["retrieval"]
        self.embed_cfg = config["embedding"]
        self.store_cfg = config["vector_store"]

        import chromadb
        from sentence_transformers import SentenceTransformer

        # Load embedding model
        self.embedder = SentenceTransformer(self.embed_cfg["model"])

        # Connect to ChromaDB (raises if documents haven't been ingested yet)
        self.client = chromadb.PersistentClient(path=self.store_cfg["persist_directory"])
        self.collection = self.client.get_collection(name=self.store_cfg["collection_name"])

        # Build BM25 index from all stored documents
        self._build_bm25_index()

    def _build_bm25_index(self):
        """Load all documents from ChromaDB and build a BM25 index."""
        results = self.collection.get(include=["documents", "metadatas"])
        self._all_ids = results["ids"]
        self._all_docs = results["documents"]
        self._all_metadatas = results["metadatas"]

        if not self._all_docs:
            raise RuntimeError("The document index is empty — run ingestion first.")

        from rank_bm25 import BM25Okapi
        self._bm25 = BM25Okapi([tokenize(doc) for doc in self._all_docs])

    @property
    def num_chunks(self) -> int:
        return len(self._all_ids)

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """
        Retrieve top-k chunks using hybrid BM25 + vector fusion.
        """
        top_k = top_k or self.ret_cfg["top_k"]
        bm25_weight = self.ret_cfg["bm25_weight"]
        vector_weight = self.ret_cfg["vector_weight"]

        # ── BM25 scores ──
        bm25_norm = normalize_bm25(self._bm25.get_scores(tokenize(query)))

        # ── Vector scores ──
        query_embedding = self.embedder.encode(query).tolist()
        # Fetch enough candidates for fusion
        n_candidates = min(len(self._all_ids), max(top_k * 3, 20))
        vec_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_candidates,
            include=["distances"],
        )

        # Build a map: chunk_id → cosine similarity
        vector_score_map = {}
        for i, cid in enumerate(vec_results["ids"][0]):
            distance = vec_results["distances"][0][i]
            vector_score_map[cid] = 1.0 - distance  # cosine distance → similarity

        # ── Fuse scores ──
        # Chunks outside the vector candidate set get a semantic score of 0.
        vector_sims = [vector_score_map.get(doc_id, 0.0) for doc_id in self._all_ids]
        fused = fuse_scores(bm25_norm, vector_sims, bm25_weight, vector_weight)

        scored = []
        for i, doc_id in enumerate(self._all_ids):
            scored.append(RetrievedChunk(
                chunk_id=doc_id,
                text=self._all_docs[i],
                score=fused[i],
                source=self._all_metadatas[i].get("source", "unknown"),
                chunk_index=self._all_metadatas[i].get("chunk_index", -1),
            ))

        scored.sort(key=lambda c: c.score, reverse=True)
        return scored[:top_k]
