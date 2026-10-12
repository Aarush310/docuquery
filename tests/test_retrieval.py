"""
test_retrieval.py — Hybrid score fusion and cross-encoder re-ranking logic.
Run: pytest tests/test_retrieval.py -v
"""

from src.reranker import Reranker
from src.retriever import fuse_scores, normalize_bm25, tokenize
from tests.fakes import make_chunk


class TestTokenize:
    def test_lowercases_and_strips_punctuation(self):
        assert tokenize("What is RAG? (BM25, ChromaDB)") == ["what", "is", "rag", "bm25", "chromadb"]


class TestNormalizeBM25:
    def test_scales_to_unit_max(self):
        assert normalize_bm25([2.0, 1.0, 0.0]) == [1.0, 0.5, 0.0]

    def test_all_zero_stays_zero(self):
        assert normalize_bm25([0.0, 0.0]) == [0.0, 0.0]

    def test_negative_scores_clamped(self):
        assert normalize_bm25([-1.0, 4.0]) == [0.0, 1.0]


class TestFusion:
    def test_weighted_sum(self):
        fused = fuse_scores([1.0, 0.0], [0.0, 1.0], bm25_weight=0.3, vector_weight=0.7)
        assert fused == [0.3, 0.7]

    def test_keyword_match_can_beat_weak_semantic_match(self):
        # exact keyword hit + decent semantics beats strong semantics with no keyword overlap
        fused = fuse_scores([1.0, 0.0], [0.6, 0.8], 0.3, 0.7)
        assert fused[0] > fused[1]


class TestReranker:
    def _reranker(self, scores, top_n=2):
        r = Reranker.__new__(Reranker)  # skip loading the real cross-encoder
        r.top_n = top_n
        r.model = type("M", (), {"predict": lambda self, pairs: scores[: len(pairs)]})()
        return r

    def test_reorders_by_cross_encoder_score_and_truncates(self):
        chunks = [make_chunk(cid=f"{i:016x}", text=f"t{i}") for i in range(3)]
        out = self._reranker([0.1, 0.9, 0.5]).rerank("q", chunks)
        assert [c.text for c in out] == ["t1", "t2"]
        assert out[0].score == 0.9

    def test_empty(self):
        assert self._reranker([]).rerank("q", []) == []
