"""
test_chunking.py — Tests for document chunking logic.

Self-contained: replicates pure functions to avoid importing heavy ML deps.
Run: pytest tests/test_chunking.py -v
"""

import hashlib
import tiktoken

_encoder = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_encoder.encode(text))


def generate_chunk_id(source: str, index: int) -> str:
    raw = f"{source}::{index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def chunk_text(text, chunk_size=600, chunk_overlap=100, min_chunk_size=50):
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    def _get_overlap(paras, budget):
        result, total = [], 0
        for p in reversed(paras):
            pt = count_tokens(p)
            if total + pt > budget:
                break
            result.insert(0, p)
            total += pt
        return result, total

    chunks, cur_paras, cur_tok = [], [], 0

    for para in paragraphs:
        pt = count_tokens(para)
        if pt > chunk_size:
            if cur_paras:
                chunks.append({"text": "\n\n".join(cur_paras), "token_count": cur_tok})
                cur_paras, cur_tok = _get_overlap(cur_paras, chunk_overlap)
            tokens = _encoder.encode(para)
            s = 0
            while s < len(tokens):
                e = min(s + chunk_size, len(tokens))
                ct = tokens[s:e]
                if len(ct) >= min_chunk_size:
                    chunks.append({"text": _encoder.decode(ct), "token_count": len(ct)})
                s += chunk_size - chunk_overlap
            continue
        if cur_tok + pt > chunk_size and cur_paras:
            chunks.append({"text": "\n\n".join(cur_paras), "token_count": cur_tok})
            cur_paras, cur_tok = _get_overlap(cur_paras, chunk_overlap)
        cur_paras.append(para)
        cur_tok += pt

    if cur_paras:
        t = "\n\n".join(cur_paras)
        tc = count_tokens(t)
        if tc >= min_chunk_size:
            chunks.append({"text": t, "token_count": tc})
    return chunks


class TestCountTokens:
    def test_basic(self):
        assert count_tokens("Hello world") > 0

    def test_empty(self):
        assert count_tokens("") == 0

    def test_long(self):
        assert count_tokens("word " * 1000) > 500


class TestChunking:
    def test_short_text_single_chunk(self):
        assert len(chunk_text("Short paragraph.", chunk_size=600)) == 1

    def test_long_text_multiple_chunks(self):
        text = "\n\n".join([f"Paragraph {i}. " * 30 for i in range(20)])
        assert len(chunk_text(text, chunk_size=200, chunk_overlap=50)) > 1

    def test_overlap_more_chunks(self):
        text = "\n\n".join([f"Para {i}. " * 40 for i in range(10)])
        few = chunk_text(text, chunk_size=200, chunk_overlap=20)
        many = chunk_text(text, chunk_size=200, chunk_overlap=100)
        assert len(many) >= len(few)

    def test_min_filter(self):
        assert len(chunk_text("Tiny.\n\nAlso tiny.", chunk_size=600, min_chunk_size=100)) <= 1

    def test_required_fields(self):
        for c in chunk_text("A test document with enough content.", chunk_size=600):
            assert "text" in c and "token_count" in c
            assert isinstance(c["text"], str) and c["token_count"] > 0

    def test_empty(self):
        assert chunk_text("") == []

    def test_whitespace_only(self):
        assert chunk_text("   \n\n  \n  ") == []

    def test_long_paragraph_splits(self):
        assert len(chunk_text("word " * 2000, chunk_size=200, chunk_overlap=50)) > 5


class TestChunkId:
    def test_deterministic(self):
        assert generate_chunk_id("f.md", 0) == generate_chunk_id("f.md", 0)

    def test_different_inputs(self):
        a, b, c = generate_chunk_id("f.md", 0), generate_chunk_id("f.md", 1), generate_chunk_id("g.md", 0)
        assert a != b and a != c

    def test_length(self):
        assert len(generate_chunk_id("t", 0)) == 16

    def test_hex(self):
        assert all(c in "0123456789abcdef" for c in generate_chunk_id("t", 42))
