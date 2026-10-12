"""
test_chunking.py — Tests for document chunking logic.

Imports the real functions from src.ingest, so a change to the chunker
is actually tested.
Run: pytest tests/test_chunking.py -v
"""

from src.ingest import chunk_text, count_tokens, generate_chunk_id


class TestCountTokens:
    def test_basic(self):
        assert count_tokens("Hello world") > 0

    def test_empty(self):
        assert count_tokens("") == 0

    def test_long(self):
        assert count_tokens("word " * 1000) > 500


class TestChunking:
    def test_short_text_single_chunk(self):
        # One paragraph of roughly 110 tokens: above min_chunk_size (50), below chunk_size (600)
        sentence = "This short paragraph repeats one sentence so it clears the minimum chunk size but stays far below the maximum. "
        text = (sentence * 5).strip()
        assert 50 < count_tokens(text) < 600
        assert len(chunk_text(text, chunk_size=600)) == 1

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
