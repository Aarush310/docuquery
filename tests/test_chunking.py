"""
test_chunking.py — Tests for document loading and chunking (src/ingest.py).
Run: pytest tests/test_chunking.py -v
"""

import pytest

from src.ingest import IngestError, chunk_text, collect_files, count_tokens, generate_chunk_id


class TestCountTokens:
    def test_basic(self):
        assert count_tokens("Hello world") > 0

    def test_empty(self):
        assert count_tokens("") == 0

    def test_long(self):
        assert count_tokens("word " * 1000) > 500


class TestChunking:
    def test_short_text_single_chunk(self):
        text = ("This is a short paragraph that has just enough words and tokens in it to clear the "
                "minimum chunk size threshold imposed by the default chunking configuration so the "
                "test passes reliably across environments.")
        assert len(chunk_text(text, chunk_size=600, min_chunk_size=10)) == 1

    def test_text_below_min_size_is_dropped(self):
        assert chunk_text("Too short to keep.", chunk_size=600, min_chunk_size=50) == []

    def test_long_text_multiple_chunks(self):
        text = "\n\n".join([f"Paragraph {i}. " * 30 for i in range(20)])
        assert len(chunk_text(text, chunk_size=200, chunk_overlap=50)) > 1

    def test_chunks_respect_size_limit(self):
        text = "\n\n".join([f"Paragraph {i}. " * 30 for i in range(20)])
        for chunk in chunk_text(text, chunk_size=200, chunk_overlap=50):
            assert chunk["token_count"] <= 200

    def test_overlap_more_chunks(self):
        text = "\n\n".join([f"Para {i}. " * 40 for i in range(10)])
        few = chunk_text(text, chunk_size=200, chunk_overlap=20)
        many = chunk_text(text, chunk_size=200, chunk_overlap=100)
        assert len(many) >= len(few)

    def test_overlap_repeats_trailing_paragraph(self):
        paras = [f"Unique paragraph {i} " + "filler " * 20 for i in range(6)]
        per_para = max(count_tokens(p) for p in paras)
        # Room for ~3 paragraphs per chunk, overlap budget fits exactly one.
        chunks = chunk_text("\n\n".join(paras), chunk_size=3 * per_para + 2,
                            chunk_overlap=per_para + 1, min_chunk_size=1)
        assert len(chunks) > 1
        # Some paragraph must appear at the end of one chunk and the start of the next.
        assert any(
            a["text"].split("\n\n")[-1] == b["text"].split("\n\n")[0]
            for a, b in zip(chunks, chunks[1:])
        )

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


class TestCollectFiles:
    def test_directory_filters_supported_types(self, tmp_path):
        (tmp_path / "a.md").write_text("x")
        (tmp_path / "b.txt").write_text("x")
        (tmp_path / "c.png").write_bytes(b"x")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "d.pdf").write_bytes(b"x")
        assert sorted(p.name for p in collect_files(tmp_path)) == ["a.md", "b.txt", "d.pdf"]

    def test_missing_path_raises(self, tmp_path):
        with pytest.raises(IngestError, match="not found"):
            collect_files(tmp_path / "nope")

    def test_no_supported_files_raises(self, tmp_path):
        (tmp_path / "c.png").write_bytes(b"x")
        with pytest.raises(IngestError, match="No supported"):
            collect_files(tmp_path)
