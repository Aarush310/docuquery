"""
test_citations.py — Citation extraction and coverage (src/generator.py).
Run: pytest tests/test_citations.py -v
"""

from src.generator import Generator

extract_citations = Generator.extract_citations
compute_citation_coverage = Generator.compute_citation_coverage


class TestCitationExtraction:
    def test_single(self):
        assert extract_citations("Claim [abc12345].") == ["abc12345"]

    def test_multiple(self):
        assert len(extract_citations("A [abc12345]. B [def67890].")) == 2

    def test_none(self):
        assert extract_citations("No citations here.") == []

    def test_ignores_non_hex(self):
        assert extract_citations("[XXXXXXXX]") == []

    def test_ignores_wrong_length(self):
        assert extract_citations("[abc1] [abc123456789]") == []

    def test_duplicates_preserved(self):
        assert len(extract_citations("[abc12345] and again [abc12345]")) == 2


class TestCitationCoverage:
    def test_full(self):
        text = "Every sentence has a citation here [abc12345]. This one too with evidence [def67890]."
        assert compute_citation_coverage(text) == 1.0

    def test_partial(self):
        text = "This sentence has a citation [abc12345]. This sentence does not have one at all."
        assert 0.0 < compute_citation_coverage(text) < 1.0

    def test_none(self):
        text = "No citations in this long enough sentence. Or in this one either for that matter."
        assert compute_citation_coverage(text) == 0.0

    def test_empty(self):
        assert compute_citation_coverage("") == 1.0

    def test_short_sentences_skipped(self):
        text = "Yes. No. But this longer sentence has a citation [abc12345]."
        assert compute_citation_coverage(text) == 1.0
