"""
test_citations.py — Tests for citation extraction and coverage.

Self-contained: replicates pure functions to avoid heavy ML deps.
Run: pytest tests/test_citations.py -v
"""

import re


def extract_citations(text: str) -> list[str]:
    return re.findall(r"\[([a-f0-9]{8})\]", text)


def compute_citation_coverage(text: str) -> float:
    sentences = [s.strip() for s in re.split(r"[.!?]", text) if len(s.strip().split()) >= 5]
    if not sentences:
        return 1.0
    cited = sum(1 for s in sentences if re.search(r"\[[a-f0-9]{8}\]", s))
    return cited / len(sentences)


class TestCitationExtraction:
    def test_single(self):
        assert extract_citations("Claim [abc12345].") == ["abc12345"]

    def test_multiple(self):
        cites = extract_citations("A [abc12345]. B [def67890].")
        assert len(cites) == 2

    def test_none(self):
        assert extract_citations("No citations here.") == []

    def test_ignores_non_hex(self):
        assert extract_citations("[XXXXXXXX]") == []

    def test_ignores_wrong_length(self):
        assert extract_citations("[abc1] [abc123456789]") == []

    def test_duplicates_preserved(self):
        cites = extract_citations("[abc12345] and again [abc12345]")
        assert len(cites) == 2


class TestCitationCoverage:
    def test_full(self):
        text = "Every sentence has a citation here [abc12345]. This one too with evidence [def67890]."
        assert compute_citation_coverage(text) == 1.0

    def test_partial(self):
        text = "This sentence has a citation [abc12345]. This sentence does not have one at all."
        cov = compute_citation_coverage(text)
        assert 0.0 < cov < 1.0

    def test_none(self):
        text = "No citations in this long enough sentence. Or in this one either for that matter."
        assert compute_citation_coverage(text) == 0.0

    def test_empty(self):
        assert compute_citation_coverage("") == 1.0

    def test_short_sentences_skipped(self):
        text = "Yes. No. But this longer sentence has a citation [abc12345]."
        assert compute_citation_coverage(text) == 1.0
