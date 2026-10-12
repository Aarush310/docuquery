"""
test_api.py — Tests for the FastAPI routes that don't need models or an API key.

Run: pytest tests/test_api.py -v
"""

from fastapi.testclient import TestClient

from src.app import app

client = TestClient(app)


class TestHealth:
    def test_healthz(self):
        res = client.get("/healthz")
        assert res.status_code == 200
        assert res.json() == {"status": "ok"}


class TestQueryValidation:
    def test_blank_query_rejected(self):
        res = client.post("/api/query", json={"query": "   "})
        assert res.status_code == 400

    def test_missing_query_rejected(self):
        res = client.post("/api/query", json={})
        assert res.status_code == 422

    def test_top_k_out_of_range_rejected(self):
        res = client.post("/api/query", json={"query": "What is RAG?", "top_k": 0})
        assert res.status_code == 422


class TestIngestPathSafety:
    def test_path_outside_data_rejected(self):
        res = client.post("/api/ingest", json={"source": "../"})
        assert res.status_code == 400

    def test_absolute_path_rejected(self):
        res = client.post("/api/ingest", json={"source": "/etc"})
        assert res.status_code == 400

    def test_missing_path_inside_data(self):
        res = client.post("/api/ingest", json={"source": "data/does-not-exist.md"})
        assert res.status_code == 404
