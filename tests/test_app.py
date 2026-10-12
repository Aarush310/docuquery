"""
test_app.py — HTTP API, health probes, metrics endpoint and ingest path safety.
Run: pytest tests/test_app.py -v
"""

import pytest
from fastapi.testclient import TestClient

from src import app as app_module
from tests.fakes import fake_answer


class FakeOrch:
    def __init__(self):
        self.retriever = type("R", (), {"num_chunks": 42})()

    def query(self, q, top_k=None):
        return fake_answer()

    def get_metrics(self):
        return {"total_requests": 1}

    def get_recent_traces(self, limit):
        return [{"trace_id": "t1"}]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(app_module, "_orchestrator", FakeOrch())
    return TestClient(app_module.app)


def test_healthz(client):
    assert client.get("/healthz").json() == {"status": "ok"}


def test_readyz_ready(client):
    body = client.get("/readyz").json()
    assert body["status"] == "ready" and body["chunks_indexed"] == 42


def test_readyz_not_ready_when_pipeline_cannot_load(monkeypatch):
    monkeypatch.setattr(app_module, "_orchestrator", None)

    def fail():
        raise RuntimeError("index missing")

    monkeypatch.setattr(app_module, "get_orchestrator", fail)
    r = TestClient(app_module.app).get("/readyz")
    assert r.status_code == 503 and "index missing" in r.json()["reason"]


def test_query_ok(client):
    r = client.post("/api/query", json={"query": "What is RAG?"})
    assert r.status_code == 200
    body = r.json()
    assert body["citations"] == ["abc12345"]
    assert body["tokens"]["total"] == 60
    assert body["chunks"][0]["source"] == "doc.md"


@pytest.mark.parametrize("payload", [{}, {"query": ""}, {"query": "x", "top_k": 0}])
def test_query_validation(client, payload):
    assert client.post("/api/query", json=payload).status_code == 422


def test_query_blank_after_strip(client):
    assert client.post("/api/query", json={"query": "   "}).status_code == 400


@pytest.mark.parametrize("source", ["/etc", "../", "data/../src", "/"])
def test_ingest_rejects_paths_outside_data_dir(client, source):
    r = client.post("/api/ingest", json={"source": source})
    assert r.status_code == 400
    assert "inside the data directory" in r.json()["error"]


def test_ingest_allows_data_dir():
    assert app_module.resolve_ingest_path("data/").name == "data"
    assert app_module.resolve_ingest_path("data/sample.md").name == "sample.md"


def test_prometheus_metrics(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "docuquery_requests_total" in r.text


def test_api_metrics_and_traces(client):
    assert client.get("/api/metrics").json() == {"total_requests": 1}
    assert client.get("/api/traces?limit=5").json() == [{"trace_id": "t1"}]


def test_index_page(client):
    r = client.get("/")
    assert r.status_code == 200 and "DocuQuery" in r.text
