# tests/test_api_smoke.py
import types
from fastapi.testclient import TestClient
from app.api import build_app

def test_query_baseline(monkeypatch):
    from app import rag
    monkeypatch.setattr(rag, "answer_with_rag_v2", lambda q, k=5: ("A", [], {"provider":"mock","model":"mock","latency_ms":1}))
    app = build_app()
    client = TestClient(app)
    resp = client.post("/query", json={"query":"hello","k":2})
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "A"
    assert "meta" in data

def test_query_advanced(monkeypatch):
    from app import rag_advanced as adv
    fake = types.SimpleNamespace(answer="B", docs=[], meta={"mode":"fusion"})
    monkeypatch.setattr(adv, "answer_advanced", lambda *a, **k: fake)
    app = build_app()
    client = TestClient(app)
    resp = client.post("/query", json={"query":"hi","k":2,"mode":"fusion","n_queries":3})
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "B"
    assert data["meta"]["mode"] == "fusion"
