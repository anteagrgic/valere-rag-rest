# tests/test_smoke.py
import httpx, pytest
BASE = "http://localhost:8000"

@pytest.mark.timeout(10)
def test_health():
    r = httpx.get(f"{BASE}/health", timeout=5)
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_collections():
    r = httpx.get(f"{BASE}/collections", timeout=5)
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data["collections"], list)
