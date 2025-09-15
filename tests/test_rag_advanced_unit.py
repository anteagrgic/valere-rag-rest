# tests/test_rag_advanced_unit.py
import types
import builtins
import pytest

from app import rag_advanced as adv
from langchain.schema import Document

class DummyChat:
    def __init__(self, responses):
        self._responses = responses
        self._i = 0
    def provider(self): return "mock"
    def invoke(self, messages):
        # vrati sljedeći odgovor (ili zadnji)
        text = self._responses[min(self._i, len(self._responses)-1)]
        self._i += 1
        return types.SimpleNamespace(content=text, provider="mock", model="mock", latency_ms=1)

class DummyVS:
    def __init__(self, docs_by_query):
        self.docs_by_query = docs_by_query
    def as_retriever(self, search_kwargs=None):
        class R:
            def __init__(self, outer): self.outer = outer
            def invoke(self, q):
                # vrati listu dokumenta po upitu q
                return [Document(page_content=t, metadata={"source": f"s-{i}"}) for i, t in enumerate(self.outer.docs_by_query.get(q, []))]
        return R(self)

@pytest.fixture(autouse=True)
def patch_make_chat(monkeypatch):
    # prvi poziv (translation) -> vraća englesku verziju,
    # drugi poziv (multi-query) -> vrati 3 retka s alternativama,
    # treći (sinteza) -> bilo što, nije bitno (odgovor)
    chat = DummyChat(["who is the ceo of openai", "CEO of OpenAI\nOpenAI chief executive\nLeadership of OpenAI", "final"])
    monkeypatch.setattr(adv, "make_chat", lambda: chat)
    yield

@pytest.fixture(autouse=True)
def patch_vectorstore(monkeypatch):
    # mapiraj tri upita na razlicite dokumente
    docs_by_query = {
        "who is the ceo of openai": ["Q1 doc A", "Q1 doc B"],
        "CEO of OpenAI": ["Q2 doc A"],
        "OpenAI chief executive": ["Q3 doc A", "Q3 doc B"],
        "Leadership of OpenAI": ["Q4 doc A"]
    }
    vs = DummyVS(docs_by_query)
    monkeypatch.setattr(adv, "get_or_create_vectorstore", lambda *a, **k: vs)
    yield

def test_translate_guard(monkeypatch):
    # force False -> ne prevodi!!
    qt, meta = adv.translate_query_if_needed("ko je CEO OpenAI?", force=False)
    assert qt == "ko je CEO OpenAI?"
    assert meta["translated"] is False

def test_multi_query_generation(monkeypatch):
    qs, m = adv.generate_multi_queries("seed", n=3)
    assert len(qs) >= 1 
    assert m.get("provider") == "mock"

def test_rrf_fusion_scores():
    d1 = [Document(page_content="A"), Document(page_content="B")]
    d2 = [Document(page_content="B"), Document(page_content="C")]
    fused = adv.reciprocal_rank_fusion([d1, d2], k_smooth=60)
    # očekujemo da "B" bude visoko jer se pojavljuje u obje liste
    texts = [d.page_content for d, _ in fused[:3]]
    assert "B" in texts

def test_answer_advanced_fusion():
    res = adv.answer_advanced("ko je CEO OpenAI?", k=3, translate=True, mode="fusion", n_queries=3)
    assert isinstance(res.answer, str)
    assert len(res.docs) <= 3
    assert res.meta["mode"] == "fusion"
