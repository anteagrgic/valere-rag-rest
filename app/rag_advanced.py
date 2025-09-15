# app/rag_advanced.py
from __future__ import annotations
from dataclasses import dataclass
from time import perf_counter
from typing import List, Dict, Any, Tuple, Union

from langchain.schema import Document
from langchain.load import dumps, loads

from .config import settings
from .vectorstore import get_or_create_vectorstore
from .llm import make_chat

# -------------------------
# Helpers
# -------------------------

def _is_english_only_embeddings(model_name: str) -> bool:
    name = (model_name or "").lower()
    multilingual_markers = [
        "multilingual", "paraphrase-multilingual", "labse", "xlm",
        "laser", "use-multilingual", "distiluse", "e5-multilingual"
    ]
    return not any(m in name for m in multilingual_markers)

def _res_content(res: Any) -> str:
    """
    Ekstrakcija teksta iz raznih povrata Chat sloja.
    Podržava: string, objekt s .content, dict s ['content'].
    """
    if res is None:
        return ""
    if isinstance(res, str):
        return res
    if hasattr(res, "content"):
        return getattr(res, "content") or ""
    if isinstance(res, dict):
        return str(res.get("content", ""))
    return str(res)

def _res_meta(res: Any) -> Dict[str, Any]:
    meta = {}
    for k in ("provider", "model", "latency_ms", "prompt_tokens", "completion_tokens"):
        if hasattr(res, k):
            meta[k] = getattr(res, k)
        elif isinstance(res, dict) and k in res:
            meta[k] = res[k]
    return meta

def translate_query_if_needed(query: str, force: bool | None = None) -> Tuple[str, Dict[str, Any]]:
    """
    Prevede korisnički upit na engleski za retrieval ako:
      - embedding model nije multilingual ili
      - force=True.
    Vraća (možda prevedeni upit, meta).
    """
    if force is False:
        return query, {"translated": False}
    
    if force is None and getattr(settings, "enable_query_translation", True) is False:
        return query, {"translated": False}

    if _is_english_only_embeddings(getattr(settings, "embedding_model", "")) or (force is True):
        chat = make_chat()
        # Ako je provider mock, preskoči
        if hasattr(chat, "provider") and callable(chat.provider) and chat.provider() == "mock":
            return query, {"translated": False, "provider": "mock"}

        prompt = (
            "Translate the following search query into **English** for semantic retrieval. "
            "Keep named entities in original form if they are proper nouns. "
            "Return ONLY the translated query; no quotes or explanations.\n\n"
            f"Query: {query}"
        )
        res = chat.invoke([{"role": "user", "content": prompt}])
        translated = _res_content(res).strip()
        if not translated:
            translated = query
        meta = {"translated": translated != query} | _res_meta(res)
        return translated, meta

    return query, {"translated": False}

def generate_multi_queries(seed_query: str, n: int = 3) -> Tuple[List[str], Dict[str, Any]]:
    """
    LLM generira N alternativnih, semantički različitih upita (uklj. original).
    Ako je provider 'mock' ili n<=1, vraća samo original.
    """
    chat = make_chat()
    if hasattr(chat, "provider") and callable(chat.provider) and chat.provider() == "mock":
        return [seed_query], {"provider": "mock"}
    if n <= 1:
        return [seed_query], {}

    prompt = (
        "Generate {n} diverse, semantically different search queries that would retrieve the "
        "same information as the original. Keep them concise. One per line. "
        "Do NOT number the lines.\n\n"
        f"Original query: {seed_query}\n\n"
        "Queries:"
    ).format(n=n)
    res = chat.invoke([{"role": "user", "content": prompt}])
    lines = [ln.strip(" •-\t") for ln in _res_content(res).splitlines() if ln.strip()]
    uniq, seen = [], set()
    for q in lines:
        key = q.lower()
        if q and key not in seen:
            uniq.append(q)
            seen.add(key)
    if seed_query.lower() not in seen:
        uniq.insert(0, seed_query)
    uniq = uniq[:max(1, n)]
    return uniq, {"count": len(uniq)} | _res_meta(res)

def reciprocal_rank_fusion(per_query_ranked_docs: List[List[Document]], k_smooth: int = 60) -> List[Tuple[Document, float]]:
    """
    Reciprocal Rank Fusion: score += 1 / (rank + 1 + k_smooth)
    Vraća listu (Document, fused_score) sortiranu silazno.
    """
    fused: Dict[str, float] = {}
    for docs in per_query_ranked_docs:
        for rank, doc in enumerate(docs):
            key = dumps(doc)  # stabilan ključ preko JSON serializacije LC Document
            fused[key] = fused.get(key, 0.0) + 1.0 / (rank + 1 + k_smooth)
    ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    return [(loads(k), score) for k, score in ranked]

def _format_context(docs: List[Document]) -> str:
    chunks = []
    for i, d in enumerate(docs, 1):
        meta = d.metadata or {}
        src = meta.get("source") or meta.get("doc_id") or meta.get("id") or f"doc-{i}"
        chunks.append(f"[{i}] ({src})\n{d.page_content}")
    return "\n\n".join(chunks)

# -------------------------
# Public entry point
# -------------------------

@dataclass
class AdvancedRAGResult:
    answer: str
    docs: List[Document]
    meta: Dict[str, Any]

def answer_advanced(
    question: str,
    *,
    k: int = 5,
    translate: bool | None = None,
    mode: str = "simple",            # 'simple' | 'multi' | 'fusion'
    n_queries: int = 4,
    k_smooth: int = 60,
) -> AdvancedRAGResult:

    t0 = perf_counter()
    vs = get_or_create_vectorstore(getattr(settings, "qdrant_collection", "default"))
    retriever = vs.as_retriever(search_kwargs={"k": k})

    # 1) Query translation (ako treba)
    q_translated, tr_meta = translate_query_if_needed(question, force=translate)

    # 2) Multi-query (opcionalno)
    queries = [q_translated]
    multi_meta: Dict[str, Any] = {}
    if mode in ("multi", "fusion"):
        queries, multi_meta = generate_multi_queries(q_translated, n=n_queries)

    # 3) Retrieval
    per_query_docs: List[List[Document]] = []
    for q in queries:
        try:
            docs = retriever.invoke(q)
        except Exception:
            docs = retriever.get_relevant_documents(q)  # starije LC verzije
        for idx, d in enumerate(docs):
            d.metadata = (d.metadata or {}) | {"_rank": idx}
        per_query_docs.append(docs)

    # 4) Spajanje lista
    if mode == "fusion":
        fused = reciprocal_rank_fusion(per_query_docs, k_smooth=k_smooth)
        used_docs = [d for d, _ in fused][:k]
        rerank_meta = {"rrf_k": k_smooth}
    elif mode == "multi":
        seen, tmp = set(), []
        for docs in per_query_docs:
            for d in docs:
                key = dumps(d)
                if key not in seen:
                    tmp.append(d)
                    seen.add(key)
        used_docs = tmp[:k]
        rerank_meta = {}
    else:
        used_docs = per_query_docs[0][:k] if per_query_docs else []
        rerank_meta = {}

    # 5) Sinteza (odgovor na jeziku pitanja)
    chat = make_chat()
    system = "You are a helpful assistant. Use ONLY the provided context. If answer is not in context, say you don't know."
    ctx = _format_context(used_docs)
    user = f"Question:\n{question}\n\nContext:\n{ctx}\n\nAnswer in the same language as the question."

    out = chat.invoke([{"role": "system", "content": system}, {"role": "user", "content": user}])
    answer = _res_content(out)

    meta: Dict[str, Any] = {
        "translated": tr_meta.get("translated"),
        "mode": mode,
        "n_queries": len(queries),
        **_res_meta(out),
        **rerank_meta,
    }
    meta["pipeline_ms"] = int((perf_counter() - t0) * 1000)
    return AdvancedRAGResult(answer=answer, docs=used_docs, meta=meta)
