# app/rag.py
from typing import List, Tuple, Dict, Any
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from .errors import ProviderError
from .config import settings
from .vectorstore import get_or_create_vectorstore
from .llm import make_chat

# -------------------------
# Prompt
# -------------------------

PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful assistant. Use ONLY the provided context to answer.
If the answer is not in the context, say you don't know.

Question:
{question}

Context:
{context}

Answer:"""
)

# -------------------------
# Helpers
# -------------------------

def format_context(docs: List[Document],
                   max_per_doc_chars: int = 1200,
                   max_docs: int | None = None) -> str:
    """
    Build a single string context from a list of Documents.
    - max_per_doc_chars: truncate each doc to this many chars
    - max_docs: if set, only use first `max_docs` documents
    """
    parts = []
    use_docs = docs[:max_docs] if (max_docs is not None and max_docs > 0) else docs
    for d in use_docs:
        meta = d.metadata or {}
        text = d.page_content or ""
        if max_per_doc_chars and len(text) > max_per_doc_chars:
            text = text[:max_per_doc_chars] + "…"
        parts.append(f"[{meta.get('target_name','unknown')}] {text}")
    return "\n---\n".join(parts)

def retrieve(question: str, k: int = 5) -> List[Document]:
    """
    Retrieve top-k documents from the vectorstore.
    """
    vs = get_or_create_vectorstore(settings.qdrant_collection)
    retriever = vs.as_retriever(search_kwargs={"k": k})
    # retriever.invoke(question) is LangChain retriever API — returns list[Document]
    return retriever.invoke(question)

# -------------------------
# Simple (legacy) RAG
# -------------------------

def answer_with_rag(question: str, k: int = 5) -> tuple[str, List[Document], dict]:
    docs = retrieve(question, k=k)
    context = format_context(docs, max_per_doc_chars=1200, max_docs=k)
    prompt = PROMPT.format_messages(question=question, context=context)
    chat = make_chat()
    msg_input = [{"role": m.type, "content": m.content} for m in prompt]
    try:
        res = chat.invoke(msg_input)          # sada vraća ChatResult
    except Exception as e:
        # podigni domensku iznimku koju API zna pretvoriti u 503
        raise ProviderError(str(e))
    text = res.content                    # uzimaš sadržaj
    meta = {
        "provider": res.provider,
        "model": res.model,
        "latency_ms": res.latency_ms,
        "prompt_tokens": res.prompt_tokens,
        "completion_tokens": res.completion_tokens,
    }

    return text, docs, meta

# -------------------------
# LCEL chain variant
# -------------------------

from langchain_core.runnables import RunnablePassthrough, RunnableLambda

def _call_llm_from_prompt(messages: List[Dict[str, Any]]):
    """Adapter: list-of-message-dicts -> make_chat.invoke  (returns ChatResult)"""
    chat = make_chat()
    try:
        return chat.invoke(messages)
    except Exception as e:
        from .errors import ProviderError
        raise ProviderError(str(e))
    #  make_chat.invoke expects a list of dicts like {"role": "...", "content": "..."}
    return chat.invoke(messages)

def _add_context(inp: Dict[str, Any]) -> Dict[str, Any]:
    """
    Input: {'question': str, 'retrieved': List[Document], 'k': int (optional)}
    Output: {'question': str, 'docs': List[Document], 'context': str, 'k': int}
    """
    docs = inp.get("retrieved", [])
    k = inp.get("k", None)
    ctx = format_context(docs, max_per_doc_chars=1200, max_docs=k)
    return {"question": inp.get("question"), "docs": docs, "context": ctx, "k": k}

def _render_and_call(inp: Dict[str, Any]) -> Dict[str, Any]:
    """
    Render PROMPT using question+context, call LLM and return standardized dict.
    Expects inp to contain 'question' and 'context' keys.
    """
    # Render prompt messages
    messages = PROMPT.format_messages(question=inp["question"], context=inp["context"])
    # Convert to simple message dicts
    msg_input = [{"role": m.type, "content": m.content} for m in messages]
    # Call LLM
    res = _call_llm_from_prompt(msg_input)   # sada vraća ChatResult
    return {
        "answer": res.content,
        "docs": inp.get("docs", []),
        "meta": {
            "provider": res.provider,
            "model": res.model,
            "latency_ms": res.latency_ms,
            "prompt_tokens": res.prompt_tokens,
            "completion_tokens": res.completion_tokens,
        },
    }

def build_rag_chain(k: int = 5):
    """
    LCEL chain: question -> retriever(top-K) -> add_context -> render -> LLM
    Returns a chain where invoking with the question string will perform retrieval internally.
    """
    vs = get_or_create_vectorstore(settings.qdrant_collection)
    retriever = vs.as_retriever(search_kwargs={"k": k})
    # The chain mapping: the retriever will be run with the question under the key 'retrieved'
    chain = (
        {"retrieved": retriever, "question": RunnablePassthrough()}
        | RunnableLambda(_add_context)
        | RunnableLambda(_render_and_call)
    )
    return chain

def answer_with_rag_v2(question: str, k: int = 5) -> tuple[str, List[Document], dict]:
    """
    Run the LCEL chain. Returns (answer_text, docs_used).
    """
    chain = build_rag_chain(k=k)
    # Invoke with the question; chain's retriever will get the question and do retrieval
    out = chain.invoke(question)
    # out expected to be {"answer": str, "docs": List[Document]}
    return out["answer"], out["docs"], out.get("meta", {})
