from typing import List, Tuple
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document

from .config import settings
from .vectorstore import get_or_create_vectorstore
from .llm import make_chat

# -------------------------
# Postojeći RAG "ručni chain"
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

def format_context(docs: List[Document]) -> str:
    parts = []
    for d in docs:
        meta = d.metadata
        parts.append(f"[{meta.get('target_name','unknown')}] {d.page_content}")
    return "\n---\n".join(parts)

def retrieve(question: str, k: int = 5) -> List[Document]:
    vs = get_or_create_vectorstore(settings.qdrant_collection)
    retriever = vs.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(question)

def answer_with_rag(question: str, k: int = 5) -> tuple[str, List[Document]]:
    docs = retrieve(question, k=k)
    context = format_context(docs)
    prompt = PROMPT.format_messages(question=question, context=context)
    chat = make_chat()
    output = chat.invoke([{"role": m.type, "content": m.content} for m in prompt])
    return output, docs

# -----------------------------------
# NOVO: LCEL "pravi" chain za RAG
# -----------------------------------

from langchain_core.runnables import RunnablePassthrough, RunnableLambda

def _call_llm_from_prompt(messages):
    """Adapter: PROMPT -> make_chat()"""
    chat = make_chat()
    msgs = []
    for m in messages:
        role = getattr(m, "type", None) or (m.get("role") if isinstance(m, dict) else None)
        content = getattr(m, "content", None) or (m.get("content") if isinstance(m, dict) else None)
        msgs.append({"role": role, "content": content})
    return chat.invoke(msgs)

def _add_context(inp: dict):
    """Ulaz: {'question': str, 'retrieved': List[Document]}  ->  Izlaz: +context string"""
    docs = inp["retrieved"]
    return {"question": inp["question"], "docs": docs, "context": format_context(docs)}

def _render_and_call(inp: dict):
    """Renderira PROMPT i poziva LLM; vraća {'answer': str, 'docs': List[Document]}"""
    messages = PROMPT.format_messages(question=inp["question"], context=inp["context"])
    answer = _call_llm_from_prompt(messages)
    return {"answer": answer, "docs": inp["docs"]}

def build_rag_chain(k: int = 5):
    """LCEL chain: question -> retriever(top-K) -> format_context -> PROMPT -> LLM"""
    vs = get_or_create_vectorstore(settings.qdrant_collection)
    retriever = vs.as_retriever(search_kwargs={"k": k})
    chain = (
        {"retrieved": retriever, "question": RunnablePassthrough()}
        | RunnableLambda(_add_context)
        | RunnableLambda(_render_and_call)
    )
    return chain

def answer_with_rag_v2(question: str, k: int = 5) -> tuple[str, List[Document]]:
    """Isti povrat kao stara: (answer, docs), ali preko LCEL chain-a."""
    chain = build_rag_chain(k=k)
    out = chain.invoke(question)
    return out["answer"], out["docs"]
