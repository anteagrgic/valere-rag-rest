from typing import List
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document

from .config import settings
from .vectorstore import get_or_create_vectorstore

from .llm import make_chat

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
