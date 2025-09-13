# app/vectorstore.py

from typing import Iterable, List
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

from .config import settings

# cache da ne kreiramo više puta
_EMB = None

def make_embeddings():
    global _EMB
    if _EMB is None:
        _EMB = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={
                "batch_size": 64,               # ← batchiranje embedanja (CPU friendly)
                "normalize_embeddings": True,   # ← standardno za cosine
            },
        )
    return _EMB

def make_qdrant_client() -> QdrantClient:
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
        timeout=60,
    )

def drop_collection_if_exists(name: str):
    client = make_qdrant_client()
    if name in [c.name for c in client.get_collections().collections]:
        client.delete_collection(name)

def _ensure_collection(client: QdrantClient, collection: str, embeddings: HuggingFaceEmbeddings):
    # Odredi dimenziju embed vektora iz stvarnog modela (bez hard-codinga 384)
    try:
        dim = len(embeddings.embed_query("dim-probe"))
    except Exception:
        # fallback ako embed_query ne uspije iz bilo kojeg razloga
        dim = 384  # MiniLM-L3/L6 su 384; možeš promijeniti ako znaš da je drugačije
    existing = [c.name for c in client.get_collections().collections]
    if collection not in existing:
        client.create_collection(
            collection_name=collection,
            vectors_config=qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE),
        )

def get_or_create_vectorstore(collection: str, recreate: bool = False) -> Qdrant:
    client = make_qdrant_client()
    emb = make_embeddings()
    if recreate:
        drop_collection_if_exists(collection)
    _ensure_collection(client, collection, emb)
    return Qdrant(client=client, collection_name=collection, embeddings=emb)

def add_documents(docs: Iterable[Document], collection: str) -> int:
    vs = get_or_create_vectorstore(collection)
    ids = vs.add_documents(list(docs))  # LangChain sada koristi batch_size iz encode_kwargs
    return len(ids)

def as_sourcedocs(docs: List[Document]):
    from .models import SourceDoc
    sdocs = []
    for i, d in enumerate(docs):
        sdocs.append(
            SourceDoc(
                id=str(d.metadata.get("doc_id", i)),
                metadata=d.metadata,
                content=d.page_content,
            )
        )
    return sdocs
