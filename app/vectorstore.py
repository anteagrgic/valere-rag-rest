
from typing import Iterable, List
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from langchain_community.vectorstores import Qdrant as LCQdrant
from langchain.schema import Document
from .errors import VectorStoreError
from .config import settings
from .providers.embedding_hf import HFEmbeddingService


_EMB_SVC: HFEmbeddingService | None = None

def get_embedding_service() -> HFEmbeddingService:
    global _EMB_SVC
    if _EMB_SVC is None:
        _EMB_SVC = HFEmbeddingService()
    return _EMB_SVC


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

        
def _ensure_collection(client: QdrantClient, collection: str, dim: int):
    existing = [c.name for c in client.get_collections().collections]
    if collection not in existing:
        client.create_collection(
            collection_name=collection,
            vectors_config=qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE),
        )

def get_or_create_vectorstore(collection: str, recreate: bool = False) -> LCQdrant:           
    try:
            client = make_qdrant_client()
            emb = get_embedding_service() # <-- adapter (sa stvarnom dimenzijom)
    except Exception as e:
        raise VectorStoreError(str(e))

    if recreate:
        try:
            drop_collection_if_exists(collection)
        except Exception as e:
            raise VectorStoreError(str(e))

    try:
        _ensure_collection(client, collection, emb.dim())
    except Exception as e:
        raise VectorStoreError(str(e))
   
    # LangChain-ov Qdrant wrapper koristi "raw" embeddings objekt
    return LCQdrant(client=client, collection_name=collection, embeddings=emb.raw)


def add_documents(docs: Iterable[Document], collection: str) -> int:
    vs = get_or_create_vectorstore(collection)
    ids = vs.add_documents(list(docs))
    return len(ids)

def as_sourcedocs(docs: List[Document]):
    from .models import SourceDoc
    sdocs = []
    for i, d in enumerate(docs):

        m = d.metadata or {}
        sdocs.append(
            SourceDoc(
                id=str(m.get("doc_id", i)),
                score=float(m.get("score") or m.get("_score") or 0.0),
                target_name=m.get("target_name"),
                metadata=m,
                content=d.page_content,

            )
        )
    return sdocs
