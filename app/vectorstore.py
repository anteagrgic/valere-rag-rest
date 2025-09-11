from typing import Iterable, List
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

from .config import settings

def make_embeddings():
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)

def make_qdrant_client() -> QdrantClient:
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
        timeout=60
    )

def drop_collection_if_exists(name: str):
    client = make_qdrant_client()
    if name in [c.name for c in client.get_collections().collections]:
        client.delete_collection(name)

def get_or_create_vectorstore(collection: str, recreate: bool = False) -> Qdrant:
    client = make_qdrant_client()
    if recreate:
        drop_collection_if_exists(collection)
    # Ensure collection exists with cosine distance
    if collection not in [c.name for c in client.get_collections().collections]:
        client.create_collection(
            collection_name=collection,
            vectors_config=qmodels.VectorParams(size=384, distance=qmodels.Distance.COSINE),  # 384 for MiniLM-L6
        )
    return Qdrant(client=client, collection_name=collection, embeddings=make_embeddings())

def add_documents(docs: Iterable[Document], collection: str) -> int:
    vs = get_or_create_vectorstore(collection)
    ids = vs.add_documents(list(docs))
    return len(ids)

def as_sourcedocs(docs: List[Document]):
    from .models import SourceDoc
    sdocs = []
    for i, d in enumerate(docs):
        sdocs.append(
            SourceDoc(
                id=str(d.metadata.get("doc_id", i)),
                metadata=d.metadata,
                content=d.page_content
            )
        )
    return sdocs
