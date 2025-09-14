# app/providers/embedding_hf.py
from typing import List
from langchain_community.embeddings import HuggingFaceEmbeddings
from ..config import settings
from ..interfaces import EmbeddingService

class HFEmbeddingService(EmbeddingService):
    """
    Adapter oko HuggingFaceEmbeddings koji:
    - računa stvarnu dimenziju vektora (dim-probe),
    - izlaže embed_texts() i dim().
    """
    def __init__(self) -> None:
        self._emb = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"batch_size": 64, "normalize_embeddings": True},
        )
        # Dim-probe (jednom), bez hard-coda
        self._dim = len(self._emb.embed_query("dim-probe"))

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return self._emb.embed_documents(texts)

    def dim(self) -> int:
        return self._dim

    @property
    def raw(self) -> HuggingFaceEmbeddings:
        """Ako LangChain Qdrant traži 'embeddings' objekt, koristimo sirovi LC adapter."""
        return self._emb
