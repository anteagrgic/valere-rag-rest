from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict


class IngestRequest(BaseModel):
    dataset: str = Field(default="20newsgroups", description="Currently supported: 20newsgroups")
    recreate: bool = Field(default=False, description="Drop and recreate collection before ingest")
    chunk_size: int | None = None
    chunk_overlap: int | None = None

class SourceDoc(BaseModel):
    id: str
    score: float | None = None
    target_name: str | None = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    content: str

class IngestResponse(BaseModel):
    collection: str
    chunks_indexed: int
    dim: int

class QueryRequest(BaseModel):
    query: str
    k: int = 5

    # Napredne opcije (opcionalno, unatrag kompatibilno)
    mode: Optional[str] = Field(
        default=None, description="simple | multi | fusion"
    )
    translate: Optional[bool] = Field(
        default=None, description="Force translate to English for retrieval (otherwise auto)"
    )
    n_queries: Optional[int] = Field(
        default=None, ge=1, le=10, description="Number of alternate queries for multi/fusion"
    )

class MetaInfo(BaseModel):
    """
    Struktuirani meta podaci o izvodenju upita / LLM-u / pipelineu.
    Dozvoljavamo dodatna polja (extra='allow') za buduca prosirenja.
    """
    provider: Optional[str] = None
    model: Optional[str] = None
    latency_ms: Optional[int] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None

    # Napredno (ako koristis translation/multi/fusion)
    translated: Optional[bool] = None
    mode: Optional[str] = None
    n_queries: Optional[int] = None
    rrf_k: Optional[int] = None

    # Ukupno vrijeme cijevi (ako ga saljes)
    pipeline_ms: Optional[int] = None

    class Config:
        extra = "allow"  # dopusti npr. dodatne kljuceve iz providera

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDoc]
    meta: Optional[MetaInfo] = None


class CollectionsResponse(BaseModel):
    collections: List[str]
