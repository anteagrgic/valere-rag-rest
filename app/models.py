from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict

class IngestRequest(BaseModel):
    dataset: str = Field(default="20newsgroups", description="Currently supported: 20newsgroups")
    recreate: bool = Field(default=False, description="Drop and recreate collection before ingest")
    chunk_size: int | None = None
    chunk_overlap: int | None = None

class SourceDoc(BaseModel):
    id: str
    metadata: Dict[str, Any]
    content: str

class IngestResponse(BaseModel):
    collection: str
    chunks_indexed: int
    dim: int

class QueryRequest(BaseModel):
    query: str
    k: int = 5

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDoc]

class CollectionsResponse(BaseModel):
    collections: List[str]
