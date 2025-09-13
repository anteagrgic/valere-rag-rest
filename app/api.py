from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .models import IngestRequest, IngestResponse, QueryRequest, QueryResponse, CollectionsResponse

# PROMJENA: koristimo v2 chain
from .rag import answer_with_rag_v2
from .vectorstore import get_or_create_vectorstore, make_qdrant_client, as_sourcedocs

app = FastAPI(title="Valere RAG REST API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok", "collection": settings.qdrant_collection}

@app.get("/collections", response_model=CollectionsResponse)
def list_collections():
    client = make_qdrant_client()
    cols = [c.name for c in client.get_collections().collections]
    return CollectionsResponse(collections=cols)

@app.delete("/collections/{name}")
def delete_collection(name: str):
    client = make_qdrant_client()
    client.delete_collection(name)
    return {"deleted": name}

@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest):
    from .ingest import ingest_20newsgroups
    if req.dataset.lower() != "20newsgroups":
        raise HTTPException(400, "Only '20newsgroups' supported in this version.")
    # Ensure collection exists (and optionally recreate)
    _ = get_or_create_vectorstore(settings.qdrant_collection, recreate=req.recreate)
    n = ingest_20newsgroups(
        collection=settings.qdrant_collection,
        recreate=req.recreate,
        chunk_size=req.chunk_size or settings.chunk_size,
        chunk_overlap=req.chunk_overlap or settings.chunk_overlap,
    )
    return IngestResponse(collection=settings.qdrant_collection, chunks_indexed=n, dim=384)

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(400, "Empty query.")
    # PROMJENA: zovi v2 chain + default k ako je None/0
    k = req.k or 5
    try:
        answer, docs = answer_with_rag_v2(req.query, k=k)
    except Exception as e:
        # zgodno za debug da dobiješ razlog umjesto "puklo"
        raise HTTPException(status_code=500, detail=f"RAG error: {e}")
    return QueryResponse(answer=answer, sources=as_sourcedocs(docs))
