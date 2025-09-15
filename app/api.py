
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .config import settings
from .models import IngestRequest, IngestResponse, QueryRequest, QueryResponse, CollectionsResponse
from .errors import register_exception_handlers, ProviderError, IngestionError, VectorStoreError
from .models import IngestRequest, IngestResponse, QueryRequest, QueryResponse, CollectionsResponse, SourceDoc, MetaInfo

# koristimo v2 chain
from .rag import answer_with_rag_v2
# napredni RAG (translation + multi-query + fusion)
from .rag_advanced import answer_advanced
from .vectorstore import (
    get_or_create_vectorstore,
    make_qdrant_client,
    as_sourcedocs,
    get_embedding_service,
)

def build_app() -> FastAPI:
    app = FastAPI(title="Valere RAG REST API", version="0.1.0")
    register_exception_handlers(app)

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

        # 1) osiguraj kolekciju u Qdrantu (može pasti na krivoj konfiguraciji)
        try:
            _ = get_or_create_vectorstore(settings.qdrant_collection, recreate=req.recreate)
        except Exception as e:
            raise VectorStoreError(str(e))

        # 2) pokreni ingest pipeline (može pasti na datasetu/chunkingu/upsertu)
        try:
            n = ingest_20newsgroups(
                collection=settings.qdrant_collection,
                recreate=req.recreate,
                chunk_size=req.chunk_size or settings.chunk_size,
                chunk_overlap=req.chunk_overlap or settings.chunk_overlap,
            )
        except Exception as e:
            raise IngestionError(str(e))

        emb = get_embedding_service()
        return IngestResponse(collection=settings.qdrant_collection, chunks_indexed=n, dim=emb.dim())

    @app.post("/query", response_model=QueryResponse)
    def query_endpoint(req: QueryRequest):
        if not req.query.strip():
            raise HTTPException(400, "Empty query.")

        k = req.k or 5
        # Ako klijent pošalje napredne opcije -> koristi advanced,
        # inače fallback na postojeći v2 chain
        use_adv = bool(getattr(req, "mode", None)) or (getattr(req, "translate", None) is not None)

        try:
            if use_adv:
                mode = (
                    req.mode
                    or ("fusion" if getattr(settings, "enable_rag_fusion", True)
                        else "multi" if getattr(settings, "enable_multi_query", True)
                        else "simple")
                )

                res = answer_advanced(
                    req.query,
                    k=k,
                    translate=getattr(req, "translate", None),
                    mode=mode,
                    n_queries=getattr(req, "n_queries", None) or getattr(settings, "multi_query_n", 4),
                    k_smooth=getattr(settings, "rrf_k_smooth", 60),
                )
                answer, docs, meta = res.answer, res.docs, res.meta
            else:
                answer, docs, meta = answer_with_rag_v2(req.query, k=k)
        except ProviderError:
            raise
        except VectorStoreError:
            raise
        except Exception as e:
            raise ProviderError(str(e))

        return QueryResponse(
            answer=answer,
            sources=as_sourcedocs(docs),
            meta=MetaInfo(**(meta or {})),
        )

    return app

# kompatibilnost za uvicorn/gunicorn: uvicorn app.api:app
app = build_app()
