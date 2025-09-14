# app/errors.py
from fastapi import FastAPI
from fastapi.responses import JSONResponse

# --- Domenske iznimke ---
class ProviderError(Exception):
    """LLM/vanjski provider nije dostupan ili je vratio grešku."""
    pass

class VectorStoreError(Exception):
    """Problemi s vektorskom bazom (Qdrant) ili store konfiguracijom."""
    pass

class IngestionError(Exception):
    """Greška prilikom ingest pipeline-a (dataset, chunking, upsert)."""
    pass


def register_exception_handlers(app: FastAPI) -> None:
    """Registrira FastAPI handlere koji mapiraju domenske iznimke na HTTP statuse."""
    @app.exception_handler(ProviderError)
    async def _provider_error_handler(_, exc: ProviderError):
        return JSONResponse(status_code=503, content={"detail": f"Provider unavailable: {exc}"})

    @app.exception_handler(VectorStoreError)
    async def _vector_error_handler(_, exc: VectorStoreError):
        return JSONResponse(status_code=503, content={"detail": f"Vector store error: {exc}"})

    @app.exception_handler(IngestionError)
    async def _ingestion_error_handler(_, exc: IngestionError):
        return JSONResponse(status_code=422, content={"detail": f"Ingestion error: {exc}"})
