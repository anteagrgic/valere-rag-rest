# app/main.py
import uvicorn
from .config import settings
from .api import build_app

def create_app():
    app = build_app()

    # (opcionalno) warm-up: iniciraj embedding servis i kolekciju jednom
    try:
        from .vectorstore import get_embedding_service, get_or_create_vectorstore
        _emb = get_embedding_service()
        _ = get_or_create_vectorstore(settings.qdrant_collection, recreate=False)
        # možeš i logirati: print(f"Warmed up embeddings dim={_emb.dim()}")
    except Exception:
        # warm-up je best-effort; ako padne, runtime će i dalje pokušati pri prvom pozivu rute
        pass

    return app

if __name__ == "__main__":
    uvicorn.run(create_app(), host=settings.host, port=settings.port)
