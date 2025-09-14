
import uvicorn
from .config import settings
from .api import build_app

def create_app():
    app = build_app()

    
    try:
        from .vectorstore import get_embedding_service, get_or_create_vectorstore
        _emb = get_embedding_service()
        _ = get_or_create_vectorstore(settings.qdrant_collection, recreate=False)
        
    except Exception:
        
        pass

    return app

if __name__ == "__main__":
    uvicorn.run(create_app(), host=settings.host, port=settings.port)

