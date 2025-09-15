# app/main.py
import time
import uvicorn
from .config import settings
from .api import build_app


def _wait_for_qdrant(timeout: int = 120, interval: float = 2.0) -> bool:
    """Čeka da se Qdrant digne tako da periodički zove get_collections()."""
    from .vectorstore import make_qdrant_client
    start = time.perf_counter()
    client = make_qdrant_client()
    while True:
        try:
            client.get_collections()
            return True
        except Exception:
            if time.perf_counter() - start > timeout:
                return False
            time.sleep(interval)


def _warmup_ollama() -> None:
    """
    Trigerira mali chat poziv kako bi Ollama model (npr. llama3.2:3b)
    bio učitan u memoriju prije prvog korisničkog upita.
    """
    try:
        from .llm import make_chat
        chat = make_chat()
        # kratki prompt — dovoljan da natjera load modela (poštuje tvoj llm_timeout)
        chat.invoke([{"role": "user", "content": "OK"}])
    except Exception:
        # ne ruši servis ako warmup ne uspije (npr. Ollama se još diže)
        pass


def create_app():
    app = build_app()

    # Pri startu: pričekaj Qdrant, pripremi embedding/vectorstore, pa ugrij LLM
    try:
        ok = _wait_for_qdrant(timeout=120, interval=2)
        if ok:
            from .vectorstore import get_embedding_service, get_or_create_vectorstore
            _ = get_embedding_service()  # warm-up HF embeddera
            _ = get_or_create_vectorstore(settings.qdrant_collection, recreate=False)
    except Exception:
        # ne ruši servis; endpointi će kasnije pokušati ponovno
        pass

    # LLM warm-up (odvojeno od Qdrant-a; nema ovisnosti)
    try:
        _warmup_ollama()
    except Exception:
        pass

    return app


if __name__ == "__main__":
    uvicorn.run(create_app(), host=settings.host, port=settings.port)
