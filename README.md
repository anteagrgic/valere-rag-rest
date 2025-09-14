# Valere RAG (REST)

Minimalni RAG demo: 20 Newsgroups → Qdrant (vector DB) → FastAPI REST → LLM (OpenAI/Ollama/Mock).

## Quickstart

1. Kloniraj i postavi env:
   ```bash
   cp .env.example .env
   # (po želji) otkomentiraj LLM_PROVIDER i model, npr.:
   # LLM_PROVIDER=ollama   # ili openai (treba API ključ)
2. Pokreni docker stack:
    ```bash
    docker compose up --build -d
3. (Opcionalno) Lokalno pokretanje umjesto Dockera:
    ```bash
    pip install -r requirements.txt
    uvicorn app.main:create_app --factory --reload
4. Ingest podataka (dataset: 20newsgroups):
    ```bash
    curl -X POST http://localhost:8000/ingest \
        -H "Content-Type: application/json" \
        -d '{"dataset":"20newsgroups","recreate":true}'
    # odgovor: {"collection":"newsgroups","chunks_indexed": N, "dim": <stvarna dimenzija>}
5. Postavi upit:
    ```bash
    curl -X POST http://localhost:8000/query \
        -H "Content-Type: application/json" \
        -d '{"query":"What is comp.graphics?", "k": 5}'



## Endpoints
- `GET /health` → `{ "status": "ok", "collection": "..." }`
- `GET /collections`→ `{ "collections": [...] }`
- `DELETE /collections/{name}`
- `POST /ingest`
    Body: `{"dataset":"20newsgroups","recreate":true}`
    Response: `{"collection":"newsgroups","chunks_indexed":N,"dim":<embedding-dim>}`
        Dimenzija se izračuna iz aktivnog embedding modela
- `POST /query`
    Body: `{"query":"...", "k":5}`
    Response:

        {
        "answer": "string",
        "sources": [
            {"id":"...", "score":0.0, "target_name":"...", "metadata": {...}, "content":"..."}
        ],
        "provider":"openai|ollama|mock",
        "model":"gpt-4o-mini|llama3.2:3b|mock",
        "latency_ms": 123,
        "prompt_tokens": 123,
        "completion_tokens": 45
        }
## Dev notes
- Embeddings: all-MiniLM-L6-v2 (dim=384)
- Vector DB: Qdrant (cosine)
- LLM: OpenAI / Ollama / Mock (fallback)
