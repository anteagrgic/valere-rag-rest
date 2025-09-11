# Valere RAG (REST)

RAG demo preko 20 Newsgroups → Qdrant (vector DB) → FastAPI REST.

## Quickstart
1. `cp .env.example .env` (po želji postavi `LLM_PROVIDER=openai` + API ključ)
2. `docker compose up --build -d`
3. Ingest: `make ingest`
4. Postavi upit: `make query`
5. Swagger: http://localhost:8000/docs

## Endpoints
- `GET /health`
- `GET /collections`
- `DELETE /collections/{name}`
- `POST /ingest` → `{ "dataset":"20newsgroups", "recreate":true }`
- `POST /query` → `{ "query":"...", "k": 5 }`

## Dev notes
- Embeddings: all-MiniLM-L6-v2 (dim=384)
- Vector DB: Qdrant (cosine)
- LLM: OpenAI / Ollama / Mock (fallback)
