# Valere RAG (REST)

RAG demo: **20 Newsgroups → Qdrant (vector DB) → FastAPI REST**.

## Quickstart
1. **Env**
   ```bash
   cp .env.example .env

Choose one provider in .env:

## a) Ollama (Compose, nije potrebno ručno pokretati)
        LLM_PROVIDER=ollama
        OLLAMA_BASE_URL=http://ollama:11434
        OLLAMA_MODEL=llama3.2:3b
Za prvi start pokrenite: `docker exec -it ollama ollama pull <model>`

## b) OpenAI (paid, billing required)
        LLM_PROVIDER=openai
        OPENAI_API_KEY=sk-...
        OPENAI_MODEL=gpt-4o-mini

2. **Run**
    docker compose up --build -d

3. **Ingest**
    make ingest
     or:
     curl -X POST http://localhost:8000/ingest \
       -H "Content-Type: application/json" \
       -d '{"dataset":"20newsgroups", "recreate":true}'

4. **Query**
    make query
     or:
     curl -s -X POST http://localhost:8000/query \
       -H "Content-Type: application/json" \
       -d '{"query":"What is discussed about space exploration?","k":5}'

5. **Docs**
    Swagger: http://localhost:8000/docs
    Health: curl http://localhost:8000/health


## Endpoints
- `GET /health`
- `GET /collections`
- `DELETE /collections/{name}`
- `POST /ingest` → `{ "dataset":"20newsgroups", "recreate":true }`
- `POST /query` → `{ "query":"...", "k": 5 }`

## Dev notes
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (dim=384)
- Vector DB: Qdrant (cosine)
- LLM provider: OpenAI / Ollama / Mock (fallback)
- Chain: LCEL `answer_with_rag_v2` (retriever → prompt → LLM)
- `.env` is git-ignored (do not commit secrets)

## Troubleshooting
- 429 insufficient_quota (OpenAI): add billing or switch to `LLM_PROVIDER=ollama`
- Slow first response (Ollama/CPU): normal; try `k=1..2` or smaller model `llama3.2:3b`



## Apendix- Ollama quick starst(local)
1) Install (macOS)
`brew install ollama`

2) Start the server (keep this window open)
`ollama serve`
or as a service:
brew services start ollama

3) Pull a model (fast on CPU)
`ollama pull llama3.2:3b`

4) Test the server
curl -s http://localhost:11434/api/tags | jq .
curl -s http://localhost:11434/api/generate \
-d '{"model":"llama3.2:3b","prompt":"Say hi in one sentence.","stream":false}'

