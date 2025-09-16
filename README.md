# Valere RAG (REST)

Minimalni RAG demo: **20 Newsgroups → Qdrant (vector DB) → FastAPI REST → LLM (OpenAI/Ollama/Mock)**  
Uz to: **Query Translation**, **Multi-Query** i **RAG-Fusion (RRF)**.

---

## Sadržaj
- [Quickstart (Docker)](#quickstart-docker)
- [Točan redoslijed pokretanja — terminal pipeline](#točan-redoslijed-pokretanja--terminal-pipeline)
- [Endpoints](#endpoints)
- [Napredni RAG (Translation / Multi-Query / Fusion)](#napredni-rag-translation--multi-query--fusion)
- [Primjeri cURL poziva](#primjeri-curl-poziva)
- [Bash testovi (kopiraj u .sh)](#bash-testovi-kopiraj-u-sh)
- [Lokalno (bez Dockera)](#lokalno-bez-dockera)
- [Konfiguracija (.env)](#konfiguracija-env)
- [Troubleshooting](#troubleshooting)
- [Dev notes](#dev-notes)

---

## Quickstart (Docker)

1. Kloniraj i postavi env:
   ```bash
   cp .env.example .env
   # preporučeno (Ollama u docker compose-u):
   # LLM_PROVIDER=ollama
   # OLLAMA_BASE_URL=http://ollama:11434
   # OLLAMA_MODEL=llama3.2:3b
   # LLM_TIMEOUT=300
   ```

2. Pokreni docker stack (sve servise):
   ```bash
   docker compose up --build -d
   ```

3. Provjere:
   ```bash
   curl -sS http://localhost:8000/health | jq
   curl -sS http://localhost:6333/collections | jq
   ```

4. Ingest podataka (dataset: 20 Newsgroups):
   ```bash
   curl -sS -X POST http://localhost:8000/ingest      -H "Content-Type: application/json"      -d '{"dataset":"20newsgroups","recreate":true}' | jq
   ```

5. Postavi upit (baseline):
   ```bash
   curl -sS -X POST http://localhost:8000/query      -H "Content-Type: application/json"      -d '{"query":"What is comp.graphics?","k":5}' | jq
   ```

---

## Točan redoslijed pokretanja — terminal pipeline

> Ako koristiš `docker compose up -d`, Compose će pokrenuti sve servise i ovisnosti.  
> U ručnom/Debug scenariju koristi slijed ispod:

```bash
# 1) Digni bazne servise
docker compose up -d qdrant ollama

# 2) Povuci model u docker Ollami (one-shot init servis ili ručno)
docker compose up -d ollama-init
# ili ručno:
# docker compose exec ollama ollama pull llama3.2:3b

# 3) Provjere (Ollama + Qdrant)
docker compose ps
curl -sS http://localhost:6333/collections | jq
docker compose exec -T ollama ollama list
# iz API kontejnera provjeri pristup Ollami (dobit ćeš listu modela)
docker compose exec -T api curl -sS http://ollama:11434/api/tags | jq || true

# 4) Digni API
docker compose up -d api
curl -sS http://localhost:8000/health | jq

# 5) Ingest (prvi put ili nakon čišćenja volumena/promjene embeddera)
curl -sS -X POST http://localhost:8000/ingest   -H "Content-Type: application/json"   -d '{"dataset":"20newsgroups","recreate":true}' | jq

# 6) Upiti (baseline i advanced)
curl -sS -X POST http://localhost:8000/query   -H "Content-Type: application/json"   -d '{"query":"What is comp.graphics?","k":5}' | jq

curl -sS -X POST http://localhost:8000/query   -H "Content-Type: application/json"   -d '{"query":"Što je X Window System?","k":5,"mode":"fusion","n_queries":4}' | jq
```

---

## Endpoints

- `GET /health` → `{ "status": "ok", "collection": "<ime_kolekcije>" }`
- `GET /collections` → `{ "collections": [...] }`
- `DELETE /collections/{name}`
- `POST /ingest`  
  **Body**: `{"dataset":"20newsgroups","recreate":true}`  
  **Response**: `{"collection":"newsgroups","chunks_indexed":N,"dim":<embedding-dim>}`
- `POST /query`  
  **Body (minimalno)**: `{"query":"...", "k":5}`  
  **Napredne opcije**: `mode`, `n_queries`, `translate` (vidi dolje)

---

## Napredni RAG (Translation / Multi-Query / Fusion)

- `mode`: `"simple" | "multi" | "fusion"`
  - **multi**: LLM generira više semantički različitih upita i **spaja** rezultate (union + dedupe).
  - **fusion**: koristi **Reciprocal Rank Fusion (RRF)** za ujedinjavanje rangova više lista.
- `n_queries`: broj alternativnih upita (default 4 u postavkama).
- `translate`: `true/false` – forsira prijevod upita na engleski za retrieval.  
  Ako embedder nije multilingual, prijevod se aktivira automatski (osim ako `translate:false`).

**Primjer `meta`:**
```json
{
  "provider":"ollama",
  "model":"llama3.2:3b",
  "latency_ms": 38446,
  "translated": true,
  "mode": "fusion",
  "n_queries": 4,
  "rrf_k": 60,
  "pipeline_ms": 54985
}
```

---

## Primjeri cURL poziva

**Baseline:**
```bash
curl -sS -X POST http://localhost:8000/query   -H "Content-Type: application/json"   -d '{"query":"What is comp.graphics?","k":5}' | jq
```

**Multi-Query:**
```bash
curl -sS -X POST http://localhost:8000/query   -H "Content-Type: application/json"   -d '{"query":"Who is the CEO of OpenAI?","k":5,"mode":"multi","n_queries":4}' | jq
```

**RAG-Fusion (RRF):**
```bash
curl -sS -X POST http://localhost:8000/query   -H "Content-Type: application/json"   -d '{"query":"Who is the CEO of OpenAI?","k":5,"mode":"fusion","n_queries":4}' | jq
```

**Query Translation (force):**
```bash
curl -sS -X POST http://localhost:8000/query   -H "Content-Type: application/json"   -d '{"query":"Ko je izvršni direktor OpenAI?","k":5,"mode":"fusion","translate":true}' | jq ".meta"
```
---

## Lokalno (bez Dockera)

```bash
pip install -r requirements.txt
uvicorn app.main:create_app --factory --reload

# u drugom terminalu:
curl -sS http://localhost:8000/health | jq
```

---

## Konfiguracija (.env)

```dotenv
HOST=0.0.0.0
PORT=8000

EMBEDDING_MODEL=sentence-transformers/paraphrase-MiniLM-L3-v2

QDRANT_URL=http://qdrant:6333

LLM_PROVIDER=ollama
LLM_TIMEOUT=300

# OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4o-mini

OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_MODEL=llama3.2:3b
```

> Ako koristiš **host** Ollama (izvan Dockera):  
> `OLLAMA_BASE_URL=http://host.docker.internal:11434`

---

## Troubleshooting

- **`/api/tags` prazan** → model nije povučen u **docker** Ollami.  
  `docker compose exec ollama ollama pull llama3.2:3b`
- **Prvi upit traje dugo** → hladni start; koristi `LLM_TIMEOUT=300` i `OLLAMA_KEEP_ALIVE=30m` (compose env u `ollama` servisu).
- **Qdrant healthcheck crveni** → možeš maknuti healthcheck; app ionako čeka Qdrant na startu.
- **Nema odgovora na temu** → 20NG nema taj sadržaj; dodaj vlastite dokumente ili implementiraj `/ingest_texts` endpoint za ad-hoc tekstove.

---

## Dev notes

- Embeddings: MiniLM (paraphrase), dimenzija se detektira runtime.
- Vector DB: Qdrant (cosine).
- LLM: OpenAI / Ollama / Mock.
- Advanced RAG: Query Translation (auto/force), Multi-Query, RAG-Fusion (RRF).
- LangChain adapteri: **langchain-huggingface** (embedding) i **langchain-qdrant** (vectorstore).
