SHELL := /bin/bash

.PHONY: up down logs ingest query test fmt

up:
\tdocker compose up --build -d

down:
\tdocker compose down

logs:
\tdocker compose logs -f api

ingest:
\tcurl -X POST http://localhost:8000/ingest -H "Content-Type: application/json" -d '{"dataset":"20newsgroups","recreate":true}'

query:
\tcurl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{"query":"What topics are discussed about space?", "k": 5}'

test:
\tpytest -q
