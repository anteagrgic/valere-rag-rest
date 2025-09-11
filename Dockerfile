FROM python:3.11-slim

WORKDIR /app

# System deps (git for sentence-transformers cache occasionally)
RUN apt-get update && apt-get install -y build-essential git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY .env ./.env

EXPOSE 8000
CMD ["python", "-m", "app.main"]
