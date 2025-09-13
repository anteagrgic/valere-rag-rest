FROM python:3.11-slim

WORKDIR /app

# System deps (curl zbog healthcheckova i općenito, build-essential zbog uvloop itd.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git \
 && rm -rf /var/lib/apt/lists/*

# Uvijek svježi pip/setuptools/wheel (bitno zbog wheelova)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 🔑Preinstall Torch (CPU wheel) iz službenog repoa — izbjegava fallback na source build
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch==2.3.1

# Sada ide ostatak — koristit će već instalirani torch
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 8000
CMD ["python", "-m", "app.main"]
