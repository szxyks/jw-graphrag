FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libpq-dev curl unzip libxml2-dev libgmp-dev libmpfr-dev libmpc-dev \
    libffi-dev libssl-dev && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip list | grep -iE "psycopg|pgvector|cryptography|requests|networkx|igraph|leidenalg"

COPY . .

CMD ["python", "ingest.py", "--help"]
