FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libpq-dev curl unzip libxml2-dev libgmp-dev libmpfr-dev libmpc-dev \
    libffi-dev libssl-dev && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements first (cache layer)
COPY ingest/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip list | grep -iE "psycopg|pgvector|cryptography|requests|networkx|igraph|leidenalg"

# Copy ingest code
COPY ingest/ /app/

# Also bake in the shared scripts (jwpub_decryptor, graphrag) so the ingest
# container doesn't depend on the host's ./scripts directory being mounted.
COPY scripts/jwpub_decryptor.py /scripts/jwpub_decryptor.py
COPY scripts/graphrag.py /scripts/graphrag.py
COPY scripts/init_db.sql /scripts/init_db.sql

# Make /scripts importable
ENV PYTHONPATH=/scripts:/app:$PYTHONPATH

CMD ["python", "ingest.py", "--help"]
