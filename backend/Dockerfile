FROM python:3.12-slim

WORKDIR /app

# System deps for psycopg2 and cryptography
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libpq-dev curl unzip libffi-dev libssl-dev && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip first (older pip versions sometimes silently fail to install)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Python deps (this layer is cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip list | grep -iE "flask|psycopg|pgvector|cryptography|requests"

# Copy app source (mounted over at runtime for dev, but bake it in for prod)
COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
