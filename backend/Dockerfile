FROM python:3.12-slim

WORKDIR /app

# System deps:
#  - gcc/g++/libpq-dev/libffi-dev/libssl-dev: build deps for psycopg2 + cryptography
#  - curl: for healthchecks
#  - docker-cli: so the backend can spawn one-shot `docker compose run --rm ingest ...`
#    commands against the host Docker daemon (via the mounted socket)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libpq-dev curl unzip libffi-dev libssl-dev \
    ca-certificates gnupg lsb-release && \
    # Install Docker CLI (no daemon — we just need the client to talk to the host socket)
    install -m 0755 -d /etc/apt/keyrings && \
    curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc && \
    chmod a+r /etc/apt/keyrings/docker.asc && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
        > /etc/apt/sources.list.d/docker.list && \
    apt-get update && apt-get install -y --no-install-recommends docker-ce-cli docker-compose-plugin && \
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
