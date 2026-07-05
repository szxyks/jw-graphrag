# JW GraphRAG — Local Knowledge-Graph RAG for JW Publications

An **intelligent GraphRAG** system for the JW publication archive (Watchtower 1950–2026, Awake!, Kingdom Ministry, Meeting Workbooks, Bibles, books, songbooks).

**All compute runs locally** via Docker. LLM = Ollama. Knowledge graph + vector store = Postgres+pgvector.

---

## What makes this GraphRAG (not plain RAG)

Plain RAG retrieves chunks by vector similarity and feeds them to the LLM. **GraphRAG** adds:

1. **Knowledge graph construction** — every chunk is mined for entities (people, places, concepts, doctrines) and relationships (Jehovah → father_of → Jesus Christ). Entities become graph nodes; relationships become edges.
2. **Entity graph traversal** — when you ask a question, we find matching entities, then expand to their 1-hop neighbors, then pull chunks that mention any of them. This catches chunks that wouldn't surface via pure vector search.
3. **Community detection (Leiden)** — we cluster the entity graph into thematic communities (e.g. "eschatology cluster", "family life cluster") and ask the LLM to summarize each one.
4. **Hybrid retrieval** — every query combines (a) vector search, (b) graph expansion, (c) community context. Three modes: `local`, `global`, `hybrid` (default).
5. **Multi-hop reasoning** — the LLM sees the entity graph + community summaries in its context window, enabling it to reason about relationships rather than just match keywords.

This is the [Microsoft GraphRAG pattern](https://microsoft.github.io/graphrag/) adapted for local LLMs (Ollama) and the JW corpus.

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│  docker compose up -d                                                   │
│                                                                        │
│  ┌────────────┐    ┌────────────┐    ┌──────────────────────────┐     │
│  │  frontend  │───▶│  backend   │───▶│  postgres + pgvector     │     │
│  │  Tailwind  │    │  Flask     │    │  ┌─ chunks (vector)      │     │
│  │  + SVG     │    │  /chat     │    │  ├─ entities (graph)    │     │
│  │  graph viz │    │  /graph    │    │  ├─ relationships       │     │
│  └────────────┘    │  /entities │    │  ├─ communities         │     │
│                    │  /communiti│    │  └─ sessions            │     │
│                    └─────┬──────┘    └──────────────────────────┘     │
│                          │                  ▲                          │
│                          ▼                  │                          │
│                   ┌────────────┐             │                          │
│                   │  ollama    │             │                          │
│                   │ llama3.2:3b│             │                          │
│                   │ nomic-embed│             │                          │
│                   └────────────┘             │                          │
│                          ▲                   │                          │
│                          │                   │                          │
│                   ┌──────────────────────────┴────────────┐            │
│                   │  ingest (Python)                       │            │
│                   │  1. download .jwpub from b.jw-cdn.org  │            │
│                   │  2. AES-128-CBC + zlib decrypt         │            │
│                   │  3. chunk XHTML by paragraph           │            │
│                   │  4. embed chunks (nomic-embed-text)    │            │
│                   │  5. extract entities + relationships   │            │
│                   │     via local LLM                      │            │
│                   │  6. run Leiden community detection     │            │
│                   │  7. summarize each community via LLM   │            │
│                   └────────────────────────────────────────┘            │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Quick start

### 1. Prerequisites
- Docker 24+ and Docker Compose v2
- ~12 GB free disk (models + ingested corpus)
- 8 GB+ RAM (16 GB recommended for GraphRAG ingestion)
- **Ollama installed on your host** (recommended) — get it from https://ollama.com
  - Or use the bundled Ollama (see "Bundled Ollama" section below)

### 2. Pull the required models (host Ollama)

If you don't already have them:
```bash
ollama pull qwen2.5:latest        # or llama3.2:3b — your choice of LLM
ollama pull nomic-embed-text      # required for embeddings
```

Verify with `ollama list`.

### 3. Configure & start

```bash
cd jw-graphrag
cp .env.example .env
# Edit .env to match the models you have pulled (LLM + embed model)

docker compose up -d
docker compose logs -f backend   # verify "Ollama reachable"
```

Open **http://localhost:8080**.

### 4. Verify

```bash
curl http://localhost:8080/api/health
# {"ollama_ready": true, "llm_model": "qwen2.5:latest", ...}
```

### 5. Ingest your first publication

Use the UI quick-ingest buttons, or:

```bash
curl -X POST http://localhost:8080/api/ingest \
  -H 'Content-Type: application/json' \
  -d '{"pub_code":"w","issue":"19800101"}'
```

This downloads the 1980-01 Watchtower, decrypts all 12 articles, chunks them into ~228 paragraphs, embeds them, AND extracts the entity graph (entities + relationships) for that issue.

### 6. Ingest more, then build communities

```bash
# Ingest several publications
docker compose run --rm --profile ingest ingest python ingest.py --catalog /scripts/STARTER_CATALOG.json

# After ingestion, run community detection on the global entity graph
docker compose run --rm --profile ingest ingest python build_graph.py --communities
```

The community detection step clusters all entities (across all publications) into thematic groups and asks the LLM to summarize each cluster. This is what enables global-mode questions like "What are the major themes across all Watchtower issues from 1980?".

### 7. Ask questions

In the UI, try:
- "What does the Bible say about hope?" → vector + graph retrieval
- "How is Jehovah related to Jesus Christ?" → graph traversal dominant
- "What are the major themes across the corpus?" → community summaries

Switch between `local`, `global`, and `hybrid` modes via the API.

---

## Bundled Ollama (alternative to host Ollama)

If you don't want to install Ollama on your host, the stack can run its own:

```bash
# Edit .env first:
#   OLLAMA_HOST=http://ollama:11434
#   OLLAMA_LLM_MODEL=llama3.2:3b
#   OLLAMA_EMBED_MODEL=nomic-embed-text

docker compose up -d --profile bundled-ollama
docker compose logs -f ollama-init   # ~5 min for first model pull
```

The `bundled-ollama` profile starts an Ollama container on port 11434 and an `ollama-init` job that pulls the required models.

**Note:** If you see `bind host port 0.0.0.0:11434/tcp: address already in use`, you already have Ollama running on the host. Either stop it (`sudo systemctl stop ollama`) or use the default host-Ollama mode (no `--profile bundled-ollama`).

---

## Three retrieval modes

```bash
curl -X POST http://localhost:8080/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"question":"What is the Kingdom of God?","mode":"hybrid","top_k":6}'
```

| Mode | Behavior | Best for |
|------|----------|----------|
| `local` | Vector search + 1-hop entity graph expansion | Specific factual questions |
| `global` | Match community summaries first, then dive into top communities | "What are the main themes?" |
| `hybrid` (default) | Both: vector + graph + community context | General purpose |

---

## API endpoints

### Core
- `GET /api/health` — service + Ollama status
- `GET /api/stats` — corpus statistics (chunks, entities, relationships, communities)
- `GET /api/publications` — list ingested publications
- `POST /api/ingest` — trigger one-shot ingestion
- `POST /api/chat` — streaming GraphRAG chat (SSE)
- `GET /api/search?q=...` — pure vector search

### Graph-specific
- `GET /api/entities/search?q=...` — vector search over entities
- `GET /api/graph?limit=40` — top entities + their relationships (for visualization)
- `GET /api/graph?entity_id=N` — 1-hop neighborhood of one entity
- `GET /api/communities` — list all community summaries

### Sessions
- `GET /api/sessions` — list conversations
- `POST /api/sessions` — create new
- `GET /api/sessions/<uid>` — get session + messages

---

## Frontend tabs

The right sidebar has three tabs:

1. **Sources** — citation cards for the current answer (vector + graph-retrieved chunks)
2. **Graph** — interactive SVG visualization of the entity knowledge graph. Click a node to focus its 1-hop neighborhood. Search entities by vector similarity.
3. **Communities** — list of Leiden-detected community summaries (themes)

---

## File structure

```
jw-graphrag/
├── docker-compose.yml
├── .env.example
├── README.md
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── app.py              # Flask API (chat, graph, communities, ingest)
│   ├── db.py               # Postgres pool
│   ├── ollama_client.py    # Ollama HTTP wrapper
│   └── rag.py              # GraphRAG retrieval logic (3 modes)
├── frontend/
│   └── index.html          # Tailwind UI + SVG graph viz
├── ingest/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── ingest.py           # JWPUB → decrypt → chunks → embed → DB
│   └── build_graph.py      # entity extraction + community detection
├── nginx/
│   └── nginx.conf
└── scripts/
    ├── init_db.sql         # Schema: chunks + entities + relationships + communities
    ├── jwpub_decryptor.py  # AES-128-CBC + zlib decryptor
    ├── graphrag.py         # LLM prompts for entity extraction + community summarization (shared)
    ├── epub_to_rag_chunks.py
    ├── ingest_one.py
    └── STARTER_CATALOG.json
```

---

## Customizing the LLM

Edit `.env`:

```env
OLLAMA_LLM_MODEL=llama3.2:3b           # default — fast
# OLLAMA_LLM_MODEL=qwen2.5:7b-instruct # better for entity extraction
# OLLAMA_LLM_MODEL=llama3.1:8b         # higher quality reasoning
# OLLAMA_LLM_MODEL=mistral:7b          # general purpose
```

For GraphRAG specifically, a stronger LLM (7B+) gives much better entity extraction quality — `llama3.2:3b` will miss entities that `qwen2.5:7b` catches.

For NVIDIA GPU acceleration, uncomment the `deploy.resources` block in `docker-compose.yml`.

---

## How entity extraction works

For every chunk (paragraph), the LLM is prompted:

```
Given the text below, identify all entities (people, places, concepts,
organizations, doctrines, books, events) and the relationships between them.

TEXT:
{text}

Return JSON:
{
  "entities": [{"name": "Jehovah", "type": "deity", "description": "..."}],
  "relationships": [{"source": "Jehovah", "target": "Jesus Christ", "type": "father_of", "description": "..."}]
}
```

Entities are canonicalized (lowercase + alias resolution: "God" → "jehovah", "Christ" → "jesus christ"). Each unique entity gets an embedding of `name: description` and is stored once. Relationships are deduplicated by (source, target, type) and weighted by co-occurrence count.

After all chunks in a publication are processed, `build_graph.py` runs the **Leiden algorithm** on the global entity graph. Each resulting community is summarized by the LLM.

---

## Cost / performance notes

| Step | Per-chunk cost | Notes |
|------|---------------:|-------|
| Decrypt | ~5 ms | AES + zlib |
| Chunk parse | ~1 ms | XML parsing |
| Embed chunk | ~50 ms CPU / ~5 ms GPU | nomic-embed-text |
| Summarize chunk | ~3 s CPU / ~0.5 s GPU | LLM call |
| Extract entities + relationships | ~10 s CPU / ~1.5 s GPU | LLM call (longer prompt) |
| Embed entities | ~50 ms each | nomic-embed-text |
| Community detection | ~1 s for 1K entities | networkx + leidenalg |
| Community summaries | ~5 s each | LLM call |

For a 1980 Watchtower issue (~228 chunks), full GraphRAG ingestion takes:
- CPU only: ~50 minutes
- GPU (RTX 3060+): ~8 minutes

For the full Watchtower 1950–2026 (~1,499 issues × ~228 chunks ≈ 342K chunks):
- CPU only: ~50 days (use a GPU)
- GPU: ~5 days

Start small (10 publications) to verify quality before committing to full ingestion.

---

## How the JWPUB decryption works

Every `.jwpub` is a ZIP containing `manifest.json` + an inner `contents` ZIP containing `*.db` (SQLite). The `Document.Content` BLOBs are encrypted:

```python
# 1. pub_card = f"{MepsLang}_{Symbol}_{Year}_{IssueTag}" (magazines)
#          OR  f"{MepsLang}_{Symbol}_{Year}" (books — when IssueTag is "0")
# 2. SHA-256(pub_card) → 32 bytes
# 3. XOR with hardcoded key: 11cbb5587e32846d4c26790c633da289f66fe5842a3a585ce1bc3a294af5ada7
# 4. First 16 bytes = AES-128 key, last 16 = AES-CBC IV
# 5. AES-128-CBC decrypt → zlib stream (starts with 78 9c)
# 6. zlib inflate → UTF-8 XHTML
```

Works for every publication 1950–2026 — all magazines, books, Bibles, songbooks, the 4,782-document Insight.

---

## Troubleshooting

**`networkx.community` missing Louvain:**
Upgrade networkx to 3.3+ (`pip install --upgrade networkx`).

**Leiden algorithm not available:**
`pip install python-igraph leidenalg`. If it fails to build, the code falls back to networkx Louvain, then to connected components.

**Out of memory during ingestion:**
- Use `llama3.2:1b` instead of 3b
- Set `OLLAMA_LLM_MODEL=phi3:mini`
- Run ingestion with `--max-chunks 100` to test first

**Embedding dimensions mismatch:**
nomic-embed-text = 768. If you switch to bge-m3 (1024), update `vector(768)` in `scripts/init_db.sql`.

---

## Disclaimer

Personal study tool. JW publication content is accessed via the public catalog API (`b.jw-cdn.org/apis/pub-media/GETPUBMEDIALINKS`). The decryption algorithm is publicly documented (e.g. `sws2apps/meeting-schedules-parser`).

Use for personal study. Don't redistribute or claim official endorsement.
