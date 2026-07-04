-- ===========================================================================
-- JW GraphRAG database schema
-- ===========================================================================
-- Combines:
--   * Vector store (pgvector) for semantic chunk search
--   * Knowledge graph (entities + relationships) for graph traversal
--   * Community detection (Leiden algorithm via networkx) for global summarization
--   * Conversation sessions for follow-up questions
-- ===========================================================================

CREATE EXTENSION IF NOT EXISTS vector;

-- ============= Publications =============
CREATE TABLE IF NOT EXISTS publications (
    id              SERIAL PRIMARY KEY,
    pub_code        TEXT NOT NULL,
    issue           TEXT,
    symbol          TEXT,
    year            INTEGER,
    title           TEXT,
    pub_type        TEXT,
    source_format   TEXT,
    file_path       TEXT,
    file_hash       TEXT,
    meps_lang       INTEGER,
    ingested_at     TIMESTAMPTZ DEFAULT NOW(),
    chunk_count     INTEGER DEFAULT 0,
    entity_count    INTEGER DEFAULT 0,
    graph_built     BOOLEAN DEFAULT FALSE,
    communities_built BOOLEAN DEFAULT FALSE,
    UNIQUE(pub_code, issue)
);
CREATE INDEX IF NOT EXISTS idx_publications_code ON publications(pub_code);
CREATE INDEX IF NOT EXISTS idx_publications_year ON publications(year);

-- ============= Documents =============
CREATE TABLE IF NOT EXISTS documents (
    id              SERIAL PRIMARY KEY,
    publication_id  INTEGER REFERENCES publications(id) ON DELETE CASCADE,
    doc_id          INTEGER,
    meps_doc_id     INTEGER,
    title           TEXT,
    toc_title       TEXT,
    class           TEXT,
    type            INTEGER,
    paragraph_count INTEGER,
    page_first      INTEGER,
    page_last       INTEGER,
    ingested_at     TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(publication_id, doc_id)
);
CREATE INDEX IF NOT EXISTS idx_documents_pub ON documents(publication_id);

-- ============= Chunks (paragraph-level) =============
CREATE TABLE IF NOT EXISTS chunks (
    id              SERIAL PRIMARY KEY,
    publication_id  INTEGER REFERENCES publications(id) ON DELETE CASCADE,
    document_id     INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    chunk_uid       TEXT NOT NULL,
    element_tag     TEXT,
    paragraph_id    TEXT,
    paragraph_index INTEGER,
    page_numbers    INTEGER[],
    bible_citations TEXT[],
    text            TEXT NOT NULL,
    char_count      INTEGER,
    embedding       vector(768),
    -- GraphRAG augmentation
    summary         TEXT,                  -- LLM-generated summary of this chunk
    keywords        TEXT[],                -- LLM-extracted keywords
    ingested_at     TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(chunk_uid)
);
CREATE INDEX IF NOT EXISTS idx_chunks_pub ON chunks(publication_id);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_citations ON chunks USING gin(bible_citations);
CREATE INDEX IF NOT EXISTS idx_chunks_pages ON chunks USING gin(page_numbers);
CREATE INDEX IF NOT EXISTS idx_chunks_keywords ON chunks USING gin(keywords);
CREATE INDEX IF NOT EXISTS idx_chunks_embedding
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- ============= Entities (Knowledge Graph nodes) =============
CREATE TABLE IF NOT EXISTS entities (
    id              SERIAL PRIMARY KEY,
    name            TEXT NOT NULL,             -- canonical name (lower-cased)
    name_original   TEXT,                      -- first-seen original casing
    type            TEXT,                      -- person, place, concept, organization, doctrine, etc.
    description     TEXT,                      -- LLM-generated description
    embedding       vector(768),               -- embedding of name+description
    chunk_count     INTEGER DEFAULT 0,         -- how many chunks mention this entity
    pub_count       INTEGER DEFAULT 0,         -- how many publications mention this
    community_id    INTEGER,                   -- assigned by Leiden
    first_seen_pub_id INTEGER REFERENCES publications(id),
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(name, type)
);
CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type);
CREATE INDEX IF NOT EXISTS idx_entities_community ON entities(community_id);
CREATE INDEX IF NOT EXISTS idx_entities_embedding
    ON entities USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- ============= Relationships (Knowledge Graph edges) =============
CREATE TABLE IF NOT EXISTS relationships (
    id              SERIAL PRIMARY KEY,
    source_id       INTEGER REFERENCES entities(id) ON DELETE CASCADE,
    target_id       INTEGER REFERENCES entities(id) ON DELETE CASCADE,
    type            TEXT NOT NULL,             -- e.g. "preached", "is_about", "mentions", "leads_to"
    description     TEXT,                      -- LLM-generated description of the relationship
    weight          FLOAT DEFAULT 1.0,         -- frequency-based
    chunk_count     INTEGER DEFAULT 1,         -- how many chunks co-mention source + target
    first_seen_pub_id INTEGER REFERENCES publications(id),
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(source_id, target_id, type)
);
CREATE INDEX IF NOT EXISTS idx_rel_source ON relationships(source_id);
CREATE INDEX IF NOT EXISTS idx_rel_target ON relationships(target_id);
CREATE INDEX IF NOT EXISTS idx_rel_type ON relationships(type);

-- ============= Entity ↔ Chunk mentions =============
CREATE TABLE IF NOT EXISTS entity_chunks (
    entity_id       INTEGER REFERENCES entities(id) ON DELETE CASCADE,
    chunk_id        INTEGER REFERENCES chunks(id) ON DELETE CASCADE,
    mention_count   INTEGER DEFAULT 1,
    PRIMARY KEY(entity_id, chunk_id)
);
CREATE INDEX IF NOT EXISTS idx_ec_entity ON entity_chunks(entity_id);
CREATE INDEX IF NOT EXISTS idx_ec_chunk ON entity_chunks(chunk_id);

-- ============= Communities (Leiden clusters of entities) =============
CREATE TABLE IF NOT EXISTS communities (
    id              SERIAL PRIMARY KEY,
    pub_scope       TEXT,                      -- 'global' or specific pub_code
    level           INTEGER DEFAULT 0,         -- hierarchy level (0 = leaf)
    title           TEXT,                      -- LLM-generated short title
    summary         TEXT,                      -- LLM-generated comprehensive summary
    entity_count    INTEGER DEFAULT 0,
    chunk_count     INTEGER DEFAULT 0,
    parent_id       INTEGER REFERENCES communities(id),
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS community_entities (
    community_id    INTEGER REFERENCES communities(id) ON DELETE CASCADE,
    entity_id       INTEGER REFERENCES entities(id) ON DELETE CASCADE,
    PRIMARY KEY(community_id, entity_id)
);
CREATE INDEX IF NOT EXISTS idx_ce_community ON community_entities(community_id);

-- ============= Bible verse citation graph =============
CREATE TABLE IF NOT EXISTS bible_verses (
    id              SERIAL PRIMARY KEY,
    book            TEXT NOT NULL,
    chapter         INTEGER NOT NULL,
    verse           INTEGER NOT NULL,
    citation_text   TEXT,
    UNIQUE(book, chapter, verse)
);

CREATE TABLE IF NOT EXISTS chunk_bible_verses (
    chunk_id        INTEGER REFERENCES chunks(id) ON DELETE CASCADE,
    verse_id        INTEGER REFERENCES bible_verses(id) ON DELETE CASCADE,
    PRIMARY KEY(chunk_id, verse_id)
);
CREATE INDEX IF NOT EXISTS idx_cbv_chunk ON chunk_bible_verses(chunk_id);
CREATE INDEX IF NOT EXISTS idx_cbv_verse ON chunk_bible_verses(verse_id);

-- ============= Conversation sessions =============
CREATE TABLE IF NOT EXISTS sessions (
    id              SERIAL PRIMARY KEY,
    session_uid     TEXT NOT NULL UNIQUE,
    title           TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    last_active     TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS messages (
    id              SERIAL PRIMARY KEY,
    session_id      INTEGER REFERENCES sessions(id) ON DELETE CASCADE,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    citations       JSONB,
    entities        JSONB,                     -- entities referenced in this answer
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_uid ON sessions(session_uid);

-- ============= Ingestion log =============
CREATE TABLE IF NOT EXISTS ingest_log (
    id              SERIAL PRIMARY KEY,
    pub_code        TEXT,
    issue           TEXT,
    stage           TEXT,                      -- 'download', 'decrypt', 'chunk', 'embed', 'extract_entities', 'build_communities'
    status          TEXT,
    message         TEXT,
    started_at      TIMESTAMPTZ DEFAULT NOW(),
    finished_at     TIMESTAMPTZ
);
