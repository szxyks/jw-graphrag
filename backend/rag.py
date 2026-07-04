"""GraphRAG retrieval — hybrid search combining:
  1. Vector similarity (chunks)
  2. Entity graph traversal (1-hop neighbors)
  3. Community summaries (global context)

Three retrieval modes:
  - "local":  vector search + expand via entity graph
  - "global": match community summaries, then dive into top communities
  - "hybrid": combination of both (default)
"""
import os
import json
import psycopg2.extras
from db import query
import ollama_client
import graphrag


SYSTEM_PROMPT = """You are a careful research assistant for the publications of Jehovah's Witnesses (Watchtower 1950-2026, Awake!, books, brochures, Bibles, Meeting Workbooks, Kingdom Ministry).

Your job:
1. Answer the user's question using ONLY the provided context.
2. Cite sources inline as [1], [2], etc. matching the source list.
3. If context is insufficient, say so honestly.
4. Do NOT use outside knowledge — even if you know the answer.
5. For follow-up questions, use prior conversation context.
6. Be concise: 2-4 paragraphs unless asked for more.
7. When entities are mentioned in the question, leverage the entity graph context to enrich the answer.

CONTEXT CHUNKS (vector + graph-retrieved):
{context}

ENTITY GRAPH (key entities + their relationships):
{entity_graph}

COMMUNITY CONTEXT (themes that bind these entities):
{community_context}

SOURCE LIST:
{sources}

Answer with inline citations [1], [2], etc.
"""


def retrieve_vector(question: str, top_k: int = 8, pub_filter: str = None) -> list:
    """Pure vector search on chunks."""
    q_vec = ollama_client.embed(question)
    vec_str = "[" + ",".join(f"{x:.6f}" for x in q_vec) + "]"

    sql = """
        SELECT c.id, c.chunk_uid, c.text, c.char_count, c.element_tag,
               c.paragraph_id, c.page_numbers, c.bible_citations, c.summary,
               c.publication_id, c.document_id,
               p.pub_code, p.issue, p.title AS pub_title, p.year, p.pub_type,
               d.title AS doc_title,
               c.embedding <=> %s::vector AS distance
        FROM chunks c
        JOIN publications p ON c.publication_id = p.id
        LEFT JOIN documents d ON c.document_id = d.id
        WHERE 1=1
    """
    params = [vec_str]
    if pub_filter:
        sql += " AND p.pub_code = %s"
        params.append(pub_filter)
    sql += " ORDER BY c.embedding <=> %s::vector LIMIT %s"
    params.extend([vec_str, top_k])
    return query(sql, params)


def retrieve_entities(question: str, top_k: int = 10) -> list:
    """Find entities whose name+description embedding matches the question."""
    q_vec = ollama_client.embed(question)
    vec_str = "[" + ",".join(f"{x:.6f}" for x in q_vec) + "]"

    rows = query("""
        SELECT id, name, name_original, type, description, community_id,
               chunk_count, pub_count,
               embedding <=> %s::vector AS distance
        FROM entities
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """, (vec_str, vec_str, top_k))
    return rows


def expand_via_graph(entity_ids: list, max_hops: int = 1, max_chunks: int = 12) -> list:
    """Given a set of entities, find chunks that mention them or their 1-hop neighbors.

    Returns chunks with extra metadata about which entities they reference.
    """
    if not entity_ids:
        return []

    # Step 1: find neighbor entities (1-hop)
    neighbor_rows = query("""
        WITH neighbors AS (
            SELECT DISTINCT
                CASE WHEN source_id = e.id THEN target_id ELSE source_id END AS neighbor_id
            FROM entities e
            JOIN relationships r ON r.source_id = e.id OR r.target_id = e.id
            WHERE e.id = ANY(%s)
        )
        SELECT id FROM entities WHERE id IN (SELECT neighbor_id FROM neighbors) AND id <> ALL(%s)
        LIMIT 50
    """, (entity_ids, entity_ids))

    all_entity_ids = entity_ids + [r["id"] for r in neighbor_rows]

    # Step 2: find chunks mentioning these entities
    rows = query("""
        SELECT c.id, c.chunk_uid, c.text, c.char_count, c.element_tag,
               c.paragraph_id, c.page_numbers, c.bible_citations, c.summary,
               c.publication_id, c.document_id,
               p.pub_code, p.issue, p.title AS pub_title, p.year, p.pub_type,
               d.title AS doc_title,
               array_agg(DISTINCT e.name) AS matched_entities
        FROM chunks c
        JOIN entity_chunks ec ON ec.chunk_id = c.id
        JOIN entities e ON e.id = ec.entity_id
        JOIN publications p ON c.publication_id = p.id
        LEFT JOIN documents d ON c.document_id = d.id
        WHERE ec.entity_id = ANY(%s)
        GROUP BY c.id, c.chunk_uid, c.text, c.char_count, c.element_tag,
                 c.paragraph_id, c.page_numbers, c.bible_citations, c.summary,
                 c.publication_id, c.document_id,
                 p.pub_code, p.issue, p.title, p.year, p.pub_type, d.title
        ORDER BY COUNT(DISTINCT e.id) DESC, c.char_count DESC
        LIMIT %s
    """, (all_entity_ids, max_chunks))
    return rows


def retrieve_communities(entity_ids: list, top_k: int = 3) -> list:
    """Get community summaries for the entities involved."""
    if not entity_ids:
        return []
    rows = query("""
        SELECT DISTINCT com.id, com.title, com.summary, com.entity_count, com.chunk_count,
               com.level
        FROM communities com
        JOIN community_entities ce ON ce.community_id = com.id
        WHERE ce.entity_id = ANY(%s) AND com.summary IS NOT NULL
        ORDER BY com.entity_count DESC
        LIMIT %s
    """, (entity_ids, top_k))
    return rows


def build_entity_graph_context(entity_ids: list) -> str:
    """Build a textual view of the entity graph for the LLM context."""
    if not entity_ids:
        return "(no entity graph available — run graphrag ingest first)"

    # Get entities
    ents = query("""
        SELECT id, name, name_original, type, description
        FROM entities WHERE id = ANY(%s)
        ORDER BY chunk_count DESC LIMIT 15
    """, (entity_ids,))

    # Get relationships among them
    rels = query("""
        SELECT
            se.name AS source_name, se.type AS source_type,
            re.name AS target_name, re.type AS target_type,
            r.type AS rel_type, r.description, r.weight
        FROM relationships r
        JOIN entities se ON r.source_id = se.id
        JOIN entities re ON r.target_id = re.id
        WHERE r.source_id = ANY(%s) AND r.target_id = ANY(%s)
        ORDER BY r.weight DESC LIMIT 30
    """, (entity_ids, entity_ids))

    lines = []
    lines.append("ENTITIES:")
    for e in ents:
        lines.append(f"  • {e['name_original'] or e['name']} ({e['type']}): {(e['description'] or '')[:120]}")
    lines.append("\nRELATIONSHIPS:")
    for r in rels:
        lines.append(f"  • {r['source_name']} --[{r['rel_type']}]--> {r['target_name']}")
        if r['description']:
            lines.append(f"      {r['description'][:100]}")
    return "\n".join(lines)


def build_community_context(communities: list) -> str:
    if not communities:
        return "(no community summaries available)"
    lines = []
    for c in communities:
        lines.append(f"[{c['id']}] {c['title'] or '(untitled community)'}")
        if c['summary']:
            lines.append(f"    {c['summary']}")
        lines.append(f"    ({c['entity_count']} entities, {c['chunk_count']} chunks)")
    return "\n\n".join(lines)


def deduplicate_chunks(chunks_lists: list, max_total: int = 12) -> list:
    """Merge chunks from multiple retrieval paths, dedupe, cap total."""
    seen = set()
    merged = []
    for chunks in chunks_lists:
        for c in chunks:
            if c["id"] in seen:
                continue
            seen.add(c["id"])
            merged.append(c)
            if len(merged) >= max_total:
                return merged
    return merged


def build_context(chunks: list) -> tuple:
    """Format chunks as context string + sources list."""
    context_parts = []
    sources = []
    for i, c in enumerate(chunks, 1):
        citation_parts = []
        if c.get("pub_code"):
            citation_parts.append(c["pub_code"])
        if c.get("year"):
            citation_parts.append(str(c["year"]))
        if c.get("doc_title"):
            citation_parts.append(c["doc_title"][:60])
        if c.get("page_numbers"):
            citation_parts.append(f"p.{c['page_numbers'][0]}")
        source_label = " ".join(citation_parts) or "source"

        summary_str = f"\n[Summary: {c['summary']}]" if c.get("summary") else ""
        entities_str = f"\n[Entities: {', '.join(c.get('matched_entities', [])[:5])}]" if c.get("matched_entities") else ""

        context_parts.append(f"[{i}] ({source_label}){summary_str}{entities_str}\n{c['text']}\n")
        sources.append({
            "index": i,
            "chunk_id": c["id"],
            "chunk_uid": c["chunk_uid"],
            "pub_code": c.get("pub_code"),
            "issue": c.get("issue"),
            "pub_title": c.get("pub_title"),
            "doc_title": c.get("doc_title"),
            "page_numbers": c.get("page_numbers"),
            "bible_citations": c.get("bible_citations"),
            "snippet": c["text"][:200] + ("..." if len(c["text"]) > 200 else ""),
            "distance": float(c.get("distance", 0.5)) if "distance" in c else None,
            "matched_entities": c.get("matched_entities", []),
        })
    return "\n---\n".join(context_parts), sources


def answer_question(
    question: str,
    history: list = None,
    top_k: int = 6,
    pub_filter: str = None,
    mode: str = "hybrid",
    model: str = None,
):
    """GraphRAG answer generation.

    mode: "local" (vector+graph), "global" (community-first), or "hybrid" (default)
    """
    # ---------- Phase 1: vector retrieval ----------
    vector_chunks = retrieve_vector(question, top_k=top_k, pub_filter=pub_filter)
    if not vector_chunks:
        yield {"type": "token", "content": "No documents have been ingested yet. Run the ingest service first."}
        yield {"type": "done"}
        return

    # ---------- Phase 2: entity retrieval + graph expansion ----------
    matched_entities = retrieve_entities(question, top_k=10)
    entity_ids = [e["id"] for e in matched_entities]

    graph_chunks = []
    if mode in ("local", "hybrid") and entity_ids:
        graph_chunks = expand_via_graph(entity_ids, max_hops=1, max_chunks=8)

    # ---------- Phase 3: community context (global mode) ----------
    communities = []
    if mode in ("global", "hybrid"):
        communities = retrieve_communities(entity_ids, top_k=3)

    # ---------- Phase 4: merge + dedupe ----------
    chunks = deduplicate_chunks([vector_chunks, graph_chunks], max_total=12)

    # ---------- Phase 5: build context ----------
    context_str, sources = build_context(chunks)
    entity_graph_str = build_entity_graph_context(entity_ids)
    community_str = build_community_context(communities)
    sources_json = json.dumps(sources, indent=2, default=str)

    # ---------- Emit sources first ----------
    yield {
        "type": "sources",
        "sources": sources,
        "entities": [dict(e) for e in matched_entities],
        "communities": [dict(c) for c in communities],
        "mode": mode,
    }

    # ---------- Phase 6: LLM generation ----------
    system = SYSTEM_PROMPT.format(
        context=context_str,
        entity_graph=entity_graph_str,
        community_context=community_str,
        sources=sources_json,
    )
    messages = [{"role": "system", "content": system}]
    if history:
        for h in history[-6:]:
            messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": question})

    for token in ollama_client.chat_stream(messages, model=model):
        yield {"type": "token", "content": token}

    yield {"type": "done"}


# ----- Session helpers (unchanged from RAG version) -----

def create_session(title: str = None) -> dict:
    import uuid
    session_uid = str(uuid.uuid4())
    row = query(
        "INSERT INTO sessions (session_uid, title) VALUES (%s, %s) RETURNING id, session_uid, title, created_at",
        (session_uid, title),
        fetch="one",
    )
    return dict(row)


def get_session(session_uid: str) -> dict:
    row = query(
        "SELECT id, session_uid, title, created_at, last_active FROM sessions WHERE session_uid = %s",
        (session_uid,),
        fetch="one",
    )
    return dict(row) if row else None


def list_sessions(limit: int = 50) -> list:
    rows = query(
        "SELECT session_uid, title, created_at, last_active FROM sessions ORDER BY last_active DESC LIMIT %s",
        (limit,),
    )
    return [dict(r) for r in rows]


def add_message(session_id: int, role: str, content: str, citations: list = None, entities: list = None) -> dict:
    row = query(
        "INSERT INTO messages (session_id, role, content, citations, entities) VALUES (%s, %s, %s, %s, %s) "
        "RETURNING id, role, content, citations, entities, created_at",
        (session_id, role, content,
         json.dumps(citations) if citations else None,
         json.dumps(entities) if entities else None),
        fetch="one",
    )
    query("UPDATE sessions SET last_active = NOW() WHERE id = %s", (session_id,), fetch="none")
    return dict(row)


def get_messages(session_id: int, limit: int = 50) -> list:
    rows = query(
        "SELECT role, content, citations, entities, created_at FROM messages WHERE session_id = %s ORDER BY id ASC LIMIT %s",
        (session_id, limit),
    )
    return [dict(r) for r in rows]


# ----- Stats / publications -----

def get_stats() -> dict:
    pubs = query("SELECT COUNT(*) AS c FROM publications", fetch="one")
    docs = query("SELECT COUNT(*) AS c FROM documents", fetch="one")
    chunks = query("SELECT COUNT(*) AS c FROM chunks", fetch="one")
    ents = query("SELECT COUNT(*) AS c FROM entities", fetch="one")
    rels = query("SELECT COUNT(*) AS c FROM relationships", fetch="one")
    coms = query("SELECT COUNT(*) AS c FROM communities WHERE summary IS NOT NULL", fetch="one")
    by_type = query("""
        SELECT pub_type, COUNT(*) AS c, SUM(chunk_count) AS chunks,
               SUM(entity_count) AS entities
        FROM publications GROUP BY pub_type ORDER BY c DESC
    """)
    by_year = query("""
        SELECT year, COUNT(DISTINCT p.id) AS pubs, COUNT(c.id) AS chunks
        FROM publications p LEFT JOIN chunks c ON c.publication_id = p.id
        WHERE p.year IS NOT NULL
        GROUP BY year ORDER BY year
    """)
    return {
        "publications": pubs["c"],
        "documents": docs["c"],
        "chunks": chunks["c"],
        "entities": ents["c"],
        "relationships": rels["c"],
        "communities": coms["c"],
        "by_type": [dict(r) for r in by_type],
        "by_year": [dict(r) for r in by_year],
    }


def list_publications() -> list:
    rows = query("""
        SELECT id, pub_code, issue, symbol, year, title, pub_type, source_format,
               ingested_at, chunk_count, entity_count, graph_built, communities_built
        FROM publications ORDER BY pub_code, year, issue
    """)
    return [dict(r) for r in rows]


# ----- Graph endpoints -----

def get_entity_graph(entity_id: int = None, limit: int = 50) -> dict:
    """Get entities + relationships for visualization.

    If entity_id given, return that entity's 1-hop neighborhood.
    Otherwise return top entities by chunk_count.
    """
    if entity_id:
        ents = query("""
            WITH target AS (SELECT id FROM entities WHERE id = %s)
            SELECT DISTINCT e.id, e.name, e.name_original, e.type, e.description,
                   e.chunk_count, e.community_id
            FROM entities e
            WHERE e.id = %s
               OR e.id IN (SELECT target_id FROM relationships WHERE source_id = %s)
               OR e.id IN (SELECT source_id FROM relationships WHERE target_id = %s)
            LIMIT %s
        """, (entity_id, entity_id, entity_id, entity_id, limit))
        ent_ids = [e["id"] for e in ents]
        if ent_ids:
            rels = query("""
                SELECT r.id, r.source_id, r.target_id, r.type, r.description, r.weight
                FROM relationships r
                WHERE r.source_id = ANY(%s) AND r.target_id = ANY(%s)
                ORDER BY r.weight DESC LIMIT 200
            """, (ent_ids, ent_ids))
        else:
            rels = []
    else:
        ents = query("""
            SELECT id, name, name_original, type, description, chunk_count, community_id
            FROM entities ORDER BY chunk_count DESC LIMIT %s
        """, (limit,))
        ent_ids = [e["id"] for e in ents]
        if ent_ids:
            rels = query("""
                SELECT r.id, r.source_id, r.target_id, r.type, r.description, r.weight
                FROM relationships r
                WHERE r.source_id = ANY(%s) AND r.target_id = ANY(%s)
                ORDER BY r.weight DESC LIMIT 300
            """, (ent_ids, ent_ids))
        else:
            rels = []

    return {
        "entities": [dict(e) for e in ents],
        "relationships": [dict(r) for r in rels],
    }


def search_entities(query_text: str, top_k: int = 10) -> list:
    """Vector search over entities (not chunks)."""
    q_vec = ollama_client.embed(query_text)
    vec_str = "[" + ",".join(f"{x:.6f}" for x in q_vec) + "]"
    rows = query("""
        SELECT id, name, name_original, type, description, chunk_count, pub_count,
               community_id,
               embedding <=> %s::vector AS distance
        FROM entities
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """, (vec_str, vec_str, top_k))
    return [dict(r) for r in rows]


def list_communities() -> list:
    rows = query("""
        SELECT com.id, com.title, com.summary, com.entity_count, com.chunk_count,
               com.level, com.created_at
        FROM communities com
        WHERE com.summary IS NOT NULL
        ORDER BY com.entity_count DESC
        LIMIT 100
    """)
    return [dict(r) for r in rows]
