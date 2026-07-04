"""GraphRAG ingestion step — runs AFTER chunks are inserted.

For each chunk:
  1. Generate a summary (1-2 sentences)
  2. Extract entities + relationships via LLM
  3. Insert entities (deduplicated by name+type) and relationships
  4. Create entity_chunks mentions

After all chunks processed:
  5. Run Leiden community detection on the entity graph
  6. For each community, ask LLM to generate a summary
  7. Update entities.community_id and insert community summaries
"""
import os
import sys
import time
import json
import psycopg2
import psycopg2.extras

sys.path.insert(0, "/scripts")
sys.path.insert(0, "/app")

import graphrag
from ingest import db_conn, embed_text


def upsert_entity(conn, name, name_original, etype, description, embed, first_pub_id):
    """Insert or update an entity. Returns entity id."""
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO entities (name, name_original, type, description, embedding, first_seen_pub_id)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (name, type) DO UPDATE
                SET chunk_count = entities.chunk_count + 1,
                    description = COALESCE(EXCLUDED.description, entities.description)
            RETURNING id
        """, (name, name_original, etype, description, embed, first_pub_id))
        return cur.fetchone()[0]


def upsert_relationship(conn, source_id, target_id, rel_type, description, first_pub_id):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO relationships (source_id, target_id, type, description, first_seen_pub_id)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (source_id, target_id, type) DO UPDATE
                SET weight = relationships.weight + 1,
                    chunk_count = relationships.chunk_count + 1,
                    description = COALESCE(EXCLUDED.description, relationships.description)
            RETURNING id
        """, (source_id, target_id, rel_type, description, first_pub_id))
        return cur.fetchone()[0]


def link_entity_chunk(conn, entity_id, chunk_id):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO entity_chunks (entity_id, chunk_id, mention_count)
            VALUES (%s, %s, 1)
            ON CONFLICT (entity_id, chunk_id) DO UPDATE
                SET mention_count = entity_chunks.mention_count + 1
        """, (entity_id, chunk_id))


def update_chunk_summary(conn, chunk_id, summary):
    with conn.cursor() as cur:
        cur.execute("UPDATE chunks SET summary = %s WHERE id = %s", (summary, chunk_id))


def fetch_all_entities(conn):
    """Return all entities as list of dicts."""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT id, name, name_original, type, description FROM entities")
        return cur.fetchall()


def fetch_all_relationships(conn):
    """Return all relationships with entity names."""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT r.id, r.source_id, r.target_id, r.type, r.description, r.weight,
                   se.name AS source_name, se.type AS source_type,
                   te.name AS target_name, te.type AS target_type
            FROM relationships r
            JOIN entities se ON r.source_id = se.id
            JOIN entities te ON r.target_id = te.id
        """)
        return cur.fetchall()


def get_sample_text_for_community(conn, entity_ids, max_chars=3000):
    """Get sample text from chunks mentioning these entities."""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT DISTINCT c.text
            FROM chunks c
            JOIN entity_chunks ec ON ec.chunk_id = c.id
            WHERE ec.entity_id = ANY(%s)
            ORDER BY c.char_count DESC
            LIMIT 5
        """, (entity_ids,))
        texts = [r["text"][:600] for r in cur.fetchall()]
        combined = "\n---\n".join(texts)
        return combined[:max_chars]


def build_graph_for_publication(pub_id, max_chunks=None):
    """Build entity graph for all chunks in a publication.

    Returns dict with stats.
    """
    conn = db_conn()
    conn.autocommit = False
    try:
        # Get all chunks for this publication that haven't been processed yet
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            sql = """
                SELECT c.id, c.chunk_uid, c.text, c.publication_id, c.summary
                FROM chunks c
                WHERE c.publication_id = %s
                  AND c.summary IS NULL
                ORDER BY c.id
            """
            if max_chunks:
                sql += f" LIMIT {max_chunks}"
            cur.execute(sql, (pub_id,))
            chunks = cur.fetchall()

        print(f"  Building graph for {len(chunks)} chunks")
        if not chunks:
            # Mark graph as built
            with conn.cursor() as cur:
                cur.execute("UPDATE publications SET graph_built = TRUE WHERE id = %s", (pub_id,))
            conn.commit()
            return {"processed": 0, "entities": 0, "relationships": 0}

        entity_cache = {}  # (name, type) -> id
        stats = {"processed": 0, "entities": 0, "relationships": 0}
        t0 = time.time()

        for i, chunk in enumerate(chunks):
            try:
                # Step 1: summarize chunk
                summary = graphrag.summarize_chunk(chunk["text"])
                if summary:
                    update_chunk_summary(conn, chunk["id"], summary)

                # Step 2: extract entities + relationships
                extracted = graphrag.extract_entities_from_chunk(chunk["text"])
                ents = extracted.get("entities", [])
                rels = extracted.get("relationships", [])

                # Step 3: insert entities (canonicalize names)
                ent_name_to_id = {}
                for e in ents:
                    name = graphrag.canonicalize_entity_name(e.get("name", ""))
                    if not name:
                        continue
                    etype = e.get("type", "concept")
                    desc = e.get("description", "")
                    cache_key = (name, etype)
                    if cache_key in entity_cache:
                        eid = entity_cache[cache_key]
                    else:
                        # Embed the entity name + description
                        embed_text_str = f"{name}: {desc}"[:500]
                        try:
                            emb = embed_text(embed_text_str)
                        except Exception:
                            emb = None
                        eid = upsert_entity(conn, name, e.get("name", name), etype,
                                            desc, emb, chunk["publication_id"])
                        entity_cache[cache_key] = eid
                    ent_name_to_id[e.get("name", "").lower()] = eid
                    # Link entity -> chunk
                    link_entity_chunk(conn, eid, chunk["id"])
                    stats["entities"] += 1

                # Step 4: insert relationships
                for r in rels:
                    src_name = graphrag.canonicalize_entity_name(r.get("source", ""))
                    tgt_name = graphrag.canonicalize_entity_name(r.get("target", ""))
                    src_id = ent_name_to_id.get(r.get("source", "").lower())
                    tgt_id = ent_name_to_id.get(r.get("target", "").lower())
                    if not (src_id and tgt_id and src_id != tgt_id):
                        continue
                    rel_type = r.get("type", "related_to").lower().replace(" ", "_")
                    rel_desc = r.get("description", "")
                    upsert_relationship(conn, src_id, tgt_id, rel_type, rel_desc,
                                       chunk["publication_id"])
                    stats["relationships"] += 1

                stats["processed"] += 1

                if (i + 1) % 5 == 0:
                    conn.commit()
                    elapsed = time.time() - t0
                    rate = (i + 1) / elapsed
                    eta = (len(chunks) - i - 1) / rate if rate > 0 else 0
                    print(f"    [{i+1}/{len(chunks)}] {rate:.1f} chunks/s, "
                          f"ent={stats['entities']} rel={stats['relationships']}, ETA {eta:.0f}s")

            except Exception as e:
                print(f"    ! chunk {chunk['chunk_uid']}: {e}")
                conn.rollback()
                conn.autocommit = False

        # Mark pub as graph-built
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE publications
                SET graph_built = TRUE,
                    entity_count = (SELECT COUNT(*) FROM entity_chunks ec
                                    JOIN chunks c ON c.id = ec.chunk_id
                                    WHERE c.publication_id = %s)
                WHERE id = %s
            """, (pub_id, pub_id))
        conn.commit()
        print(f"  ✓ Graph built: {stats['processed']} chunks, "
              f"{stats['entities']} entity mentions, {stats['relationships']} relationships")
        return stats
    finally:
        conn.close()


def detect_and_summarize_communities(pub_scope="global"):
    """Run community detection on the global entity graph and summarize each cluster."""
    print(f"\n{'='*70}")
    print(f"COMMUNITY DETECTION ({pub_scope})")
    print(f"{'='*70}")

    conn = db_conn()
    try:
        entities = fetch_all_entities(conn)
        relationships = fetch_all_relationships(conn)
        print(f"  Total entities: {len(entities)}")
        print(f"  Total relationships: {len(relationships)}")

        if not entities:
            print("  No entities to cluster")
            return

        # Run community detection
        ent_list = [{"name": e["name"], "type": e["type"], "description": e.get("description", "")}
                    for e in entities]
        rel_list = [{"source": r["source_name"], "target": r["target_name"],
                     "type": r["type"], "description": r.get("description", "")}
                    for r in relationships]

        communities = graphrag.detect_communities(ent_list, rel_list)
        print(f"  Detected {len(communities)} communities")

        # Clear existing communities for this scope
        with conn.cursor() as cur:
            cur.execute("DELETE FROM communities WHERE pub_scope = %s", (pub_scope,))

        # For each community, generate summary and insert
        for c_id, ent_indices in communities.items():
            if len(ent_indices) < 2:
                continue
            ent_ids = [entities[i]["id"] for i in ent_indices]
            ent_subset = [ent_list[i] for i in ent_indices]

            # Get sample text
            sample_text = get_sample_text_for_community(conn, ent_ids)

            # Generate summary
            summary = graphrag.summarize_community(ent_subset, sample_text) if sample_text else ""
            title = ent_subset[0]["name"][:60] if ent_subset else f"Community {c_id}"

            # Insert community
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO communities (pub_scope, level, title, summary, entity_count, chunk_count)
                    VALUES (%s, 0, %s, %s, %s, %s) RETURNING id
                """, (pub_scope, title, summary, len(ent_indices), 0))
                new_com_id = cur.fetchone()[0]

                # Link entities to community
                for eid in ent_ids:
                    cur.execute("""
                        INSERT INTO community_entities (community_id, entity_id)
                        VALUES (%s, %s) ON CONFLICT DO NOTHING
                    """, (new_com_id, eid))

                # Update entities.community_id
                cur.execute("""
                    UPDATE entities SET community_id = %s WHERE id = ANY(%s)
                """, (new_com_id, ent_ids))

            print(f"    Community {new_com_id}: '{title}' — {len(ent_indices)} entities")

        # Update publications.communities_built
        with conn.cursor() as cur:
            cur.execute("UPDATE publications SET communities_built = TRUE")
        conn.commit()
        print(f"  ✓ {len(communities)} communities stored")

    finally:
        conn.close()


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--pub-id", type=int, help="Build graph for one publication")
    p.add_argument("--pub-code", help="Build graph for publications matching code")
    p.add_argument("--communities", action="store_true", help="Run community detection after extraction")
    p.add_argument("--max-chunks", type=int, help="Limit chunks per publication (debug)")
    args = p.parse_args()

    if args.pub_id:
        build_graph_for_publication(args.pub_id, args.max_chunks)
        if args.communities:
            detect_and_summarize_communities()
    elif args.pub_code:
        conn = db_conn()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT id, pub_code, issue, title FROM publications WHERE pub_code = %s",
                       (args.pub_code,))
            pubs = cur.fetchall()
        conn.close()
        for pub in pubs:
            print(f"\n--- {pub['pub_code']} {pub['issue'] or '-'} : {pub['title']} ---")
            build_graph_for_publication(pub["id"], args.max_chunks)
        if args.communities:
            detect_and_summarize_communities()
    else:
        p.print_help()


if __name__ == "__main__":
    main()
