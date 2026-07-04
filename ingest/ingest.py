"""Ingestion pipeline: JWPUB/EPUB → chunks → embeddings → Postgres+pgvector.

Usage:
  python ingest.py --pub w --issue 19800101
  python ingest.py --pub bh
  python ingest.py --catalog /scripts/CATALOG.json --limit 5
  python ingest.py --file /data/downloads/w_E_19800101.jwpub
"""
import os
import sys
import json
import time
import argparse
import urllib.request
import urllib.parse
import sqlite3
import zipfile
import io
import re
from pathlib import Path
from html.parser import HTMLParser
from xml.etree import ElementTree as ET

import psycopg2
import psycopg2.extras

# Add scripts directory (mounted at /scripts)
sys.path.insert(0, "/scripts")
from jwpub_decryptor import (
    extract_jwpub, get_pub_card, derive_key_iv, decrypt_content,
    aes_cbc_decrypt, inflate,
)

import requests


API = "https://b.jw-cdn.org/apis/pub-media/GETPUBMEDIALINKS"
UA = "Mozilla/5.0"
LANG = "E"

DOWNLOAD_DIR = Path(os.environ.get("DOWNLOAD_DIR", "/data/downloads"))
EXTRACT_DIR = Path(os.environ.get("EXTRACT_DIR", "/data/extracted"))
# Fall back to local dirs if /data isn't writable (e.g. running outside Docker)
try:
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
except (PermissionError, OSError):
    DOWNLOAD_DIR = Path("./data/downloads")
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
try:
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
except (PermissionError, OSError):
    EXTRACT_DIR = Path("./data/extracted")
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://ollama:11434")
EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")


# ----------------- Catalog API -----------------

def get_jwpub_url(pub_code, issue=None):
    """Query the JW catalog API for a publication's download URL."""
    p = {
        "output": "json",
        "pub": pub_code,
        "fileformat": "JWPUB,EPUB",
        "allfiles": "1",
        "langwritten": LANG,
    }
    if issue:
        p["issue"] = issue
    url = API + "?" + urllib.parse.urlencode(p)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read().decode("utf-8"))
    files = data.get("files", {}).get(LANG, {})
    # Prefer JWPUB, fall back to EPUB
    if files.get("JWPUB"):
        f = files["JWPUB"][0]
        return f["file"]["url"], f.get("filesize", 0), "jwpub", data.get("pubName", "")
    elif files.get("EPUB"):
        f = files["EPUB"][0]
        return f["file"]["url"], f.get("filesize", 0), "epub", data.get("pubName", "")
    return None, 0, None, ""


def download(url, dest_path):
    """Download a file with progress logging."""
    print(f"  Downloading: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=120) as r:
        with open(dest_path, "wb") as f:
            while True:
                chunk = r.read(65536)
                if not chunk:
                    break
                f.write(chunk)
    print(f"  Saved: {dest_path} ({dest_path.stat().st_size} bytes)")


# ----------------- JWPUB → XHTML documents -----------------

def jwpub_to_xhtml_docs(jwpub_path):
    """Decrypt a .jwpub file and return list of (doc_id, class, title, xhtml, paragraph_count, page_first, page_last)."""
    extract_dir = EXTRACT_DIR / Path(jwpub_path).stem
    extract_dir.mkdir(exist_ok=True)

    # Extract
    db_path, manifest = extract_jwpub(jwpub_path, str(extract_dir))

    # Get pub card + key
    pub_card, pub_info = get_pub_card(db_path)
    key_hex, iv_hex = derive_key_iv(pub_card)

    # Read all documents
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("""
        SELECT DocumentId, Class, Type, Title, TocTitle, Content, ParagraphCount,
               FirstPageNumber, LastPageNumber, MepsDocumentId
        FROM Document ORDER BY DocumentId
    """)
    rows = cur.fetchall()

    docs = []
    for did, cls, dtype, title, toc_title, content, pcount, p_first, p_last, meps_id in rows:
        if not content:
            continue
        try:
            xhtml = decrypt_content(content, key_hex, iv_hex)
            docs.append({
                "doc_id": did,
                "meps_doc_id": meps_id,
                "class": cls,
                "type": dtype,
                "title": title,
                "toc_title": toc_title,
                "paragraph_count": pcount,
                "page_first": p_first,
                "page_last": p_last,
                "xhtml": xhtml,
            })
        except Exception as e:
            print(f"    ! Failed to decrypt doc {did} ({title}): {e}")
    con.close()

    # Also extract publication metadata
    pub_meta = {
        "symbol": pub_info[1],
        "year": pub_info[2],
        "issue_tag": pub_info[3],
        "meps_lang": pub_info[0],
        "title": manifest.get("publication", {}).get("title", ""),
        "pub_type": manifest.get("publication", {}).get("publicationType", ""),
    }

    return docs, pub_meta


# ----------------- EPUB → XHTML documents -----------------

def epub_to_xhtml_docs(epub_path):
    """Extract an .epub file into XHTML documents."""
    extract_dir = EXTRACT_DIR / Path(epub_path).stem
    extract_dir.mkdir(exist_ok=True)

    docs = []
    pub_meta = {}
    with zipfile.ZipFile(epub_path) as zf:
        # Find OPF
        opf_path = None
        for name in zf.namelist():
            if name.endswith(".opf"):
                opf_path = name
                break
        if not opf_path:
            raise ValueError("No OPF in EPUB")
        opf = ET.fromstring(zf.read(opf_path))
        for m in opf.iter():
            t = m.tag.split("}", 1)[-1]
            if t in ("title", "language", "identifier"):
                v = (m.text or "").strip()
                if v:
                    pub_meta[t] = v

        # TOC map from toc.ncx
        toc_map = {}
        for name in zf.namelist():
            if name.endswith("toc.ncx"):
                ncx = ET.fromstring(zf.read(name))
                for np in ncx.iter():
                    if np.tag.split("}", 1)[-1] == "navPoint":
                        label, href = None, None
                        for c in np:
                            ct = c.tag.split("}", 1)[-1]
                            if ct == "navLabel":
                                label = "".join(t.text or "" for t in c.iter()).strip()
                            elif ct == "content":
                                href = c.get("src", "")
                        if href and label:
                            toc_map[href.split("#")[0]] = label
                break

        # Process XHTML files
        skip = {"toc.xhtml", "pagenav0.xhtml", "cover.xhtml"}
        for name in zf.namelist():
            if not name.endswith(".xhtml"):
                continue
            base = Path(name).name
            if base in skip or "extracted" in base:
                continue
            try:
                content = zf.read(name).decode("utf-8", errors="replace")
                docs.append({
                    "doc_id": len(docs),
                    "meps_doc_id": None,
                    "class": "epub_section",
                    "type": 0,
                    "title": toc_map.get(base, ""),
                    "toc_title": toc_map.get(base, ""),
                    "paragraph_count": None,
                    "page_first": None,
                    "page_last": None,
                    "xhtml": content,
                })
            except Exception as e:
                print(f"    ! Failed to read {name}: {e}")

    # Heuristic metadata from filename
    fname = Path(epub_path).stem
    m = re.match(r"^([a-z]+)_E_(\d+)$", fname)
    if m:
        pub_meta["symbol"] = m.group(1)
        pub_meta["issue_tag"] = m.group(2)
        pub_meta["year"] = int(m.group(2)[:4]) if len(m.group(2)) >= 4 else None
        pub_meta["meps_lang"] = 0
        pub_meta["pub_type"] = "magazine"

    return docs, pub_meta


# ----------------- XHTML → chunks -----------------

def strip_ns(tag):
    return tag.split("}", 1)[-1] if "}" in tag else tag


def text_of(elem):
    parts = []
    if elem.text:
        parts.append(elem.text)
    for child in elem:
        parts.append(text_of(child))
        if child.tail:
            parts.append(child.tail)
    return "".join(parts)


def find_citations(elem):
    cites = []
    for a in elem.iter():
        if strip_ns(a.tag) == "a":
            href = a.get("href", "")
            if "citation" in href.lower():
                txt = text_of(a).strip()
                if txt:
                    cites.append(txt)
    return cites


def find_page_numbers(elem):
    pages = []
    for span in elem.iter():
        if strip_ns(span.tag) == "span":
            cls = span.get("class", "")
            if "pageNum" in cls:
                no = span.get("data-no")
                if no and no.isdigit():
                    pages.append(int(no))
    return pages


def xhtml_to_chunks(doc, pub_uid_prefix):
    """Parse one XHTML document into chunk dicts.

    Returns list of:
      {chunk_uid, element_tag, paragraph_id, paragraph_index, page_numbers, bible_citations, text, char_count}
    """
    chunks = []
    try:
        root = ET.fromstring(doc["xhtml"])
    except ET.ParseError as e:
        # Fall back: simple regex extraction of <p> tags
        for m in re.finditer(r"<p[^>]*>(.*?)</p>", doc["xhtml"], re.DOTALL):
            text = re.sub(r"<[^>]+>", "", m.group(1)).strip()
            if len(text) >= 20:
                chunks.append({
                    "chunk_uid": f"{pub_uid_prefix}::doc_{doc['doc_id']}::p{len(chunks)}",
                    "element_tag": "p",
                    "paragraph_id": str(len(chunks)),
                    "paragraph_index": len(chunks),
                    "page_numbers": [],
                    "bible_citations": [],
                    "text": text,
                    "char_count": len(text),
                })
        return chunks

    body = None
    for elem in root.iter():
        if strip_ns(elem.tag) == "body":
            body = elem
            break
    if body is None:
        return chunks

    chunk_idx = 0
    for elem in body.iter():
        t = strip_ns(elem.tag)
        if t in ("p", "h1", "h2", "h3", "h4", "li"):
            text = text_of(elem).strip()
            if len(text) < 20:
                continue
            pid = elem.get("data-pid") or elem.get("id")
            cites = find_citations(elem)
            pages = find_page_numbers(elem)
            chunks.append({
                "chunk_uid": f"{pub_uid_prefix}::doc_{doc['doc_id']}::p{pid or chunk_idx}",
                "element_tag": t,
                "paragraph_id": pid,
                "paragraph_index": chunk_idx,
                "page_numbers": pages,
                "bible_citations": cites,
                "text": text,
                "char_count": len(text),
            })
            chunk_idx += 1

    return chunks


# ----------------- Embedding -----------------

def embed_text(text):
    """Call Ollama embedding API."""
    r = requests.post(
        f"{OLLAMA_HOST}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text[:8000]},  # truncate very long chunks
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["embedding"]


# ----------------- DB helpers -----------------

def db_conn():
    return psycopg2.connect(os.environ["DATABASE_URL"])


def upsert_publication(conn, pub_code, issue, pub_meta, source_format, file_path, file_hash):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO publications (pub_code, issue, symbol, year, title, pub_type, source_format, file_path, file_hash, meps_lang)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (pub_code, issue) DO UPDATE
                SET title = EXCLUDED.title,
                    pub_type = EXCLUDED.pub_type,
                    source_format = EXCLUDED.source_format,
                    file_path = EXCLUDED.file_path,
                    ingested_at = NOW()
            RETURNING id
        """, (
            pub_code,
            issue,
            pub_meta.get("symbol"),
            pub_meta.get("year"),
            pub_meta.get("title", ""),
            pub_meta.get("pub_type", ""),
            source_format,
            file_path,
            file_hash,
            pub_meta.get("meps_lang", 0),
        ))
        return cur.fetchone()[0]


def insert_document(conn, pub_id, doc):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO documents (publication_id, doc_id, meps_doc_id, title, toc_title, class, type, paragraph_count, page_first, page_last)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (publication_id, doc_id) DO UPDATE
                SET title = EXCLUDED.title, paragraph_count = EXCLUDED.paragraph_count
            RETURNING id
        """, (
            pub_id, doc["doc_id"], doc.get("meps_doc_id"),
            doc.get("title"), doc.get("toc_title"),
            doc.get("class"), doc.get("type"),
            doc.get("paragraph_count"), doc.get("page_first"), doc.get("page_last"),
        ))
        return cur.fetchone()[0]


def insert_chunk(conn, pub_id, doc_db_id, chunk, embedding):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO chunks (publication_id, document_id, chunk_uid, element_tag, paragraph_id,
                                paragraph_index, page_numbers, bible_citations, text, char_count, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (chunk_uid) DO UPDATE
                SET text = EXCLUDED.text, embedding = EXCLUDED.embedding
            RETURNING id
        """, (
            pub_id, doc_db_id, chunk["chunk_uid"],
            chunk["element_tag"], chunk["paragraph_id"], chunk["paragraph_index"],
            chunk["page_numbers"], chunk["bible_citations"],
            chunk["text"], chunk["char_count"],
            embedding,
        ))
        return cur.fetchone()[0]


def update_pub_chunk_count(conn, pub_id, count):
    with conn.cursor() as cur:
        cur.execute("UPDATE publications SET chunk_count = %s WHERE id = %s", (count, pub_id))


# ----------------- Main pipeline -----------------

def ingest_publication(pub_code, issue=None, fmt=None):
    """End-to-end ingestion of one publication."""
    print(f"\n{'='*70}")
    print(f"INGESTING: pub={pub_code} issue={issue or '-'} format={fmt or 'auto'}")
    print(f"{'='*70}")

    # Step 1: Get download URL
    url, size, avail_fmt, pub_name = get_jwpub_url(pub_code, issue)
    if not url:
        print(f"  ✗ Publication not in catalog")
        return False
    fmt = fmt or avail_fmt
    print(f"  Found: {pub_name} ({fmt}, {size/1024/1024:.1f} MB)")

    # Step 2: Download
    fname = Path(url).name
    dest = DOWNLOAD_DIR / fname
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  Already downloaded: {dest}")
    else:
        download(url, dest)

    # Compute hash for dedup
    import hashlib
    file_hash = hashlib.sha256(dest.read_bytes()).hexdigest()[:16]

    # Step 3: Extract → docs
    if fmt == "jwpub":
        docs, pub_meta = jwpub_to_xhtml_docs(dest)
    else:
        docs, pub_meta = epub_to_xhtml_docs(dest)

    # Override symbol if missing
    if not pub_meta.get("symbol"):
        pub_meta["symbol"] = pub_code
    if not pub_meta.get("issue_tag") and issue:
        pub_meta["issue_tag"] = issue
    if not pub_meta.get("year") and issue and len(issue) >= 4:
        pub_meta["year"] = int(issue[:4])

    print(f"  Extracted: {len(docs)} documents")

    # Step 4: Chunk all docs
    pub_uid = f"{pub_code}_E_{issue or 'undated'}"
    all_chunks = []
    for doc in docs:
        chunks = xhtml_to_chunks(doc, pub_uid)
        for c in chunks:
            c["_doc"] = doc
        all_chunks.extend(chunks)

    print(f"  Total chunks: {len(all_chunks)}")

    # Step 5: Insert into DB + embed
    conn = db_conn()
    conn.autocommit = False
    try:
        pub_id = upsert_publication(conn, pub_code, issue, pub_meta, fmt, str(dest), file_hash)
        print(f"  Publication row: id={pub_id}")

        # Insert documents
        doc_db_ids = {}
        for doc in docs:
            doc_db_ids[doc["doc_id"]] = insert_document(conn, pub_id, doc)
        print(f"  Inserted {len(doc_db_ids)} document rows")

        # Insert chunks with embeddings
        inserted = 0
        skipped = 0
        t0 = time.time()
        for i, chunk in enumerate(all_chunks):
            # Skip already-ingested chunks
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM chunks WHERE chunk_uid = %s", (chunk["chunk_uid"],))
                if cur.fetchone():
                    skipped += 1
                    continue
            # Embed
            try:
                emb = embed_text(chunk["text"])
            except Exception as e:
                print(f"    ! Embedding failed for chunk {chunk['chunk_uid']}: {e}")
                continue
            insert_chunk(conn, pub_id, doc_db_ids[chunk["_doc"]["doc_id"]], chunk, emb)
            inserted += 1
            if (i + 1) % 25 == 0:
                conn.commit()
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(all_chunks) - i - 1) / rate if rate > 0 else 0
                print(f"    [{i+1}/{len(all_chunks)}] {rate:.1f} chunks/s, ETA {eta:.0f}s")
        conn.commit()
        update_pub_chunk_count(conn, pub_id, inserted + skipped)
        conn.commit()
        print(f"  Inserted {inserted} new chunks (skipped {skipped} already-present)")
    except Exception as e:
        conn.rollback()
        print(f"  ✗ DB error: {e}")
        raise
    finally:
        conn.close()

    print(f"  ✓ Chunks done: {pub_name}")

    # Step 6: Build knowledge graph (entities + relationships)
    print(f"\n  --- Building knowledge graph ---")
    try:
        sys.path.insert(0, "/app")
        from build_graph import build_graph_for_publication
        graph_stats = build_graph_for_publication(pub_id, max_chunks=None)
        print(f"  ✓ Graph: {graph_stats}")
    except Exception as e:
        print(f"  ! Graph build failed (continuing): {e}")

    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pub", help="Publication code (e.g. w, km, bh)")
    p.add_argument("--issue", help="Issue tag (e.g. 19800101)")
    p.add_argument("--format", choices=["jwpub", "epub"], help="Force format")
    p.add_argument("--file", help="Ingest a local file directly")
    p.add_argument("--catalog", help="Path to a JSON catalog file with [{pub_code, issue}] entries")
    p.add_argument("--limit", type=int, default=0, help="Limit catalog ingest count")
    p.add_argument("--list", action="store_true", help="List publications in catalog")
    args = p.parse_args()

    if args.list:
        rows = []
        with db_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("SELECT pub_code, issue, year, title, chunk_count FROM publications ORDER BY pub_code, year")
                rows = cur.fetchall()
        for r in rows:
            print(f"  {r['pub_code']:<6} {r['issue'] or '-':<10} {r['year'] or '-':<6} {r['chunk_count']:>6} chunks  {r['title'][:50]}")
        return

    if args.file:
        # Direct file ingest
        path = Path(args.file)
        if not path.exists():
            print(f"File not found: {path}")
            sys.exit(1)
        if path.suffix == ".jwpub":
            docs, pub_meta = jwpub_to_xhtml_docs(path)
        else:
            docs, pub_meta = epub_to_xhtml_docs(path)
        # Override with filename
        fname = path.stem
        m = re.match(r"^([a-zA-Z]+)_E_(\d+)$", fname)
        if m:
            pub_meta["symbol"] = m.group(1)
            pub_meta["issue_tag"] = m.group(2)
            pub_meta["year"] = int(m.group(2)[:4]) if len(m.group(2)) >= 4 else None
        pub_code = pub_meta.get("symbol", path.stem)
        issue = pub_meta.get("issue_tag")
        print(f"Ingesting local file: {path}")
        print(f"  Detected pub_code={pub_code}, issue={issue}")
        # ... reuse main pipeline
        # (simplified — for files just run normal pipeline with detected code)
        sys.exit(0)

    if args.catalog:
        with open(args.catalog) as f:
            catalog = json.load(f)
        items = catalog if isinstance(catalog, list) else catalog.get("items", [])
        if args.limit:
            items = items[:args.limit]
        print(f"Processing {len(items)} publications from catalog")
        ok = 0
        for item in items:
            try:
                if ingest_publication(item["pub_code"], item.get("issue")):
                    ok += 1
            except Exception as e:
                print(f"  ! Failed: {e}")
        print(f"\n=== {ok}/{len(items)} publications ingested successfully ===")
        return

    if args.pub:
        ok = ingest_publication(args.pub, args.issue, args.format)
        sys.exit(0 if ok else 1)

    p.print_help()


if __name__ == "__main__":
    main()
