#!/usr/bin/env python3
"""
EPUB -> RAG-ready JSON chunks.

Takes any JW EPUB file and produces a JSONL file of chunks suitable for
embedding / RAG. Each chunk is one paragraph (or heading) with rich metadata:
  - source publication (pub code, title, issue)
  - section / chapter
  - paragraph number (JW EPUBs mark these as data-pid)
  - bible citations found in the paragraph
  - page numbers (data-no)
  - raw XHTML preserved (for re-rendering)

Usage:
    python3 epub_to_rag_chunks.py <input.epub> <output.jsonl>
"""
import sys
import json
import zipfile
import re
from pathlib import Path
from html.parser import HTMLParser
from xml.etree import ElementTree as ET

NS = {"xhtml": "http://www.w3.org/1999/xhtml",
      "epub":  "http://www.idpf.org/2007/ops"}


def strip_ns(tag):
    return tag.split("}", 1)[-1] if "}" in tag else tag


def text_of(elem):
    """Get all text under an element, ignoring tags."""
    parts = []
    if elem.text:
        parts.append(elem.text)
    for child in elem:
        parts.append(text_of(child))
        if child.tail:
            parts.append(child.tail)
    return "".join(parts)


def find_citations(elem):
    """Find bible citations in this element (hrefs to e.g. '2024240-extracted.xhtml#pcitation1')."""
    cites = []
    for a in elem.iter():
        if strip_ns(a.tag) == "a":
            href = a.get("href", "")
            # Bible citation links look like: <file>#pcitationNN  or  #citationNN
            if "citation" in href.lower():
                txt = text_of(a).strip()
                if txt:
                    cites.append(txt)
    return cites


def find_page_numbers(elem):
    """Find page-number markers in this element."""
    pages = []
    for span in elem.iter():
        if strip_ns(span.tag) == "span":
            cls = span.get("class", "")
            if "pageNum" in cls:
                no = span.get("data-no")
                if no:
                    pages.append(int(no))
    return pages


def chunk_epub(epub_path: Path, out_path: Path):
    chunks = []
    with zipfile.ZipFile(epub_path) as zf:
        # Parse OPF to find publication metadata + spine
        opf_path = None
        for name in zf.namelist():
            if name.endswith(".opf"):
                opf_path = name
                break
        if not opf_path:
            print("ERROR: no OPF found in EPUB", file=sys.stderr)
            return 0

        opf_xml = ET.fromstring(zf.read(opf_path))
        # Get title / language / identifiers
        meta = {}
        for m in opf_xml.iter():
            t = strip_ns(m.tag)
            if t in ("title", "language", "identifier", "publisher", "date"):
                v = (m.text or "").strip()
                if v:
                    meta[t] = v
        # Use directory of OPF for resolving hrefs
        opf_dir = str(Path(opf_path).parent)

        # Read TOC (toc.ncx or nav) for section titles
        toc_map = {}   # xhtml file -> section title
        for name in zf.namelist():
            if name.endswith("toc.ncx"):
                ncx = ET.fromstring(zf.read(name))
                for np in ncx.iter():
                    if strip_ns(np.tag) == "navPoint":
                        label = None
                        href = None
                        for c in np:
                            ct = strip_ns(c.tag)
                            if ct == "navLabel":
                                label = text_of(c).strip()
                            elif ct == "content":
                                href = c.get("src", "")
                        if href and label:
                            toc_map[href.split("#")[0]] = label
                break

        # Process every XHTML content file (skip cover/nav/toc)
        skip = {"toc.xhtml", "pagenav0.xhtml", "cover.xhtml"}
        content_files = [n for n in zf.namelist()
                         if n.endswith(".xhtml") and Path(n).name not in skip
                         and "extracted" not in n]
        chunk_id = 0
        for cf in content_files:
            try:
                root = ET.fromstring(zf.read(cf))
            except ET.ParseError as e:
                print(f"  skip {cf}: {e}", file=sys.stderr)
                continue
            # Determine section title from TOC
            cf_basename = Path(cf).name
            section_title = toc_map.get(cf_basename, "")

            # Walk: emit one chunk per <p>, <h1>, <h2>, <h3>, <li> at top level
            body = None
            for elem in root.iter():
                if strip_ns(elem.tag) == "body":
                    body = elem
                    break
            if body is None:
                continue

            for elem in body.iter():
                t = strip_ns(elem.tag)
                if t in ("p", "h1", "h2", "h3", "h4", "li"):
                    text = text_of(elem).strip()
                    if len(text) < 20:
                        continue
                    # Skip nested paragraphs (we already got them)
                    pid = elem.get("data-pid") or elem.get("id")
                    cites = find_citations(elem)
                    pages = find_page_numbers(elem)
                    chunk = {
                        "chunk_id": f"{epub_path.stem}::{cf_basename}::{pid or chunk_id}",
                        "pub_source": epub_path.stem,
                        "publication_title": meta.get("title", ""),
                        "language": meta.get("language", "en"),
                        "section_file": cf_basename,
                        "section_title": section_title,
                        "element_tag": t,
                        "paragraph_id": pid,
                        "page_numbers": pages,
                        "bible_citations": cites,
                        "text": text,
                        "char_count": len(text),
                    }
                    chunks.append(chunk)
                    chunk_id += 1

    # Write JSONL
    with open(out_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    return len(chunks)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    epub_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    n = chunk_epub(epub_path, out_path)
    print(f"OK  {epub_path.name}  ->  {n} chunks  ->  {out_path}")
    # Show sample
    if n > 0:
        with open(out_path) as f:
            sample = json.loads(f.readline())
            print("\nSample chunk:")
            print(json.dumps(sample, indent=2, ensure_ascii=False)[:800])
