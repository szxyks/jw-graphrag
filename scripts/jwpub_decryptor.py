#!/usr/bin/env python3
"""
JWPUB Document.Content decryptor + decompressor.

Algorithm (per meeting-schedules-parser/src/common/jwpub_parser.ts):
  1. Build "pubCard" = f"{MepsLanguageIndex}_{Symbol}_{Year}_{IssueTagNumber}" from Publication table
  2. Derive key/IV:
       - hardcoded XOR key = base64-decode("MTFjYmI1NTg3ZTMyODQ2ZDRjMjY3OTBjNjMzZGEyODlmNjZmZTU4NDJhM2E1ODVjZTFiYzNhMjk0YWY1YWRhNw==")
                            = hex "11cbb5587e32846d4c26790c633da289f66fe5842a3a585ce1bc3a294af5ada7"
       - SHA256(pubCard) gives 32 bytes
       - XOR those 32 bytes with the 32-byte hardcoded key → produces 32-byte hex string
       - First 16 bytes (hex chars 0-31) = AES-128 key
       - Last 16 bytes (hex chars 32-63) = AES-128 IV (CBC mode)
  3. AES-128-CBC decrypt the Document.Content BLOB
  4. zlib inflate (raw deflate) the decrypted bytes
  5. UTF-8 decode → XHTML text

Usage:
    python3 jwpub_decryptor.py <input.jwpub> <output_dir>
"""
import sys
import os
import json
import base64
import hashlib
import zlib
import sqlite3
import zipfile
import shutil
from pathlib import Path
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend


# Hardcoded XOR key (base64-encoded), as found in the jwpub_parser.ts source
XOR_KEY_B64 = "MTFjYmI1NTg3ZTMyODQ2ZDRjMjY3OTBjNjMzZGEyODlmNjZmZTU4NDJhM2E1ODVjZTFiYzNhMjk0YWY1YWRhNw=="
XOR_KEY_HEX = base64.b64decode(XOR_KEY_B64).decode("ascii")  # This is itself a hex string
XOR_KEY_BYTES = bytes.fromhex(XOR_KEY_HEX)  # 32 raw bytes


def hex_to_bytes(hex_str):
    """Convert hex string to bytes."""
    clean = "".join(c for c in hex_str if c in "0123456789abcdefABCDEF")
    return bytes(int(clean[i:i+2], 16) for i in range(0, len(clean), 2))


def bytes_to_hex(b):
    """Convert bytes to lowercase hex string."""
    return "".join(f"{x:02x}" for x in b)


def sha256_hex(text):
    """SHA-256 of UTF-8 encoded text, returned as hex string (64 chars)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def xor_bytes(a, b):
    """XOR two equal-length byte sequences."""
    if len(a) != len(b):
        raise ValueError(f"Buffer length mismatch: {len(a)} vs {len(b)}")
    return bytes(x ^ y for x, y in zip(a, b))


def derive_key_iv(pub_card):
    """
    Derive AES-128-CBC key and IV from publication card string.
    pub_card format: f"{MepsLanguageIndex}_{Symbol}_{Year}_{IssueTagNumber}"
    e.g. "0_w24_2024_20240100"
    """
    # Step 1: SHA-256 of pub_card → 64-char hex string
    hash_hex = sha256_hex(pub_card)
    hash_bytes = bytes.fromhex(hash_hex)  # 32 raw bytes

    # Step 2: XOR the 32-byte hash with the 32-byte hardcoded key
    xored = xor_bytes(hash_bytes, XOR_KEY_BYTES)
    xored_hex = bytes_to_hex(xored)  # 64-char hex string

    # Step 3: First 32 hex chars = AES-128 key (16 bytes); last 32 = IV (16 bytes)
    key_hex = xored_hex[:32]
    iv_hex = xored_hex[32:]

    return key_hex, iv_hex


def aes_cbc_decrypt(data, key_hex, iv_hex):
    """AES-128-CBC decrypt."""
    key = bytes.fromhex(key_hex)
    iv = bytes.fromhex(iv_hex)
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    return decryptor.update(data) + decryptor.finalize()


def inflate(data):
    """zlib inflate — auto-detect header (zlib magic 78 9c, gzip 1f 8b, or raw)."""
    # Try with zlib header first (most common — JW uses this)
    try:
        return zlib.decompress(data)  # wbits=ZLIB_DEFAULT, auto-detects
    except zlib.error:
        pass
    # Try gzip
    try:
        return zlib.decompress(data, 16 + 15)
    except zlib.error:
        pass
    # Try raw deflate (no header)
    return zlib.decompress(data, -15)


def get_pub_card(db_path):
    """Read Publication table to construct the pub card string.

    Format depends on publication type:
      - Magazines (w, wp, g, km, mwb): f"{MepsLanguageIndex}_{Symbol}_{Year}_{IssueTagNumber}"
        e.g. "0_w24_2024_20240100"
      - Books/Bibles/Brochures: f"{MepsLanguageIndex}_{Symbol}_{Year}"
        e.g. "0_bh_2014" (no IssueTagNumber when it's "0")
    """
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("SELECT MepsLanguageIndex, Symbol, Year, IssueTagNumber, Type, PublicationType FROM Publication")
    row = cur.fetchone()
    con.close()
    if not row:
        raise ValueError("No Publication row found in DB")

    meps_lang, symbol, year, issue_tag, pub_type, pub_type_name = row

    # Heuristic: if IssueTagNumber is "0" or empty, it's an undated publication (book/bible/brochure)
    if issue_tag and issue_tag != "0" and issue_tag != "":
        pub_card = f"{meps_lang}_{symbol}_{year}_{issue_tag}"
    else:
        pub_card = f"{meps_lang}_{symbol}_{year}"

    return pub_card, row


def decrypt_content(content_blob, key_hex, iv_hex):
    """Decrypt a Document.Content BLOB → XHTML text."""
    # Step 1: AES-128-CBC decrypt
    decrypted = aes_cbc_decrypt(content_blob, key_hex, iv_hex)

    # Step 2: zlib inflate (raw deflate)
    decompressed = inflate(decrypted)

    # Step 3: UTF-8 decode
    return decompressed.decode("utf-8", errors="replace")


def extract_jwpub(jwpub_path, output_dir):
    """Extract a .jwpub file → output_dir/{db, manifest.json, *.jpg}."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(jwpub_path) as zf:
        # Read manifest
        manifest = json.loads(zf.read("manifest.json"))
        # Extract inner 'contents' ZIP
        contents_data = zf.read("contents")

    # Extract inner contents ZIP
    inner_dir = output_dir / "_inner_extracted"
    inner_dir.mkdir(exist_ok=True)
    import io
    with zipfile.ZipFile(io.BytesIO(contents_data)) as inner_zf:
        inner_zf.extractall(inner_dir)

    # Find the .db file
    db_files = list(inner_dir.glob("*.db"))
    if not db_files:
        raise FileNotFoundError(f"No .db file found in JWPUB contents")
    db_path = db_files[0]
    final_db = output_dir / db_path.name
    shutil.copy(db_path, final_db)

    # Save manifest
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    return final_db, manifest


def decrypt_all_documents(db_path, output_dir):
    """Decrypt all Document.Content blobs and save as separate XHTML files."""
    pub_card, pub_info = get_pub_card(db_path)
    print(f"  Publication card: {pub_card}")
    print(f"  Publication info: MepsLanguageIndex={pub_info[0]} Symbol={pub_info[1]} Year={pub_info[2]} IssueTag={pub_info[3]}")

    key_hex, iv_hex = derive_key_iv(pub_card)
    print(f"  AES-128 key: {key_hex}")
    print(f"  AES-128 IV:  {iv_hex}")

    con = sqlite3.connect(db_path)
    cur = con.cursor()

    # Decrypt all documents
    cur.execute("SELECT DocumentId, Class, Title, Content FROM Document ORDER BY DocumentId")
    docs = cur.fetchall()
    print(f"  Found {len(docs)} documents to decrypt")

    decrypted_dir = output_dir / "decrypted"
    decrypted_dir.mkdir(exist_ok=True)

    success = 0
    failed = 0
    for doc_id, doc_class, title, content in docs:
        if not content:
            continue
        try:
            xhtml = decrypt_content(content, key_hex, iv_hex)
            # Save
            safe_title = "".join(c for c in (title or f"doc_{doc_id}") if c.isalnum() or c in " -_")[:60]
            out_file = decrypted_dir / f"doc_{doc_id:03d}_{safe_title}.xhtml"
            out_file.write_text(xhtml, encoding="utf-8")
            success += 1
            print(f"    ✓ Doc {doc_id} (class={doc_class}): {title[:50]!r} → {len(xhtml)} chars")
        except Exception as e:
            failed += 1
            print(f"    ✗ Doc {doc_id} ({title[:40]!r}): {e}")

    # Also decrypt Question.Content if any
    try:
        cur.execute("SELECT QuestionId, DocumentId, QuestionIndex, Content FROM Question ORDER BY QuestionId")
        questions = cur.fetchall()
        if questions:
            print(f"  Found {len(questions)} questions to decrypt")
            q_success = 0
            for qid, did, qidx, content in questions:
                if not content: continue
                try:
                    xhtml = decrypt_content(content, key_hex, iv_hex)
                    out_file = decrypted_dir / f"question_{qid:04d}_doc{did}.xhtml"
                    out_file.write_text(xhtml, encoding="utf-8")
                    q_success += 1
                except: pass
            print(f"    ✓ {q_success}/{len(questions)} questions decrypted")
    except Exception as e:
        print(f"  (No Question table or error: {e})")

    # Footnotes
    try:
        cur.execute("SELECT FootnoteId, DocumentId, Content FROM Footnote ORDER BY FootnoteId")
        footnotes = cur.fetchall()
        if footnotes:
            print(f"  Found {len(footnotes)} footnotes to decrypt")
            fn_success = 0
            for fnid, did, content in footnotes:
                if not content: continue
                try:
                    xhtml = decrypt_content(content, key_hex, iv_hex)
                    out_file = decrypted_dir / f"footnote_{fnid:04d}_doc{did}.xhtml"
                    out_file.write_text(xhtml, encoding="utf-8")
                    fn_success += 1
                except: pass
            print(f"    ✓ {fn_success}/{len(footnotes)} footnotes decrypted")
    except Exception as e:
        print(f"  (No Footnote table or error: {e})")

    con.close()
    print(f"\n=== Summary: {success}/{len(docs)} documents decrypted, {failed} failed ===")
    return success, failed


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nExamples:")
        print("  python3 jwpub_decryptor.py /path/to/w_E_19800101.jwpub")
        print("  python3 jwpub_decryptor.py /path/to/w_E_19800101.jwpub /output/dir")
        sys.exit(1)

    jwpub_path = sys.argv[1]
    if len(sys.argv) >= 3:
        output_dir = sys.argv[2]
    else:
        # Default: same name as jwpub, with _extracted suffix
        output_dir = Path(jwpub_path).stem + "_decrypted"

    print(f"=== JWPUB Decryptor ===")
    print(f"Input:  {jwpub_path}")
    print(f"Output: {output_dir}")
    print()

    # Step 1: Extract JWPUB
    print("[1/2] Extracting JWPUB...")
    db_path, manifest = extract_jwpub(jwpub_path, output_dir)
    print(f"  DB: {db_path}")
    print(f"  Publication: {manifest.get('publication',{}).get('title','?')}")

    # Step 2: Decrypt all documents
    print(f"\n[2/2] Decrypting Document.Content BLOBs...")
    success, failed = decrypt_all_documents(db_path, Path(output_dir))

    if success > 0:
        print(f"\n✓ SUCCESS — {success} documents decrypted to {output_dir}/decrypted/")
        # Show sample
        sample_files = sorted(Path(output_dir, "decrypted").glob("doc_*.xhtml"))[:1]
        if sample_files:
            print(f"\n=== Sample (first 800 chars of {sample_files[0].name}) ===")
            print(sample_files[0].read_text(encoding="utf-8")[:800])
    else:
        print(f"\n✗ FAILED — no documents decrypted")


if __name__ == "__main__":
    main()
