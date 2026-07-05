"""
Bulk ingest script — download + decrypt + chunk + embed + extract entities
for ALL verified-downloadable JW publications (1950-2026 Watchtower, all
books, etc.).

Generates a catalog from the verified probes, then calls the backend's
/api/ingest endpoint for each publication in sequence.

Usage:
  # From the host (with the stack running):
  python bulk_ingest.py                    # ingest everything (~weeks on CPU)
  python bulk_ingest.py --limit 10         # just the first 10 publications
  python bulk_ingest.py --only w           # only Watchtower Study
  python bulk_ingest.py --only w --start-year 1980 --end-year 1990
  python bulk_ingest.py --dry-run          # list what would be ingested, don't actually ingest

The script calls http://localhost:8080/api/ingest_sync for each publication
and prints progress. Failed ingests are logged but don't stop the run.

Prerequisites:
  - Docker stack running: docker compose up -d
  - At least one model pulled: ollama pull qwen2.5:latest nomic-embed-text
"""
import argparse
import json
import sys
import time
import urllib.request
import urllib.parse
import urllib.error
from pathlib import Path


# ============= Verified-downloadable publications =============
# Built from the historical probes documented in /home/z/my-project/jw_extract/inventory/

def gen_watchtower_issues(start_year=1950, end_year=2026):
    """Generate Watchtower Study issues.
    Pre-2016: semi-monthly (1st + 15th of each month) — format YYYYMMDD
    2016+: monthly — format YYYYMM00
    """
    for year in range(start_year, end_year + 1):
        if year < 2016:
            for month in range(1, 13):
                for day in [1, 15]:
                    yield ("w", f"{year}{month:02d}{day:02d}")
        else:
            for month in range(1, 13):
                yield ("w", f"{year}{month:02d}00")


def gen_kingdom_ministry_issues(start_year=1970, end_year=2015):
    """Kingdom Ministry: monthly, YYYYMM00, 1970-2015."""
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            yield ("km", f"{year}{month:02d}00")


def gen_awake_issues(start_year=2006, end_year=2024):
    """Awake!: monthly through 2015, bimonthly/quarterly after.
    Use YYYYMM00 format — only months that actually exist will succeed."""
    monthly_years = list(range(2006, 2016))
    bimonthly_years = list(range(2016, 2025))  # 2016+ is bimonthly
    for year in monthly_years:
        if year > end_year: break
        for month in range(1, 13):
            yield ("g", f"{year}{month:02d}00")
    for year in bimonthly_years:
        if year > end_year: break
        for month in [1, 3, 5, 7, 9, 11]:  # bimonthly
            yield ("g", f"{year}{month:02d}00")


def gen_meeting_workbook_issues(start_year=2016, end_year=2026):
    """Meeting Workbook: bimonthly, YYYYMM00, 2016-2026."""
    for year in range(start_year, end_year + 1):
        for month in [1, 3, 5, 7, 9, 11]:
            yield ("mwb", f"{year}{month:02d}00")


def gen_watchtower_public_issues(start_year=2016, end_year=2026):
    """Watchtower Public Edition: bimonthly, 2016-2026."""
    for year in range(start_year, end_year + 1):
        for month in [1, 3, 5, 7, 9, 11]:
            yield ("wp", f"{year}{month:02d}00")


# ============= Books / Brochures / Manuals =============
# These have no issue — they're undated publications.
BOOKS = [
    # === Bibles ===
    ("nwt", None),     # New World Translation (2013 revision)
    ("nwtsty", None),  # NWT Study Edition
    ("bi12", None),    # Reference Bible (1984)
    ("by", None),      # Byington Bible

    # === Current Books (with EPUB) ===
    ("bh", None),      # What Does the Bible Really Teach?
    ("cl", None),      # Draw Close to Jehovah
    ("jy", None),      # Jesus: The Way, the Truth, the Life
    ("kr", None),      # God's Kingdom Rules!
    ("my", None),      # My Book of Bible Stories
    ("lr", None),      # Learn From the Great Teacher
    ("bt", None),      # Bearing Thorough Witness
    ("bt", None),
    ("lfb", None),     # Lessons You Can Learn From the Bible
    ("lff", None),     # Enjoy Life Forever!
    ("lmd", None),     # Love People — Make Disciples
    ("ll", None),      # Listen to God and Live Forever
    ("lv", None),      # Keep Yourselves in God's Love
    ("hf", None),      # Your Family Can Be Happy
    ("fy", None),      # The Secret of Family Happiness
    ("th", None),      # Apply Yourself to Reading and Teaching
    ("sjj", None),     # Sing Out Joyfully (songbook)
    ("sn", None),      # Sing to Jehovah (older songbook)
    ("yp1", None),     # Questions Young People Ask Vol 1
    ("yp2", None),     # Questions Young People Ask Vol 2
    ("fg", None),      # Good News From God!
    ("lc", None),      # Was Life Created?
    ("lf", None),      # Origin of Life — Five Questions
    ("yc", None),      # Teach Your Children

    # === Historical Books (JWPUB only) ===
    ("it", None),      # Insight on the Scriptures (4,782 docs!)
    ("rs", None),      # Reasoning From the Scriptures
    ("si", None),      # All Scripture Is Inspired of God
    ("tr", None),      # Truth That Leads to Eternal Life (1968)
    ("uw", None),      # United in Worship of the Only True God (1983)
    ("br", None),      # What Does the Bible Really Teach? (Br variant)
    ("bm", None),      # The Bible — What Is Its Message?
    ("be", None),      # Benefit From Theocratic Ministry School Education
    ("ed", None),      # Jehovah's Witnesses and Education
    ("cf", None),      # Come Be My Follower
    ("ct", None),      # Is There a Creator Who Cares?
    ("ce", None),      # Life — How Did It Get Here?
    ("dg", None),      # Does God Really Care About Us?
    ("dn", None),      # Divine Rulership — The Only Hope
    ("dp", None),      # Pay Attention to Daniel's Prophecy!
    ("dt", None),      # Path of Divine Truth
    ("dy", None),      # Divine Victory
    ("gh", None),      # Good News — To Make You Happy
    ("gm", None),      # The Bible — God's Word or Man's?
    ("go", None),      # Our Incoming World Government
    ("gt", None),      # The Greatest Man Who Ever Lived
    ("hb", None),      # How Can Blood Save Your Life?
    ("hl", None),      # How Can You Have a Happy Life?
    ("hp", None),      # Happiness — How to Find It
    ("hs", None),      # Holy Spirit (1976)
    ("hu", None),      # Human Plans Failing
    ("kp", None),      # Keep on the Watch!
    ("le", None),      # Enjoy Life on Earth Forever!
    ("lp", None),      # Life Does Have a Purpose (1965)
    ("ml", None),      # There Is Much More to Life!
    ("mn", None),      # Look! I Am Making All Things New
    ("op", None),      # Our Problems — Who Will Help?
    ("pc", None),      # Lasting Peace and Happiness
    ("ph", None),      # Pathway to Peace and Happiness
    ("pm", None),      # Paradise Restored to Mankind
    ("pr", None),      # What Is the Purpose of Life?
    ("sc", None),      # In Search of a Father
    ("sg", None),      # Theocratic Ministry School Guidebook
    ("sh", None),      # Mankind's Search for God
    ("sj", None),      # School and Jehovah's Witnesses
    ("sl", None),      # Man's Salvation out of World Distress
    ("tp", None),      # True Peace and Security
    ("ts", None),      # Is This Life All There Is?
    ("ws", None),      # Worldwide Security Under the Prince of Peace
    ("wt", None),      # Worship the Only True God
]


# ============= Main =============

def build_catalog(only=None, start_year=None, end_year=None, include_books=True):
    """Build the full list of (pub_code, issue) to ingest."""
    items = []
    if only is None or only == "w":
        items.extend(gen_watchtower_issues(start_year or 1950, end_year or 2026))
    if only is None or only == "km":
        items.extend(gen_kingdom_ministry_issues(start_year or 1970, end_year or 2015))
    if only is None or only == "g":
        items.extend(gen_awake_issues(start_year or 2006, end_year or 2024))
    if only is None or only == "mwb":
        items.extend(gen_meeting_workbook_issues(start_year or 2016, end_year or 2026))
    if only is None or only == "wp":
        items.extend(gen_watchtower_public_issues(start_year or 2016, end_year or 2026))
    if include_books and (only is None or only == "books"):
        items.extend(BOOKS)
    return items


def call_ingest(pub_code, issue, backend_url="http://localhost:8080"):
    """Call the backend's /api/ingest_sync endpoint."""
    payload = json.dumps({"pub_code": pub_code, "issue": issue}).encode()
    req = urllib.request.Request(
        f"{backend_url}/api/ingest_sync",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=3600) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        try:
            err_body = json.loads(e.read().decode())
        except Exception:
            err_body = {"error": str(e)}
        return {"ok": False, "error": err_body}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def main():
    p = argparse.ArgumentParser(description="Bulk ingest JW publications into GraphRAG")
    p.add_argument("--only", choices=["w", "wp", "g", "km", "mwb", "books"],
                   help="Only ingest one category")
    p.add_argument("--start-year", type=int, help="Start year (for dated publications)")
    p.add_argument("--end-year", type=int, help="End year (for dated publications)")
    p.add_argument("--limit", type=int, help="Max publications to ingest")
    p.add_argument("--dry-run", action="store_true", help="List what would be ingested, don't actually call the API")
    p.add_argument("--backend", default="http://localhost:8080", help="Backend URL")
    p.add_argument("--sleep", type=float, default=2.0, help="Sleep between ingests (seconds)")
    p.add_argument("--skip-failed", action="store_true", help="Don't retry failed ingests")
    args = p.parse_args()

    catalog = build_catalog(
        only=args.only,
        start_year=args.start_year,
        end_year=args.end_year,
        include_books=True,
    )

    if args.limit:
        catalog = catalog[:args.limit]

    print(f"\n{'='*70}")
    print(f"Bulk Ingest Plan")
    print(f"{'='*70}")
    print(f"Total publications to ingest: {len(catalog)}")
    if args.only:
        print(f"Category filter: {args.only}")
    if args.start_year or args.end_year:
        print(f"Year range: {args.start_year or 'start'} to {args.end_year or 'end'}")
    print(f"Backend: {args.backend}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print()

    if args.dry_run:
        for i, (pub, issue) in enumerate(catalog, 1):
            print(f"  [{i:4d}/{len(catalog)}] {pub:<5} {issue or 'undated':<10}")
        print(f"\nTotal: {len(catalog)} publications")
        # Estimate time
        # Watchtower: ~30 min each on CPU
        # Books: 30min-3hr depending on size (Insight = ~12 hours!)
        wt_count = sum(1 for p, _ in catalog if p == "w")
        book_count = sum(1 for p, _ in catalog if p in ("bh", "cl", "jy", "it", "nwt", "nwtsty", "bi12", "by"))
        other_count = len(catalog) - wt_count - book_count
        print(f"\nEstimated time on CPU (qwen2.5:latest, ~10s/chunk for entity extraction):")
        print(f"  Watchtower issues ({wt_count}): ~{wt_count * 30 // 60} hours")
        print(f"  Books ({book_count}, Insight alone = ~12hr): ~{book_count * 60 // 60} hours")
        print(f"  Other ({other_count}): ~{other_count * 20 // 60} hours")
        print(f"  TOTAL: ~{(wt_count * 30 + book_count * 60 + other_count * 20) // 3600} hours")
        print(f"\nWith a modern GPU (RTX 3060+), divide by ~6x.")
        return

    # Live ingest
    succeeded = 0
    failed = 0
    failed_items = []
    t0 = time.time()

    for i, (pub, issue) in enumerate(catalog, 1):
        elapsed = time.time() - t0
        eta = (elapsed / i) * (len(catalog) - i) if i > 0 else 0
        print(f"\n[{i}/{len(catalog)}] {pub} {issue or 'undated'}  "
              f"(elapsed {elapsed/60:.1f}min, ETA {eta/60:.1f}min)")

        result = call_ingest(pub, issue, args.backend)

        if result.get("ok"):
            succeeded += 1
            # Try to extract chunk count from stdout
            stdout = result.get("stdout", "")
            chunk_line = [l for l in stdout.split("\n") if "Total chunks:" in l]
            graph_line = [l for l in stdout.split("\n") if "Graph built:" in l]
            chunks = chunk_line[0].split(":")[1].strip() if chunk_line else "?"
            graph = graph_line[0].split(":", 1)[1].strip() if graph_line else "?"
            print(f"  ✓ chunks={chunks}, graph={graph}")
        else:
            failed += 1
            failed_items.append((pub, issue, result.get("error", "unknown")))
            err_str = str(result.get("error", ""))[:200]
            print(f"  ✗ FAILED: {err_str}")
            if not args.skip_failed:
                # Don't retry, just continue — the user can re-run later
                pass

        if i < len(catalog):
            time.sleep(args.sleep)

    # Summary
    print(f"\n{'='*70}")
    print(f"Bulk Ingest Summary")
    print(f"{'='*70}")
    print(f"Total: {len(catalog)}")
    print(f"Succeeded: {succeeded}")
    print(f"Failed: {failed}")
    print(f"Time: {(time.time() - t0) / 60:.1f} minutes")
    if failed_items:
        print(f"\nFailed items (re-run with --skip-failed to skip these):")
        for pub, issue, err in failed_items[:20]:
            print(f"  {pub} {issue or 'undated'}: {str(err)[:100]}")
        if len(failed_items) > 20:
            print(f"  ... and {len(failed_items) - 20} more")

    # Save failed list for re-running
    if failed_items:
        failed_path = Path("failed_ingests.json")
        failed_path.write_text(json.dumps([
            {"pub_code": p, "issue": i, "error": str(e)[:500]}
            for p, i, e in failed_items
        ], indent=2))
        print(f"\nFailed items saved to: {failed_path}")


if __name__ == "__main__":
    main()
