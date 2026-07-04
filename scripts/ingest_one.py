"""Quick one-shot ingest entrypoint (called by backend /ingest endpoint)."""
import sys
sys.path.insert(0, "/app")
from ingest import ingest_publication

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: ingest_one.py <pub_code> [issue] [--format FMT]")
        sys.exit(1)
    pub_code = sys.argv[1]
    issue = None
    fmt = None
    for i, arg in enumerate(sys.argv[2:], 2):
        if arg == "--format":
            fmt = sys.argv[i + 1]
        elif not arg.startswith("--") and not issue:
            issue = arg
    ok = ingest_publication(pub_code, issue, fmt)
    sys.exit(0 if ok else 1)
