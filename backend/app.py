"""Flask backend — JW RAG API."""
import os
import json
import time
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS

import ollama_client
import rag

app = Flask(__name__, static_folder=None)
CORS(app)


@app.route("/health")
def health():
    """Service health + Ollama status."""
    return jsonify({
        "status": "ok",
        "ollama_ready": ollama_client.is_ready(),
        "ollama_models": ollama_client.list_models(),
        "llm_model": ollama_client.DEFAULT_LLM,
        "embed_model": ollama_client.DEFAULT_EMBED,
    })


@app.route("/stats")
def stats():
    return jsonify(rag.get_stats())


@app.route("/publications")
def publications():
    return jsonify(rag.list_publications())


@app.route("/search")
def search():
    """Pure vector search (no LLM)."""
    q = request.args.get("q", "").strip()
    top_k = int(request.args.get("top_k", 20))
    pub = request.args.get("pub", None) or None
    if not q:
        return jsonify({"error": "missing 'q' param"}), 400
    try:
        results = rag.search_chunks(q, top_k=top_k) if not pub else rag.search_chunks(q, top_k=top_k)
        return jsonify({"query": q, "results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ===== Graph endpoints =====

@app.route("/entities/search")
def search_entities():
    """Vector search over entities."""
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify({"error": "missing 'q'"}), 400
    return jsonify({"query": q, "entities": rag.search_entities(q, top_k=int(request.args.get("top_k", 10)))})


@app.route("/graph")
def get_graph():
    """Get entities + relationships for visualization."""
    entity_id = request.args.get("entity_id", type=int)
    limit = request.args.get("limit", 50, type=int)
    return jsonify(rag.get_entity_graph(entity_id=entity_id, limit=limit))


@app.route("/communities")
def list_communities():
    return jsonify(rag.list_communities())


@app.route("/sessions", methods=["GET", "POST"])
def sessions():
    if request.method == "POST":
        body = request.get_json(silent=True) or {}
        s = rag.create_session(body.get("title"))
        return jsonify(s), 201
    else:
        return jsonify(rag.list_sessions())


@app.route("/sessions/<session_uid>", methods=["GET"])
def get_session(session_uid):
    s = rag.get_session(session_uid)
    if not s:
        return jsonify({"error": "not found"}), 404
    messages = rag.get_messages(s["id"])
    return jsonify({"session": s, "messages": messages})


@app.route("/chat", methods=["POST"])
def chat():
    """Streaming RAG chat endpoint.

    Request: {"question": "...", "session_uid": "...", "pub_filter": null, "top_k": 8, "model": null}
    Response: Server-Sent Events stream with:
      data: {"type":"sources","sources":[...]}
      data: {"type":"token","content":"..."}
      data: {"type":"done"}
    """
    body = request.get_json(silent=True) or {}
    question = body.get("question", "").strip()
    session_uid = body.get("session_uid")
    pub_filter = body.get("pub_filter")
    top_k = int(body.get("top_k", 8))
    model = body.get("model")

    if not question:
        return jsonify({"error": "missing 'question'"}), 400

    # Resolve session
    history = []
    session = None
    if session_uid:
        session = rag.get_session(session_uid)
        if session:
            history = rag.get_messages(session["id"])

    def generate():
        full_answer = []
        sources_out = []
        for event in rag.answer_question(
            question, history=history, top_k=top_k,
            pub_filter=pub_filter, model=model
        ):
            if event["type"] == "sources":
                sources_out = event["sources"]
                yield f"data: {json.dumps(event)}\n\n"
            elif event["type"] == "token":
                full_answer.append(event["content"])
                yield f"data: {json.dumps(event)}\n\n"
            elif event["type"] == "done":
                # Persist to DB
                if session:
                    rag.add_message(session["id"], "user", question)
                    rag.add_message(session["id"], "assistant", "".join(full_answer), sources_out)
                # Send final sources again with full content
                yield f"data: {json.dumps({'type':'done','sources':sources_out,'answer':''.join(full_answer)})}\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


@app.route("/ingest", methods=["POST"])
def trigger_ingest():
    """Trigger an ingestion job by spawning a one-shot ingest container.

    Uses the Docker socket (mounted into the backend container) to run:
        docker compose --project-name jw-graphrag run --rm ingest \
            python ingest.py --pub <pub_code> [--issue <issue>]

    Request: {"pub_code": "w", "issue": "19800101"}
    Response: streamed JSON lines with status updates
    """
    body = request.get_json(silent=True) or {}
    pub_code = body.get("pub_code")
    issue = body.get("issue")
    if not pub_code:
        return jsonify({"error": "missing 'pub_code'"}), 400

    import subprocess, shlex

    cmd = [
        "docker", "compose", "--project-name", "jw-graphrag",
        "-f", "/docker-compose.yml",
        "run", "--rm", "ingest",
        "python", "ingest.py", "--pub", pub_code,
    ]
    if issue:
        cmd.extend(["--issue", issue])

    def stream():
        yield f"data: {json.dumps({'type':'start','command':' '.join(shlex.quote(c) for c in cmd),'pub_code':pub_code,'issue':issue})}\n\n"
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            for line in proc.stdout:
                # Strip trailing newline; emit as a log event
                yield f"data: {json.dumps({'type':'log','line':line.rstrip()})}\n\n"
            proc.wait()
            ok = proc.returncode == 0
            yield f"data: {json.dumps({'type':'done','ok':ok,'exit_code':proc.returncode})}\n\n"
        except FileNotFoundError:
            yield f"data: {json.dumps({'type':'error','error':'docker CLI not found in backend container'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type':'error','error':str(e)})}\n\n"

    return Response(stream_with_context(stream()), mimetype="text/event-stream")


@app.route("/ingest_sync", methods=["POST"])
def trigger_ingest_sync():
    """Synchronous version of /ingest (waits for completion). Useful for curl testing."""
    body = request.get_json(silent=True) or {}
    pub_code = body.get("pub_code")
    issue = body.get("issue")
    if not pub_code:
        return jsonify({"error": "missing 'pub_code'"}), 400

    import subprocess, shlex
    cmd = [
        "docker", "compose", "--project-name", "jw-graphrag",
        "-f", "/docker-compose.yml",
        "run", "--rm", "ingest",
        "python", "ingest.py", "--pub", pub_code,
    ]
    if issue:
        cmd.extend(["--issue", issue])
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        return jsonify({
            "ok": result.returncode == 0,
            "stdout": result.stdout[-5000:],
            "stderr": result.stderr[-5000:],
            "command": " ".join(cmd),
        })
    except subprocess.TimeoutExpired:
        return jsonify({"ok": False, "error": "ingest timed out (30 min)"}), 504
    except FileNotFoundError:
        return jsonify({"ok": False, "error": "docker CLI not installed in backend container — rebuild with: docker compose build backend"}), 500
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/")
def root():
    return jsonify({
        "service": "JW RAG Backend",
        "endpoints": [
            "GET  /health",
            "GET  /stats",
            "GET  /publications",
            "GET  /search?q=...",
            "GET  /sessions",
            "POST /sessions",
            "GET  /sessions/<uid>",
            "POST /chat",
            "POST /ingest",
        ],
    })


if __name__ == "__main__":
    # Don't block startup waiting for Ollama — just warn.
    # The /health endpoint will report its status, and chat calls will fail
    # with a clear message if Ollama isn't reachable.
    print(f"Configured OLLAMA_HOST = {ollama_client.OLLAMA_HOST}")
    print(f"Configured LLM model   = {ollama_client.DEFAULT_LLM}")
    print(f"Configured embed model = {ollama_client.DEFAULT_EMBED}")
    if ollama_client.is_ready():
        print("✓ Ollama reachable. Models:", ollama_client.list_models()[:5])
    else:
        print("⚠ Ollama not reachable at the configured host.")
        print("  If you're using host Ollama, make sure it's running:")
        print("    ollama serve   # or check `systemctl status ollama`")
        print("  And make sure these models are pulled:")
        print(f"    ollama pull {ollama_client.DEFAULT_LLM}")
        print(f"    ollama pull {ollama_client.DEFAULT_EMBED}")
        print("  Backend will still start — chat will return an error until Ollama is up.")

    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, threaded=True, debug=False)
