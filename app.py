"""
RAGAgent Flask Server — Enhanced Web API (Multi-User Hosted Mode)

Endpoints:
  GET  /              — Serve the web UI (includes setup overlay)
  POST /api/setup     — Validate & store HF token in session
  GET  /api/session-status — Check if current session has a valid token
  POST /api/logout    — Clear session token
  POST /ask           — Query the RAG pipeline (requires session token)
  POST /upload        — Upload and ingest a document
  GET  /documents     — List ingested documents
  DELETE /documents/<name> — Remove a document
  POST /clear         — Clear conversation memory
  GET  /stats         — Collection statistics
  GET  /health        — Health check
"""

import os
import secrets
import logging
from pathlib import Path
from functools import wraps

from flask import Flask, render_template, request, jsonify, session
from werkzeug.utils import secure_filename
from rag_module import RAGAgent

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "documents")
ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", secrets.token_hex(32))
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_HTTPONLY"] = True

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Initialise shared RAG Agent (embedding + reranker + ChromaDB — no LLM)
# The LLM backend is injected per-request via the user's session token.
# ---------------------------------------------------------------------------
print("=" * 50)
print("  Initializing RAG Agent (shared core)...")
print("=" * 50)
rag_agent = RAGAgent()  # Will init without HF_TOKEN if not in .env — that's fine
stats = rag_agent.get_stats()
print(f"  Backend : {stats['backend']} (per-user tokens used at runtime)")
print(f"  Model   : {stats['model']}")
print(f"  Chunks  : {stats['total_chunks']}")
print("=" * 50)
print("  RAG Agent ready!")
print("=" * 50)


# ---------------------------------------------------------------------------
# Auth guard decorator
# ---------------------------------------------------------------------------
def require_token(f):
    """Decorator that ensures a valid HF token is stored in the session."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("hf_token"):
            return jsonify({"error": "No API token configured. Please set up your HuggingFace token first."}), 401
        return f(*args, **kwargs)
    return decorated


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


# ---------------------------------------------------------------------------
# Setup / Auth Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    """Serve the main web UI (includes setup overlay if no token)."""
    return render_template("index.html")


@app.route("/api/session-status", methods=["GET"])
def session_status():
    """Check if the current session has a valid HF token."""
    has_token = bool(session.get("hf_token"))
    return jsonify({
        "authenticated": has_token,
        "model": session.get("hf_model", rag_agent.model_name) if has_token else None,
    })


@app.route("/api/setup", methods=["POST"])
def setup_token():
    """Validate & store the HF token in the session.

    Request:  { "token": "hf_..." }
    Response: { "status": "ok", "model": "..." } or { "error": "..." }
    """
    try:
        data = request.get_json(silent=True)
        if not data or not data.get("token", "").strip():
            return jsonify({"error": "Please provide a HuggingFace API token."}), 400

        token = data["token"].strip()

        # Validate the token with a lightweight test call
        try:
            from huggingface_hub import InferenceClient
            model = data.get("model", rag_agent.model_name)
            client = InferenceClient(model=model, token=token)
            # Make a minimal test call to verify the token works
            client.chat_completion(
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5,
            )
        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg or "Unauthorized" in error_msg or "Invalid" in error_msg:
                return jsonify({"error": "Invalid API token. Please check your token and try again."}), 401
            elif "404" in error_msg or "not found" in error_msg.lower():
                return jsonify({"error": f"Model '{model}' not found or not accessible with this token."}), 404
            else:
                return jsonify({"error": f"Token validation failed: {error_msg}"}), 400

        # Token is valid — store in session
        session["hf_token"] = token
        session["hf_model"] = model
        logger.info("User session configured with HF token (model: %s)", model)

        return jsonify({"status": "ok", "model": model})

    except Exception as e:
        logger.exception("Error in /api/setup")
        return jsonify({"error": str(e)}), 500


@app.route("/api/logout", methods=["POST"])
def logout():
    """Clear the session token."""
    session.pop("hf_token", None)
    session.pop("hf_model", None)
    return jsonify({"status": "logged_out"})


# ---------------------------------------------------------------------------
# App Routes (all require a valid session token)
# ---------------------------------------------------------------------------
@app.route("/ask", methods=["POST"])
@require_token
def ask_question():
    """Query the RAG pipeline using the session's HF token.

    Request:  { "question": "..." }
    Response: { "answer": "...", "sources": [...], "backend": "...", "stats": {...} }
    """
    try:
        data = request.get_json(silent=True)
        if not data or not data.get("question", "").strip():
            return jsonify({"error": "Please provide a question."}), 400

        question = data["question"].strip()
        hf_token = session["hf_token"]
        hf_model = session.get("hf_model")

        # Use per-token query method
        result = rag_agent.query_with_token(question, hf_token, model_name=hf_model)
        return jsonify(result)

    except Exception as e:
        logger.exception("Error in /ask")
        return jsonify({"error": str(e)}), 500


@app.route("/upload", methods=["POST"])
@require_token
def upload_file():
    """Upload and ingest a document into the knowledge base.

    Accepts multipart/form-data with a 'file' field.
    Response: { "source": "...", "chunks_added": N, "total_collection_size": N }
    """
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided."}), 400

        file = request.files["file"]
        if not file.filename:
            return jsonify({"error": "Empty filename."}), 400

        if not _allowed_file(file.filename):
            return jsonify({
                "error": f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            }), 400

        filename = secure_filename(file.filename)
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        file_path = os.path.join(UPLOAD_DIR, filename)
        file.save(file_path)

        # Ingest into RAG pipeline
        result = rag_agent.load_file(file_path)
        return jsonify(result)

    except Exception as e:
        logger.exception("Error in /upload")
        return jsonify({"error": str(e)}), 500


@app.route("/documents", methods=["GET"])
@require_token
def list_documents():
    """List all ingested documents with their chunk counts.

    Response: { "documents": [{ "name": "...", "chunks": N }], "total_chunks": N }
    """
    try:
        stats = rag_agent.get_stats()
        documents = []

        # Count chunks per source
        if stats["total_chunks"] > 0:
            try:
                all_meta = rag_agent.collection.get(include=["metadatas"])
                source_counts: dict[str, int] = {}
                for m in all_meta["metadatas"]:
                    if m and "source" in m:
                        source_counts[m["source"]] = source_counts.get(m["source"], 0) + 1
                documents = [
                    {"name": name, "chunks": count}
                    for name, count in sorted(source_counts.items())
                ]
            except Exception:
                documents = [{"name": s, "chunks": 0} for s in stats["sources"]]

        return jsonify({
            "documents": documents,
            "total_chunks": stats["total_chunks"],
        })

    except Exception as e:
        logger.exception("Error in /documents")
        return jsonify({"error": str(e)}), 500


@app.route("/documents/<name>", methods=["DELETE"])
@require_token
def delete_document(name):
    """Remove all chunks for a specific document from the knowledge base.

    Response: { "source": "...", "chunks_deleted": N }
    """
    try:
        result = rag_agent.delete_document(name)
        return jsonify(result)
    except Exception as e:
        logger.exception("Error in DELETE /documents/%s", name)
        return jsonify({"error": str(e)}), 500


@app.route("/clear", methods=["POST"])
@require_token
def clear_memory():
    """Clear conversation memory.

    Response: { "status": "cleared", "memory_turns": 0 }
    """
    rag_agent.clear_memory()
    return jsonify({"status": "cleared", "memory_turns": 0})


@app.route("/stats", methods=["GET"])
def get_stats():
    """Return collection and system statistics.

    Response: { "total_chunks": N, "total_documents": N, "backend": "...", ... }
    """
    try:
        stats = rag_agent.get_stats()
        # Override backend info based on session
        if session.get("hf_token"):
            stats["backend"] = "huggingface_api"
            stats["model"] = session.get("hf_model", stats["model"])
        return jsonify(stats)
    except Exception as e:
        logger.exception("Error in /stats")
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "backend": rag_agent.backend,
        "collection_size": rag_agent.collection.count(),
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs("templates", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    port = int(os.getenv("PORT", 7860))

    print("\n>> Starting RAG Agent server (multi-user hosted mode)...")
    print(f"   Open: http://localhost:{port}")
    print(f"   Health: http://localhost:{port}/health\n")

    app.run(debug=False, host="0.0.0.0", port=port)