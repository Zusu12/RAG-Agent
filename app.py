"""
RAGAgent Flask Server — Enhanced Web API

Endpoints:
  GET  /            — Serve the web UI
  POST /ask         — Query the RAG pipeline
  POST /upload      — Upload and ingest a document
  GET  /documents   — List ingested documents
  DELETE /documents/<name> — Remove a document
  POST /clear       — Clear conversation memory
  GET  /stats       — Collection statistics
  GET  /health      — Health check
"""

import os
import logging
from pathlib import Path

from flask import Flask, render_template, request, jsonify
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

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Initialise RAG Agent
# ---------------------------------------------------------------------------
print("=" * 50)
print("  Initializing RAG Agent...")
print("=" * 50)
rag_agent = RAGAgent()
stats = rag_agent.get_stats()
print(f"  Backend : {stats['backend']}")
print(f"  Model   : {stats['model']}")
print(f"  Chunks  : {stats['total_chunks']}")
print("=" * 50)
print("  RAG Agent ready!")
print("=" * 50)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    """Serve the main web UI."""
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask_question():
    """Query the RAG pipeline.

    Request:  { "question": "..." }
    Response: { "answer": "...", "sources": [...], "backend": "...", "stats": {...} }
    """
    try:
        data = request.get_json(silent=True)
        if not data or not data.get("question", "").strip():
            return jsonify({"error": "Please provide a question."}), 400

        question = data["question"].strip()
        result = rag_agent.query(question)
        return jsonify(result)

    except Exception as e:
        logger.exception("Error in /ask")
        return jsonify({"error": str(e)}), 500


@app.route("/upload", methods=["POST"])
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
        return jsonify(rag_agent.get_stats())
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

    print("\n>> Starting RAG Agent server...")
    print("   Open: http://localhost:5000")
    print("   Health: http://localhost:5000/health\n")

    app.run(debug=False, host="0.0.0.0", port=5000)