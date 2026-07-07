# 🧠 RAG Agent — Production-Ready Retrieval-Augmented Generation

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/framework-Flask-lightgrey.svg)](https://flask.palletsprojects.com/)
[![ChromaDB](https://img.shields.io/badge/vector_db-ChromaDB-orange.svg)](https://www.trychroma.com/)
[![HuggingFace](https://img.shields.io/badge/LLM_API-HuggingFace-yellow.svg)](https://huggingface.co/docs/api-inference/index)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An intelligent, production-ready Retrieval-Augmented Generation (RAG) assistant featuring a premium glassmorphic dark UI, high-performance hybrid search, and cross-encoder reranking. 

This repository is self-contained, clean, and ready for deployment or GitHub distribution.

---

## 🌟 Key Features

- ⚡ **HuggingFace Inference API Integration** — Seamlessly query advanced models like `Qwen2.5-7B-Instruct`, `Mistral-7B-Instruct-v0.3`, or `Llama-3.1` using high-speed API endpoints. Includes automatic fallback if no key is provided.
- 📂 **Multi-Format Document Ingestion** — Upload and parse PDF, TXT, DOCX, and Markdown files instantly.
- 🔀 **High-Recall Hybrid Search** — Combines dense semantic vector search (via ChromaDB & `sentence-transformers`) with lexical search (`BM25Okapi`) for optimal document retrieval.
- 🏆 **Cross-Encoder Reranking** — Employs `ms-marco-MiniLM-L-6-v2` as a second-stage ranker to ensure only the most relevant document chunks populate the LLM context window.
- 💬 **Stateful Conversation Memory** — Automatically maintains dialogue state across multi-turn queries.
- 💾 **Persistent Vector Indexing** — Disk-backed storage using ChromaDB that persists database state across server restarts.
- 🌙 **Premium Dark Web UI** — Sleek responsive frontend with drag-and-drop file uploading, ingestion statistics, and source citations.

---

## 🛠️ Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────┐
│ Hybrid Search Retrieval Engine  │
│ ├─ Semantic (ChromaDB + Cosine) │
│ └─ Lexical (BM25 Keyword Match) │
└────────────────┬────────────────┘
                 │ Top-K Candidates
                 ▼
┌─────────────────────────────────┐
│     Cross-Encoder Reranking     │ ← Evaluates full passage-query relevance
└────────────────┬────────────────┘
                 │ Top-3 Highest Scoring Chunks
                 ▼
┌─────────────────────────────────┐
│      LLM Generation Context     │ ← Appends conversation history memory
└────────────────┬────────────────┘
                 │
                 ▼
          Answer + Sources
```

---

## 📂 Repository Structure

```text
RAGAgent-GitHub/
├── documents/                # Sample/Uploaded documents
│   ├── agent_info.txt
│   └── world_landmarks.txt
├── static/                   # Placeholder for static files
├── templates/
│   └── index.html            # Web UI template (HTML/CSS/JS)
├── .env.example              # Environment variables template
├── .gitignore                # Production ignore settings
├── app.py                    # Flask Web API entrypoint
├── rag_module.py             # Core RAG pipeline logic
└── requirements.txt          # Project dependencies
```

---

## 🚀 Quick Start

### 1. Prerequisites
- **Python 3.10 or higher**
- A Hugging Face account (to generate a free API token for high-quality text generation).

### 2. Installation
Clone or copy this repository to your local machine, navigate to the folder, and install the required Python packages:

```bash
pip install -r requirements.txt
```

### 3. Environment Setup
Copy the `.env.example` file to `.env`:

```bash
cp .env.example .env
```

Open `.env` in your text editor and add your **HuggingFace API Token**:

```env
HF_TOKEN=hf_your_actual_token_here
```
> [!NOTE]
> You can generate a free token under your profile settings on Hugging Face: [Settings -> Access Tokens](https://huggingface.co/settings/tokens). A "Read" access token is all that is required.

### 4. Running the Application
Start the Flask development server:

```bash
python app.py
```

Open your browser and navigate to:
- **Web UI:** [http://localhost:5000](http://localhost:5000)
- **Health Check:** [http://localhost:5000/health](http://localhost:5000/health)

---

## 🌐 API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Serves the web-based interactive UI. |
| `/ask` | POST | Submits a query. Request body: `{"question": "Your question"}`. |
| `/upload` | POST | Uploads a file. Expects multipart/form-data. |
| `/documents` | GET | Returns a list of all ingested files and their chunk counts. |
| `/documents/<name>` | DELETE | Deletes all chunks associated with `<name>` from the vector database. |
| `/clear` | POST | Resets conversational memory context. |
| `/stats` | GET | Returns database capacity, model configurations, and backend status. |
| `/health` | GET | System status endpoint. |

---

## ⚙️ Configuration Reference

Adjust these variables inside your `.env` file to customize pipeline behaviors:

| Variable | Default | Description |
|---|---|---|
| `HF_TOKEN` | — | Hugging Face Access Token. |
| `HF_MODEL` | `Qwen/Qwen2.5-7B-Instruct` | Target generation LLM. |
| `EMBEDDING_MODEL` | `all-mpnet-base-v2` | Embedding model used for sentence representation. |
| `CHUNK_SIZE` | `500` | Target character length for each document chunk. |
| `CHUNK_OVERLAP` | `50` | Characters shared between contiguous chunks. |
| `CHROMA_DIR` | `./chroma_db` | Storage path on disk for ChromaDB. |

---
## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.



