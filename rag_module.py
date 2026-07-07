"""
RAGAgent — Enhanced Retrieval-Augmented Generation Module

Features:
  - HuggingFace Inference API for text generation (with local fallback)
  - Persistent ChromaDB vector storage
  - Smart document chunking with overlap
  - Multi-format document loading (PDF, TXT, DOCX, MD)
  - Hybrid search (vector + BM25 keyword)
  - Cross-encoder reranking
  - Conversation memory for multi-turn Q&A
"""

import os
import re
import hashlib
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from rank_bm25 import BM25Okapi

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------------------------------------------------------------------
# Constants & Config
# ---------------------------------------------------------------------------
DEFAULT_HF_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_EMBEDDING_MODEL = "all-mpnet-base-v2"
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
DEFAULT_CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
MAX_MEMORY_TURNS = 5


class RAGAgent:
    """Enhanced RAG Agent with HuggingFace Inference API, hybrid search, and reranking."""

    def __init__(
        self,
        hf_token: Optional[str] = None,
        model_name: Optional[str] = None,
        embedding_model_name: Optional[str] = None,
        collection_name: str = "rag_collection",
        chroma_dir: Optional[str] = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ):
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.model_name = model_name or os.getenv("HF_MODEL", DEFAULT_HF_MODEL)
        self.embedding_model_name = embedding_model_name or os.getenv(
            "EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL
        )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.conversation_memory: list[dict] = []
        self.backend = "unknown"

        # --- Embedding Model (always local for speed) ---
        logger.info("Loading embedding model: %s", self.embedding_model_name)
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        # --- Reranker (local cross-encoder) ---
        logger.info("Loading reranker: %s", DEFAULT_RERANKER_MODEL)
        self.reranker = CrossEncoder(DEFAULT_RERANKER_MODEL)

        # --- Persistent ChromaDB ---
        chroma_path = chroma_dir or DEFAULT_CHROMA_DIR
        logger.info("Initializing ChromaDB at: %s", chroma_path)
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB collection '%s' has %d documents.",
            collection_name,
            self.collection.count(),
        )

        # --- LLM Backend ---
        self._init_llm_backend()

    # ------------------------------------------------------------------
    # LLM Backend Initialisation
    # ------------------------------------------------------------------
    def _init_llm_backend(self):
        """Set up HuggingFace Inference API."""
        if not self.hf_token:
            logger.error("No HF_TOKEN found. Set it in your .env file.")
            self.backend = "none"
            return

        try:
            from huggingface_hub import InferenceClient

            self.hf_client = InferenceClient(
                model=self.model_name, token=self.hf_token
            )
            self.backend = "huggingface_api"
            logger.info(
                "[OK] HuggingFace Inference API ready -- model: %s", self.model_name
            )
        except Exception as e:
            logger.error("HF Inference API init failed: %s", e)
            self.backend = "none"

    # ------------------------------------------------------------------
    # Text Generation (Chat Completion API)
    # ------------------------------------------------------------------
    def _generate(self, messages: list[dict], max_tokens: int = 512) -> str:
        """Generate text using the HuggingFace Chat Completion API."""
        if self.backend != "huggingface_api":
            return "[No LLM backend available. Set HF_TOKEN in your .env file.]"

        try:
            response = self.hf_client.chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error("HF API generation failed: %s", e)
            return f"[Error generating response: {e}]"

    # ------------------------------------------------------------------
    # Document Loading
    # ------------------------------------------------------------------
    def load_file(self, file_path: str) -> dict:
        """Load and chunk a single file into the vector store.

        Supported formats: .txt, .md, .pdf, .docx

        Returns:
            dict with keys: source, chunks_added, total_collection_size
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = path.suffix.lower()
        text = ""

        if ext in (".txt", ".md"):
            text = path.read_text(encoding="utf-8", errors="ignore")

        elif ext == ".pdf":
            try:
                from PyPDF2 import PdfReader

                reader = PdfReader(str(path))
                text = "\n".join(
                    page.extract_text() or "" for page in reader.pages
                )
            except ImportError:
                raise ImportError("Install PyPDF2: pip install PyPDF2")

        elif ext == ".docx":
            try:
                from docx import Document

                doc = Document(str(path))
                text = "\n".join(p.text for p in doc.paragraphs)
            except ImportError:
                raise ImportError("Install python-docx: pip install python-docx")

        else:
            raise ValueError(f"Unsupported file format: {ext}")

        if not text.strip():
            return {"source": path.name, "chunks_added": 0, "total_collection_size": self.collection.count()}

        chunks = self._chunk_text(text, source=path.name)
        self._add_chunks(chunks)

        return {
            "source": path.name,
            "chunks_added": len(chunks),
            "total_collection_size": self.collection.count(),
        }

    def load_directory(self, dir_path: str) -> dict:
        """Load all supported files from a directory.

        Returns:
            dict with keys: files_processed, total_chunks, errors
        """
        supported = {".txt", ".md", ".pdf", ".docx"}
        directory = Path(dir_path)
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")

        results = {"files_processed": 0, "total_chunks": 0, "errors": []}
        for file_path in sorted(directory.iterdir()):
            if file_path.suffix.lower() in supported:
                try:
                    info = self.load_file(str(file_path))
                    results["files_processed"] += 1
                    results["total_chunks"] += info["chunks_added"]
                except Exception as e:
                    results["errors"].append({"file": file_path.name, "error": str(e)})

        return results

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------
    def _chunk_text(self, text: str, source: str = "unknown") -> list[dict]:
        """Split text into overlapping chunks, respecting sentence boundaries.

        Returns:
            List of dicts: {text, source, chunk_index}
        """
        # Clean up whitespace
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if not text:
            return []

        # Split into sentences (rough but effective)
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks = []
        current_chunk: list[str] = []
        current_length = 0

        for sentence in sentences:
            sentence_len = len(sentence)
            if current_length + sentence_len > self.chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(
                    {"text": chunk_text, "source": source, "chunk_index": len(chunks)}
                )
                # Overlap: keep last few sentences
                overlap_chars = 0
                overlap_sentences: list[str] = []
                for s in reversed(current_chunk):
                    if overlap_chars + len(s) > self.chunk_overlap:
                        break
                    overlap_sentences.insert(0, s)
                    overlap_chars += len(s)
                current_chunk = overlap_sentences
                current_length = overlap_chars

            current_chunk.append(sentence)
            current_length += sentence_len

        # Don't forget the last chunk
        if current_chunk:
            chunks.append(
                {"text": " ".join(current_chunk), "source": source, "chunk_index": len(chunks)}
            )

        return chunks

    # ------------------------------------------------------------------
    # Vector Store Operations
    # ------------------------------------------------------------------
    def _content_hash(self, text: str) -> str:
        """Generate a deterministic ID from content to avoid duplicates."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def _add_chunks(self, chunks: list[dict]):
        """Add text chunks to ChromaDB with embeddings and metadata."""
        if not chunks:
            return

        texts = [c["text"] for c in chunks]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=False).tolist()
        ids = [self._content_hash(t) for t in texts]
        metadatas = [
            {"source": c["source"], "chunk_index": c["chunk_index"]} for c in chunks
        ]

        # Upsert to handle re-uploads gracefully
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    def delete_document(self, source_name: str) -> dict:
        """Remove all chunks belonging to a specific source file.

        Returns:
            dict with keys: source, chunks_deleted
        """
        # Find all chunks from this source
        results = self.collection.get(where={"source": source_name})
        if not results["ids"]:
            return {"source": source_name, "chunks_deleted": 0}

        self.collection.delete(ids=results["ids"])
        return {"source": source_name, "chunks_deleted": len(results["ids"])}

    # ------------------------------------------------------------------
    # Retrieval: Hybrid Search
    # ------------------------------------------------------------------
    def _vector_search(self, query: str, k: int = 10) -> list[dict]:
        """Pure vector similarity search."""
        if self.collection.count() == 0:
            return []

        query_embedding = self.embedding_model.encode([query], show_progress_bar=False).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=min(k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        docs = []
        for i in range(len(results["ids"][0])):
            docs.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
                "score": 1 - results["distances"][0][i],  # cosine: 1 = identical
            })
        return docs

    def _hybrid_search(self, query: str, k: int = 10) -> list[dict]:
        """Combine vector search with BM25 keyword search for better recall."""
        vector_results = self._vector_search(query, k=k)
        if not vector_results:
            return []

        # BM25 keyword scoring on the vector-retrieved documents
        corpus = [doc["text"] for doc in vector_results]
        tokenized_corpus = [doc.lower().split() for doc in corpus]
        tokenized_query = query.lower().split()

        try:
            bm25 = BM25Okapi(tokenized_corpus)
            bm25_scores = bm25.get_scores(tokenized_query)
        except Exception:
            # If BM25 fails (e.g., empty corpus), use vector scores only
            return vector_results

        # Normalise both score sets to [0, 1]
        vector_scores = np.array([doc["score"] for doc in vector_results])
        bm25_scores = np.array(bm25_scores)

        if vector_scores.max() > 0:
            vector_scores = vector_scores / vector_scores.max()
        if bm25_scores.max() > 0:
            bm25_scores = bm25_scores / bm25_scores.max()

        # Weighted combination: 70% vector, 30% keyword
        hybrid_scores = 0.7 * vector_scores + 0.3 * bm25_scores

        for i, doc in enumerate(vector_results):
            doc["hybrid_score"] = float(hybrid_scores[i])

        # Sort by hybrid score descending
        vector_results.sort(key=lambda d: d["hybrid_score"], reverse=True)
        return vector_results

    # ------------------------------------------------------------------
    # Reranking
    # ------------------------------------------------------------------
    def _rerank(self, query: str, documents: list[dict], k: int = 3) -> list[dict]:
        """Use a cross-encoder to rerank retrieved documents."""
        if not documents:
            return []

        pairs = [[query, doc["text"]] for doc in documents]
        scores = self.reranker.predict(pairs)

        for i, doc in enumerate(documents):
            doc["rerank_score"] = float(scores[i])

        documents.sort(key=lambda d: d["rerank_score"], reverse=True)
        return documents[:k]

    # ------------------------------------------------------------------
    # Conversation Memory
    # ------------------------------------------------------------------

    def clear_memory(self):
        """Reset conversation history."""
        self.conversation_memory.clear()

    # ------------------------------------------------------------------
    # Main Query Pipeline
    # ------------------------------------------------------------------
    def query(self, question: str, k_retrieve: int = 8, k_rerank: int = 3) -> dict:
        """Run the full RAG pipeline: retrieve → rerank → generate.

        Returns:
            dict with keys: answer, sources, backend, stats
        """
        # 1. Hybrid search
        retrieved = self._hybrid_search(question, k=k_retrieve)

        # 2. Rerank
        top_docs = self._rerank(question, retrieved, k=k_rerank)

        # 3. Build prompt with context + memory
        context = "\n\n".join(
            f"[Source: {doc['metadata'].get('source', 'unknown')}] {doc['text']}"
            for doc in top_docs
        )
        # 3. Build chat messages with context + memory
        messages = self._build_chat_messages(question, context)

        # 4. Generate
        answer = self._generate(messages)

        # 5. Store in memory
        self.conversation_memory.append({"question": question, "answer": answer})

        # 6. Build source info
        sources = [
            {
                "text": doc["text"][:200] + ("..." if len(doc["text"]) > 200 else ""),
                "source": doc["metadata"].get("source", "unknown"),
                "score": round(doc.get("rerank_score", doc.get("hybrid_score", 0)), 3),
            }
            for doc in top_docs
        ]

        return {
            "answer": answer,
            "sources": sources,
            "backend": self.backend,
            "stats": {
                "retrieved": len(retrieved),
                "reranked": len(top_docs),
                "collection_size": self.collection.count(),
            },
        }

    # ------------------------------------------------------------------
    # Chat Message Builder
    # ------------------------------------------------------------------
    def _build_chat_messages(self, question: str, context: str) -> list[dict]:
        """Build chat messages for the HF Chat Completion API."""
        system_prompt = (
            "You are a helpful AI assistant. Answer the question based on the "
            "provided context. If the context doesn't contain enough information, "
            "say so honestly. Be concise and accurate. Do not repeat the context "
            "or the question in your answer."
        )

        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history
        for turn in self.conversation_memory[-MAX_MEMORY_TURNS:]:
            messages.append({"role": "user", "content": turn["question"]})
            messages.append({"role": "assistant", "content": turn["answer"]})

        # Add current question with context
        user_message = f"Context:\n{context}\n\nQuestion: {question}"
        messages.append({"role": "user", "content": user_message})

        return messages


    # ------------------------------------------------------------------
    # Stats & Info
    # ------------------------------------------------------------------
    def get_stats(self) -> dict:
        """Return collection statistics."""
        count = self.collection.count()

        # Get unique sources
        sources = set()
        if count > 0:
            try:
                all_meta = self.collection.get(include=["metadatas"])
                for m in all_meta["metadatas"]:
                    if m and "source" in m:
                        sources.add(m["source"])
            except Exception:
                pass

        return {
            "total_chunks": count,
            "total_documents": len(sources),
            "sources": sorted(sources),
            "backend": self.backend,
            "model": self.model_name,
            "embedding_model": self.embedding_model_name,
            "memory_turns": len(self.conversation_memory),
        }


# ---------------------------------------------------------------------------
# CLI entry point for testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  RAGAgent — Enhanced RAG Pipeline")
    print("=" * 60)

    agent = RAGAgent()
    stats = agent.get_stats()
    print(f"\nBackend: {stats['backend']}")
    print(f"Model: {stats['model']}")
    print(f"Collection: {stats['total_chunks']} chunks from {stats['total_documents']} documents")

    # Load documents from the documents/ directory if it exists and has files
    docs_dir = Path("documents")
    if docs_dir.is_dir() and any(docs_dir.iterdir()):
        print(f"\nLoading documents from {docs_dir}/...")
        result = agent.load_directory(str(docs_dir))
        print(f"  Processed {result['files_processed']} files, {result['total_chunks']} chunks")
        if result["errors"]:
            for err in result["errors"]:
                print(f"  [!] {err['file']}: {err['error']}")

    # Interactive mode
    print("\n--- Interactive mode (type 'quit' to exit) ---")
    while True:
        question = input("\nYou: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        result = agent.query(question)
        print(f"\nAssistant: {result['answer']}")
        if result["sources"]:
            print("\nSources:")
            for s in result["sources"]:
                print(f"  [{s['score']}] {s['source']}: {s['text']}")