"""
indexer.py — Step 1: Build the vector index from your documents.

Run once (or when you add new documents):
    python src/indexer.py

The index persists to ./vector_store/ — querying does NOT re-run this.

Flags:
    --force / -f   → wipe the existing index and re-index everything from scratch
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import DATA_DIR, VECTOR_STORE_DIR, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL
from ingestion import load_documents
from chunking import chunk_documents
from embeddings import get_embedding_model
from vector_store import VectorStore


def run_indexing(force_reindex: bool = False) -> None:
    print("\n" + "=" * 60)
    print("  RAG Q&A Bot — Indexing Step")
    print("=" * 60)

    # ── 1. Load documents ─────────────────────────────────────────────────────
    print(f"\n[1/4] Loading documents from: {DATA_DIR}")
    docs = load_documents(DATA_DIR)
    print(f"      → {len(docs)} document(s) loaded.\n")

    # ── 2. Chunk ──────────────────────────────────────────────────────────────
    print(f"[2/4] Chunking  (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}) …")
    chunks = chunk_documents(docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    print(f"      → {len(chunks)} total chunks.\n")

    # ── 3. Embed (batched) ────────────────────────────────────────────────────
    print(f"[3/4] Embedding chunks  (model={EMBEDDING_MODEL}) …")
    t0         = time.time()
    model      = get_embedding_model()
    texts      = [c.text for c in chunks]
    embeddings = model.embed_texts(texts, show_progress=True)
    print(f"      → Embedded {len(embeddings)} chunks in {time.time() - t0:.1f}s.\n")

    # ── 4. Store ──────────────────────────────────────────────────────────────
    print("[4/4] Storing in ChromaDB …")
    store = VectorStore(persist_dir=VECTOR_STORE_DIR)

    if force_reindex:
        store.clear()

    store.add_chunks(chunks, embeddings, skip_existing=not force_reindex)

    stats = store.stats()
    print(f"\n{'=' * 60}")
    print(f"  ✅ Indexing complete!")
    print(f"     Total chunks in store : {stats['total_chunks']}")
    print(f"     Persist location      : {stats['persist_dir']}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    force = "--force" in sys.argv or "-f" in sys.argv
    if force:
        print("⚠️  --force flag detected: re-indexing all documents.")
    run_indexing(force_reindex=force)