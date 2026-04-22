"""
vector_store.py — ChromaDB-backed persistent vector store.

Key improvements:
  1. PERSISTS to disk — no re-indexing on every run
  2. Checks for existing index — only re-indexes new/changed docs
  3. MMR (Maximal Marginal Relevance) retrieval — reduces redundant chunks
  4. Similarity threshold — rejects chunks below a cosine floor
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List, Tuple

import numpy as np

from config import (
    COLLECTION_NAME,
    VECTOR_STORE_DIR,
    TOP_K,
    MMR_ENABLED,
    MMR_LAMBDA,
    SIMILARITY_THRESHOLD,
)
from chunking import Chunk
from embeddings import EmbeddingModel


# ─── MMR Helpers ─────────────────────────────────────────────────────────────

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def _mmr(
    query_vec: np.ndarray,
    candidates: List[Tuple[Chunk, np.ndarray, float]],
    top_k: int,
    mmr_lambda: float,
) -> List[Tuple[Chunk, float]]:
    """
    Maximal Marginal Relevance reranking.
    mmr_lambda=1.0 → pure relevance | mmr_lambda=0.0 → pure diversity
    """
    if not candidates:
        return []

    selected:  List[Tuple[Chunk, np.ndarray, float]] = []
    remaining = list(candidates)

    while len(selected) < top_k and remaining:
        scores: List[float] = []
        for chunk, emb, relevance in remaining:
            if not selected:
                mmr_score = relevance
            else:
                max_sim   = max(_cosine_sim(emb, s_emb) for _, s_emb, _ in selected)
                mmr_score = mmr_lambda * relevance - (1 - mmr_lambda) * max_sim
            scores.append(mmr_score)

        best_idx = int(np.argmax(scores))
        selected.append(remaining.pop(best_idx))

    return [(c, score) for c, _, score in selected]


# ─── VectorStore ─────────────────────────────────────────────────────────────

class VectorStore:
    """Persistent ChromaDB collection with MMR retrieval."""

    def __init__(
        self,
        persist_dir: str | Path = VECTOR_STORE_DIR,
        collection_name: str = COLLECTION_NAME,
    ):
        self.persist_dir     = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self._client         = None
        self._collection     = None

    def _get_collection(self):
        if self._collection is None:
            try:
                import chromadb
            except ImportError:
                raise ImportError("chromadb is required: pip install chromadb")

            self._client     = chromadb.PersistentClient(path=str(self.persist_dir))
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    # ── Index Management ─────────────────────────────────────────────────────

    def is_empty(self) -> bool:
        return self._get_collection().count() == 0

    def clear(self) -> None:
        if self._client is not None:
            try:
                self._client.delete_collection(self.collection_name)
            except Exception:
                pass
        self._collection = None
        print("  🗑️  Vector store cleared.")

    def _doc_already_indexed(self, doc_id: str) -> bool:
        coll    = self._get_collection()
        results = coll.get(where={"doc_id": doc_id}, limit=1, include=[])
        return len(results["ids"]) > 0

    # ── Indexing ─────────────────────────────────────────────────────────────

    def add_chunks(
        self,
        chunks: List[Chunk],
        embeddings: np.ndarray,
        skip_existing: bool = True,
    ) -> int:
        """Store chunks + embeddings in ChromaDB. Returns number inserted."""
        coll                         = self._get_collection()
        ids, docs, metas, vecs       = [], [], [], []
        skipped                      = 0

        for chunk, emb in zip(chunks, embeddings):
            if skip_existing and self._doc_already_indexed(chunk.doc_id):
                skipped += 1
                continue
            ids.append(chunk.chunk_id)
            docs.append(chunk.text)
            metas.append(chunk.to_metadata())
            vecs.append(emb.tolist())

        if skipped:
            print(f"  ⏭️  Skipped {skipped} already-indexed chunks.")

        if not ids:
            return 0

        BATCH = 500
        for i in range(0, len(ids), BATCH):
            coll.upsert(
                ids=ids[i:i + BATCH],
                documents=docs[i:i + BATCH],
                metadatas=metas[i:i + BATCH],
                embeddings=vecs[i:i + BATCH],
            )

        print(f"  💾 Stored {len(ids)} chunks in ChromaDB.")
        return len(ids)

    # ── Retrieval ────────────────────────────────────────────────────────────

    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = TOP_K,
        use_mmr: bool = MMR_ENABLED,
        mmr_lambda: float = MMR_LAMBDA,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
        model: EmbeddingModel | None = None,
    ) -> List[Tuple[Chunk, float]]:
        """
        Retrieve top-k relevant chunks for a query embedding.

        Steps:
          1. Fetch (top_k * 2) from ChromaDB — tighter pool for quality
          2. Filter by similarity_threshold
          3. Apply MMR reranking (optional)
          4. Return top_k results sorted by relevance
        """
        coll    = self._get_collection()
        fetch_k = top_k * 2   # tighter pool than top_k*3 — better quality candidates

        results = coll.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(fetch_k, coll.count()),
            include=["documents", "metadatas", "distances", "embeddings"],
        )

        docs      = results["documents"][0]
        metas     = results["metadatas"][0]
        distances = results["distances"][0]
        raw_embs  = results["embeddings"][0]

        candidates: List[Tuple[Chunk, np.ndarray, float]] = []
        for doc_text, meta, dist, emb in zip(docs, metas, distances, raw_embs):
            similarity = 1.0 - dist   # ChromaDB cosine distance → similarity
            if similarity < similarity_threshold:
                continue

            chunk = Chunk(
                text=doc_text,
                source=meta.get("source", "unknown"),
                chunk_index=int(meta.get("chunk_index", 0)),
                page_hint=int(meta.get("page_hint", 0)),
                char_start=int(meta.get("char_start", 0)),
                doc_id=meta.get("doc_id", ""),
            )
            candidates.append((chunk, np.array(emb), similarity))

        if not candidates:
            return []

        if use_mmr and len(candidates) > top_k:
            return _mmr(query_embedding, candidates, top_k, mmr_lambda)

        candidates.sort(key=lambda x: x[2], reverse=True)
        return [(c, score) for c, _, score in candidates[:top_k]]

    def stats(self) -> dict:
        coll = self._get_collection()
        return {
            "total_chunks": coll.count(),
            "collection":   self.collection_name,
            "persist_dir":  str(self.persist_dir),
        }