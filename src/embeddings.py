"""
embeddings.py — Embedding model wrapper with mandatory batching.

Uses sentence-transformers (local, free, no API key needed).
All embeddings are produced in batches — never one-at-a-time — as required.
"""

from __future__ import annotations

from typing import List
import numpy as np

from config import EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE


class EmbeddingModel:
    """Thin wrapper around a SentenceTransformer model."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        print(f"  🔢 Loading embedding model: {model_name} …")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("sentence-transformers is required: pip install sentence-transformers")

        self.model      = SentenceTransformer(model_name)
        self.model_name = model_name
        print(f"     ✅ Embedding dim: {self.dim}")

    # ── Dimension helper (handles deprecated API gracefully) ──────────────────

    @property
    def dim(self) -> int:
        """Return embedding dimension — compatible with all ST versions."""
        # get_embedding_dimension() is the new name (ST >= 3.x)
        # get_sentence_embedding_dimension() is the old name (ST < 3.x)
        if hasattr(self.model, "get_embedding_dimension"):
            return self.model.get_embedding_dimension()
        return self.model.get_sentence_embedding_dimension()

    # ── Batched embedding ─────────────────────────────────────────────────────

    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = EMBEDDING_BATCH_SIZE,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Embed a list of texts in batches.
        Returns shape (N, embedding_dim) as a numpy array.
        """
        if not texts:
            return np.empty((0, self.dim))

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,   # unit vectors → cosine sim == dot product
        )
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string. Returns shape (dim,)."""
        vec = self.model.encode(
            [query],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vec[0]


# Singleton — reused across indexing and querying steps
_model_instance: EmbeddingModel | None = None


def get_embedding_model() -> EmbeddingModel:
    global _model_instance
    if _model_instance is None:
        _model_instance = EmbeddingModel()
    return _model_instance