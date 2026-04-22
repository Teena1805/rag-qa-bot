"""
query_engine.py — Retrieval + Generation pipeline for a single question.

This is the core RAG query logic, called by both the CLI and the Streamlit UI.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    VECTOR_STORE_DIR,
    TOP_K,
    MMR_ENABLED,
    MMR_LAMBDA,
    SIMILARITY_THRESHOLD,
)
from chunking import Chunk
from embeddings import get_embedding_model
from vector_store import VectorStore
from generator import AnswerGenerator


class QueryEngine:
    """
    End-to-end RAG query pipeline:
      query → embed → retrieve (MMR) → generate → return answer + sources
    """

    def __init__(self):
        self._embedding_model = None
        self._vector_store    = None
        self._generator       = None

    # ── Lazy loading ──────────────────────────────────────────────────────────

    @property
    def embedding_model(self):
        if self._embedding_model is None:
            self._embedding_model = get_embedding_model()
        return self._embedding_model

    @property
    def vector_store(self):
        if self._vector_store is None:
            self._vector_store = VectorStore(persist_dir=VECTOR_STORE_DIR)
            if self._vector_store.is_empty():
                raise RuntimeError(
                    "Vector store is empty!\n"
                    "Run  python src/indexer.py  first to index your documents."
                )
        return self._vector_store

    @property
    def generator(self):
        if self._generator is None:
            self._generator = AnswerGenerator()
        return self._generator

    # ── Public API ────────────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        top_k: int = TOP_K,
        use_mmr: bool = MMR_ENABLED,
    ) -> Tuple[str, List[Tuple[Chunk, float]]]:
        """
        Answer *question* using the RAG pipeline.

        Returns
        -------
        answer    : str — LLM-generated, grounded answer
        retrieved : List[(Chunk, similarity_score)] — chunks used as context
        """
        if not question.strip():
            return "Please enter a question.", []

        # Step 1: Embed the query
        query_vec = self.embedding_model.embed_query(question)

        # Step 2: Retrieve relevant chunks (with MMR)
        retrieved = self.vector_store.query(
            query_embedding=query_vec,
            top_k=top_k,
            use_mmr=use_mmr,
            mmr_lambda=MMR_LAMBDA,
            similarity_threshold=SIMILARITY_THRESHOLD,
        )

        # Step 3: Generate grounded answer
        answer, _ = self.generator.generate(question, retrieved)

        return answer, retrieved