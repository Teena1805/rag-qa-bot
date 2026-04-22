"""
config.py — Central configuration for the RAG Q&A Bot.
All tunable parameters live here. Change values here to experiment.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR         = Path(__file__).resolve().parent.parent
DATA_DIR         = BASE_DIR / "data"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"

# ─── Embedding Model ──────────────────────────────────────────────────────────
EMBEDDING_MODEL      = "all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 64

# ─── Chunking ─────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 512
CHUNK_OVERLAP = 100

# ─── Vector Database (ChromaDB) ───────────────────────────────────────────────
COLLECTION_NAME = "rag_documents"

# ─── Retrieval ────────────────────────────────────────────────────────────────
TOP_K                = 3      # reduced from 6 — avoids redundant chunks from same doc
MMR_ENABLED          = True
MMR_LAMBDA           = 0.7    # raised slightly — favour relevance over diversity
SIMILARITY_THRESHOLD = 0.45   # raised from 0.30 — filters out weak matches

# ─── LLM ──────────────────────────────────────────────────────────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "llama3.2")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL   = "gpt-3.5-turbo"

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL   = "claude-3-haiku-20240307"

MAX_TOKENS  = 1024
TEMPERATURE = 0.1

# ─── Supported File Types ─────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}