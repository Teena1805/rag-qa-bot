# RAG Document Q&A Bot

A Retrieval-Augmented Generation (RAG) pipeline that lets you ask natural language questions against a collection of documents and receive accurate, grounded answers with source citations. Answers are generated strictly from the document content — the LLM is explicitly prevented from using its own training knowledge.

---

## Demo Video

[Click here to watch](https://www.loom.com/share/d8439003640b4a0cb5d22aa239a6c9d9)

---

## Tech Stack

| Component | Library / Tool | Version |
|-----------|---------------|---------|
| Document loading | `pypdf`, `python-docx` | ≥4.2, ≥1.1 |
| Text chunking | Custom recursive splitter | — |
| Embedding model | `sentence-transformers` — `all-MiniLM-L6-v2` | ≥3.0 |
| Vector database | `chromadb` (persistent) | ≥0.5 |
| LLM (answer gen) | OpenAI GPT-3.5-turbo **or** Anthropic Claude Haiku | via API |
| CLI display | `rich` | ≥13.7 |
| Web UI | `streamlit` | ≥1.35 |
| Config / secrets | `python-dotenv` | ≥1.0 |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                       INDEXING (run once)                       │
│                                                                 │
│  /data/*.pdf/.txt/.docx                                         │
│        │                                                        │
│        ▼                                                        │
│  [ingestion.py]  Load + clean text (strip headers/footers)     │
│        │                                                        │
│        ▼                                                        │
│  [chunking.py]   Recursive char split  (512 chars, 100 overlap)│
│        │                                                        │
│        ▼                                                        │
│  [embeddings.py] Batch embed with SentenceTransformer          │
│        │                                                        │
│        ▼                                                        │
│  [vector_store.py] Persist to ChromaDB on disk                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    QUERYING (interactive)                       │
│                                                                 │
│  User Question                                                  │
│        │                                                        │
│        ▼                                                        │
│  [embeddings.py]   Embed query                                  │
│        │                                                        │
│        ▼                                                        │
│  [vector_store.py] Cosine similarity search → MMR reranking    │
│        │                                                        │
│        ▼                                                        │
│  Top-K diverse, relevant chunks                                │
│        │                                                        │
│        ▼                                                        │
│  [generator.py]    Strict "grounded only" prompt → LLM answer  │
│        │                                                        │
│        ▼                                                        │
│  Answer + Source Citations                                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Chunking Strategy

**Strategy:** Recursive Character Split (paragraph → sentence → word → character)

**Why this strategy:**
- Pure fixed-size splits often cut sentences mid-way, destroying the semantic unit and confusing the LLM.
- Recursive splitting respects natural text boundaries: it first tries to split on paragraph breaks (`\n\n`), then sentence endings (`. ! ?`), and only falls back to word/character splits as a last resort.
- **Chunk size = 512 characters** — small enough for precise retrieval; large enough to contain a full thought.
- **Overlap = 100 characters** — each chunk inherits the last 100 characters of the previous chunk, preventing context loss at boundaries.

---

## Embedding Model and Vector Database

**Embedding model: `all-MiniLM-L6-v2` (SentenceTransformers)**
- Fast (< 1 second per batch), runs entirely locally — no API key needed.
- 384-dimensional output, excellent semantic similarity quality for its size.
- All embedding calls are **batched** (never one-at-a-time in a loop).

**Vector database: ChromaDB (persistent)**
- Simple setup — no external server needed.
- Persists to `./vector_store/` on disk — indexing runs once, not on every query.
- Supports cosine similarity via HNSW index.

**Retrieval improvement — MMR (Maximal Marginal Relevance):**
- Standard top-k retrieval often returns several near-duplicate chunks from the same passage.
- MMR balances **relevance to the query** against **diversity among selected chunks**.
- Result: the LLM receives a richer, more varied context → more accurate answers.

---

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/rag-qa-bot.git
cd rag-qa-bot
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
```bash
cp .env.example .env
# The default config uses Ollama (free, local) — no API key needed!
# Only edit .env if you want to use OpenAI or Anthropic instead.
```

### 5. Install Ollama (one-time, for the free local LLM)
```bash
# Visit https://ollama.com/download and install for your OS, then:
ollama pull llama3.2   # downloads the model (~2 GB)
```
> **Using OpenAI or Anthropic instead?** Set `LLM_PROVIDER=openai` (or `anthropic`) in `.env` and add your API key. Ollama step not needed.

### 6. Add your documents
Place 4–5 documents (`.pdf`, `.txt`, or `.docx`) into the `/data` folder.

### 6. Index the documents (run once)
```bash
python src/indexer.py
```
To force a full re-index (e.g. after replacing documents):
```bash
python src/indexer.py --force
```

### 7. Start the bot

**Command-line interface:**
```bash
python src/main.py
```

**Streamlit web UI:**
```bash
streamlit run src/app.py
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `LLM_PROVIDER` | No | `ollama` (default, free) / `openai` / `anthropic` |
| `OLLAMA_BASE_URL` | No | Ollama server URL (default: `http://localhost:11434`) |
| `OLLAMA_MODEL` | No | Model name (default: `llama3.2`) |
| `OPENAI_API_KEY` | Only if `LLM_PROVIDER=openai` | Your OpenAI API key |
| `ANTHROPIC_API_KEY` | Only if `LLM_PROVIDER=anthropic` | Your Anthropic API key |

**The default setup requires NO API key.** Ollama runs the LLM entirely on your machine.

**Never commit your `.env` file.** It is listed in `.gitignore`.

---

## Example Queries

Below are 5 sample questions based on the documents in this project:

| # | Question | Expected Answer Theme | Source File  |
|---|----------|-----------------------|--------------|
| 1 | "What is a healthy lifestyle?" | Proper nutrition, regular exercise, adequate sleep, stress management | Healthy Lifestyle.txt |
| 2 | "What are the steps in the machine learning process?" | Data Collection, Preprocessing, Feature Selection, Model Training, Evaluation, Deployment | Introduction to Machine Learning.pdf |
| 3 | "What is entrepreneurship?" | Starting and managing a business, risk-taking, innovation | Entrepreneurship.docx |
| 4 | "How does digital technology impact daily life?" | Communication, education, work, social interaction | Digital Life and Technology.docx |
| 5 | "What is the difference between supervised and unsupervised learning?" | Labeled vs unlabeled data, classification vs clustering | Introduction to Machine Learning.pdf |

### Unanswerable Example (bot will refuse):
> "What is the current stock price of Apple?"

→ *"I could not find an answer to that question in the provided documents."*

---

## Known Limitations

- **PDF quality:** Scanned PDFs without OCR will produce garbled text — use text-based PDFs.
- **Very long documents:** Extremely long documents (200+ pages) may produce many chunks; increase `TOP_K` in `config.py` for better coverage.
- **Multi-hop reasoning:** Questions requiring information combined from different, distant parts of the documents may not always be answered correctly.
- **Table extraction:** Tables in PDFs are often extracted as plain text without structure, which can reduce accuracy for tabular queries.
- **Language:** Optimised for English documents; performance degrades on other languages.

---

## Project Structure

```
rag-qa-bot/
├── data/                   # Your documents go here
├── vector_store/           # ChromaDB index (auto-generated, gitignored)
├── src/
│   ├── config.py           # All tunable parameters
│   ├── ingestion.py        # PDF / TXT / DOCX loading
│   ├── chunking.py         # Recursive character splitter
│   ├── embeddings.py       # SentenceTransformer wrapper (batched)
│   ├── vector_store.py     # ChromaDB persistence + MMR retrieval
│   ├── generator.py        # LLM answer generation with grounding prompt
│   ├── query_engine.py     # End-to-end query pipeline
│   ├── indexer.py          # CLI: build the index
│   ├── main.py             # CLI: interactive Q&A loop
│   └── app.py              # Streamlit web UI
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```