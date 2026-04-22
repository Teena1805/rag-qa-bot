"""
app.py — Streamlit Web UI for the RAG Q&A Bot.

Run with:
    streamlit run src/app.py
"""

import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

# ─── Page config (must be first Streamlit call) ───────────────────────────────

st.set_page_config(
    page_title="RAG Q&A Bot",
    page_icon="🔍",
    layout="wide",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
.answer-box {
    background: #f0f7ff;
    border-left: 4px solid #2196F3;
    padding: 1rem 1.2rem;
    border-radius: 4px;
    margin-bottom: 1rem;
    white-space: pre-wrap;
}
.source-chip {
    display: inline-block;
    background: #e8f5e9;
    color: #2e7d32;
    border: 1px solid #a5d6a7;
    border-radius: 12px;
    padding: 2px 10px;
    font-size: 0.82rem;
    margin: 2px;
}
.chunk-box {
    background: #fafafa;
    border: 1px solid #ddd;
    border-radius: 6px;
    padding: 0.8rem;
    font-size: 0.85rem;
    line-height: 1.5;
}
.score-badge {
    color: #888;
    font-size: 0.75rem;
}
</style>
""", unsafe_allow_html=True)


# ─── Cached engine init ───────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading RAG engine…")
def load_engine():
    from query_engine import QueryEngine
    return QueryEngine()


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Settings")
    top_k       = st.slider("Chunks to retrieve (top-k)", 1, 6, 3)
    use_mmr     = st.toggle("Use MMR (diverse retrieval)", value=True)
    show_chunks = st.toggle("Show retrieved chunks", value=True)

    st.divider()
    st.caption("**About this bot**")
    st.caption("Answers questions using only the documents in `/data`. "
               "It will refuse to answer if the information is not in the documents.")

    st.divider()
    if st.button("🗑️ Clear chat history"):
        st.session_state.history = []
        st.rerun()


# ─── Main ─────────────────────────────────────────────────────────────────────

st.title("🔍 RAG Document Q&A Bot")
st.caption("Ask questions grounded in your uploaded documents.")

try:
    engine = load_engine()
    stats  = engine.vector_store.stats()
    st.success(f"✅ Index loaded — {stats['total_chunks']} chunks across your documents.")
except RuntimeError as e:
    st.error(str(e))
    st.info("Run `python src/indexer.py` first, then restart Streamlit.")
    st.stop()

# ── Chat history ──────────────────────────────────────────────────────────────

if "history" not in st.session_state:
    st.session_state.history = []

for item in st.session_state.history:
    with st.chat_message("user"):
        st.write(item["question"])
    with st.chat_message("assistant"):
        st.markdown(f'<div class="answer-box">{item["answer"]}</div>', unsafe_allow_html=True)
        if item.get("sources"):
            st.markdown(
                f'<div style="margin-top:0.5rem">'
                f'<span class="source-chip">📄 Source: {item["sources"][0]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

# ── Input ─────────────────────────────────────────────────────────────────────

question = st.chat_input("Ask a question about your documents…")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating answer…"):
            t0 = time.time()
            answer, retrieved = engine.query(question, top_k=top_k, use_mmr=use_mmr)
            elapsed = time.time() - t0

        st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

        if retrieved:
            from generator import _extract_sources
            sources = _extract_sources(retrieved)   # single highest-scoring source
            if sources:
                st.markdown(
                    f'<div style="margin-top:0.5rem">'
                    f'<span class="source-chip">📄 Source: {sources[0]}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            st.caption(f"⏱ {elapsed:.2f}s | {len(retrieved)} chunks retrieved")

            if show_chunks:
                with st.expander("📚 View retrieved chunks"):
                    for i, (chunk, score) in enumerate(retrieved, 1):
                        page_str = f" • page {chunk.page_hint}" if chunk.page_hint else ""
                        st.markdown(
                            f'<div class="chunk-box">'
                            f'<b>[{i}] {chunk.source}{page_str}</b> '
                            f'<span class="score-badge">relevance: {score:.3f}</span><br><br>'
                            f'{chunk.text}</div>',
                            unsafe_allow_html=True,
                        )
        else:
            st.warning("No relevant chunks found. Try rephrasing your question.")
            sources = []

        st.session_state.history.append({
            "question": question,
            "answer":   answer,
            "sources":  sources if retrieved else [],
        })