"""
chunking.py — Recursive Character Text Splitter with semantic awareness.

Strategy chosen: Recursive Character Split
  WHY:
    ✓ Tries to split on paragraph breaks first → preserves topic coherence
    ✓ Falls back to sentence boundaries (. ! ?)
    ✓ Only splits on characters/words as last resort
    ✓ Configurable overlap prevents losing context at chunk boundaries

Page hint logic:
  - PDF with page_boundaries → exact page from boundary map
  - PDF without page_boundaries → heuristic estimate
  - TXT / DOCX → always page 1 (no real page concept)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from ingestion import RawDocument


# ─── Data Model ───────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    """A single text chunk ready for embedding."""
    text:        str
    source:      str
    chunk_index: int
    page_hint:   int   # accurate for PDFs; always 1 for TXT/DOCX
    char_start:  int
    doc_id:      str

    @property
    def chunk_id(self) -> str:
        return f"{self.doc_id}_chunk{self.chunk_index}"

    def to_metadata(self) -> dict:
        return {
            "source":      self.source,
            "chunk_index": self.chunk_index,
            "page_hint":   self.page_hint,
            "char_start":  self.char_start,
            "doc_id":      self.doc_id,
        }


# ─── Splitter ─────────────────────────────────────────────────────────────────

_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", " ", ""]


def _split_on_separator(text: str, sep: str, chunk_size: int) -> List[str]:
    if sep == "":
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    parts   = text.split(sep)
    merged: List[str] = []
    current = ""

    for part in parts:
        candidate = (current + sep + part).lstrip(sep) if current else part
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                merged.append(current)
            current = part

    if current:
        merged.append(current)

    return [m for m in merged if m.strip()]


def _recursive_split(text: str, chunk_size: int, separators: List[str]) -> List[str]:
    if len(text) <= chunk_size:
        return [text]

    sep, *rest = separators
    pieces     = _split_on_separator(text, sep, chunk_size)
    result: List[str] = []

    for piece in pieces:
        if len(piece) <= chunk_size:
            result.append(piece)
        elif rest:
            result.extend(_recursive_split(piece, chunk_size, rest))
        else:
            for i in range(0, len(piece), chunk_size):
                result.append(piece[i:i + chunk_size])

    return result


def _add_overlap(pieces: List[str], overlap: int) -> List[str]:
    if overlap <= 0 or len(pieces) < 2:
        return pieces

    overlapped: List[str] = [pieces[0]]
    for i in range(1, len(pieces)):
        tail = pieces[i - 1][-overlap:]
        overlapped.append(tail + " " + pieces[i])

    return overlapped


# ─── Page estimation (PDFs only) ──────────────────────────────────────────────

def _estimate_page(char_start: int, full_text: str, num_pages: int) -> int:
    """Rough page estimate for PDFs that have no page_boundaries."""
    if num_pages <= 1:
        return 1
    fraction = char_start / max(len(full_text), 1)
    return max(1, int(fraction * num_pages) + 1)


def _is_pdf(source: str) -> bool:
    return source.lower().endswith(".pdf")


# ─── Public API ───────────────────────────────────────────────────────────────

def chunk_document(
    doc: RawDocument,
    chunk_size: int = 512,
    chunk_overlap: int = 100,
) -> List[Chunk]:
    """Split a RawDocument into Chunks using recursive character splitting."""
    raw_pieces = _recursive_split(doc.text, chunk_size, _SEPARATORS)
    pieces     = _add_overlap(raw_pieces, chunk_overlap)

    chunks: List[Chunk] = []
    search_offset = 0

    for idx, piece in enumerate(pieces):
        stripped = piece.strip()
        if not stripped:
            continue

        core = stripped[:40]
        pos  = doc.text.find(core, search_offset)
        if pos == -1:
            pos = search_offset
        else:
            search_offset = pos + 1

        # ── Page assignment ───────────────────────────────────────────────────
        if _is_pdf(doc.source):
            if doc.page_boundaries:
                page = doc.get_page_for_char(pos)
            else:
                page = _estimate_page(pos, doc.text, doc.num_pages)
        else:
            # TXT / DOCX — no real page numbers, always 1
            page = 1

        chunks.append(Chunk(
            text=stripped,
            source=doc.source,
            chunk_index=idx,
            page_hint=page,
            char_start=pos,
            doc_id=doc.doc_id,
        ))

    return chunks


def chunk_documents(
    docs: List[RawDocument],
    chunk_size: int = 512,
    chunk_overlap: int = 100,
) -> List[Chunk]:
    """Chunk all documents in *docs*."""
    all_chunks: List[Chunk] = []
    for doc in docs:
        doc_chunks = chunk_document(doc, chunk_size, chunk_overlap)
        all_chunks.extend(doc_chunks)
        print(f"  ✂️  {doc.source} → {len(doc_chunks)} chunks")
    return all_chunks