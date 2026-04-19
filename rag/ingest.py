"""
Build the FAISS vector store from files under rag/knowledge_base/.

Usage:
    python -m rag.ingest

This chunks every .md/.txt file, embeds the chunks with a local
sentence-transformers model (all-MiniLM-L6-v2), and persists a FAISS
index plus the chunk texts + source metadata as a pickle sidecar.

Re-run this whenever the knowledge base changes.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from rag._types import Chunk

_BASE = Path(__file__).resolve().parent
KB_DIR = _BASE / "knowledge_base"
INDEX_DIR = _BASE / "vector_store"
INDEX_PATH = INDEX_DIR / "index.faiss"
META_PATH = INDEX_DIR / "chunks.pkl"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500       # characters
CHUNK_OVERLAP = 80     # characters


def _read_documents() -> list[tuple[str, str]]:
    """Return list of (filename, full_text) for every .md/.txt in KB_DIR."""
    docs: list[tuple[str, str]] = []
    for path in sorted(KB_DIR.iterdir()):
        if path.suffix.lower() not in {".md", ".txt"}:
            continue
        docs.append((path.name, path.read_text(encoding="utf-8")))
    if not docs:
        raise RuntimeError(f"No .md/.txt files found in {KB_DIR}")
    return docs


def _chunk(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Simple sliding-window character chunker that snaps to sentence boundaries."""
    text = text.strip()
    if len(text) <= size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        # snap to the nearest sentence boundary within the last 120 chars of window
        if end < len(text):
            boundary = max(
                text.rfind(". ", start, end),
                text.rfind("\n", start, end),
            )
            if boundary > start + size - 120:
                end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def build_index() -> None:
    print(f"[ingest] Loading embedding model: {EMBED_MODEL_NAME}")
    model = SentenceTransformer(EMBED_MODEL_NAME)

    print(f"[ingest] Reading documents from {KB_DIR}")
    docs = _read_documents()
    print(f"[ingest]   Found {len(docs)} document(s)")

    all_chunks: list[Chunk] = []
    for filename, text in docs:
        for i, piece in enumerate(_chunk(text)):
            all_chunks.append(Chunk(text=piece, source=filename, chunk_id=i))

    print(f"[ingest] Total chunks: {len(all_chunks)}")

    texts = [c.text for c in all_chunks]
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,  # enables cosine similarity via inner product
        convert_to_numpy=True,
    ).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product == cosine on normalized vectors
    index.add(embeddings)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    with open(META_PATH, "wb") as f:
        pickle.dump(all_chunks, f)

    print(f"[ingest] Wrote index → {INDEX_PATH}")
    print(f"[ingest] Wrote metadata → {META_PATH}")
    print(f"[ingest] Embedding dim: {dim}, vectors: {index.ntotal}")


if __name__ == "__main__":
    build_index()
