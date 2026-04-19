"""
Loads the FAISS index + chunk metadata built by rag.ingest, and exposes
a `retrieve(query, k)` function the agent uses to fetch grounded market
insights and regulatory context.

The embedding model and FAISS index are loaded lazily on first call and
cached at module level so subsequent agent invocations are fast.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from rag._types import Chunk

_BASE = Path(__file__).resolve().parent
INDEX_PATH = _BASE / "vector_store" / "index.faiss"
META_PATH = _BASE / "vector_store" / "chunks.pkl"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class RetrievedChunk:
    text: str
    source: str
    score: float


class Retriever:
    def __init__(self) -> None:
        if not INDEX_PATH.exists() or not META_PATH.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {INDEX_PATH}. "
                f"Run `python -m rag.ingest` first to build it."
            )
        self.model = SentenceTransformer(EMBED_MODEL_NAME)
        self.index = faiss.read_index(str(INDEX_PATH))
        with open(META_PATH, "rb") as f:
            self.chunks: list[Chunk] = pickle.load(f)

    def retrieve(self, query: str, k: int = 4) -> list[RetrievedChunk]:
        q_emb = self.model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        ).astype("float32")
        scores, idxs = self.index.search(q_emb, k)
        results: list[RetrievedChunk] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            c = self.chunks[idx]
            results.append(RetrievedChunk(text=c.text, source=c.source, score=float(score)))
        return results


_RETRIEVER: Optional[Retriever] = None


def get_retriever() -> Retriever:
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = Retriever()
    return _RETRIEVER


def retrieve(query: str, k: int = 4) -> list[RetrievedChunk]:
    """Top-k chunks from the knowledge base for the given query."""
    return get_retriever().retrieve(query, k)


def format_context(results: list[RetrievedChunk]) -> str:
    """Concatenate retrieved chunks into a single context block with source tags."""
    if not results:
        return ""
    blocks = []
    for r in results:
        blocks.append(f"[Source: {r.source}]\n{r.text}")
    return "\n\n---\n\n".join(blocks)
