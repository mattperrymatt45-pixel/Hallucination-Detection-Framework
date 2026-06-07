"""
retriever.py — FAISS-based document chunk retriever using Sentence Transformers.
"""

import os
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
_CHUNK_SIZE = 500
_TOP_K = 3


class DocumentRetriever:
    """
    Loads .txt files from a directory, chunks them, embeds them with
    Sentence Transformers, and stores them in a FAISS flat-L2 index
    for nearest-neighbour retrieval.
    """

    def __init__(self, docs_dir: str = "docs") -> None:
        self.docs_dir: Path = Path(docs_dir)
        self.model: SentenceTransformer = SentenceTransformer(_EMBEDDING_MODEL)
        self.chunks: list[str] = []
        self.index: Optional[faiss.IndexFlatIP] = None
        self._embeddings: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_index(self) -> int:
        """
        Read all .txt files, chunk them, embed them, and build the FAISS index.

        Returns:
            Number of chunks indexed.

        Raises:
            FileNotFoundError: If docs_dir does not exist.
            ValueError: If no text chunks could be extracted.
        """
        if not self.docs_dir.exists():
            raise FileNotFoundError(f"docs directory not found: {self.docs_dir}")

        raw_text = self._load_documents()
        self.chunks = self._chunk_text(raw_text)

        if not self.chunks:
            raise ValueError(
                "No text chunks found. Add .txt files to the docs/ directory."
            )

        logger.info("Embedding %d chunks…", len(self.chunks))
        embeddings: np.ndarray = self.model.encode(
            self.chunks,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).astype(np.float32)

        self._embeddings = embeddings
        dimension: int = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # inner-product ≡ cosine on L2-normalised vecs
        self.index.add(embeddings)

        logger.info("FAISS index built with %d vectors (dim=%d).", len(self.chunks), dimension)
        return len(self.chunks)

    def retrieve(self, query: str, top_k: int = _TOP_K) -> list[dict]:
        """
        Retrieve the top_k most similar passages for a query.

        Args:
            query: The search string.
            top_k: Number of results to return.

        Returns:
            List of dicts with keys ``text`` and ``score``.

        Raises:
            RuntimeError: If build_index() has not been called.
        """
        if self.index is None:
            raise RuntimeError("Index is not built. Call build_index() first.")

        query_vec: np.ndarray = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        k = min(top_k, len(self.chunks))
        scores, indices = self.index.search(query_vec, k)

        results: list[dict] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append({"text": self.chunks[idx], "score": float(score)})

        return results

    def get_evidence_text(self, query: str, top_k: int = _TOP_K) -> list[str]:
        """Convenience wrapper — returns only the passage texts."""
        return [r["text"] for r in self.retrieve(query, top_k=top_k)]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_documents(self) -> str:
        """Read all .txt files in docs_dir and concatenate their contents."""
        texts: list[str] = []
        for path in sorted(self.docs_dir.glob("*.txt")):
            try:
                content = path.read_text(encoding="utf-8", errors="ignore").strip()
                if content:
                    texts.append(content)
                    logger.debug("Loaded: %s (%d chars)", path.name, len(content))
            except OSError as exc:
                logger.warning("Could not read %s: %s", path, exc)

        if not texts:
            logger.warning("No .txt files found in %s.", self.docs_dir)

        return "\n\n".join(texts)

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = _CHUNK_SIZE) -> list[str]:
        """
        Split text into non-overlapping chunks of ``chunk_size`` characters,
        breaking on the nearest whitespace boundary where possible.
        """
        chunks: list[str] = []
        start = 0
        length = len(text)

        while start < length:
            end = min(start + chunk_size, length)

            # Try to break on a whitespace boundary
            if end < length:
                boundary = text.rfind(" ", start, end)
                if boundary > start:
                    end = boundary

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end

        return chunks
