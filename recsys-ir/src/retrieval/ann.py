"""Embedding index — FAISS / brute-force approximate nearest-neighbour search.

Provides a unified ``ArticleIndex`` interface for:
  1. FAISS ``IndexFlatIP`` — exact inner-product search on L2-normalized
     vectors (= cosine similarity).  Fast enough at 10K–100K articles
     (single-digit ms/query on CPU) and avoids recall-degrading bugs that
     approximate indexes can silently introduce.
  2. NumPy brute-force fallback — ``embeddings @ query.T`` — for environments
     where FAISS won't install.

The interface is designed so an approximate index (IVFFlat / HNSW) can be
swapped in later without changing callers.  Only ``FLAT`` is used now.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ── Try to import FAISS ───────────────────────────────────────────

try:
    import faiss
    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False
    logger.warning(
        "faiss not available — falling back to NumPy brute-force search. "
        "Install faiss-cpu for better performance."
    )


class IndexType(Enum):
    """Supported index types (only FLAT is used currently)."""
    FLAT = "flat"         # Exact search via IndexFlatIP
    IVFFLAT = "ivfflat"   # Approximate — reserved for later
    HNSW = "hnsw"         # Approximate — reserved for later


class ArticleIndex:
    """Embedding-based article search index.

    Wraps either FAISS ``IndexFlatIP`` (preferred) or a NumPy brute-force
    fallback, providing a consistent interface for nearest-neighbour search.

    Parameters
    ----------
    embeddings : np.ndarray
        L2-normalized embeddings, shape ``(n_articles, dim)``.
    article_ids : list[str]
        Article IDs corresponding to embedding rows.
    index_type : IndexType
        Which index backend to use.

    Usage::

        index = ArticleIndex(embeddings, article_ids)
        results = index.search(query_vec, k=10)
        results = index.search_restricted(query_vec, candidate_ids, k=10)
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        article_ids: list[str],
        index_type: IndexType = IndexType.FLAT,
    ) -> None:
        assert embeddings.ndim == 2, f"Expected 2-D embeddings, got {embeddings.ndim}-D"
        assert len(article_ids) == embeddings.shape[0], (
            f"Mismatch: {len(article_ids)} IDs vs {embeddings.shape[0]} embeddings"
        )
        self._embeddings = embeddings.astype(np.float32)
        self._article_ids = list(article_ids)
        self._id_to_idx = {aid: i for i, aid in enumerate(article_ids)}
        self._dim = embeddings.shape[1]
        self._index_type = index_type
        self._faiss_index = None

        if index_type != IndexType.FLAT:
            raise NotImplementedError(
                f"Index type {index_type} is reserved for later. "
                f"Only IndexType.FLAT is currently supported."
            )

        if _HAS_FAISS:
            self._build_faiss_index()
        else:
            logger.info("Using NumPy brute-force backend (dim=%d, n=%d)", self._dim, len(article_ids))

    def _build_faiss_index(self) -> None:
        """Build a FAISS IndexFlatIP from the embeddings."""
        self._faiss_index = faiss.IndexFlatIP(self._dim)
        self._faiss_index.add(self._embeddings)
        logger.info(
            "Built FAISS IndexFlatIP: dim=%d, n=%d",
            self._dim, self._faiss_index.ntotal,
        )

    # ── Core search methods ────────────────────────────────────────

    def search(
        self,
        query: np.ndarray,
        k: int = 100,
    ) -> list[tuple[str, float]]:
        """Search the full index for the *k* nearest neighbours.

        Parameters
        ----------
        query : np.ndarray
            L2-normalized query vector, shape ``(dim,)`` or ``(1, dim)``.
        k : int
            Number of results.

        Returns
        -------
        list[tuple[str, float]]
            ``[(article_id, cosine_similarity)]`` sorted descending.
        """
        query = query.reshape(1, -1).astype(np.float32)

        if self._faiss_index is not None:
            k_eff = min(k, self._faiss_index.ntotal)
            scores, indices = self._faiss_index.search(query, k_eff)
            results = []
            for idx, score in zip(indices[0], scores[0]):
                if idx >= 0:  # FAISS returns -1 for missing results
                    results.append((self._article_ids[idx], float(score)))
            return results
        else:
            return self._numpy_search(query, k)

    def search_restricted(
        self,
        query: np.ndarray,
        candidate_ids: list[str],
        k: int = 100,
    ) -> list[tuple[str, float]]:
        """Search only within a restricted set of candidate article IDs.

        This is the method used for per-impression retrieval (matching the
        BM25 protocol of scoring only within the impression's candidates).

        Parameters
        ----------
        query : np.ndarray
            L2-normalized query vector, shape ``(dim,)`` or ``(1, dim)``.
        candidate_ids : list[str]
            Article IDs to restrict the search to.
        k : int
            Number of results.

        Returns
        -------
        list[tuple[str, float]]
            ``[(article_id, cosine_similarity)]`` sorted descending.
            Includes all candidates (even those without embeddings, scored 0.0).
        """
        query = query.reshape(1, -1).astype(np.float32)

        # Find embeddings for candidates that exist in the index
        valid_indices = []
        valid_ids = []
        missing_ids = []
        for cid in candidate_ids:
            idx = self._id_to_idx.get(cid)
            if idx is not None:
                valid_indices.append(idx)
                valid_ids.append(cid)
            else:
                missing_ids.append(cid)

        if not valid_indices:
            # No candidates have embeddings — return all with score 0
            return [(cid, 0.0) for cid in candidate_ids[:k]]

        # Compute dot products for valid candidates
        candidate_embeds = self._embeddings[valid_indices]  # (n_valid, dim)
        scores = (candidate_embeds @ query.T).flatten()     # (n_valid,)

        # Build results
        results = list(zip(valid_ids, scores.tolist()))

        # Add missing candidates with score 0
        for cid in missing_ids:
            results.append((cid, 0.0))

        # Sort by score descending, then by ID for stability
        results.sort(key=lambda x: (-x[1], x[0]))

        return results[:k]

    def _numpy_search(
        self,
        query: np.ndarray,
        k: int,
    ) -> list[tuple[str, float]]:
        """NumPy brute-force fallback for full-index search."""
        scores = (self._embeddings @ query.T).flatten()  # (n_articles,)
        # Get top-k indices
        if k >= len(scores):
            top_k_idx = np.argsort(-scores)
        else:
            top_k_idx = np.argpartition(-scores, k)[:k]
            top_k_idx = top_k_idx[np.argsort(-scores[top_k_idx])]

        return [
            (self._article_ids[idx], float(scores[idx]))
            for idx in top_k_idx
        ]

    # ── Sanity checks ──────────────────────────────────────────────

    def self_similarity_check(self, sample_size: int = 5) -> bool:
        """Verify that an article's cosine similarity to itself is ~1.0.

        Checks the first *sample_size* articles in the index.

        Returns True if all pass, raises AssertionError otherwise.
        """
        n = min(sample_size, len(self._article_ids))
        for i in range(n):
            aid = self._article_ids[i]
            query = self._embeddings[i]
            results = self.search_restricted(query, [aid], k=1)
            score = results[0][1]
            assert abs(score - 1.0) < 1e-4, (
                f"Self-similarity check failed for {aid}: "
                f"expected ~1.0, got {score:.6f}"
            )
        logger.info("Self-similarity check passed (%d articles)", n)
        return True

    # ── Properties ─────────────────────────────────────────────────

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def n_articles(self) -> int:
        return len(self._article_ids)

    def get_embedding(self, article_id: str) -> np.ndarray | None:
        """Return the embedding for a single article, or None."""
        idx = self._id_to_idx.get(article_id)
        if idx is None:
            return None
        return self._embeddings[idx]

    def get_embeddings_batch(self, article_ids: list[str]) -> tuple[np.ndarray, list[str]]:
        """Return embeddings for a batch of article IDs.

        Returns
        -------
        embeddings : np.ndarray
            Shape ``(n_found, dim)`` — only articles with embeddings.
        found_ids : list[str]
            The article IDs that were found (parallel to embeddings rows).
        """
        indices = []
        found_ids = []
        for aid in article_ids:
            idx = self._id_to_idx.get(aid)
            if idx is not None:
                indices.append(idx)
                found_ids.append(aid)
        if not indices:
            return np.empty((0, self._dim), dtype=np.float32), []
        return self._embeddings[indices], found_ids

    def __repr__(self) -> str:
        backend = "FAISS" if self._faiss_index is not None else "NumPy"
        return f"ArticleIndex(n={self.n_articles}, dim={self._dim}, backend={backend})"
