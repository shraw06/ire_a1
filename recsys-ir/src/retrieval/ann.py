"""Exact cosine-similarity article index with optional FAISS full-index search."""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    import faiss
    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False
    faiss = None  # type: ignore[assignment]


class IndexType(Enum):
    FLAT = "flat"
    IVFFLAT = "ivfflat"
    HNSW = "hnsw"


class ArticleIndex:
    """Article embedding index.

    `build_full_index=False` is recommended for Codabench submission because the
    submission path only ranks a small candidate list per impression and therefore
    does not need a FAISS catalog-wide search structure.
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        article_ids: list[str],
        index_type: IndexType = IndexType.FLAT,
        build_full_index: bool = True,
    ) -> None:
        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2-D embeddings, got {embeddings.ndim}-D")
        if len(article_ids) != embeddings.shape[0]:
            raise ValueError(f"Mismatch: {len(article_ids)} IDs vs {embeddings.shape[0]} embeddings")
        if index_type != IndexType.FLAT:
            raise NotImplementedError("Only IndexType.FLAT is currently supported")
        self._embeddings = np.asarray(embeddings, dtype=np.float32)
        self._article_ids = list(article_ids)
        self._id_to_idx = {aid: i for i, aid in enumerate(article_ids)}
        self._dim = embeddings.shape[1]
        self._faiss_index = None
        if build_full_index and _HAS_FAISS:
            self._faiss_index = faiss.IndexFlatIP(self._dim)
            self._faiss_index.add(self._embeddings)
            logger.info("Built FAISS IndexFlatIP: dim=%d, n=%d", self._dim, self._faiss_index.ntotal)
        elif build_full_index:
            logger.info("FAISS unavailable; using NumPy for full-index search")

    def search(self, query: np.ndarray, k: int = 100) -> list[tuple[str, float]]:
        query = np.asarray(query, dtype=np.float32).reshape(1, -1)
        if self._faiss_index is not None:
            k_eff = min(k, self._faiss_index.ntotal)
            scores, indices = self._faiss_index.search(query, k_eff)
            return [
                (self._article_ids[int(idx)], float(score))
                for idx, score in zip(indices[0], scores[0]) if idx >= 0
            ]
        scores = (self._embeddings @ query.T).ravel()
        if k >= len(scores):
            order = np.argsort(-scores)
        else:
            order = np.argpartition(-scores, k - 1)[:k]
            order = order[np.argsort(-scores[order])]
        return [(self._article_ids[int(i)], float(scores[int(i)])) for i in order]

    def search_restricted(
        self,
        query: np.ndarray,
        candidate_ids: list[str],
        k: int | None = None,
    ) -> list[tuple[str, float]]:
        """Exact scoring on only the supplied candidates."""
        if not candidate_ids:
            return []
        query = np.asarray(query, dtype=np.float32).reshape(-1)
        valid_indices: list[int] = []
        valid_ids: list[str] = []
        missing: list[str] = []
        for cid in candidate_ids:
            idx = self._id_to_idx.get(str(cid))
            if idx is None:
                missing.append(str(cid))
            else:
                valid_indices.append(idx)
                valid_ids.append(str(cid))
        if valid_indices:
            scores = self._embeddings[valid_indices] @ query
            results = list(zip(valid_ids, scores.astype(float).tolist()))
        else:
            results = []
        results.extend((cid, 0.0) for cid in missing)
        results.sort(key=lambda x: (-x[1], x[0]))
        return results[: (len(candidate_ids) if k is None else min(k, len(candidate_ids)))]

    def get_embedding(self, article_id: str) -> np.ndarray | None:
        idx = self._id_to_idx.get(str(article_id))
        return None if idx is None else self._embeddings[idx]

    def get_embeddings_batch(self, article_ids: list[str]) -> tuple[np.ndarray, list[str]]:
        indices: list[int] = []
        found_ids: list[str] = []
        for aid in article_ids:
            idx = self._id_to_idx.get(str(aid))
            if idx is not None:
                indices.append(idx)
                found_ids.append(str(aid))
        if not indices:
            return np.empty((0, self._dim), dtype=np.float32), []
        return self._embeddings[indices], found_ids

    def self_similarity_check(self, sample_size: int = 5) -> bool:
        n = min(sample_size, len(self._article_ids))
        for i in range(n):
            score = self.search_restricted(self._embeddings[i], [self._article_ids[i]], k=1)[0][1]
            if abs(score - 1.0) >= 1e-4:
                raise AssertionError(f"Self-similarity failed for {self._article_ids[i]}: {score}")
        return True

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def n_articles(self) -> int:
        return len(self._article_ids)

    def __repr__(self) -> str:
        backend = "FAISS" if self._faiss_index is not None else "NumPy"
        return f"ArticleIndex(n={self.n_articles}, dim={self.dim}, backend={backend})"
