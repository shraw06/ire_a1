"""Shared candidate ranking primitives."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from src.retrieval.ann import ArticleIndex


def rank_candidates(
    query_vector: np.ndarray,
    candidate_ids: Sequence[str],
    index: ArticleIndex,
) -> list[str]:
    """Rank an impression's candidates by embedding similarity."""
    return [article_id for article_id, _ in index.search_restricted(query_vector, list(candidate_ids), k=len(candidate_ids))]


def rank_candidate_batch(
    user_vectors: np.ndarray,
    candidate_batches: Sequence[Sequence[str]],
    index: ArticleIndex,
) -> list[list[str]]:
    """Rank each candidate list independently while reusing one loaded index."""
    if len(user_vectors) != len(candidate_batches):
        raise ValueError("user_vectors and candidate_batches must have the same length")
    return [
        rank_candidates(vector, candidates, index)
        for vector, candidates in zip(user_vectors, candidate_batches)
    ]
