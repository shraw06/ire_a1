"""Hybrid re-ranking: recency-weighted user vectors + embedding/popularity blend.

Additive module — does not modify src/retrieval/{bm25,embeddings,ann,
candidate_gen,user_representation}.py or any checkpointed retrieval/eval
code. Reuses ArticleIndex.search_restricted() for per-candidate cosine
similarity and reuses src.evaluation.run_eval._load_popularity for a
leakage-safe (train-split-only) popularity signal, consistent with the
Q9 anti-gaming audit (see design_note/design.md).
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from src.retrieval.ann import ArticleIndex


def load_train_popularity(dataset: str, scale: str = "large") -> dict[str, int]:
    """Train-split-only candidate popularity (serving-time-safe)."""
    from src.evaluation.run_eval import _load_popularity
    return _load_popularity(dataset, "train", scale)


def recency_weighted_user_vector(
    history: Sequence[dict[str, Any]],
    index: ArticleIndex,
    history_cap: int = 20,
    decay: float = 0.85,
) -> np.ndarray:
    """Exponentially recency-weighted mean of history embeddings, L2-normalized.

    `history` is assumed chronological (oldest first, most recent last),
    matching the convention used elsewhere in this pipeline. decay=1.0
    reduces to the existing uniform mean (build_mean_user_vector).
    """
    recent = list(history[-history_cap:])
    if not recent:
        return np.zeros(index.dim, dtype=np.float32)
    ids = [str(entry["article_id"]) for entry in recent]
    embeddings, found_ids = index.get_embeddings_batch(ids)
    if embeddings.shape[0] == 0:
        return np.zeros(index.dim, dtype=np.float32)
    pos = {aid: i for i, aid in enumerate(ids)}
    weights = np.array(
        [decay ** (len(ids) - 1 - pos[aid]) for aid in found_ids],
        dtype=np.float32,
    )
    weights = weights / weights.sum()
    vector = (embeddings * weights[:, None]).sum(axis=0, dtype=np.float32)
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    return vector.astype(np.float32, copy=False)


def recency_weighted_user_vectors(
    histories: Sequence[Sequence[dict[str, Any]]],
    index: ArticleIndex,
    history_cap: int = 20,
    decay: float = 0.85,
) -> np.ndarray:
    """Batch version. Loops per-user (ragged history lengths); still cheap
    relative to the embedding lookups it wraps."""
    vectors = np.zeros((len(histories), index.dim), dtype=np.float32)
    for i, history in enumerate(histories):
        vectors[i] = recency_weighted_user_vector(history, index, history_cap, decay)
    return vectors


def _pop_score(article_id: str, popularity: dict[str, int]) -> float:
    return math.log1p(popularity.get(article_id, 0))


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def hybrid_rank_candidates(
    query_vector: np.ndarray,
    candidate_ids: Sequence[str],
    index: ArticleIndex,
    popularity: dict[str, int],
    alpha: float = 0.8,
) -> list[str]:
    """Rank candidates by alpha*cos_sim + (1-alpha)*popularity_prior.

    Both signals are min-max normalized within THIS impression's candidate
    set before blending, so MIND's and EB-NeRD's very different popularity
    scales don't need a separately-tuned global constant.
    """
    candidate_ids = list(candidate_ids)
    sim_results = dict(index.search_restricted(query_vector, candidate_ids, k=len(candidate_ids)))
    sims = np.array([sim_results.get(cid, 0.0) for cid in candidate_ids], dtype=np.float64)
    pops = np.array([_pop_score(cid, popularity) for cid in candidate_ids], dtype=np.float64)

    blended = alpha * _minmax(sims) + (1.0 - alpha) * _minmax(pops)
    order = np.argsort(-blended, kind="stable")
    return [candidate_ids[i] for i in order]


def hybrid_rank_candidate_batch(
    user_vectors: np.ndarray,
    candidate_batches: Sequence[Sequence[str]],
    index: ArticleIndex,
    popularity: dict[str, int],
    alpha: float = 0.8,
) -> list[list[str]]:
    if len(user_vectors) != len(candidate_batches):
        raise ValueError("user_vectors and candidate_batches must have the same length")
    return [
        hybrid_rank_candidates(vector, candidates, index, popularity, alpha)
        for vector, candidates in zip(user_vectors, candidate_batches)
    ]