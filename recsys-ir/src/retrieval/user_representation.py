"""User representation builders shared by offline retrieval and submission."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

import numpy as np

from src.retrieval.ann import ArticleIndex


def build_mean_user_vector(
    history: Sequence[dict[str, Any]],
    index: ArticleIndex,
    history_cap: int = 20,
) -> np.ndarray:
    """Mean-pool the most recent history embeddings and L2-normalize."""
    recent = history[-history_cap:]
    ids = [str(entry["article_id"]) for entry in recent]
    if not ids:
        return np.zeros(index.dim, dtype=np.float32)
    embeddings, _ = index.get_embeddings_batch(ids)
    if embeddings.shape[0] == 0:
        return np.zeros(index.dim, dtype=np.float32)
    vector = embeddings.mean(axis=0, dtype=np.float32)
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    return vector.astype(np.float32, copy=False)


def build_mean_user_vectors(
    histories: Sequence[Sequence[dict[str, Any]]],
    index: ArticleIndex,
    history_cap: int = 20,
) -> np.ndarray:
    """Batch version that resolves all unique history article IDs once."""
    unique_ids: list[str] = []
    seen: set[str] = set()
    truncated: list[list[str]] = []
    for history in histories:
        ids = [str(entry["article_id"]) for entry in history[-history_cap:]]
        truncated.append(ids)
        for aid in ids:
            if aid not in seen:
                seen.add(aid)
                unique_ids.append(aid)

    vectors = np.zeros((len(histories), index.dim), dtype=np.float32)
    if not unique_ids:
        return vectors

    embeddings, found_ids = index.get_embeddings_batch(unique_ids)
    row_by_id = {aid: i for i, aid in enumerate(found_ids)}
    for i, ids in enumerate(truncated):
        rows = [row_by_id[aid] for aid in ids if aid in row_by_id]
        if not rows:
            continue
        vector = embeddings[rows].mean(axis=0, dtype=np.float32)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vectors[i] = vector / norm
    return vectors
