"""Ranking metrics — AUC, MRR, nDCG@5, nDCG@10, recall@K.

Currently implements recall@K for BM25 candidate retrieval evaluation.
Other metrics (AUC, MRR, nDCG) will be added for the ranking stage.
"""

from __future__ import annotations


def recall_at_k(
    ranked_ids: list[str],
    ground_truth: set[str],
    k: int,
) -> float:
    """Compute recall@K: fraction of ground-truth items in the top-K.

    Parameters
    ----------
    ranked_ids : list[str]
        Candidate IDs ordered by decreasing score.
    ground_truth : set[str]
        The set of relevant (clicked) article IDs.
    k : int
        Cutoff — consider only the first *k* entries of *ranked_ids*.

    Returns
    -------
    float
        ``|top_K ∩ ground_truth| / |ground_truth|``.
        Returns 0.0 if ground_truth is empty.
    """
    if not ground_truth:
        return 0.0
    top_k = set(ranked_ids[:k])
    return len(top_k & ground_truth) / len(ground_truth)
