"""Ranking metrics — AUC, MRR, nDCG@5, nDCG@10, recall@K.

Currently implements recall@K for BM25 candidate retrieval evaluation.
Other metrics (AUC, MRR, nDCG) will be added for the ranking stage.
"""

from __future__ import annotations
import numpy as np


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


def mrr(labels: list[int], scores: list[float]) -> float:
    """Compute Mean Reciprocal Rank (MRR).
    
    Parameters
    ----------
    labels : list[int]
        Binary labels (1 for clicked, 0 for not clicked).
    scores : list[float]
        Predicted scores.
        
    Returns
    -------
    float
        1 / rank of the highest scored relevant item, or 0 if none.
    """
    if not labels or sum(labels) == 0:
        return 0.0
    
    # Sort labels based on scores in descending order
    # Handle ties by preserving original order or we can use argsort
    # Python's sort is stable. We sort by score descending.
    # We pair (score, label)
    paired = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)
    
    for i, (_, label) in enumerate(paired):
        if label == 1:
            return 1.0 / (i + 1)
    return 0.0


def dcg_at_k(labels: list[int], scores: list[float], k: int) -> float:
    """Compute Discounted Cumulative Gain (DCG) at K."""
    if not labels or sum(labels) == 0:
        return 0.0
        
    paired = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)
    dcg = 0.0
    for i, (_, label) in enumerate(paired[:k]):
        if label == 1:
            # 0-indexed i means rank is i+1. Formula: 1 / log2(rank + 1) = 1 / log2(i + 2)
            dcg += 1.0 / np.log2(i + 2)
    return dcg


def ndcg_at_k(labels: list[int], scores: list[float], k: int) -> float:
    """Compute Normalized Discounted Cumulative Gain (nDCG) at K."""
    if not labels or sum(labels) == 0:
        return 0.0
        
    actual_dcg = dcg_at_k(labels, scores, k)
    
    # Ideal ranking: all relevant items at the top
    ideal_labels = sorted(labels, reverse=True)
    ideal_scores = [float(lbl) for lbl in ideal_labels]
    ideal_dcg = dcg_at_k(ideal_labels, ideal_scores, k)
    
    if ideal_dcg == 0.0:
        return 0.0
        
    return actual_dcg / ideal_dcg


def auc_score(labels: list[int], scores: list[float]) -> float:
    """Compute Area Under the ROC Curve (AUC)."""
    if not labels or sum(labels) == 0 or sum(labels) == len(labels):
        return 0.5
        
    from sklearn.metrics import roc_auc_score
    try:
        return roc_auc_score(labels, scores)
    except ValueError:
        return 0.5
