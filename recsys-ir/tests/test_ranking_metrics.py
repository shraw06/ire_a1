"""Test suite for ranking metrics including perfect/random-ranking sanity checks."""

import numpy as np
import pytest

from src.evaluation.ranking_metrics import auc_score, mrr, ndcg_at_k
from src.evaluation.bootstrap import compute_bootstrap_ci

def test_perfect_ranking():
    # Synthetic perfect ranking
    labels = [1, 1, 0, 0, 0]
    scores = [0.9, 0.8, 0.3, 0.2, 0.1]
    
    assert np.isclose(auc_score(labels, scores), 1.0)
    assert np.isclose(mrr(labels, scores), 1.0)
    assert np.isclose(ndcg_at_k(labels, scores, 5), 1.0)
    assert np.isclose(ndcg_at_k(labels, scores, 10), 1.0)

def test_random_ranking():
    # Synthetic random ranking (multiple impressions to get a good bootstrap CI)
    rng = np.random.default_rng(42)
    n_impressions = 200
    
    auc_list = []
    for _ in range(n_impressions):
        # 10 candidates, 2 positives
        labels = [1, 1] + [0] * 8
        rng.shuffle(labels)
        scores = rng.random(10).tolist()
        
        # Sort descending by score just to simulate ranked output (metrics don't technically require it,
        # but our metrics sort internally anyway).
        # Actually auc_score doesn't sort, it takes them as is.
        auc_list.append(auc_score(labels, scores))
        
    auc_arr = np.array(auc_list)
    mean_auc, ci_low, ci_high = compute_bootstrap_ci(auc_arr, b=1000, random_state=42)
    
    # AUC should be roughly 0.5 and 0.5 should fall within the CI
    assert ci_low <= 0.5 <= ci_high
    assert np.isclose(mean_auc, 0.5, atol=0.05)
