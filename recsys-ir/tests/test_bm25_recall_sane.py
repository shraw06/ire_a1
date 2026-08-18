"""Sanity-check BM25 recall — ensure non-degenerate retrieval on known inputs.

Tests:
  1. recall@K is monotonically non-decreasing as K grows.
  2. recall@K ∈ [0, 1].
  3. A query matching a known document retrieves it at K=1.
  4. recall@K computed from the ranking_metrics module is consistent.
"""

from __future__ import annotations

import pytest

from src.evaluation.ranking_metrics import recall_at_k
from src.retrieval.bm25 import BM25Engine


# ── Test corpus ───────────────────────────────────────────────────

CORPUS = [
    ("D1", "machine learning algorithms for classification tasks"),
    ("D2", "deep neural networks and representation learning"),
    ("D3", "natural language processing with transformers"),
    ("D4", "computer vision object detection methods"),
    ("D5", "reinforcement learning in game environments"),
    ("D6", "statistical methods for data analysis"),
    ("D7", "optimization algorithms gradient descent convergence"),
    ("D8", "machine learning model evaluation and metrics"),
]


# ── Tests ─────────────────────────────────────────────────────────

class TestBM25RecallSanity:
    """Sanity checks on BM25 recall behavior."""

    @pytest.fixture()
    def engine(self):
        return BM25Engine.from_corpus(
            CORPUS, dataset="mind", use_stopwords=False, use_stemming=False,
        )

    def test_recall_monotonically_nondecreasing(self, engine):
        """recall@K must be non-decreasing as K increases."""
        query = "machine learning algorithms"
        all_ids = [d[0] for d in CORPUS]
        results = engine.rank(query, candidate_ids=all_ids, top_k=len(all_ids))
        ranked_ids = [r[0] for r in results]

        ground_truth = {"D1", "D8"}  # Both mention "machine learning"
        ks = [1, 2, 3, 4, 5, 6, 7, 8]
        recalls = [recall_at_k(ranked_ids, ground_truth, k) for k in ks]

        for i in range(1, len(recalls)):
            assert recalls[i] >= recalls[i - 1], (
                f"recall@{ks[i]}={recalls[i]:.4f} < recall@{ks[i-1]}={recalls[i-1]:.4f} — "
                f"NOT monotonically non-decreasing"
            )

    def test_recall_values_in_unit_range(self, engine):
        """All recall values must be in [0, 1]."""
        query = "deep learning neural networks"
        all_ids = [d[0] for d in CORPUS]
        results = engine.rank(query, candidate_ids=all_ids, top_k=len(all_ids))
        ranked_ids = [r[0] for r in results]

        ground_truth = {"D2"}
        for k in [1, 2, 4, 8]:
            r = recall_at_k(ranked_ids, ground_truth, k)
            assert 0.0 <= r <= 1.0, f"recall@{k}={r} out of [0,1] range"

    def test_known_document_retrieved_at_top(self, engine):
        """A highly specific query should retrieve its exact match at K=1."""
        # D3 is the only doc about "natural language processing transformers"
        query = "natural language processing transformers"
        all_ids = [d[0] for d in CORPUS]
        results = engine.rank(query, candidate_ids=all_ids, top_k=1)
        assert results[0][0] == "D3", (
            f"Expected D3 at rank 1, got {results[0][0]}"
        )

    def test_recall_at_k_full_retrieval(self, engine):
        """At K = corpus size, recall should be 1.0 if ground truth ⊆ candidates."""
        query = "machine learning"
        all_ids = [d[0] for d in CORPUS]
        results = engine.rank(query, candidate_ids=all_ids, top_k=len(all_ids))
        ranked_ids = [r[0] for r in results]

        ground_truth = {"D1", "D2", "D8"}
        r = recall_at_k(ranked_ids, ground_truth, len(all_ids))
        assert r == 1.0, f"recall@{len(all_ids)} should be 1.0 but got {r}"

    def test_empty_ground_truth_returns_zero(self):
        """recall@K with empty ground truth should be 0.0."""
        assert recall_at_k(["D1", "D2"], set(), 2) == 0.0

    def test_recall_metric_direct(self):
        """Direct unit test of recall_at_k function."""
        ranked = ["D1", "D3", "D2", "D5", "D4"]
        gt = {"D1", "D2"}

        assert recall_at_k(ranked, gt, 1) == 0.5   # D1 in top-1
        assert recall_at_k(ranked, gt, 2) == 0.5   # D1 in top-2, D2 not yet
        assert recall_at_k(ranked, gt, 3) == 1.0   # D1 + D2 in top-3
        assert recall_at_k(ranked, gt, 5) == 1.0   # Still 1.0
