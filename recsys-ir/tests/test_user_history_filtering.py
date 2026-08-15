"""Unit tests for user history timestamp filtering (leakage prevention).

Tests the critical ``get_user_history`` method of ``UserFeatureStore``:

  - EB-NeRD (primary test): synthetic user history spanning a timestamp
    boundary.  Asserts that ``get_user_history`` excludes entries at or
    after ``as_of_ts``.  This is where the invariant is actually load-bearing.

  - MIND (lighter test): synthetic user history with no timestamps.
    Confirms the pass-through returns the full snapshot unchanged.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest


# Helpers

def _write_user_features_parquet(
    user_id: str,
    dataset: str,
    history: list[dict],
    tmpdir: Path,
) -> Path:
    """Create a minimal user_features.parquet for testing."""
    out_dir = tmpdir / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "user_features.parquet"

    df = pl.DataFrame([{
        "user_id": user_id,
        "dataset": dataset,
        "all_history": json.dumps(history),
        "history_len": len(history),
        "last_active_at": datetime(2023, 6, 1),
    }], schema={
        "user_id": pl.Utf8,
        "dataset": pl.Utf8,
        "all_history": pl.Utf8,
        "history_len": pl.Int64,
        "last_active_at": pl.Datetime("us"),
    })
    df.write_parquet(out_path)
    return out_dir


# EB-NeRD tests (primary — invariant is load-bearing) 

class TestEbnerdTimestampFiltering:
    """Verify that get_user_history excludes entries at/after as_of_ts for EB-NeRD."""

    def test_excludes_entries_at_or_after_cutoff(self, tmp_path: Path) -> None:
        """Synthetic history with 5 entries spanning a cutoff boundary.

        History entries:
          - 3 entries BEFORE cutoff → should be included
          - 1 entry EXACTLY AT cutoff → should be EXCLUDED
          - 1 entry AFTER cutoff → should be EXCLUDED

        Expected: exactly 3 entries returned, all with clicked_at < cutoff.
        """
        from src.feature_store.user_store import UserFeatureStore

        cutoff = datetime(2023, 5, 25, 7, 0, 0)
        history = [
            {"article_id": "A1", "clicked_at": "2023-05-20T10:00:00"},  # before
            {"article_id": "A2", "clicked_at": "2023-05-22T14:30:00"},  # before
            {"article_id": "A3", "clicked_at": "2023-05-24T23:59:59"},  # before
            {"article_id": "A4", "clicked_at": "2023-05-25T07:00:00"},  # AT cutoff → excluded
            {"article_id": "A5", "clicked_at": "2023-05-26T12:00:00"},  # after → excluded
        ]

        processed_dir = _write_user_features_parquet("U1", "ebnerd", history, tmp_path)
        store = UserFeatureStore("ebnerd", processed_dir=processed_dir)

        result = store.get_user_history("U1", as_of_ts=cutoff, dataset="ebnerd")

        assert len(result) == 3, f"Expected 3 entries before cutoff, got {len(result)}"

        # All returned entries should have clicked_at < cutoff
        for entry in result:
            clicked_at = datetime.fromisoformat(entry["clicked_at"])
            assert clicked_at < cutoff, (
                f"Entry {entry['article_id']} has clicked_at={clicked_at} which is "
                f">= cutoff={cutoff}"
            )

        # Verify the right articles were returned
        returned_ids = {e["article_id"] for e in result}
        assert returned_ids == {"A1", "A2", "A3"}

    def test_all_before_cutoff_returns_all(self, tmp_path: Path) -> None:
        """When all entries are before the cutoff, all should be returned."""
        from src.feature_store.user_store import UserFeatureStore

        cutoff = datetime(2023, 6, 1, 0, 0, 0)
        history = [
            {"article_id": "A1", "clicked_at": "2023-05-20T10:00:00"},
            {"article_id": "A2", "clicked_at": "2023-05-25T14:00:00"},
        ]

        processed_dir = _write_user_features_parquet("U1", "ebnerd", history, tmp_path)
        store = UserFeatureStore("ebnerd", processed_dir=processed_dir)
        result = store.get_user_history("U1", as_of_ts=cutoff, dataset="ebnerd")

        assert len(result) == 2

    def test_all_after_cutoff_returns_empty(self, tmp_path: Path) -> None:
        """When all entries are at/after the cutoff, none should be returned."""
        from src.feature_store.user_store import UserFeatureStore

        cutoff = datetime(2023, 5, 19, 0, 0, 0)
        history = [
            {"article_id": "A1", "clicked_at": "2023-05-20T10:00:00"},
            {"article_id": "A2", "clicked_at": "2023-05-25T14:00:00"},
        ]

        processed_dir = _write_user_features_parquet("U1", "ebnerd", history, tmp_path)
        store = UserFeatureStore("ebnerd", processed_dir=processed_dir)
        result = store.get_user_history("U1", as_of_ts=cutoff, dataset="ebnerd")

        assert len(result) == 0

    def test_unknown_user_returns_empty(self, tmp_path: Path) -> None:
        """Looking up a non-existent user should return empty list."""
        from src.feature_store.user_store import UserFeatureStore

        history = [{"article_id": "A1", "clicked_at": "2023-05-20T10:00:00"}]
        processed_dir = _write_user_features_parquet("U1", "ebnerd", history, tmp_path)
        store = UserFeatureStore("ebnerd", processed_dir=processed_dir)

        result = store.get_user_history("NONEXISTENT", as_of_ts=datetime(2023, 6, 1))
        assert result == []


# MIND tests (lighter - pass-through confirmation) 

class TestMindPassThrough:
    """Verify that get_user_history for MIND returns the full snapshot unchanged.

    MIND's clicked_history has no per-item timestamps — it was already a
    pre-trimmed, as-of-impression snapshot at parsing time.  The as_of_ts
    parameter is accepted but has no effect.
    """

    def test_full_snapshot_returned_unchanged(self, tmp_path: Path) -> None:
        """MIND history has no timestamps → full snapshot returned regardless of as_of_ts."""
        from src.feature_store.user_store import UserFeatureStore

        history = [
            {"article_id": "N001", "clicked_at": None},
            {"article_id": "N002", "clicked_at": None},
            {"article_id": "N003", "clicked_at": None},
            {"article_id": "N004", "clicked_at": None},
        ]

        processed_dir = _write_user_features_parquet("U42", "mind", history, tmp_path)
        store = UserFeatureStore("mind", processed_dir=processed_dir)

        # Any cutoff should return full history unchanged
        result = store.get_user_history("U42", as_of_ts=datetime(2019, 11, 14), dataset="mind")

        assert len(result) == 4, f"Expected 4 entries (full snapshot), got {len(result)}"
        returned_ids = [e["article_id"] for e in result]
        assert returned_ids == ["N001", "N002", "N003", "N004"]

    def test_different_cutoff_same_result(self, tmp_path: Path) -> None:
        """Varying as_of_ts should not change the result for MIND."""
        from src.feature_store.user_store import UserFeatureStore

        history = [
            {"article_id": "N001", "clicked_at": None},
            {"article_id": "N002", "clicked_at": None},
        ]

        processed_dir = _write_user_features_parquet("U42", "mind", history, tmp_path)
        store = UserFeatureStore("mind", processed_dir=processed_dir)

        result_early = store.get_user_history("U42", as_of_ts=datetime(2019, 11, 10), dataset="mind")
        result_late = store.get_user_history("U42", as_of_ts=datetime(2019, 11, 15), dataset="mind")

        assert result_early == result_late
        assert len(result_early) == 2


# Recency weight tests

class TestRecencyWeights:
    """Verify recency-decayed weights are computed correctly."""

    def test_ebnerd_recent_clicks_higher_weight(self, tmp_path: Path) -> None:
        """More recent clicks should have higher recency weight."""
        from src.feature_store.user_store import UserFeatureStore

        cutoff = datetime(2023, 5, 25, 12, 0, 0)
        history = [
            {"article_id": "A1", "clicked_at": "2023-05-10T10:00:00"},  # 15 days ago
            {"article_id": "A2", "clicked_at": "2023-05-24T10:00:00"},  # 1 day ago
        ]

        processed_dir = _write_user_features_parquet("U1", "ebnerd", history, tmp_path)
        store = UserFeatureStore("ebnerd", processed_dir=processed_dir)

        features = store.get_user_features("U1", as_of_ts=cutoff, dataset="ebnerd")
        weights = features["recency_weights"]

        assert len(weights) == 2
        # More recent click (A2) should have higher weight
        assert weights[1] > weights[0], (
            f"Recent click weight ({weights[1]:.4f}) should be > "
            f"old click weight ({weights[0]:.4f})"
        )
        # All weights should be in (0, 1]
        for w in weights:
            assert 0 < w <= 1.0, f"Weight {w} out of range (0, 1]"

    def test_mind_uniform_weights(self, tmp_path: Path) -> None:
        """MIND (no timestamps) → all weights should be 1.0."""
        from src.feature_store.user_store import UserFeatureStore

        history = [
            {"article_id": "N001", "clicked_at": None},
            {"article_id": "N002", "clicked_at": None},
        ]

        processed_dir = _write_user_features_parquet("U42", "mind", history, tmp_path)
        store = UserFeatureStore("mind", processed_dir=processed_dir)

        features = store.get_user_features("U42", as_of_ts=datetime(2019, 11, 14), dataset="mind")
        weights = features["recency_weights"]

        assert weights == [1.0, 1.0]
