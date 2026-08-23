"""Scale-aware user feature stores for the large datasets.

The logical user-feature interface is shared by offline retrieval and
Codabench inference, but the physical representation differs by dataset:

* MIND-large: the behavior row already contains the as-of-impression history
  snapshot, so we expose that snapshot directly without duplicating it into a
  second 2.6M-row table.
* EB-NeRD-large: the compact MemoryMappedHistoryStore provides timestamped
  lifetime history; this class adds the strict as-of filtering and recency
  weights required at serving time.

This module deliberately reuses the large artifacts already produced by the
pipeline; it does not require reparsing the datasets.
"""

from __future__ import annotations

from datetime import datetime
import math
from pathlib import Path
from typing import Any

from src.feature_store.history_store import MemoryMappedHistoryStore


DECAY_LAMBDA = 1.0 / (7 * 86400)


class MindLargeUserFeatureStore:
    """Logical user-feature store backed by impression-level MIND snapshots."""

    @staticmethod
    def from_behavior_row(row: dict[str, Any]) -> dict[str, Any]:
        import json

        raw_history = row.get("clicked_history")
        if raw_history is None:
            history: list[dict[str, Any]] = []
        elif isinstance(raw_history, str):
            history = json.loads(raw_history) if raw_history else []
        else:
            history = list(raw_history)

        return {
            "user_id": str(row["user_id"]),
            "impression_id": str(row["impression_id"]),
            "history": history,
            "history_len": len(history),
            "recency_weights": [1.0] * len(history),
            "last_click_at": None,
        }


class EbnerdLargeUserFeatureStore:
    """User features backed by a memory-mapped timestamped history index."""

    def __init__(self, index_dir: Path) -> None:
        self.history_store = MemoryMappedHistoryStore(index_dir)

    def get_features(
        self,
        user_id: str | int,
        as_of_ts: datetime,
    ) -> dict[str, Any]:
        history = self.history_store.get_history(user_id, as_of_ts)

        weights: list[float] = []
        last_click_at: str | None = None

        for entry in history:
            clicked_at = entry.get("clicked_at")
            if clicked_at is None:
                weights.append(1.0)
                continue

            clicked_dt = datetime.fromisoformat(clicked_at)
            delta = max((as_of_ts - clicked_dt).total_seconds(), 0.0)
            weights.append(float(math.exp(-DECAY_LAMBDA * delta)))
            if last_click_at is None or clicked_at > last_click_at:
                last_click_at = clicked_at

        return {
            "user_id": str(user_id),
            "history": history,
            "history_len": len(history),
            "recency_weights": weights,
            "last_click_at": last_click_at,
        }


class LargeUserFeatureStore:
    """Factory exposing the dataset-specific large user-feature store."""

    def __init__(self, dataset: str, processed_dir: Path) -> None:
        if dataset == "ebnerd":
            self.dataset = dataset
            index_dir = processed_dir / "history_index_validation"
            if not index_dir.exists():
                raise FileNotFoundError(
                    f"EB-NeRD validation history index not found: {index_dir}"
                )
            self._store = EbnerdLargeUserFeatureStore(index_dir)
        elif dataset == "mind":
            self.dataset = dataset
            self._store = MindLargeUserFeatureStore()
        else:
            raise ValueError(f"Unsupported large user-feature dataset: {dataset}")

    def from_behavior_row(self, row: dict[str, Any]) -> dict[str, Any]:
        if self.dataset != "mind":
            raise TypeError("from_behavior_row is only valid for MIND")
        return self._store.from_behavior_row(row)

    def get_features(
        self,
        user_id: str | int,
        as_of_ts: datetime,
    ) -> dict[str, Any]:
        if self.dataset != "ebnerd":
            raise TypeError("get_features is only valid for EB-NeRD")
        return self._store.get_features(user_id, as_of_ts)
