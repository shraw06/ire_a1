"""User feature store - click history, recency-decayed weights, history length.

Builds per-user features from the interim behaviors table and exposes them
via DuckDB-backed selective reads.  The critical method is
``get_user_history(user_id, as_of_ts, dataset)`` which performs
timestamp-aware filtering to prevent data leakage.

IMPORTANT — dataset-specific handling (confirmed by EDA_SUMMARY.md):
  - EB-NeRD: ``clicked_history[]`` carries real per-click timestamps from
    ``history.parquet``'s ``impression_time_fixed``.  ``as_of_ts`` filtering
    is the ACTUAL leakage-prevention mechanism — the filter strictly excludes
    any history entry with ``clicked_at >= as_of_ts``.
  - MIND: ``clicked_history[]`` has no per-item timestamps.  It was already a
    pre-trimmed, as-of-impression snapshot at parsing time (per MIND's own
    ``behaviors.tsv`` design).  ``as_of_ts`` filtering is a pass-through
    (nothing to filter, since there's no timestamp to filter on).

    DOCUMENTED ASSUMPTION (carried from parse_mind.py): MIND's leakage guard
    is enforced upstream by the dataset's own construction, not by this
    function.  This is a documented assumption, not a verified guarantee —
    MIND provides no per-article published_time to independently check it.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import polars as pl

from src.feature_store.store_backend import ParquetStore

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Recency decay constant: λ = 1 / (7 * 86400) → ~7-day half-life.
# weight_i = exp(-λ * (as_of_ts - clicked_at_i).total_seconds())
_DECAY_LAMBDA = 1.0 / (7 * 86400)


class UserFeatureStore:
    """Query layer over user features backed by Parquet + DuckDB.

    Typical usage::

        store = UserFeatureStore("mind")
        history = store.get_user_history("U123", as_of_ts, "mind")
        features = store.get_user_features("U123", as_of_ts, "mind")
    """

    def __init__(self, dataset: str, processed_dir: Path | None = None, scale: str = "small") -> None:
        if processed_dir is None:
            from src.common.paths import processed_dir as scale_processed_dir
            processed_dir = scale_processed_dir(dataset, scale)
        self._path = processed_dir / "user_features.parquet"
        if not self._path.exists():
            raise FileNotFoundError(
                f"User features not found at {self._path}. "
                f"Run `python -m src.feature_store.build_features --dataset {dataset}` first."
            )
        self._store = ParquetStore(self._path, table_alias="users")
        self._dataset = dataset

    # Core history retrieval

    def get_user_history(
        self,
        user_id: str,
        as_of_ts: datetime,
        dataset: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return a user's click history filtered to entries BEFORE ``as_of_ts``.

        Parameters
        ----------
        user_id : str
            User ID to look up.
        as_of_ts : datetime
            Timestamp cutoff for filtering.
        dataset : str, optional
            ``"mind"`` or ``"ebnerd"``.  Defaults to the store's dataset.

        Returns
        -------
        list[dict]
            List of ``{article_id: str, clicked_at: str | None}`` dicts,
            ordered chronologically (oldest first), filtered according to
            dataset-specific rules.

        Dataset-specific filtering
        --------------------------
        EB-NeRD:
            Strictly excludes any history entry with ``clicked_at >= as_of_ts``.
            This is the ACTUAL leakage-prevention mechanism — EB-NeRD's
            ``clicked_history[]`` carries real per-click timestamps from
            ``history.parquet``'s ``impression_time_fixed``, representing the
            user's FULL lifetime history.

        MIND:
            Returns the full snapshot unchanged (pass-through / no-op).
            MIND's ``clicked_history[]`` has no per-item timestamps — it was
            already a pre-trimmed, as-of-impression snapshot at parsing time
            (per MIND's own ``behaviors.tsv`` design).

            DOCUMENTED ASSUMPTION: MIND's leakage guard is enforced upstream
            by the dataset's own construction, not by this function.  This is
            a documented assumption, not a verified guarantee — MIND provides
            no per-article published_time to independently check it.  This
            caveat is flagged identically in ``parse_mind.py``.
        """
        ds = dataset or self._dataset

        # Look up the user's full history from the feature store.
        row = self._store.get_by_id("user_id", user_id, columns=["all_history"])
        if row is None:
            return []

        all_history = json.loads(row["all_history"])

        if ds == "mind":
            # MIND: pass-through — no timestamps to filter on.
            # Leakage guard is enforced upstream by MIND's own construction
            # (each behaviors.tsv row's history column is a pre-trimmed snapshot).
            # See module docstring and parse_mind.py for the documented caveat.
            return all_history

        # EB-NeRD: filter strictly — exclude any entry with clicked_at >= as_of_ts.
        # This is the real leakage-prevention mechanism.
        filtered = []
        for entry in all_history:
            clicked_at_str = entry.get("clicked_at")
            if clicked_at_str is None:
                # Defensive: if an EB-NeRD entry somehow has no timestamp,
                # include it (conservative — same as MIND path).
                filtered.append(entry)
                continue
            # Parse ISO datetime string
            clicked_at = datetime.fromisoformat(clicked_at_str)
            if clicked_at < as_of_ts:
                filtered.append(entry)
            # else: clicked_at >= as_of_ts → excluded (leakage prevention)

        return filtered

    # Enriched features

    def get_user_features(
        self,
        user_id: str,
        as_of_ts: datetime,
        dataset: str | None = None,
    ) -> dict[str, Any]:
        """Return filtered history + recency-decayed weights + history length.

        Returns
        -------
        dict with keys:
          - user_id: str
          - history: list[dict]  (filtered by as_of_ts)
          - history_len: int
          - recency_weights: list[float]  (parallel to history)
          - last_click_at: str | None
        """
        ds = dataset or self._dataset
        history = self.get_user_history(user_id, as_of_ts, ds)

        # Compute recency-decayed weights.
        # For EB-NeRD: weight_i = exp(-λ * (as_of_ts - clicked_at_i).total_seconds())
        # For MIND: no timestamps → all weights = 1.0
        weights = []
        last_click_at = None
        for entry in history:
            clicked_at_str = entry.get("clicked_at")
            if clicked_at_str is None:
                # MIND (no timestamps) → uniform weight.
                weights.append(1.0)
            else:
                clicked_at = datetime.fromisoformat(clicked_at_str)
                delta_seconds = (as_of_ts - clicked_at).total_seconds()
                weight = math.exp(-_DECAY_LAMBDA * max(delta_seconds, 0.0))
                weights.append(weight)
                if last_click_at is None or clicked_at_str > last_click_at:
                    last_click_at = clicked_at_str

        return {
            "user_id": user_id,
            "history": history,
            "history_len": len(history),
            "recency_weights": weights,
            "last_click_at": last_click_at,
        }

    @property
    def row_count(self) -> int:
        return self._store.row_count()

    # ── Build step ─────────────────────────────────────────────────────

    @staticmethod
    def build_features(
        dataset: str,
        interim_dir: Path | None = None,
        processed_dir: Path | None = None,
    ) -> Path:
        """Read interim behaviors and write user_features.parquet.

        Derives per-user features:
          - all_history: full list of {article_id, clicked_at} dicts, de-duped
            by article_id (keeping earliest click for EB-NeRD, or first
            occurrence for MIND).
          - history_len: count of unique articles in history.
          - last_active_at: max impression timestamp across all impressions.

        Parameters
        ----------
        dataset : str
            ``"mind"`` or ``"ebnerd"``.
        interim_dir : Path, optional
            Override for ``data/interim/{dataset}/``.
        processed_dir : Path, optional
            Override for ``data/processed/{dataset}/``.

        Returns
        -------
        Path
            Path to the written Parquet file.
        """
        if interim_dir is None:
            interim_dir = _PROJECT_ROOT / "data" / "interim" / dataset
        if processed_dir is None:
            processed_dir = _PROJECT_ROOT / "data" / "processed" / dataset

        beh_path = interim_dir / "behaviors.parquet"
        if not beh_path.exists():
            raise FileNotFoundError(
                f"Interim behaviors not found at {beh_path}. "
                f"Run parse_{dataset} first."
            )

        logger.info("Building user features for %s from %s", dataset, beh_path)
        df = pl.read_parquet(beh_path)
        logger.info("  Read %d behavior rows", len(df))

        # Aggregate per-user: merge clicked_history across all impressions,
        # de-dup by article_id (keep earliest clicked_at for EB-NeRD).
        user_data: dict[str, dict] = {}
        for row in df.to_dicts():
            uid = row["user_id"]
            ts = row["timestamp"]
            history = json.loads(row["clicked_history"])

            if uid not in user_data:
                user_data[uid] = {
                    "user_id": uid,
                    "dataset": dataset,
                    # article_id → {article_id, clicked_at} (keep earliest)
                    "history_map": {},
                    "last_active_at": ts,
                }

            for entry in history:
                aid = entry["article_id"]
                clicked_at = entry.get("clicked_at")
                existing = user_data[uid]["history_map"].get(aid)
                if existing is None:
                    user_data[uid]["history_map"][aid] = entry
                elif clicked_at is not None and existing.get("clicked_at") is not None:
                    # Keep the earlier timestamp
                    if clicked_at < existing["clicked_at"]:
                        user_data[uid]["history_map"][aid] = entry

            if ts and (user_data[uid]["last_active_at"] is None
                       or ts > user_data[uid]["last_active_at"]):
                user_data[uid]["last_active_at"] = ts

        # Build the output rows
        unified_rows = []
        for ud in user_data.values():
            # Sort history chronologically (by clicked_at if available)
            history_list = list(ud["history_map"].values())
            history_list.sort(
                key=lambda e: e.get("clicked_at") or ""
            )
            unified_rows.append({
                "user_id": ud["user_id"],
                "dataset": ud["dataset"],
                "all_history": json.dumps(history_list),
                "history_len": len(history_list),
                "last_active_at": ud["last_active_at"],
            })

        result = pl.DataFrame(unified_rows, schema={
            "user_id": pl.Utf8,
            "dataset": pl.Utf8,
            "all_history": pl.Utf8,
            "history_len": pl.Int64,
            "last_active_at": pl.Datetime("us"),
        })

        # Write to processed dir
        processed_dir.mkdir(parents=True, exist_ok=True)
        out_path = processed_dir / "user_features.parquet"
        result.write_parquet(out_path)

        logger.info(
            "  Wrote user features: %d users → %s (%.1f MB)",
            len(result),
            out_path,
            out_path.stat().st_size / 1024**2,
        )
        return out_path
