"""Compact, memory-mapped EB-NeRD history index for large-scale inference.

The provided test `history.parquet` contains one row per user with list-valued
article IDs and timestamps. Expanding the complete file into Python dictionaries
would be prohibitively memory-heavy. This module builds a compact three-array
index once and memory-maps it during inference:

    user_ids.npy       sorted user IDs
    offsets.npy        row boundaries into the flattened arrays
    article_ids.npy    flattened clicked article IDs
    timestamps.npy     flattened click timestamps (Unix seconds)

This keeps inference RAM roughly constant and makes history lookup O(log U + H).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


class MemoryMappedHistoryStore:
    """Read EB-NeRD user histories from compact memory-mapped arrays."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.user_ids = np.load(self.directory / "user_ids.npy", mmap_mode="r")
        self.offsets = np.load(self.directory / "offsets.npy", mmap_mode="r")
        self.article_ids = np.load(self.directory / "article_ids.npy", mmap_mode="r")
        self.timestamps = np.load(self.directory / "timestamps.npy", mmap_mode="r")

    @classmethod
    def build(cls, history_path: Path, output_dir: Path, force: bool = False) -> "MemoryMappedHistoryStore":
        output_dir.mkdir(parents=True, exist_ok=True)
        required = ["user_ids.npy", "offsets.npy", "article_ids.npy", "timestamps.npy", "metadata.json"]
        if not force and all((output_dir / name).exists() for name in required):
            logger.info("Loading existing EB-NeRD history index from %s", output_dir)
            return cls(output_dir)

        logger.info("Building compact history index from %s", history_path)
        parquet = pq.ParquetFile(history_path)
        total_users = parquet.metadata.num_rows

        total_clicks = 0
        for batch in parquet.iter_batches(batch_size=50_000, columns=["article_id_fixed"]):
            values = batch.column(0).to_pylist()
            total_clicks += sum(len(v or []) for v in values)
        logger.info("History index dimensions: %d users, %d clicks", total_users, total_clicks)

        user_ids = np.empty(total_users, dtype=np.int64)
        offsets = np.empty(total_users + 1, dtype=np.int64)
        article_ids = np.empty(total_clicks, dtype=np.int64)
        timestamps = np.empty(total_clicks, dtype=np.int64)

        user_pos = 0
        click_pos = 0
        offsets[0] = 0
        for batch in parquet.iter_batches(
            batch_size=20_000,
            columns=["user_id", "impression_time_fixed", "article_id_fixed"],
        ):
            user_col = batch.column(0).to_pylist()
            time_col = batch.column(1).to_pylist()
            article_col = batch.column(2).to_pylist()
            for uid, times, articles in zip(user_col, time_col, article_col):
                if user_pos >= total_users:
                    raise RuntimeError("History row count changed during index build")
                articles = articles or []
                times = times or []
                user_ids[user_pos] = int(uid)
                for j, aid in enumerate(articles):
                    article_ids[click_pos] = int(aid)
                    ts = times[j] if j < len(times) else None
                    if ts is None:
                        timestamps[click_pos] = -1
                    else:
                        # Arrow returns datetime-like Python objects here.
                        dt = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        timestamps[click_pos] = int(dt.timestamp())
                    click_pos += 1
                user_pos += 1
                offsets[user_pos] = click_pos

        if user_pos != total_users or click_pos != total_clicks:
            raise RuntimeError(
                f"History index count mismatch: users {user_pos}/{total_users}, "
                f"clicks {click_pos}/{total_clicks}"
            )

        # Binary-search lookup requires sorted user IDs. Avoid a full reorder when
        # the source history is already sorted (which is common for EB-NeRD files).
        if np.all(user_ids[:-1] <= user_ids[1:]):
            sorted_users = user_ids
            sorted_offsets = offsets
            sorted_article_ids = article_ids
            sorted_timestamps = timestamps
        else:
            order = np.argsort(user_ids, kind="stable")
            sorted_users = user_ids[order]
            sorted_offsets = np.empty_like(offsets)
            sorted_article_ids = np.empty_like(article_ids)
            sorted_timestamps = np.empty_like(timestamps)
            out_pos = 0
            sorted_offsets[0] = 0
            for i, original_idx in enumerate(order):
                start, end = offsets[original_idx], offsets[original_idx + 1]
                length = int(end - start)
                if length:
                    sorted_article_ids[out_pos:out_pos + length] = article_ids[start:end]
                    sorted_timestamps[out_pos:out_pos + length] = timestamps[start:end]
                out_pos += length
                sorted_offsets[i + 1] = out_pos

        np.save(output_dir / "user_ids.npy", sorted_users)
        np.save(output_dir / "offsets.npy", sorted_offsets)
        np.save(output_dir / "article_ids.npy", sorted_article_ids)
        np.save(output_dir / "timestamps.npy", sorted_timestamps)
        metadata = {
            "source": str(history_path),
            "users": int(total_users),
            "clicks": int(total_clicks),
        }
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
        logger.info("History index built at %s", output_dir)
        return cls(output_dir)

    def get_history(self, user_id: str | int, cutoff: datetime | None = None) -> list[dict[str, str | None]]:
        uid = int(user_id)
        pos = int(np.searchsorted(self.user_ids, uid))
        if pos >= len(self.user_ids) or int(self.user_ids[pos]) != uid:
            return []
        start = int(self.offsets[pos])
        end = int(self.offsets[pos + 1])
        if end <= start:
            return []

        cutoff_epoch = None
        if cutoff is not None:
            if cutoff.tzinfo is None:
                cutoff = cutoff.replace(tzinfo=timezone.utc)
            cutoff_epoch = int(cutoff.timestamp())

        out: list[dict[str, str | None]] = []
        for aid, ts in zip(self.article_ids[start:end], self.timestamps[start:end]):
            ts_int = int(ts)
            if cutoff_epoch is not None and ts_int >= 0 and ts_int >= cutoff_epoch:
                continue
            if ts_int < 0:
                clicked_at = None
            else:
                clicked_at = datetime.fromtimestamp(ts_int, tz=timezone.utc).replace(tzinfo=None).isoformat()
            out.append({"article_id": str(int(aid)), "clicked_at": clicked_at})
        return out

    def get_histories(self, user_ids: list[str], cutoffs: list[datetime | None]) -> list[list[dict[str, str | None]]]:
        if len(user_ids) != len(cutoffs):
            raise ValueError("user_ids and cutoffs must have equal length")
        return [self.get_history(uid, cutoff) for uid, cutoff in zip(user_ids, cutoffs)]
