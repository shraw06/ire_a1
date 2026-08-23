"""Streaming readers for the large Codabench test sets.

The development pipeline parses the small/demo datasets into unified Parquet files.
The Codabench test sets are much larger, so submission inference deliberately reads
only the columns needed for prediction in bounded batches and never materializes the
complete test set.
"""

from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Impression:
    impression_id: str
    user_id: str
    timestamp: datetime | None
    history: list[dict[str, str | None]]
    candidates: list[str]

    @property
    def history_ids(self) -> tuple[str, ...]:
        """Backward-compatible view of the article IDs in user history."""
        return tuple(
            str(item["article_id"])
            for item in self.history
            if item.get("article_id") is not None
        )

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        """Backward-compatible immutable view of candidate article IDs."""
        return tuple(self.candidates)


_MIND_TIME_FMT = "%m/%d/%Y %I:%M:%S %p"


def _parse_mind_time(value: str) -> datetime:
    return datetime.strptime(value, _MIND_TIME_FMT)


def _parse_mind_candidates(raw: str | None) -> list[str]:
    """Parse either `N1-0 N2-1` or unlabeled `N1 N2` test candidates."""
    if not raw:
        return []
    out: list[str] = []
    for token in raw.split():
        match = re.match(r"^(.*)-[01]$", token)
        out.append(match.group(1) if match else token)
    return out


def _parse_mind_history(raw: str | None) -> list[dict[str, str | None]]:
    if not raw:
        return []
    return [{"article_id": article_id, "clicked_at": None} for article_id in raw.split()]


def find_mind_test_behaviors(raw_dir: Path) -> Path:
    candidates = [
        raw_dir / "MINDlarge_test" / "behaviors.tsv",
        raw_dir / "MINDlarge_test" / "behaviors_test.tsv",
    ]
    for path in candidates:
        if path.exists():
            return path
    matches = sorted(raw_dir.glob("MINDlarge_test*/behaviors*.tsv"))
    if matches:
        return matches[0]
    matches = sorted(raw_dir.rglob("behaviors.tsv"))
    for path in matches:
        if "test" in path.parent.name.lower() or "large" in path.parent.name.lower():
            return path
    raise FileNotFoundError(f"MIND large test behaviors.tsv not found below {raw_dir}")


def iter_mind_test(path: Path, batch_size: int = 50_000) -> Iterator[list[Impression]]:
    """Yield bounded lists of MIND test impressions."""
    read_options = pacsv.ReadOptions(
        use_threads=True,
        block_size=8 * 1024 * 1024,
        column_names=["impression_id", "user_id", "time", "history", "impressions"],
    )
    parse_options = pacsv.ParseOptions(
        delimiter="\t",
        quote_char=False,
    )
    convert_options = pacsv.ConvertOptions(column_types={
        "impression_id": pa.string(),
        "user_id": pa.string(),
        "time": pa.string(),
        "history": pa.string(),
        "impressions": pa.string(),
    })

    reader = pacsv.open_csv(
        path,
        read_options=read_options,
        parse_options=parse_options,
        convert_options=convert_options,
    )

    batch: list[Impression] = []
    total = 0
    for record_batch in reader:
        columns = {name: record_batch[name].to_pylist() for name in record_batch.schema.names}
        n = record_batch.num_rows
        for i in range(n):
            timestamp = None
            raw_time = columns["time"][i]
            if raw_time:
                try:
                    timestamp = _parse_mind_time(raw_time)
                except ValueError:
                    logger.warning("Could not parse MIND timestamp %r", raw_time)

            item = Impression(
                impression_id=str(columns["impression_id"][i]),
                user_id=str(columns["user_id"][i]),
                timestamp=timestamp,
                history=_parse_mind_history(columns["history"][i]),
                candidates=_parse_mind_candidates(columns["impressions"][i]),
            )
            batch.append(item)
            total += 1
            if len(batch) >= batch_size:
                yield batch
                batch = []

    if batch:
        yield batch
    logger.info("Streamed %d MIND test impressions from %s", total, path)


def _find_ebnerd_test_dir(raw_dir: Path) -> Path:
    candidates = [
        raw_dir / "ebnerd_testset" / "test",
        raw_dir / "test",
    ]
    for path in candidates:
        if (path / "behaviors.parquet").exists():
            return path
    matches = sorted(raw_dir.rglob("behaviors.parquet"))
    for path in matches:
        if path.parent.name.lower() == "test" and "test" in str(path.parent.parent).lower():
            return path.parent
    raise FileNotFoundError(f"EB-NeRD test behaviors.parquet not found below {raw_dir}")


def find_ebnerd_test_files(raw_dir: Path) -> tuple[Path, Path]:
    test_dir = _find_ebnerd_test_dir(raw_dir)
    behaviors = test_dir / "behaviors.parquet"
    history = test_dir / "history.parquet"
    if not history.exists():
        raise FileNotFoundError(f"EB-NeRD test history.parquet not found at {history}")
    return behaviors, history


def iter_ebnerd_test(path: Path, batch_size: int = 100_000) -> Iterator[list[Impression]]:
    """Yield bounded lists of EB-NeRD test impressions."""
    parquet = pq.ParquetFile(path)
    columns = ["impression_id", "user_id", "impression_time", "article_ids_inview"]
    batch: list[Impression] = []
    total = 0

    for record_batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        data = record_batch.to_pydict()
        n = record_batch.num_rows
        for i in range(n):
            raw_candidates = data["article_ids_inview"][i] or []
            candidates = [str(article_id) for article_id in raw_candidates]
            timestamp = data["impression_time"][i]
            batch.append(Impression(
                impression_id=str(data["impression_id"][i]),
                user_id=str(data["user_id"][i]),
                timestamp=timestamp.to_pydatetime() if hasattr(timestamp, "to_pydatetime") else timestamp,
                history=[],
                candidates=candidates,
            ))
            total += 1
            if len(batch) >= batch_size:
                yield batch
                batch = []

    if batch:
        yield batch
    logger.info("Streamed %d EB-NeRD test impressions from %s", total, path)
