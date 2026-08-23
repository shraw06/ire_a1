from datetime import datetime
from pathlib import Path

import polars as pl

from src.feature_store.history_store import MemoryMappedHistoryStore


def test_memory_mapped_history_store(tmp_path: Path):
    history = pl.DataFrame({
        "user_id": [2, 1],
        "article_id_fixed": [[20, 21], [10]],
        "impression_time_fixed": [
            [datetime(2023, 1, 2), datetime(2023, 1, 3)],
            [datetime(2023, 1, 1)],
        ],
    })
    source = tmp_path / "history.parquet"
    history.write_parquet(source)
    store = MemoryMappedHistoryStore.build(source, tmp_path / "index")
    result = store.get_history("2", datetime(2023, 1, 3))
    assert [x["article_id"] for x in result] == ["20"]
