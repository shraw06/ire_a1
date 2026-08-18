"""Q9 anti-leakage assertion - verifiable test that no future data leaks into training. Run as pytest."""

import json
from pathlib import Path
from datetime import timedelta

import polars as pl
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("dataset", ["mind", "ebnerd"])
def test_no_future_leakage(dataset: str) -> None:
    """Assert no candidate/history timestamp exceeds the impression's own timestamp."""
    behaviors_path = _PROJECT_ROOT / "data" / "interim" / dataset / "behaviors.parquet"
    articles_path = _PROJECT_ROOT / "data" / "interim" / dataset / "articles.parquet"
    
    if not behaviors_path.exists() or not articles_path.exists():
        pytest.skip(f"Data not found for dataset {dataset}")
        
    df_behaviors = pl.read_parquet(behaviors_path)
    df_articles = pl.read_parquet(articles_path)
    
    # Create article -> published_at mapping
    article_timestamps = dict(
        zip(df_articles["article_id"].to_list(), df_articles["published_at"].to_list())
    )
    
    # Allow 24-hour buffer for dataset noise/embargoes/timezone skew
    tolerance = timedelta(hours=24)
    
    for row in df_behaviors.iter_rows(named=True):
        imp_ts = row["timestamp"]
        if imp_ts is None:
            continue
            
        candidates_str = row["candidates"]
        history_str = row["clicked_history"]
        
        candidates = json.loads(candidates_str) if candidates_str else []
        history = json.loads(history_str) if history_str else []
        
        for item in candidates:
            item_ts = article_timestamps.get(item)
            if item_ts is not None and item_ts > imp_ts + tolerance:
                pytest.fail(f"Leakage detected in {dataset}: candidate {item} published at {item_ts} is after impression at {imp_ts}")
                
        for h_item in history:
            # Handle EB-NeRD list of dicts vs MIND list of strings
            aid = h_item["article_id"] if isinstance(h_item, dict) else h_item
            item_ts = article_timestamps.get(aid)
            if item_ts is not None and item_ts > imp_ts + tolerance:
                pytest.fail(f"Leakage detected in {dataset}: history item {aid} published at {item_ts} is after impression at {imp_ts}")
