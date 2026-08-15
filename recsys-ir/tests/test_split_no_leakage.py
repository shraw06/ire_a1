"""Verify temporal split has no data leakage - no future impressions in training data.

Three leakage checks:
  1. No behaviors row in our ``train`` has timestamp >= val_cutoff.
  2. No behaviors row in our ``val`` has timestamp >= native_test_start
     (val must not spill into what we call test).
  3. No article that appears ONLY in a later split's click history is
     unresolvable in the article table as of that point in time.
     (Proxy check: every article ID referenced in candidates/labels across
      ALL splits exists in the articles table.)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import polars as pl
import pytest

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_MIND_INTERIM = _PROJECT_ROOT / "data" / "interim" / "mind"
_EBNERD_INTERIM = _PROJECT_ROOT / "data" / "interim" / "ebnerd"

# Import split boundaries so tests stay in sync with the implementation.
from src.splitting.temporal_split import SPLIT_CONFIG


# Fixtures

@pytest.fixture(scope="session")
def mind_behaviors() -> pl.DataFrame:
    path = _MIND_INTERIM / "behaviors.parquet"
    if not path.exists():
        pytest.skip("MIND interim not yet generated")
    df = pl.read_parquet(path)
    if "split" not in df.columns:
        pytest.skip("MIND behaviors has no 'split' column - run temporal_split first")
    return df


@pytest.fixture(scope="session")
def ebnerd_behaviors() -> pl.DataFrame:
    path = _EBNERD_INTERIM / "behaviors.parquet"
    if not path.exists():
        pytest.skip("EB-NeRD interim not yet generated")
    df = pl.read_parquet(path)
    if "split" not in df.columns:
        pytest.skip("EB-NeRD behaviors has no 'split' column - run temporal_split first")
    return df


@pytest.fixture(scope="session")
def mind_articles() -> pl.DataFrame:
    path = _MIND_INTERIM / "articles.parquet"
    if not path.exists():
        pytest.skip("MIND interim not yet generated")
    return pl.read_parquet(path)


@pytest.fixture(scope="session")
def ebnerd_articles() -> pl.DataFrame:
    path = _EBNERD_INTERIM / "articles.parquet"
    if not path.exists():
        pytest.skip("EB-NeRD interim not yet generated")
    return pl.read_parquet(path)


#  Check 1: No train row has timestamp >= val_cutoff

class TestNoTrainLeakIntoVal:
    """Train rows must have timestamps strictly before the val cutoff."""

    def test_mind_train_before_val_cutoff(self, mind_behaviors: pl.DataFrame) -> None:
        val_cutoff = SPLIT_CONFIG["mind"]["val_cutoff"]
        train_rows = mind_behaviors.filter(pl.col("split") == "train")
        leakers = train_rows.filter(pl.col("timestamp") >= val_cutoff)
        assert leakers.height == 0, (
            f"MIND: {leakers.height} train rows have timestamp >= val_cutoff "
            f"({val_cutoff}). Max train timestamp: {train_rows['timestamp'].max()}"
        )

    def test_ebnerd_train_before_val_cutoff(self, ebnerd_behaviors: pl.DataFrame) -> None:
        val_cutoff = SPLIT_CONFIG["ebnerd"]["val_cutoff"]
        train_rows = ebnerd_behaviors.filter(pl.col("split") == "train")
        leakers = train_rows.filter(pl.col("timestamp") >= val_cutoff)
        assert leakers.height == 0, (
            f"EB-NeRD: {leakers.height} train rows have timestamp >= val_cutoff "
            f"({val_cutoff}). Max train timestamp: {train_rows['timestamp'].max()}"
        )


#  Check 2: No val row has timestamp >= native_test_start

class TestNoValLeakIntoTest:
    """Val rows must have timestamps strictly before the native held-out start."""

    def test_mind_val_before_test_start(self, mind_behaviors: pl.DataFrame) -> None:
        native_test_start = SPLIT_CONFIG["mind"]["native_test_start"]
        val_rows = mind_behaviors.filter(pl.col("split") == "val")
        leakers = val_rows.filter(pl.col("timestamp") >= native_test_start)
        assert leakers.height == 0, (
            f"MIND: {leakers.height} val rows have timestamp >= native_test_start "
            f"({native_test_start}). Max val timestamp: {val_rows['timestamp'].max()}"
        )

    def test_ebnerd_val_before_test_start(self, ebnerd_behaviors: pl.DataFrame) -> None:
        native_test_start = SPLIT_CONFIG["ebnerd"]["native_test_start"]
        val_rows = ebnerd_behaviors.filter(pl.col("split") == "val")
        leakers = val_rows.filter(pl.col("timestamp") >= native_test_start)
        assert leakers.height == 0, (
            f"EB-NeRD: {leakers.height} val rows have timestamp >= native_test_start "
            f"({native_test_start}). Max val timestamp: {val_rows['timestamp'].max()}"
        )


# Check 3: Article resolvability

class TestArticleResolvability:
    """Every article referenced in candidates across any split must exist in
    the articles table.

    This is a proxy for the harder check "no article present only in a later
    split's click history is unresolvable in the article table as of that point
    in time".  Since neither MIND nor EB-NeRD articles have reliable
    published_at for all rows (MIND has none), we check the simpler invariant:
    all candidate article IDs are present in the articles table, period.
    """

    @staticmethod
    def _extract_candidate_ids(behaviors: pl.DataFrame) -> set[str]:
        """Extract all unique article IDs from the candidates column."""
        all_ids: set[str] = set()
        for row in behaviors.to_dicts():
            cands = json.loads(row["candidates"])
            all_ids.update(cands)
        return all_ids

    def test_mind_candidates_resolvable(
        self, mind_behaviors: pl.DataFrame, mind_articles: pl.DataFrame
    ) -> None:
        candidate_ids = self._extract_candidate_ids(mind_behaviors)
        article_ids = set(mind_articles["article_id"].to_list())
        unresolvable = candidate_ids - article_ids
        assert len(unresolvable) == 0, (
            f"MIND: {len(unresolvable)} candidate article IDs not found in articles table. "
            f"Sample: {list(unresolvable)[:10]}"
        )

    def test_ebnerd_candidates_resolvable(
        self, ebnerd_behaviors: pl.DataFrame, ebnerd_articles: pl.DataFrame
    ) -> None:
        candidate_ids = self._extract_candidate_ids(ebnerd_behaviors)
        article_ids = set(ebnerd_articles["article_id"].to_list())
        unresolvable = candidate_ids - article_ids
        assert len(unresolvable) == 0, (
            f"EB-NeRD: {len(unresolvable)} candidate article IDs not found in articles table. "
            f"Sample: {list(unresolvable)[:10]}"
        )


# Sanity: split column exhaustive & non-empty

class TestSplitSanity:
    """Basic sanity checks on the split column itself."""

    def test_mind_split_values(self, mind_behaviors: pl.DataFrame) -> None:
        actual = set(mind_behaviors["split"].unique().to_list())
        assert actual == {"train", "val", "test"}, f"Expected {{train, val, test}}, got {actual}"

    def test_ebnerd_split_values(self, ebnerd_behaviors: pl.DataFrame) -> None:
        actual = set(ebnerd_behaviors["split"].unique().to_list())
        assert actual == {"train", "val", "test"}, f"Expected {{train, val, test}}, got {actual}"

    def test_mind_no_empty_split(self, mind_behaviors: pl.DataFrame) -> None:
        for s in ("train", "val", "test"):
            count = mind_behaviors.filter(pl.col("split") == s).height
            assert count > 0, f"MIND split '{s}' is empty"

    def test_ebnerd_no_empty_split(self, ebnerd_behaviors: pl.DataFrame) -> None:
        for s in ("train", "val", "test"):
            count = ebnerd_behaviors.filter(pl.col("split") == s).height
            assert count > 0, f"EB-NeRD split '{s}' is empty"

    def test_mind_total_preserved(self, mind_behaviors: pl.DataFrame) -> None:
        """Split column assignment must not change row count."""
        total_by_split = sum(
            mind_behaviors.filter(pl.col("split") == s).height
            for s in ("train", "val", "test")
        )
        assert total_by_split == len(mind_behaviors), (
            f"MIND: sum of splits ({total_by_split}) != total rows ({len(mind_behaviors)})"
        )

    def test_ebnerd_total_preserved(self, ebnerd_behaviors: pl.DataFrame) -> None:
        total_by_split = sum(
            ebnerd_behaviors.filter(pl.col("split") == s).height
            for s in ("train", "val", "test")
        )
        assert total_by_split == len(ebnerd_behaviors), (
            f"EB-NeRD: sum of splits ({total_by_split}) != total rows ({len(ebnerd_behaviors)})"
        )


# Val size sanity

class TestValSizeNotDegenerate:
    """Val must not be degenerately small (< 10% of train+val impressions)."""

    def test_mind_val_size(self, mind_behaviors: pl.DataFrame) -> None:
        train_count = mind_behaviors.filter(pl.col("split") == "train").height
        val_count = mind_behaviors.filter(pl.col("split") == "val").height
        ratio = val_count / (train_count + val_count)
        assert ratio >= 0.10, (
            f"MIND val is too small: {val_count} / {train_count + val_count} = "
            f"{ratio:.1%} (expected >= 10%)"
        )

    def test_ebnerd_val_size(self, ebnerd_behaviors: pl.DataFrame) -> None:
        train_count = ebnerd_behaviors.filter(pl.col("split") == "train").height
        val_count = ebnerd_behaviors.filter(pl.col("split") == "val").height
        ratio = val_count / (train_count + val_count)
        assert ratio >= 0.10, (
            f"EB-NeRD val is too small: {val_count} / {train_count + val_count} = "
            f"{ratio:.1%} (expected >= 10%)"
        )
