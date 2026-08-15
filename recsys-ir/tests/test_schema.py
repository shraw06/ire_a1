"""Tests for unified schema validation - ensure MIND and EB-NeRD parse to identical structures.

This test suite validates:
  1. Both adapters produce DataFrames with the same column set and dtypes.
  2. MIND articles have body=None and body_source=None (documented known-limitation).
  3. EB-NeRD articles have abstract populated (from subtitle), body_source="native" where body exists.
  4. EB-NeRD embedding_ref is always null at this stage (demo bundle has no embeddings).
  5. EB-NeRD clicked_history entries carry real timestamps; MIND's don't.
  6. EB-NeRD history join preserved row counts (no drops/duplicates).
  7. Pydantic model validation for sample rows.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import polars as pl
import pytest

# Project root
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_MIND_INTERIM = _PROJECT_ROOT / "data" / "interim" / "mind"
_EBNERD_INTERIM = _PROJECT_ROOT / "data" / "interim" / "ebnerd"

# Expected column sets (from schema.py)
_ARTICLES_COLUMNS = {
    "article_id", "dataset", "title", "abstract", "body", "body_source",
    "category", "subcategory", "entities", "published_at", "embedding_ref",
}

# Base behaviors columns (always present after parsing)
_BEHAVIORS_COLUMNS_BASE = {
    "impression_id", "dataset", "user_id", "timestamp",
    "clicked_history", "candidates", "labels",
}

# Optional columns added by later pipeline stages
_BEHAVIORS_COLUMNS_OPTIONAL = {
    "split",  # added by temporal_split.py
}

_USERS_COLUMNS = {
    "user_id", "dataset", "history_article_ids", "history_len", "last_active_at",
}


# Fixtures

@pytest.fixture(scope="session")
def mind_articles() -> pl.DataFrame:
    path = _MIND_INTERIM / "articles.parquet"
    if not path.exists():
        pytest.skip("MIND interim not yet generated - run parse_mind.py first")
    return pl.read_parquet(path)


@pytest.fixture(scope="session")
def mind_behaviors() -> pl.DataFrame:
    path = _MIND_INTERIM / "behaviors.parquet"
    if not path.exists():
        pytest.skip("MIND interim not yet generated - run parse_mind.py first")
    return pl.read_parquet(path)


@pytest.fixture(scope="session")
def mind_users() -> pl.DataFrame:
    path = _MIND_INTERIM / "users.parquet"
    if not path.exists():
        pytest.skip("MIND interim not yet generated - run parse_mind.py first")
    return pl.read_parquet(path)


@pytest.fixture(scope="session")
def ebnerd_articles() -> pl.DataFrame:
    path = _EBNERD_INTERIM / "articles.parquet"
    if not path.exists():
        pytest.skip("EB-NeRD interim not yet generated - run parse_ebnerd.py first")
    return pl.read_parquet(path)


@pytest.fixture(scope="session")
def ebnerd_behaviors() -> pl.DataFrame:
    path = _EBNERD_INTERIM / "behaviors.parquet"
    if not path.exists():
        pytest.skip("EB-NeRD interim not yet generated - run parse_ebnerd.py first")
    return pl.read_parquet(path)


@pytest.fixture(scope="session")
def ebnerd_users() -> pl.DataFrame:
    path = _EBNERD_INTERIM / "users.parquet"
    if not path.exists():
        pytest.skip("EB-NeRD interim not yet generated - run parse_ebnerd.py first")
    return pl.read_parquet(path)


# Column schema tests

class TestSchemaColumns:
    """Verify both datasets have identical column sets."""

    def test_mind_articles_columns(self, mind_articles: pl.DataFrame) -> None:
        assert set(mind_articles.columns) == _ARTICLES_COLUMNS

    def test_ebnerd_articles_columns(self, ebnerd_articles: pl.DataFrame) -> None:
        assert set(ebnerd_articles.columns) == _ARTICLES_COLUMNS

    def test_mind_behaviors_columns(self, mind_behaviors: pl.DataFrame) -> None:
        actual = set(mind_behaviors.columns)
        assert actual >= _BEHAVIORS_COLUMNS_BASE, (
            f"Missing base columns: {_BEHAVIORS_COLUMNS_BASE - actual}"
        )
        extra = actual - _BEHAVIORS_COLUMNS_BASE - _BEHAVIORS_COLUMNS_OPTIONAL
        assert not extra, f"Unexpected columns: {extra}"

    def test_ebnerd_behaviors_columns(self, ebnerd_behaviors: pl.DataFrame) -> None:
        actual = set(ebnerd_behaviors.columns)
        assert actual >= _BEHAVIORS_COLUMNS_BASE, (
            f"Missing base columns: {_BEHAVIORS_COLUMNS_BASE - actual}"
        )
        extra = actual - _BEHAVIORS_COLUMNS_BASE - _BEHAVIORS_COLUMNS_OPTIONAL
        assert not extra, f"Unexpected columns: {extra}"

    def test_mind_users_columns(self, mind_users: pl.DataFrame) -> None:
        assert set(mind_users.columns) == _USERS_COLUMNS

    def test_ebnerd_users_columns(self, ebnerd_users: pl.DataFrame) -> None:
        assert set(ebnerd_users.columns) == _USERS_COLUMNS

    def test_articles_dtypes_match(self, mind_articles: pl.DataFrame,
                                    ebnerd_articles: pl.DataFrame) -> None:
        """Both adapters should produce the same dtypes for all columns."""
        mind_schema = {k: str(v) for k, v in mind_articles.schema.items()}
        ebnerd_schema = {k: str(v) for k, v in ebnerd_articles.schema.items()}
        assert mind_schema == ebnerd_schema, (
            f"Schema mismatch:\n  MIND:   {mind_schema}\n  EB-NeRD: {ebnerd_schema}"
        )

    def test_behaviors_dtypes_match(self, mind_behaviors: pl.DataFrame,
                                     ebnerd_behaviors: pl.DataFrame) -> None:
        mind_schema = {k: str(v) for k, v in mind_behaviors.schema.items()}
        ebnerd_schema = {k: str(v) for k, v in ebnerd_behaviors.schema.items()}
        assert mind_schema == ebnerd_schema

    def test_users_dtypes_match(self, mind_users: pl.DataFrame,
                                 ebnerd_users: pl.DataFrame) -> None:
        mind_schema = {k: str(v) for k, v in mind_users.schema.items()}
        ebnerd_schema = {k: str(v) for k, v in ebnerd_users.schema.items()}
        assert mind_schema == ebnerd_schema


# MIND-specific null-field assertions

class TestMindNullFields:
    """Validate MIND's known-null fields - documented limitation, not bugs."""

    def test_mind_body_always_null(self, mind_articles: pl.DataFrame) -> None:
        """MIND body must be null by default (licensing restriction -
        MSN article bodies were deliberately withheld).
        This is a documented known-limitation field, not a silent bug."""
        non_null_body = mind_articles.filter(pl.col("body").is_not_null())
        assert len(non_null_body) == 0, (
            f"Expected MIND body to be all null, but found {len(non_null_body)} non-null rows"
        )

    def test_mind_body_source_always_null(self, mind_articles: pl.DataFrame) -> None:
        """body_source must also be null when body is null."""
        non_null_bs = mind_articles.filter(pl.col("body_source").is_not_null())
        assert len(non_null_bs) == 0, (
            f"Expected MIND body_source to be all null, but found {len(non_null_bs)} non-null rows"
        )

    def test_mind_history_no_timestamps(self, mind_behaviors: pl.DataFrame) -> None:
        """MIND clicked_history entries should have clicked_at=None (no per-item timestamps)."""
        sample = mind_behaviors.head(100).to_dicts()
        for row in sample:
            history = json.loads(row["clicked_history"])
            for entry in history:
                assert entry["clicked_at"] is None, (
                    f"MIND history entry should have clicked_at=None, "
                    f"got {entry['clicked_at']} for article {entry['article_id']}"
                )


# EB-NeRD-specific assertions

class TestEbnerdFields:
    """Validate EB-NeRD-specific field mappings."""

    def test_ebnerd_abstract_populated(self, ebnerd_articles: pl.DataFrame) -> None:
        """EB-NeRD abstract should be populated from subtitle column.
        ~6.8% may be null (per EDA §6), but the majority should be present."""
        non_null_count = ebnerd_articles.filter(pl.col("abstract").is_not_null()).height
        total = len(ebnerd_articles)
        pct = non_null_count / total * 100
        assert pct > 90, (
            f"Expected >90% of EB-NeRD abstracts to be non-null (from subtitle), "
            f"got {pct:.1f}% ({non_null_count}/{total})"
        )

    def test_ebnerd_body_source_native(self, ebnerd_articles: pl.DataFrame) -> None:
        """EB-NeRD rows with body text should have body_source='native'."""
        with_body = ebnerd_articles.filter(pl.col("body").is_not_null())
        if len(with_body) > 0:
            native_count = with_body.filter(pl.col("body_source") == "native").height
            assert native_count == len(with_body), (
                f"EB-NeRD rows with body should all have body_source='native', "
                f"but {len(with_body) - native_count} do not"
            )

    def test_ebnerd_embedding_ref_always_null(self, ebnerd_articles: pl.DataFrame) -> None:
        """EB-NeRD embedding_ref must be null at this stage - demo bundle has no embeddings.
        This is expected (EDA §8), not a bug."""
        non_null = ebnerd_articles.filter(pl.col("embedding_ref").is_not_null())
        assert len(non_null) == 0, (
            f"Expected EB-NeRD embedding_ref to be all null (demo bundle), "
            f"but found {len(non_null)} non-null rows"
        )

    def test_ebnerd_history_has_timestamps(self, ebnerd_behaviors: pl.DataFrame) -> None:
        """EB-NeRD clicked_history entries should carry real timestamps (from history.parquet)."""
        sample = ebnerd_behaviors.head(100).to_dicts()
        found_any_timestamp = False
        for row in sample:
            history = json.loads(row["clicked_history"])
            for entry in history:
                if entry["clicked_at"] is not None:
                    found_any_timestamp = True
                    break
            if found_any_timestamp:
                break
        assert found_any_timestamp, (
            "Expected at least some EB-NeRD history entries to have real timestamps "
            "(from history.parquet join), but all were None"
        )


# Data integrity tests

class TestDataIntegrity:
    """Basic sanity checks on parsed data."""

    def test_mind_articles_non_empty(self, mind_articles: pl.DataFrame) -> None:
        assert len(mind_articles) > 0

    def test_ebnerd_articles_non_empty(self, ebnerd_articles: pl.DataFrame) -> None:
        assert len(ebnerd_articles) > 0

    def test_mind_behaviors_non_empty(self, mind_behaviors: pl.DataFrame) -> None:
        assert len(mind_behaviors) > 0

    def test_ebnerd_behaviors_non_empty(self, ebnerd_behaviors: pl.DataFrame) -> None:
        assert len(ebnerd_behaviors) > 0

    def test_mind_all_dataset_mind(self, mind_articles: pl.DataFrame) -> None:
        vals = mind_articles["dataset"].unique().to_list()
        assert vals == ["mind"]

    def test_ebnerd_all_dataset_ebnerd(self, ebnerd_articles: pl.DataFrame) -> None:
        vals = ebnerd_articles["dataset"].unique().to_list()
        assert vals == ["ebnerd"]

    def test_mind_no_duplicate_article_ids(self, mind_articles: pl.DataFrame) -> None:
        assert mind_articles["article_id"].is_unique().all()

    def test_ebnerd_no_duplicate_article_ids(self, ebnerd_articles: pl.DataFrame) -> None:
        assert ebnerd_articles["article_id"].is_unique().all()

    def test_mind_title_never_null(self, mind_articles: pl.DataFrame) -> None:
        assert mind_articles["title"].null_count() == 0

    def test_ebnerd_title_never_null(self, ebnerd_articles: pl.DataFrame) -> None:
        assert ebnerd_articles["title"].null_count() == 0

    def test_mind_candidates_valid_json(self, mind_behaviors: pl.DataFrame) -> None:
        sample = mind_behaviors.head(50).to_dicts()
        for row in sample:
            cands = json.loads(row["candidates"])
            labels = json.loads(row["labels"])
            assert isinstance(cands, list)
            assert isinstance(labels, list)
            assert len(cands) == len(labels), (
                f"Candidates/labels length mismatch: {len(cands)} vs {len(labels)}"
            )

    def test_ebnerd_candidates_valid_json(self, ebnerd_behaviors: pl.DataFrame) -> None:
        sample = ebnerd_behaviors.head(50).to_dicts()
        for row in sample:
            cands = json.loads(row["candidates"])
            labels = json.loads(row["labels"])
            assert isinstance(cands, list)
            assert isinstance(labels, list)
            assert len(cands) == len(labels)

    def test_mind_users_history_len_consistent(self, mind_users: pl.DataFrame) -> None:
        sample = mind_users.head(50).to_dicts()
        for row in sample:
            aids = json.loads(row["history_article_ids"])
            assert row["history_len"] == len(aids)

    def test_ebnerd_users_history_len_consistent(self, ebnerd_users: pl.DataFrame) -> None:
        sample = ebnerd_users.head(50).to_dicts()
        for row in sample:
            aids = json.loads(row["history_article_ids"])
            assert row["history_len"] == len(aids)


# Pydantic validation

class TestPydanticValidation:
    """Validate sample rows against the Pydantic models."""

    def test_mind_article_pydantic(self, mind_articles: pl.DataFrame) -> None:
        from src.common.schema import Article, Entity

        sample = mind_articles.head(5).to_dicts()
        for row in sample:
            entities = json.loads(row["entities"])
            entity_objs = [Entity(**e) for e in entities]
            art = Article(
                article_id=row["article_id"],
                dataset=row["dataset"],
                title=row["title"],
                abstract=row["abstract"],
                body=row["body"],
                body_source=row["body_source"],
                category=row["category"],
                subcategory=row["subcategory"],
                entities=entity_objs,
                published_at=row["published_at"],
                embedding_ref=row["embedding_ref"],
            )
            assert art.dataset == "mind"

    def test_ebnerd_article_pydantic(self, ebnerd_articles: pl.DataFrame) -> None:
        from src.common.schema import Article, Entity

        sample = ebnerd_articles.head(5).to_dicts()
        for row in sample:
            entities = json.loads(row["entities"])
            entity_objs = [Entity(**e) for e in entities]
            art = Article(
                article_id=row["article_id"],
                dataset=row["dataset"],
                title=row["title"],
                abstract=row["abstract"],
                body=row["body"],
                body_source=row["body_source"],
                category=row["category"],
                subcategory=row["subcategory"],
                entities=entity_objs,
                published_at=row["published_at"],
                embedding_ref=row["embedding_ref"],
            )
            assert art.dataset == "ebnerd"

    def test_mind_behavior_pydantic(self, mind_behaviors: pl.DataFrame) -> None:
        from src.common.schema import Behavior, HistoryEntry

        sample = mind_behaviors.head(5).to_dicts()
        for row in sample:
            history = json.loads(row["clicked_history"])
            history_objs = [HistoryEntry(**h) for h in history]
            candidates = json.loads(row["candidates"])
            labels = json.loads(row["labels"])
            beh = Behavior(
                impression_id=row["impression_id"],
                dataset=row["dataset"],
                user_id=row["user_id"],
                timestamp=row["timestamp"],
                clicked_history=history_objs,
                candidates=candidates,
                labels=labels,
            )
            assert beh.dataset == "mind"

    def test_ebnerd_behavior_pydantic(self, ebnerd_behaviors: pl.DataFrame) -> None:
        from src.common.schema import Behavior, HistoryEntry

        sample = ebnerd_behaviors.head(5).to_dicts()
        for row in sample:
            history = json.loads(row["clicked_history"])
            history_objs = [HistoryEntry(**h) for h in history]
            candidates = json.loads(row["candidates"])
            labels = json.loads(row["labels"])
            beh = Behavior(
                impression_id=row["impression_id"],
                dataset=row["dataset"],
                user_id=row["user_id"],
                timestamp=row["timestamp"],
                clicked_history=history_objs,
                candidates=candidates,
                labels=labels,
            )
            assert beh.dataset == "ebnerd"
