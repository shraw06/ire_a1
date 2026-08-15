"""End-to-end smoke test on a tiny synthetic fixture — must run in CI in seconds.

Creates synthetic MIND-format raw data (10 users, 30 articles), then runs
the full pipeline (parse → split → build_features) programmatically.
Asserts non-empty outputs at every stage.

Target: completes in under 10 seconds.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest


# ── Synthetic data generation ─────────────────────────────────────

def _generate_synthetic_mind_data(raw_dir: Path, n_users: int = 10, n_articles: int = 30) -> None:
    """Generate minimal synthetic MIND-format TSV files.

    Creates MINDsmall_train/ and MINDsmall_dev/ directories with
    news.tsv and behaviors.tsv files.
    """
    # ── Articles (news.tsv) ──────────────────────────────────────
    # Format: news_id \t category \t subcategory \t title \t abstract \t url \t title_entities \t abstract_entities
    articles = []
    for i in range(1, n_articles + 1):
        news_id = f"N{i:05d}"
        category = f"cat{(i % 5) + 1}"
        subcategory = f"sub{(i % 3) + 1}"
        title = f"Test Article {i} Title"
        abstract = f"Abstract for test article {i} about {category}"
        url = f"https://example.com/{news_id}"
        title_entities = json.dumps([{"Label": f"Entity{i}", "Type": "O", "WikidataId": f"Q{i}", "Confidence": 0.9}])
        abstract_entities = "[]"
        articles.append(
            f"{news_id}\t{category}\t{subcategory}\t{title}\t{abstract}\t{url}\t{title_entities}\t{abstract_entities}"
        )

    # ── Behaviors (behaviors.tsv) ────────────────────────────────
    # Format: impression_id \t user_id \t time \t history \t impressions
    #
    # We create behaviors spanning Nov 9-14 (train range) and Nov 15 (dev range)
    # to exercise the temporal split.
    base_date_train = datetime(2019, 11, 9, 8, 0, 0)
    base_date_dev = datetime(2019, 11, 15, 8, 0, 0)

    train_behaviors = []
    dev_behaviors = []
    impression_counter = 1

    for u in range(1, n_users + 1):
        user_id = f"U{u:05d}"
        # Each user has a history of 3-5 articles
        history_ids = [f"N{((u + j) % n_articles) + 1:05d}" for j in range(min(5, n_articles))]
        history_str = " ".join(history_ids[:5])

        # 3 train behaviors per user spanning Nov 9, 12, 14
        # Nov 9 + Nov 12 -> train (before val_cutoff = Nov 14 00:00)
        # Nov 14         -> val   (>= val_cutoff, < native_test_start = Nov 15)
        train_day_offsets = [0, 3, 5]  # Nov 9, Nov 12, Nov 14
        for b, day_offset in enumerate(train_day_offsets):
            imp_id = str(impression_counter)
            impression_counter += 1
            ts = base_date_train + timedelta(days=day_offset, hours=u)
            ts_str = ts.strftime("%-m/%-d/%Y %-I:%M:%S %p")
            # 3 candidate articles, 1 clicked
            cands = [f"N{((u + b * 3 + c) % n_articles) + 1:05d}" for c in range(3)]
            impressions = " ".join(
                f"{cid}-{1 if c == 0 else 0}" for c, cid in enumerate(cands)
            )
            train_behaviors.append(
                f"{imp_id}\t{user_id}\t{ts_str}\t{history_str}\t{impressions}"
            )

        # 1 dev behavior per user (Nov 15)
        imp_id = str(impression_counter)
        impression_counter += 1
        ts = base_date_dev + timedelta(hours=u)
        ts_str = ts.strftime("%-m/%-d/%Y %-I:%M:%S %p")
        cands = [f"N{((u + 10 + c) % n_articles) + 1:05d}" for c in range(3)]
        impressions = " ".join(
            f"{cid}-{1 if c == 0 else 0}" for c, cid in enumerate(cands)
        )
        dev_behaviors.append(
            f"{imp_id}\t{user_id}\t{ts_str}\t{history_str}\t{impressions}"
        )

    # Write files
    train_dir = raw_dir / "MINDsmall_train"
    train_dir.mkdir(parents=True, exist_ok=True)
    dev_dir = raw_dir / "MINDsmall_dev"
    dev_dir.mkdir(parents=True, exist_ok=True)

    (train_dir / "news.tsv").write_text("\n".join(articles) + "\n")
    (train_dir / "behaviors.tsv").write_text("\n".join(train_behaviors) + "\n")
    (dev_dir / "news.tsv").write_text("\n".join(articles) + "\n")
    (dev_dir / "behaviors.tsv").write_text("\n".join(dev_behaviors) + "\n")


# ── Test ──────────────────────────────────────────────────────────

@pytest.mark.timeout(10)
class TestPipelineE2ESmoke:
    """Full pipeline smoke test on synthetic data (10 users, 30 articles)."""

    def test_full_pipeline(self, tmp_path: Path) -> None:
        """Run parse → split → features on synthetic MIND data.

        Asserts non-empty outputs at every stage.
        """
        t0 = time.time()

        # 1. Generate synthetic raw data
        raw_dir = tmp_path / "data" / "raw" / "mind"
        interim_dir = tmp_path / "data" / "interim" / "mind"
        processed_dir = tmp_path / "data" / "processed" / "mind"

        _generate_synthetic_mind_data(raw_dir, n_users=10, n_articles=30)

        # Verify raw data was created
        assert (raw_dir / "MINDsmall_train" / "news.tsv").exists()
        assert (raw_dir / "MINDsmall_train" / "behaviors.tsv").exists()
        assert (raw_dir / "MINDsmall_dev" / "news.tsv").exists()
        assert (raw_dir / "MINDsmall_dev" / "behaviors.tsv").exists()

        # 2. Parse (programmatic — not via Makefile)
        from src.parsing.parse_mind import (
            parse_mind_articles,
            parse_mind_behaviors,
            derive_mind_users,
            write_interim,
        )

        articles = parse_mind_articles(raw_dir)
        behaviors = parse_mind_behaviors(raw_dir)
        users = derive_mind_users(behaviors)
        write_interim(articles, behaviors, users, interim_dir)

        # Assert interim outputs exist and are non-empty
        assert (interim_dir / "articles.parquet").exists()
        assert (interim_dir / "behaviors.parquet").exists()
        assert (interim_dir / "users.parquet").exists()
        assert len(articles) > 0, "Articles should be non-empty"
        assert len(behaviors) > 0, "Behaviors should be non-empty"
        assert len(users) > 0, "Users should be non-empty"

        # 3. Split
        from src.splitting.temporal_split import assign_temporal_split

        behaviors = assign_temporal_split(behaviors, "mind")

        # Assert split column
        assert "split" in behaviors.columns
        split_values = set(behaviors["split"].unique().to_list())
        assert split_values == {"train", "val", "test"}, f"Expected 3 splits, got {split_values}"

        # Overwrite the behaviors parquet with split column for features step
        behaviors.write_parquet(interim_dir / "behaviors.parquet")

        # 4. Build features
        from src.feature_store.article_store import ArticleFeatureStore
        from src.feature_store.user_store import UserFeatureStore

        article_path = ArticleFeatureStore.build_features(
            "mind", interim_dir=interim_dir, processed_dir=processed_dir
        )
        user_path = UserFeatureStore.build_features(
            "mind", interim_dir=interim_dir, processed_dir=processed_dir
        )

        # Assert feature outputs exist and are non-empty
        assert article_path.exists()
        assert user_path.exists()

        article_features = pl.read_parquet(article_path)
        user_features = pl.read_parquet(user_path)

        assert len(article_features) > 0, "Article features should be non-empty"
        assert len(user_features) > 0, "User features should be non-empty"

        # Assert expected columns
        assert "cleaned_text" in article_features.columns, "Missing cleaned_text column"
        assert "all_history" in user_features.columns, "Missing all_history column"

        # 5. Verify feature store can be queried via DuckDB
        article_store = ArticleFeatureStore("mind", processed_dir=processed_dir)
        row = article_store.get_article("N00001")
        assert row is not None, "Should be able to look up article N00001"
        assert row["cleaned_text"], "cleaned_text should be non-empty"

        user_store = UserFeatureStore("mind", processed_dir=processed_dir)
        history = user_store.get_user_history(
            "U00001", as_of_ts=datetime(2019, 11, 14), dataset="mind"
        )
        assert len(history) > 0, "User U00001 should have non-empty history"

        elapsed = time.time() - t0
        print(f"\n  ✓ E2E smoke test passed in {elapsed:.1f}s")
        assert elapsed < 10, f"E2E smoke test took {elapsed:.1f}s — must be under 10s"

    def test_article_count_matches(self, tmp_path: Path) -> None:
        """The article feature store should have the same count as interim articles."""
        raw_dir = tmp_path / "data" / "raw" / "mind"
        interim_dir = tmp_path / "data" / "interim" / "mind"
        processed_dir = tmp_path / "data" / "processed" / "mind"

        _generate_synthetic_mind_data(raw_dir, n_users=10, n_articles=30)

        from src.parsing.parse_mind import (
            parse_mind_articles, parse_mind_behaviors, derive_mind_users, write_interim,
        )
        articles = parse_mind_articles(raw_dir)
        behaviors = parse_mind_behaviors(raw_dir)
        users = derive_mind_users(behaviors)
        write_interim(articles, behaviors, users, interim_dir)

        from src.feature_store.article_store import ArticleFeatureStore
        ArticleFeatureStore.build_features("mind", interim_dir=interim_dir, processed_dir=processed_dir)

        store = ArticleFeatureStore("mind", processed_dir=processed_dir)
        assert store.row_count == len(articles), (
            f"Article store has {store.row_count} rows but interim has {len(articles)}"
        )

    def test_user_count_matches(self, tmp_path: Path) -> None:
        """The user feature store should have the expected user count."""
        raw_dir = tmp_path / "data" / "raw" / "mind"
        interim_dir = tmp_path / "data" / "interim" / "mind"
        processed_dir = tmp_path / "data" / "processed" / "mind"

        _generate_synthetic_mind_data(raw_dir, n_users=10, n_articles=30)

        from src.parsing.parse_mind import (
            parse_mind_articles, parse_mind_behaviors, derive_mind_users, write_interim,
        )
        articles = parse_mind_articles(raw_dir)
        behaviors = parse_mind_behaviors(raw_dir)
        users = derive_mind_users(behaviors)
        write_interim(articles, behaviors, users, interim_dir)

        from src.feature_store.user_store import UserFeatureStore
        UserFeatureStore.build_features("mind", interim_dir=interim_dir, processed_dir=processed_dir)

        store = UserFeatureStore("mind", processed_dir=processed_dir)
        assert store.row_count == 10, f"Expected 10 users, got {store.row_count}"
