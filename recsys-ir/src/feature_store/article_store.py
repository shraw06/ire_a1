"""Article feature store - cleaned text, category, entities, embedding pointer.

Builds per-article features from the interim articles table and exposes them
via DuckDB-backed selective reads.

Feature columns:
  - article_id:    primary key (str)
  - dataset:       "mind" | "ebnerd"
  - cleaned_text:  title + " " + abstract  (for BM25 — the confirmed required scope).
                   Body text stays available on the underlying interim article record
                   for the optional ablation later, but is NOT part of this default
                   cleaned-text feature.
  - category:      pass-through from interim
  - subcategory:   pass-through from interim
  - entities:      pass-through JSON string
  - embedding_ref: lazy pointer to its embedding row (never materialized here)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import polars as pl

from src.feature_store.store_backend import ParquetStore

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ArticleFeatureStore:
    """Query layer over article features backed by Parquet + DuckDB.

    Typical usage::

        store = ArticleFeatureStore("mind")
        row = store.get_article("N12345")
        bm25_df = store.get_articles_for_bm25()
    """

    FEATURE_COLUMNS = [
        "article_id",
        "dataset",
        "cleaned_text",
        "category",
        "subcategory",
        "entities",
        "embedding_ref",
    ]

    def __init__(self, dataset: str, processed_dir: Path | None = None) -> None:
        if processed_dir is None:
            processed_dir = _PROJECT_ROOT / "data" / "processed" / dataset
        self._path = processed_dir / "article_features.parquet"
        if not self._path.exists():
            raise FileNotFoundError(
                f"Article features not found at {self._path}. "
                f"Run `python -m src.feature_store.build_features --dataset {dataset}` first."
            )
        self._store = ParquetStore(self._path, table_alias="articles")
        self._dataset = dataset

    # Lookups

    def get_article(self, article_id: str) -> dict[str, Any] | None:
        """Return all feature columns for a single article, or ``None``."""
        return self._store.get_by_id("article_id", article_id)

    def get_articles_batch(
        self, article_ids: list[str], columns: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """Return features for a batch of article IDs."""
        return self._store.batch_get("article_id", article_ids, columns=columns)

    def get_articles_for_bm25(
        self, article_ids: list[str] | None = None
    ) -> pl.DataFrame:
        """Return ``(article_id, cleaned_text)`` for BM25 indexing.

        If *article_ids* is provided, only those articles are returned.
        Otherwise, all articles in the store are returned.
        """
        cols = ["article_id", "cleaned_text"]
        if article_ids is None:
            return self._store.get_dataframe(columns=cols)
        # Use batch lookup for selective read
        return self._store.get_dataframe(
            columns=cols,
            where=f"article_id IN ({','.join(['$' + str(i+1) for i in range(len(article_ids))])})",
            params=article_ids,
        )

    def get_article_metadata(self, article_id: str) -> dict[str, Any] | None:
        """Return category, entities, embedding_ref for an article."""
        return self._store.get_by_id(
            "article_id",
            article_id,
            columns=["article_id", "category", "subcategory", "entities", "embedding_ref"],
        )

    @property
    def row_count(self) -> int:
        return self._store.row_count()

    # Build step

    @staticmethod
    def build_features(
        dataset: str,
        interim_dir: Path | None = None,
        processed_dir: Path | None = None,
    ) -> Path:
        """Read interim articles and write article_features.parquet.

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

        articles_path = interim_dir / "articles.parquet"
        if not articles_path.exists():
            raise FileNotFoundError(
                f"Interim articles not found at {articles_path}. "
                f"Run parse_{dataset} first."
            )

        logger.info("Building article features for %s from %s", dataset, articles_path)
        df = pl.read_parquet(articles_path)
        logger.info("  Read %d articles", len(df))

        # Build cleaned_text = title + " " + abstract
        # Handle nulls: if abstract is null, cleaned_text = title only.
        # Body text stays available on the interim record but is NOT part of
        # cleaned_text — body is optional/extra-credit scope.
        df = df.with_columns(
            pl.when(pl.col("abstract").is_not_null())
            .then(pl.col("title") + pl.lit(" ") + pl.col("abstract"))
            .otherwise(pl.col("title"))
            .alias("cleaned_text")
        )

        # Select only the feature columns
        feature_df = df.select([
            "article_id",
            "dataset",
            "cleaned_text",
            "category",
            "subcategory",
            "entities",
            "embedding_ref",
        ])

        # Write to processed dir
        processed_dir.mkdir(parents=True, exist_ok=True)
        out_path = processed_dir / "article_features.parquet"
        feature_df.write_parquet(out_path)

        logger.info(
            "  Wrote article features: %d rows → %s (%.1f MB)",
            len(feature_df),
            out_path,
            out_path.stat().st_size / 1024**2,
        )
        return out_path
