"""Pydantic / dataclass definitions of the unified schema shared by MIND and EB-NeRD.

The goal is ONE shared schema so all downstream code never needs to branch on
which dataset it's processing.  Two real structural differences are resolved
explicitly by the parsing adapters:

  1. MIND has no native body text (URL only, licensing restriction) → body=None.
     EB-NeRD has body natively → body_source="native".
  2. MIND history entries carry no per-item timestamps → clicked_at=None.
     EB-NeRD history carries real timestamps via history.parquet join.

See EDA_SUMMARY.md §10 for column-level source mappings.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


# Sub-models

class Entity(BaseModel):
    """Named entity extracted from article title or abstract.

    MIND entities carry WikidataId + confidence; EB-NeRD entities do not.
    Both wikidata_id and confidence are therefore optional.
    """

    label: str
    type: str
    wikidata_id: Optional[str] = None
    confidence: Optional[float] = None


class HistoryEntry(BaseModel):
    """A single article in a user's click history.

    For MIND, clicked_at is always None (no per-item timestamps available).
    For EB-NeRD, clicked_at is the real impression_time_fixed from history.parquet.
    """

    article_id: str
    clicked_at: Optional[datetime] = None


# Core tables

class Article(BaseModel):
    """Unified article record.

    Field mapping (confirmed from EDA_SUMMARY.md §10):
      - article_id:    MIND news_id (str) | EB-NeRD article_id (cast to str)
      - dataset:       "mind" | "ebnerd"
      - title:         direct from both datasets
      - abstract:      MIND abstract | EB-NeRD subtitle (functional equivalent)
      - body:          None for MIND (licensing; see docstring) | EB-NeRD body
      - body_source:   None for MIND | "native" for EB-NeRD
      - category:      direct from both (EB-NeRD uses category_str)
      - subcategory:   direct from MIND | EB-NeRD subcategory list → joined str
      - entities:      parsed from JSON (MIND) / list cols (EB-NeRD)
      - published_at:  None for MIND (not in TSV) | EB-NeRD published_time
      - embedding_ref: None for both at this stage (EB-NeRD demo has no embeddings)
    """

    article_id: str
    dataset: Literal["mind", "ebnerd"]
    title: str
    abstract: Optional[str] = None
    body: Optional[str] = None
    body_source: Optional[Literal["native", "scraped"]] = None
    category: Optional[str] = None
    subcategory: Optional[str] = None
    entities: list[Entity] = Field(default_factory=list)
    published_at: Optional[datetime] = None
    embedding_ref: Optional[str] = None


class Behavior(BaseModel):
    """Unified impression / behavior record.

    Field mapping (confirmed from EDA_SUMMARY.md §10):
      - impression_id: direct (cast to str for both)
      - dataset:       "mind" | "ebnerd"
      - user_id:       direct (cast to str for both)
      - timestamp:     MIND time (parsed M/D/YYYY h:mm:ss AM/PM) | EB-NeRD impression_time
      - clicked_history: MIND history col (no timestamps) | EB-NeRD via history.parquet join
      - candidates:    MIND impressions (split) | EB-NeRD article_ids_inview
      - labels:        MIND impressions (split) | EB-NeRD derived from article_ids_clicked
    """

    impression_id: str
    dataset: Literal["mind", "ebnerd"]
    user_id: str
    timestamp: datetime
    clicked_history: list[HistoryEntry] = Field(default_factory=list)
    candidates: list[str] = Field(default_factory=list)
    labels: list[int] = Field(default_factory=list)


class User(BaseModel):
    """Derived user record - aggregated from behaviors.

    Not a raw table; derived after parsing to provide a user-level view.
    """

    user_id: str
    dataset: Literal["mind", "ebnerd"]
    history_article_ids: list[str] = Field(default_factory=list)
    history_len: int = 0
    last_active_at: Optional[datetime] = None


# Polars schema constants
# These define the Parquet column types for the interim tables written by each
# adapter.  Downstream code reads these Parquets, not the Pydantic models
# directly (Pydantic is for validation / tests only).

import polars as pl

ARTICLES_SCHEMA = {
    "article_id": pl.Utf8,
    "dataset": pl.Utf8,
    "title": pl.Utf8,
    "abstract": pl.Utf8,
    "body": pl.Utf8,
    "body_source": pl.Utf8,
    "category": pl.Utf8,
    "subcategory": pl.Utf8,
    # entities stored as JSON string (list of dicts) — deserialized on demand
    "entities": pl.Utf8,
    "published_at": pl.Datetime("us"),
    "embedding_ref": pl.Utf8,
}

BEHAVIORS_SCHEMA = {
    "impression_id": pl.Utf8,
    "dataset": pl.Utf8,
    "user_id": pl.Utf8,
    "timestamp": pl.Datetime("us"),
    # clicked_history stored as JSON string (list of {article_id, clicked_at})
    "clicked_history": pl.Utf8,
    # candidates stored as JSON string (list of article_id strings)
    "candidates": pl.Utf8,
    # labels stored as JSON string (list of ints)
    "labels": pl.Utf8,
    # split: added by temporal_split.py - "train", "val", or "test".
    # Optional before splitting; present after running the split step.
    "split": pl.Utf8,
}

USERS_SCHEMA = {
    "user_id": pl.Utf8,
    "dataset": pl.Utf8,
    # history_article_ids stored as JSON string (list of article_id strings)
    "history_article_ids": pl.Utf8,
    "history_len": pl.Int64,
    "last_active_at": pl.Datetime("us"),
}
