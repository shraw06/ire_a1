"""Parse MIND TSV files into the unified schema (articles, behaviors, users).

Source files (per EDA_SUMMARY.md §10):
  - news.tsv — 8 columns, tab-separated, NO header:
      news_id | category | subcategory | title | abstract | url |
      title_entities (JSON) | abstract_entities (JSON)
  - behaviors.tsv — 5 columns, tab-separated, NO header:
      impression_id | user_id | time (M/D/YYYY h:mm:ss AM/PM) |
      history (space-separated news IDs) |
      impressions (space-separated NewsID-Label pairs)

Output: three Parquet files in data/interim/mind/:
  - articles.parquet
  - behaviors.parquet
  - users.parquet

Usage:
    python -m src.parsing.parse_mind [--split train] [--split dev]
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import polars as pl

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# MIND splits and their directory names
_SPLITS = {
    "train": "MINDsmall_train",
    "dev": "MINDsmall_dev",
}

# news.tsv column names (no header in the file)
_NEWS_COLUMNS = [
    "news_id",
    "category",
    "subcategory",
    "title",
    "abstract",
    "url",
    "title_entities",
    "abstract_entities",
]

# behaviors.tsv column names (no header in the file)
_BEHAVIOR_COLUMNS = [
    "impression_id",
    "user_id",
    "time",
    "history",
    "impressions",
]

# Explicit strptime format for MIND's timestamp string (EDA §2/§10):
# "M/D/YYYY h:mm:ss AM/PM" → e.g. "11/11/2019 9:05:58 AM"
# Do NOT rely on a generic/inferred datetime parser; this format is easy to
# mis-parse silently (e.g. day/month swap).
_MIND_TIME_FMT = "%m/%d/%Y %I:%M:%S %p"


# Entity parsing

def _parse_entity_json(raw: Optional[str]) -> list[dict]:
    """Parse a MIND entity JSON column into a list of {label, type, wikidata_id, confidence}.

    MIND entity format (per EDA §7):
      [{"Label": "...", "Type": "P", "WikidataId": "Q...", "Confidence": 1.0,
        "OccurrenceOffsets": [...], "SurfaceForms": [...]}]

    We keep only the fields relevant to the unified schema.
    """
    if not raw or raw.strip() in ("", "[]"):
        return []
    try:
        entities = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        logger.warning("Failed to parse entity JSON: %s", raw[:100])
        return []

    result = []
    for e in entities:
        result.append({
            "label": e.get("Label", ""),
            "type": e.get("Type", ""),
            "wikidata_id": e.get("WikidataId"),
            "confidence": e.get("Confidence"),
        })
    return result


# News/Articles parsing

def parse_mind_articles(raw_dir: Path, splits: list[str] | None = None) -> pl.DataFrame:
    """Parse MIND news.tsv into the unified articles schema.

    Body field:
      MIND's TSV only provides a URL for body text. MSN's article bodies were
      deliberately withheld from the dataset for licensing reasons (confirmed:
      "the full content body of MSN news articles are not made available for
      download, due to licensing structure").

      Body is confirmed OPTIONAL/extra-credit only per the assignment scope
      clarification (title + abstract is the required pipeline scope).

      → body = None, body_source = None for all MIND articles.
    """
    if splits is None:
        splits = list(_SPLITS.keys())

    frames = []
    for split in splits:
        split_dir = raw_dir / _SPLITS[split]
        news_path = split_dir / "news.tsv"
        if not news_path.exists():
            raise FileNotFoundError(f"MIND news.tsv not found at {news_path}")

        logger.info("Parsing MIND articles from %s", news_path)

        df = pl.read_csv(
            news_path,
            separator="\t",
            has_header=False,
            new_columns=_NEWS_COLUMNS,
            infer_schema_length=0,  # read everything as strings first
            truncate_ragged_lines=True,
            quote_char=None,  # MIND TSV has unescaped quotes inside fields
        )
        frames.append(df)

    # Combine and deduplicate (same article can appear in train + dev)
    combined = pl.concat(frames).unique(subset=["news_id"])
    logger.info("MIND articles combined: %d rows (after dedup across splits)", len(combined))

    # Parse entities from both title and abstract entity columns, merge
    def _merge_entities(row: dict) -> str:
        title_ents = _parse_entity_json(row.get("title_entities"))
        abstract_ents = _parse_entity_json(row.get("abstract_entities"))
        # Merge, dedup by (label, type)
        seen = set()
        merged = []
        for e in title_ents + abstract_ents:
            key = (e["label"], e["type"])
            if key not in seen:
                seen.add(key)
                merged.append(e)
        return json.dumps(merged)

    # Build unified DataFrame
    rows = combined.to_dicts()
    unified_rows = []
    for r in rows:
        unified_rows.append({
            "article_id": r["news_id"],
            "dataset": "mind",
            "title": r["title"] or "",
            # abstract: ~5% missing per EDA §6 - leave as null, do not impute
            "abstract": r["abstract"] if r["abstract"] else None,
            # body: null by default - licensing restriction, see docstring above
            "body": None,
            "body_source": None,
            "category": r["category"] if r["category"] else None,
            "subcategory": r["subcategory"] if r["subcategory"] else None,
            "entities": _merge_entities(r),
            # MIND TSV has no published_time column
            "published_at": None,
            # No embeddings at this stage
            "embedding_ref": None,
        })

    result = pl.DataFrame(unified_rows, schema={
        "article_id": pl.Utf8,
        "dataset": pl.Utf8,
        "title": pl.Utf8,
        "abstract": pl.Utf8,
        "body": pl.Utf8,
        "body_source": pl.Utf8,
        "category": pl.Utf8,
        "subcategory": pl.Utf8,
        "entities": pl.Utf8,
        "published_at": pl.Datetime("us"),
        "embedding_ref": pl.Utf8,
    })

    logger.info("MIND articles parsed: %d rows", len(result))
    return result


# Behaviors parsing

def parse_mind_behaviors(raw_dir: Path, splits: list[str] | None = None) -> pl.DataFrame:
    """Parse MIND behaviors.tsv into the unified behaviors schema.

    clicked_history:
      MIND's history column contains space-separated news IDs with NO per-item
      timestamps available.

      ASSUMPTION (cannot independently verify - MIND provides no per-article
      published_time to cross-check against): this list represents the user's
      state AS OF this specific impression (i.e. a pre-trimmed snapshot, not a
      full lifetime history needing further filtering).

      → Each entry stored as {article_id, clicked_at: null}.

      DOCUMENTED LIMITATION: we cannot verify whether this assumption holds.
      This should be carried into the design note for the assignment.

    candidates / labels:
      MIND's impressions column is a space-separated string of "NewsID-Label"
      pairs (e.g. "N123-1 N456-0"). Split on whitespace, then split each
      token on "-" to get candidate id and binary label.
    """
    if splits is None:
        splits = list(_SPLITS.keys())

    frames = []
    for split in splits:
        split_dir = raw_dir / _SPLITS[split]
        beh_path = split_dir / "behaviors.tsv"
        if not beh_path.exists():
            raise FileNotFoundError(f"MIND behaviors.tsv not found at {beh_path}")

        logger.info("Parsing MIND behaviors from %s", beh_path)

        df = pl.read_csv(
            beh_path,
            separator="\t",
            has_header=False,
            new_columns=_BEHAVIOR_COLUMNS,
            infer_schema_length=0,  # read everything as strings
            truncate_ragged_lines=True,
            quote_char=None,  # consistency with news.tsv parsing
        )

        # Add split marker for user derivation later if needed
        df = df.with_columns(pl.lit(split).alias("_split"))
        frames.append(df)

    combined = pl.concat(frames)
    logger.info("MIND behaviors combined: %d rows", len(combined))

    rows = combined.to_dicts()
    unified_rows = []
    for r in rows:
        # Parse timestamp with explicit format
        ts_str = r["time"]
        try:
            timestamp = datetime.strptime(ts_str, _MIND_TIME_FMT)
        except (ValueError, TypeError):
            logger.warning("Failed to parse MIND timestamp: %s", ts_str)
            # Fallback - should not happen with valid data
            timestamp = datetime(2019, 1, 1)

        # Parse history - space-separated news IDs, no timestamps
        # ASSUMPTION: pre-trimmed snapshot (see docstring above)
        history_str = r.get("history") or ""
        history_ids = history_str.split() if history_str.strip() else []
        clicked_history = [
            {"article_id": aid, "clicked_at": None}
            for aid in history_ids
        ]

        # Parse impressions - space-separated "NewsID-Label" pairs
        impressions_str = r.get("impressions") or ""
        candidates = []
        labels = []
        if impressions_str.strip():
            for token in impressions_str.split():
                # Split on last '-' to handle news IDs that might contain hyphens
                # (though MIND IDs are typically N12345 format)
                parts = token.rsplit("-", 1)
                if len(parts) == 2:
                    candidates.append(parts[0])
                    labels.append(int(parts[1]))
                else:
                    logger.warning("Malformed impression token: %s", token)

        unified_rows.append({
            "impression_id": str(r["impression_id"]),
            "dataset": "mind",
            "user_id": r["user_id"],
            "timestamp": timestamp,
            "clicked_history": json.dumps(clicked_history),
            "candidates": json.dumps(candidates),
            "labels": json.dumps(labels),
        })

    result = pl.DataFrame(unified_rows, schema={
        "impression_id": pl.Utf8,
        "dataset": pl.Utf8,
        "user_id": pl.Utf8,
        "timestamp": pl.Datetime("us"),
        "clicked_history": pl.Utf8,
        "candidates": pl.Utf8,
        "labels": pl.Utf8,
    })

    logger.info("MIND behaviors parsed: %d rows", len(result))
    return result


# User derivation

def derive_mind_users(behaviors: pl.DataFrame) -> pl.DataFrame:
    """Derive per-user summary from parsed behaviors.

    Aggregates across all impressions to build:
      - history_article_ids: union of all history entries across impressions
      - history_len: count of unique history article IDs
      - last_active_at: max timestamp across impressions
    """
    rows = behaviors.to_dicts()

    user_data: dict[str, dict] = {}
    for r in rows:
        uid = r["user_id"]
        ts = r["timestamp"]
        history = json.loads(r["clicked_history"])
        article_ids = [h["article_id"] for h in history]

        if uid not in user_data:
            user_data[uid] = {
                "user_id": uid,
                "dataset": "mind",
                "article_ids": set(),
                "last_active_at": ts,
            }

        user_data[uid]["article_ids"].update(article_ids)
        if ts and (user_data[uid]["last_active_at"] is None or ts > user_data[uid]["last_active_at"]):
            user_data[uid]["last_active_at"] = ts

    unified_rows = []
    for ud in user_data.values():
        aids = sorted(ud["article_ids"])
        unified_rows.append({
            "user_id": ud["user_id"],
            "dataset": "mind",
            "history_article_ids": json.dumps(aids),
            "history_len": len(aids),
            "last_active_at": ud["last_active_at"],
        })

    result = pl.DataFrame(unified_rows, schema={
        "user_id": pl.Utf8,
        "dataset": pl.Utf8,
        "history_article_ids": pl.Utf8,
        "history_len": pl.Int64,
        "last_active_at": pl.Datetime("us"),
    })

    logger.info("MIND users derived: %d users", len(result))
    return result


#  Write interim output

def write_interim(articles: pl.DataFrame, behaviors: pl.DataFrame,
                  users: pl.DataFrame, interim_dir: Path) -> None:
    """Write parsed DataFrames to Parquet in the interim directory."""
    interim_dir.mkdir(parents=True, exist_ok=True)

    articles_path = interim_dir / "articles.parquet"
    behaviors_path = interim_dir / "behaviors.parquet"
    users_path = interim_dir / "users.parquet"

    articles.write_parquet(articles_path)
    behaviors.write_parquet(behaviors_path)
    users.write_parquet(users_path)

    logger.info("MIND interim written to %s", interim_dir)
    logger.info("  articles:  %d rows → %s", len(articles), articles_path)
    logger.info("  behaviors: %d rows → %s", len(behaviors), behaviors_path)
    logger.info("  users:     %d rows → %s", len(users), users_path)


# Main

def main(splits: list[str] | None = None) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Parse MIND and write to interim. Returns (articles, behaviors, users)."""
    raw_dir = _PROJECT_ROOT / "data" / "raw" / "mind"
    interim_dir = _PROJECT_ROOT / "data" / "interim" / "mind"

    articles = parse_mind_articles(raw_dir, splits=splits)
    behaviors = parse_mind_behaviors(raw_dir, splits=splits)
    users = derive_mind_users(behaviors)
    write_interim(articles, behaviors, users, interim_dir)

    return articles, behaviors, users


if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(description="Parse MIND TSV → unified schema Parquet")
    parser.add_argument(
        "--split", action="append", choices=["train", "dev"],
        help="Splits to parse (default: all). Can be specified multiple times.",
    )
    args = parser.parse_args()

    splits = args.split if args.split else None
    articles, behaviors, users = main(splits=splits)

    print(f"\n{'='*60}")
    print(f"MIND parsing complete.")
    print(f"  Articles:  {len(articles):>8,d} rows")
    print(f"  Behaviors: {len(behaviors):>8,d} rows")
    print(f"  Users:     {len(users):>8,d} rows")
    print(f"{'='*60}")

    sys.exit(0)
