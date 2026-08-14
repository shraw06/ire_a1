"""Parse EB-NeRD Parquet files into the unified schema (articles, behaviors, users).

Source files (per EDA_SUMMARY.md §10):
  - articles.parquet — 21 columns (shared across splits):
      article_id, title, subtitle, last_modified_time, premium, body,
      published_time, image_ids, article_type, url, ner_clusters,
      entity_groups, topics, category, subcategory, category_str,
      total_inviews, total_pageviews, total_read_time, sentiment_score,
      sentiment_label
  - behaviors.parquet — 17 columns (per split):
      impression_id, article_id, impression_time, read_time, scroll_percentage,
      device_type, article_ids_inview (List[Int32]),
      article_ids_clicked (List[Int32]), user_id, is_sso_user, gender,
      postcode, age, is_subscriber, session_id, next_read_time,
      next_scroll_percentage
  - history.parquet — 5 columns (per split):
      user_id, impression_time_fixed (List[Datetime]),
      scroll_percentage_fixed (List[Float32]),
      article_id_fixed (List[Int32]), read_time_fixed (List[Float32])

Output: three Parquet files in data/interim/ebnerd/:
  - articles.parquet
  - behaviors.parquet
  - users.parquet

Usage:
    python -m src.parsing.parse_ebnerd [--split train] [--split validation]
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

# EB-NeRD splits
_SPLITS = ["train", "validation"]


# Entity parsing

def _parse_ebnerd_entities(ner_clusters: Optional[list], entity_groups: Optional[list]) -> str:
    """Parse EB-NeRD ner_clusters and entity_groups into unified entity format.

    Unlike MIND, EB-NeRD entities do NOT carry WikidataId or confidence scores.
    We populate {label, type} and leave wikidata_id/confidence as null.
    """
    entities = []
    seen = set()

    # ner_clusters is List[String] - each string is an entity label
    if ner_clusters:
        for label in ner_clusters:
            if label and isinstance(label, str):
                key = (label, "NER")
                if key not in seen:
                    seen.add(key)
                    entities.append({
                        "label": label,
                        "type": "NER",
                        "wikidata_id": None,
                        "confidence": None,
                    })

    # entity_groups is List[String] - each string is an entity group/type label
    if entity_groups:
        for label in entity_groups:
            if label and isinstance(label, str):
                key = (label, "GROUP")
                if key not in seen:
                    seen.add(key)
                    entities.append({
                        "label": label,
                        "type": "GROUP",
                        "wikidata_id": None,
                        "confidence": None,
                    })

    return json.dumps(entities)


# Articles parsing

def parse_ebnerd_articles(raw_dir: Path) -> pl.DataFrame:
    """Parse EB-NeRD articles.parquet into the unified articles schema.

    Key mappings:
      - abstract ← subtitle (EB-NeRD has no 'abstract' column; 'subtitle' is the
        closest functional equivalent: short teaser text alongside the title).
      - body ← body column directly, body_source = "native" (natively available,
        licensing-clean). ~8% missing per EDA §6.
      - category ← category_str (the human-readable category name)
      - subcategory ← subcategory list (List[Int16]) → joined as comma-separated string
      - embedding_ref ← null (demo bundle has no embedding files — EDA §8)
      - published_at ← published_time
    """
    articles_path = raw_dir / "articles.parquet"
    if not articles_path.exists():
        raise FileNotFoundError(f"EB-NeRD articles.parquet not found at {articles_path}")

    logger.info("Parsing EB-NeRD articles from %s", articles_path)
    df = pl.read_parquet(articles_path)
    logger.info("EB-NeRD articles raw: %d rows, %d columns", len(df), len(df.columns))

    rows = df.to_dicts()
    unified_rows = []
    for r in rows:
        # subcategory: List[Int16] → join as comma-separated string
        subcat = r.get("subcategory")
        subcat_str = None
        if subcat and isinstance(subcat, list) and len(subcat) > 0:
            subcat_str = ",".join(str(s) for s in subcat)

        entities_json = _parse_ebnerd_entities(
            r.get("ner_clusters"),
            r.get("entity_groups"),
        )

        # published_time: already a native Parquet Datetime - no string parsing needed
        published_at = r.get("published_time")

        unified_rows.append({
            "article_id": str(r["article_id"]),
            "dataset": "ebnerd",
            "title": r.get("title") or "",
            # abstract ← subtitle: the closest functional equivalent
            # Map explicitly; do not leave abstract null just because the column
            # name doesn't match.
            "abstract": r.get("subtitle") if r.get("subtitle") else None,
            # body: natively available in EB-NeRD, licensing-clean
            # Still OPTIONAL for the required pipeline (title+abstract scope),
            # but body_source="native" distinguishes from MIND's null
            "body": r.get("body") if r.get("body") else None,
            "body_source": "native" if r.get("body") else None,
            "category": r.get("category_str") if r.get("category_str") else None,
            "subcategory": subcat_str,
            "entities": entities_json,
            "published_at": published_at,
            # embedding_ref: null - demo bundle has no embeddings (EDA §8).
            # Do not error or warn; this is expected at this stage.
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

    logger.info("EB-NeRD articles parsed: %d rows", len(result))
    return result


# Behaviors parsing

def _build_history_lookup(history_path: Path) -> dict[int, list[dict]]:
    """Build a user_id → clicked_history lookup from history.parquet.

    history.parquet contains per-user FULL lifetime click history with parallel
    list columns:
      - article_id_fixed (List[Int32])
      - impression_time_fixed (List[Datetime])

    Unlike MIND, this is NOT pre-trimmed - it genuinely needs later
    as-of-timestamp filtering (the real leakage-prevention step in the feature
    store part).

    Returns: {user_id: [{article_id: str, clicked_at: ISO datetime str}, ...]}
    """
    logger.info("Building history lookup from %s", history_path)
    df = pl.read_parquet(history_path)
    logger.info("  history.parquet: %d users", len(df))

    lookup: dict[int, list[dict]] = {}
    for row in df.to_dicts():
        uid = row["user_id"]
        article_ids = row.get("article_id_fixed") or []
        timestamps = row.get("impression_time_fixed") or []

        history_entries = []
        for i, aid in enumerate(article_ids):
            ts = timestamps[i] if i < len(timestamps) else None
            clicked_at_str = ts.isoformat() if isinstance(ts, datetime) else None
            history_entries.append({
                "article_id": str(aid),
                "clicked_at": clicked_at_str,
            })

        lookup[uid] = history_entries

    return lookup


def parse_ebnerd_behaviors(
    raw_dir: Path, splits: list[str] | None = None
) -> tuple[pl.DataFrame, dict[str, int]]:
    """Parse EB-NeRD behaviors.parquet + history.parquet into unified schema.

    clicked_history:
      NOT inline in behaviors.parquet (unlike MIND). Built by joining
      history.parquet onto behaviors.parquet on user_id - each history entry
      keeps its real click timestamp ({article_id, clicked_at: impression_time_fixed[i]}).

      Unlike MIND, this is the user's FULL lifetime history, not pre-trimmed.

    candidates / labels:
      - candidates = article_ids_inview (List[Int32])
      - labels = 1 if article_id in article_ids_clicked else 0, for each candidate

    Returns:
      (behaviors_df, join_stats) where join_stats maps split name →
      original row count, for validation that the join didn't drop rows.
    """
    if splits is None:
        splits = list(_SPLITS)

    all_frames = []
    join_stats: dict[str, int] = {}

    for split in splits:
        split_dir = raw_dir / split
        beh_path = split_dir / "behaviors.parquet"
        hist_path = split_dir / "history.parquet"

        if not beh_path.exists():
            raise FileNotFoundError(f"EB-NeRD behaviors.parquet not found at {beh_path}")
        if not hist_path.exists():
            raise FileNotFoundError(f"EB-NeRD history.parquet not found at {hist_path}")

        logger.info("Parsing EB-NeRD behaviors from %s", beh_path)
        beh_df = pl.read_parquet(beh_path)
        original_count = len(beh_df)
        join_stats[split] = original_count
        logger.info("  behaviors.parquet (%s): %d rows", split, original_count)

        # Build history lookup for this split
        history_lookup = _build_history_lookup(hist_path)

        rows = beh_df.to_dicts()
        for r in rows:
            uid = r["user_id"]
            # impression_time: already a native Parquet Datetime - confirm dtype
            timestamp = r["impression_time"]
            if not isinstance(timestamp, datetime):
                logger.warning("EB-NeRD impression_time is not datetime: %s (type %s)",
                               timestamp, type(timestamp))

            # History: join from history.parquet on user_id
            clicked_history = history_lookup.get(uid, [])

            # Candidates: article_ids_inview (List[Int32])
            inview = r.get("article_ids_inview") or []
            candidates = [str(aid) for aid in inview]

            # Labels: 1 if in clicked set, else 0
            clicked_set = set(r.get("article_ids_clicked") or [])
            labels = [1 if aid in clicked_set else 0 for aid in inview]

            all_frames.append({
                "impression_id": str(r["impression_id"]),
                "dataset": "ebnerd",
                "user_id": str(uid),
                "timestamp": timestamp,
                "clicked_history": json.dumps(clicked_history),
                "candidates": json.dumps(candidates),
                "labels": json.dumps(labels),
            })

    result = pl.DataFrame(all_frames, schema={
        "impression_id": pl.Utf8,
        "dataset": pl.Utf8,
        "user_id": pl.Utf8,
        "timestamp": pl.Datetime("us"),
        "clicked_history": pl.Utf8,
        "candidates": pl.Utf8,
        "labels": pl.Utf8,
    })

    logger.info("EB-NeRD behaviors parsed: %d rows", len(result))
    return result, join_stats


# User derivation

def derive_ebnerd_users(behaviors: pl.DataFrame) -> pl.DataFrame:
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
                "dataset": "ebnerd",
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
            "dataset": "ebnerd",
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

    logger.info("EB-NeRD users derived: %d users", len(result))
    return result


# Write interim output

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

    logger.info("EB-NeRD interim written to %s", interim_dir)
    logger.info("  articles:  %d rows → %s", len(articles), articles_path)
    logger.info("  behaviors: %d rows → %s", len(behaviors), behaviors_path)
    logger.info("  users:     %d rows → %s", len(users), users_path)


# Main

def main(splits: list[str] | None = None) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, dict[str, int]]:
    """Parse EB-NeRD and write to interim. Returns (articles, behaviors, users, join_stats)."""
    raw_dir = _PROJECT_ROOT / "data" / "raw" / "ebnerd"
    interim_dir = _PROJECT_ROOT / "data" / "interim" / "ebnerd"

    articles = parse_ebnerd_articles(raw_dir)
    behaviors, join_stats = parse_ebnerd_behaviors(raw_dir, splits=splits)
    users = derive_ebnerd_users(behaviors)
    write_interim(articles, behaviors, users, interim_dir)

    return articles, behaviors, users, join_stats


if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(description="Parse EB-NeRD Parquet → unified schema Parquet")
    parser.add_argument(
        "--split", action="append", choices=["train", "validation"],
        help="Splits to parse (default: all). Can be specified multiple times.",
    )
    args = parser.parse_args()

    splits = args.split if args.split else None
    articles, behaviors, users, join_stats = main(splits=splits)

    print(f"\n{'='*60}")
    print(f"EB-NeRD parsing complete.")
    print(f"  Articles:  {len(articles):>8,d} rows")
    print(f"  Behaviors: {len(behaviors):>8,d} rows")
    print(f"  Users:     {len(users):>8,d} rows")
    print(f"\nHistory join stats (should match original row counts):")
    for split_name, orig_count in join_stats.items():
        print(f"  {split_name}: original = {orig_count:,d}")
    joined_total = len(behaviors)
    original_total = sum(join_stats.values())
    print(f"  TOTAL: original = {original_total:,d}, joined = {joined_total:,d}")
    if joined_total == original_total:
        print(f"  ✓ Row count preserved — join is correct")
    else:
        print(f"  ✗ ROW COUNT MISMATCH — investigate!")
    print(f"{'='*60}")

    sys.exit(0)
