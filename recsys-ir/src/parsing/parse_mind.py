"""Memory-safe MIND parser for small and large bundles.

The large behavior files contain millions of rows. The large-scale path is
streamed in bounded batches and written directly to Parquet; it never builds a
Python list containing the complete behavior table.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import polars as pl
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

_MIND_SPLITS = {
    "small": {"train": "MINDsmall_train", "dev": "MINDsmall_dev"},
    "large": {"train": "MINDlarge_train", "dev": "MINDlarge_dev"},
}
_NEWS_COLUMNS = [
    "news_id", "category", "subcategory", "title", "abstract", "url",
    "title_entities", "abstract_entities",
]
_BEHAVIOR_COLUMNS = ["impression_id", "user_id", "time", "history", "impressions"]
_MIND_TIME_FMT = "%m/%d/%Y %I:%M:%S %p"
_OUTPUT_SCHEMA = {
    "impression_id": pl.Utf8,
    "dataset": pl.Utf8,
    "user_id": pl.Utf8,
    "timestamp": pl.Datetime("us"),
    "clicked_history": pl.Utf8,
    "candidates": pl.Utf8,
    "labels": pl.Utf8,
}


def _raw_dir_for_scale(scale: str) -> Path:
    return _PROJECT_ROOT / "data" / "raw" / "mind"


def _split_map(scale: str) -> dict[str, str]:
    if scale not in _MIND_SPLITS:
        raise ValueError(f"Unknown MIND scale: {scale}")
    return _MIND_SPLITS[scale]


def _parse_entity_json(raw: Optional[str]) -> list[dict]:
    if not raw or raw.strip() in ("", "[]"):
        return []
    try:
        entities = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        logger.warning("Failed to parse entity JSON: %s", str(raw)[:100])
        return []
    return [
        {
            "label": e.get("Label", ""),
            "type": e.get("Type", ""),
            "wikidata_id": e.get("WikidataId"),
            "confidence": e.get("Confidence"),
        }
        for e in entities
    ]


def _merge_entities(row: dict) -> str:
    seen = set()
    merged = []
    for ent in _parse_entity_json(row.get("title_entities")) + _parse_entity_json(row.get("abstract_entities")):
        key = (ent["label"], ent["type"])
        if key not in seen:
            seen.add(key)
            merged.append(ent)
    return json.dumps(merged, separators=(",", ":"))


def parse_mind_articles(raw_dir: Path, splits: list[str] | None = None, scale: str = "small") -> pl.DataFrame:
    split_map = _split_map(scale)
    splits = splits or list(split_map)
    frames = []
    for split in splits:
        path = raw_dir / split_map[split] / "news.tsv"
        if not path.exists():
            raise FileNotFoundError(path)
        frames.append(
            pl.read_csv(
                path,
                separator="\t",
                has_header=False,
                new_columns=_NEWS_COLUMNS,
                infer_schema_length=0,
                truncate_ragged_lines=True,
                quote_char=None,
            )
        )
    combined = pl.concat(frames).unique(subset=["news_id"])
    rows = [
        {
            "article_id": r["news_id"],
            "dataset": "mind",
            "title": r["title"] or "",
            "abstract": r["abstract"] if r["abstract"] else None,
            "body": None,
            "body_source": None,
            "category": r["category"] if r["category"] else None,
            "subcategory": r["subcategory"] if r["subcategory"] else None,
            "entities": _merge_entities(r),
            "published_at": None,
            "embedding_ref": None,
        }
        for r in combined.to_dicts()
    ]
    result = pl.DataFrame(rows, schema={
        "article_id": pl.Utf8, "dataset": pl.Utf8, "title": pl.Utf8,
        "abstract": pl.Utf8, "body": pl.Utf8, "body_source": pl.Utf8,
        "category": pl.Utf8, "subcategory": pl.Utf8, "entities": pl.Utf8,
        "published_at": pl.Datetime("us"), "embedding_ref": pl.Utf8,
    })
    logger.info("MIND articles parsed: %d rows", result.height)
    return result


def _parse_behavior_batch(df: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for r in df.to_dicts():
        try:
            timestamp = datetime.strptime(r["time"], _MIND_TIME_FMT)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid MIND timestamp {r.get('time')!r}") from exc

        history = r.get("history") or ""
        clicked_history = [
            {"article_id": aid, "clicked_at": None}
            for aid in history.split()
            if aid
        ]

        candidates, labels = [], []
        impressions = r.get("impressions") or ""
        for token in impressions.split():
            article_id, sep, label = token.rpartition("-")
            if not sep or not article_id:
                logger.warning("Malformed MIND impression token: %s", token)
                continue
            try:
                label_int = int(label)
            except ValueError:
                logger.warning("Malformed MIND label token: %s", token)
                continue
            candidates.append(article_id)
            labels.append(label_int)

        rows.append({
            "impression_id": str(r["impression_id"]),
            "dataset": "mind",
            "user_id": str(r["user_id"]),
            "timestamp": timestamp,
            "clicked_history": json.dumps(clicked_history, separators=(",", ":")),
            "candidates": json.dumps(candidates, separators=(",", ":")),
            "labels": json.dumps(labels, separators=(",", ":")),
        })
    return pl.DataFrame(rows, schema=_OUTPUT_SCHEMA)


def _write_polars_batches(batches, output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    total = 0
    try:
        for batch in batches:
            if batch.height == 0:
                continue
            table = batch.to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
            writer.write_table(table)
            total += batch.height
            if total % 500_000 < batch.height:
                logger.info(f"  wrote {total:,} behavior rows")
    finally:
        if writer is not None:
            writer.close()
    return total


def _stream_mind_behaviors(
    raw_dir: Path,
    splits: list[str],
    output_path: Path,
    batch_size: int = 50_000,
    scale: str = "large",
) -> int:
    split_map = _split_map(scale)

    def batches():
        for split in splits:
            path = raw_dir / split_map[split] / "behaviors.tsv"
            if not path.exists():
                raise FileNotFoundError(path)
            logger.info("Streaming MIND behaviors from %s (batch=%d)", path, batch_size)
            reader = pl.read_csv_batched(
                path,
                separator="\t",
                has_header=False,
                new_columns=_BEHAVIOR_COLUMNS,
                infer_schema_length=0,
                truncate_ragged_lines=True,
                quote_char=None,
                batch_size=batch_size,
                low_memory=True,
                rechunk=False,
            )
            while True:
                chunks = reader.next_batches(1)
                if not chunks:
                    break
                yield _parse_behavior_batch(chunks[0])

    return _write_polars_batches(batches(), output_path)


def parse_mind_behaviors(raw_dir: Path, splits: list[str] | None = None, scale: str = "small") -> pl.DataFrame:
    """Compatibility API for small/test-sized runs."""
    split_map = _split_map(scale)
    splits = splits or list(split_map)
    frames = []
    for split in splits:
        df = pl.read_csv(
            raw_dir / split_map[split] / "behaviors.tsv",
            separator="\t", has_header=False, new_columns=_BEHAVIOR_COLUMNS,
            infer_schema_length=0, truncate_ragged_lines=True, quote_char=None,
        )
        frames.append(_parse_behavior_batch(df))
    return pl.concat(frames)


def derive_mind_users(behaviors: pl.DataFrame) -> pl.DataFrame:
    user_data: dict[str, dict] = {}
    for row in behaviors.iter_rows(named=True):
        uid = row["user_id"]
        history = json.loads(row["clicked_history"])
        state = user_data.setdefault(uid, {
            "user_id": uid, "dataset": "mind", "article_ids": set(), "last_active_at": row["timestamp"]
        })
        state["article_ids"].update(h["article_id"] for h in history)
        state["last_active_at"] = max(state["last_active_at"], row["timestamp"])

    rows = []
    for state in user_data.values():
        ids = sorted(state["article_ids"])
        rows.append({
            "user_id": state["user_id"],
            "dataset": "mind",
            "history_article_ids": json.dumps(ids),
            "history_len": len(ids),
            "last_active_at": state["last_active_at"],
        })
    return pl.DataFrame(rows, schema={
        "user_id": pl.Utf8, "dataset": pl.Utf8,
        "history_article_ids": pl.Utf8, "history_len": pl.Int64,
        "last_active_at": pl.Datetime("us"),
    })


def _derive_large_users(behaviors_path: Path, users_path: Path) -> None:
    """Build one-row-per-user summary through DuckDB + streamed Arrow batches."""
    import duckdb
    tmp = users_path.with_suffix(".summary.parquet")
    con = duckdb.connect()
    try:
        behaviors_sql = str(behaviors_path).replace("'", "''")
        tmp_sql = str(tmp).replace("'", "''")

        con.execute(
            f"""
            COPY (
                SELECT user_id,
                    max(timestamp) AS last_active_at,
                    arg_max(clicked_history, timestamp) AS latest_history
                FROM read_parquet('{behaviors_sql}')
                GROUP BY user_id
                ORDER BY user_id
            ) TO '{tmp_sql}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
    finally:
        con.close()

    import pyarrow.parquet as pq
    reader = pq.ParquetFile(tmp)
    writer = None
    total = 0
    try:
        for batch in reader.iter_batches(batch_size=25_000, columns=["user_id", "last_active_at", "latest_history"]):
            users, datasets, histories, lengths, lasts = [], [], [], [], []
            for uid, ts, hist_json in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist(), batch.column(2).to_pylist()):
                history = json.loads(hist_json or "[]")
                ids = sorted({h.get("article_id") for h in history if h.get("article_id")})
                users.append(str(uid))
                datasets.append("mind")
                histories.append(json.dumps(ids, separators=(",", ":")))
                lengths.append(len(ids))
                lasts.append(ts)
            df = pl.DataFrame({
                "user_id": users, "dataset": datasets,
                "history_article_ids": histories, "history_len": lengths,
                "last_active_at": lasts,
            })
            table = df.to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(users_path, table.schema, compression="zstd")
            writer.write_table(table)
            total += df.height
    finally:
        if writer is not None:
            writer.close()
        tmp.unlink(missing_ok=True)
    logger.info(f"MIND large user summary written: {total:,} users")


def write_interim(articles: pl.DataFrame, behaviors: pl.DataFrame, users: pl.DataFrame, interim_dir: Path) -> None:
    interim_dir.mkdir(parents=True, exist_ok=True)
    articles.write_parquet(interim_dir / "articles.parquet", compression="zstd")
    behaviors.write_parquet(interim_dir / "behaviors.parquet", compression="zstd")
    users.write_parquet(interim_dir / "users.parquet", compression="zstd")


def main(splits: list[str] | None = None, scale: str = "small"):
    from src.common.paths import interim_dir
    raw_dir = _raw_dir_for_scale(scale)
    split_map = _split_map(scale)
    splits = splits or list(split_map)
    out_dir = interim_dir("mind", scale)
    out_dir.mkdir(parents=True, exist_ok=True)
    if scale == "large":
        (out_dir / ".parse_complete").unlink(missing_ok=True)

    articles = parse_mind_articles(raw_dir, splits=splits, scale=scale)
    articles.write_parquet(out_dir / "articles.parquet", compression="zstd")

    if scale == "large":
        behavior_count = _stream_mind_behaviors(
            raw_dir, splits, out_dir / "behaviors.parquet", batch_size=50_000, scale=scale
        )
        logger.info(f"MIND-large behaviors written: {behavior_count:,} rows")
        _derive_large_users(out_dir / "behaviors.parquet", out_dir / "users.parquet")
        (out_dir / ".parse_complete").write_text("ok\n")
        return articles, None, None

    behaviors = parse_mind_behaviors(raw_dir, splits=splits, scale=scale)
    users = derive_mind_users(behaviors)
    write_interim(articles, behaviors, users, out_dir)
    return articles, behaviors, users


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Parse MIND into unified Parquet")
    parser.add_argument("--split", action="append", choices=["train", "dev"])
    parser.add_argument("--scale", choices=["small", "large"], default="small")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
    articles, behaviors, users = main(args.split, args.scale)
    print(
        f"MIND parsing complete: articles={len(articles):,}; "
        f"behaviors={'streamed' if behaviors is None else len(behaviors)}; "
        f"users={'streamed' if users is None else len(users)}"
    )
