"""Memory-safe EB-NeRD parser.

For EB-NeRD-large, the old implementation duplicated each user's complete
history JSON into every impression row and materialized all 25M+ behavior rows
as Python dictionaries. On a 15-GB machine this is enough to exhaust RAM and
can also create an unnecessarily huge intermediate file.

The large path therefore:
  * streams behaviors in Arrow batches;
  * stores candidates/labels per impression but does NOT duplicate history;
  * stores the native per-user history once in ``users.parquet``;
  * leaves timestamp-aware history retrieval to ``MemoryMappedHistoryStore``.

Small/demo behavior is kept compatible with the previous API.
"""
from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import polars as pl
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SPLITS = ["train", "validation"]

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
    return _PROJECT_ROOT / "data" / "raw" / "ebnerd" / ("ebnerd_large" if scale == "large" else "")


def _parse_ebnerd_entities(ner_clusters: Optional[list], entity_groups: Optional[list]) -> str:
    entities, seen = [], set()
    for values, kind in ((ner_clusters, "NER"), (entity_groups, "GROUP")):
        if not values:
            continue
        for label in values:
            if not label or not isinstance(label, str):
                continue
            key = (label, kind)
            if key in seen:
                continue
            seen.add(key)
            entities.append({"label": label, "type": kind, "wikidata_id": None, "confidence": None})
    return json.dumps(entities, separators=(",", ":"))


def parse_ebnerd_articles(raw_dir: Path) -> pl.DataFrame:
    articles_path = raw_dir / "articles.parquet"
    if not articles_path.exists():
        raise FileNotFoundError(articles_path)
    df = pl.read_parquet(articles_path)
    rows = []
    for r in df.iter_rows(named=True):
        subcat = r.get("subcategory")
        subcat_str = ",".join(str(x) for x in (subcat or [])) or None
        rows.append({
            "article_id": str(r["article_id"]),
            "dataset": "ebnerd",
            "title": r.get("title") or "",
            "abstract": r.get("subtitle") if r.get("subtitle") else None,
            "body": r.get("body") if r.get("body") else None,
            "body_source": "native" if r.get("body") else None,
            "category": r.get("category_str") if r.get("category_str") else None,
            "subcategory": subcat_str,
            "entities": _parse_ebnerd_entities(r.get("ner_clusters"), r.get("entity_groups")),
            "published_at": r.get("published_time"),
            "embedding_ref": None,
        })
    out = pl.DataFrame(rows, schema={
        "article_id": pl.Utf8, "dataset": pl.Utf8, "title": pl.Utf8,
        "abstract": pl.Utf8, "body": pl.Utf8, "body_source": pl.Utf8,
        "category": pl.Utf8, "subcategory": pl.Utf8, "entities": pl.Utf8,
        "published_at": pl.Datetime("us"), "embedding_ref": pl.Utf8,
    })
    logger.info("EB-NeRD articles parsed: %d rows", out.height)
    return out


def _history_lookup_small(history_path: Path) -> dict[int, list[dict]]:
    df = pl.read_parquet(history_path)
    lookup = {}
    for row in df.iter_rows(named=True):
        entries = []
        for i, aid in enumerate(row.get("article_id_fixed") or []):
            ts = (row.get("impression_time_fixed") or [None])[i] if i < len(row.get("impression_time_fixed") or []) else None
            entries.append({"article_id": str(aid), "clicked_at": ts.isoformat() if isinstance(ts, datetime) else None})
        lookup[int(row["user_id"])] = entries
    return lookup


def _parse_behavior_batch(batch, include_history: bool, history_lookup: dict[int, list[dict]] | None = None) -> pl.DataFrame:
    rows = []
    cols = {name: batch.column(i).to_pylist() for i, name in enumerate(batch.schema.names)}
    n = batch.num_rows
    for i in range(n):
        uid = cols["user_id"][i]
        inview = cols["article_ids_inview"][i] or []
        clicked = set(cols["article_ids_clicked"][i] or [])
        candidates = [str(aid) for aid in inview]
        labels = [1 if aid in clicked else 0 for aid in inview]
        history = history_lookup.get(int(uid), []) if include_history and history_lookup is not None else []
        rows.append({
            "impression_id": str(cols["impression_id"][i]),
            "dataset": "ebnerd",
            "user_id": str(uid),
            "timestamp": cols["impression_time"][i],
            "clicked_history": json.dumps(history, separators=(",", ":")),
            "candidates": json.dumps(candidates, separators=(",", ":")),
            "labels": json.dumps(labels, separators=(",", ":")),
        })
    return pl.DataFrame(rows, schema=_OUTPUT_SCHEMA)


def _write_batches(batches, output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    total = 0
    try:
        for df in batches:
            if df.height == 0:
                continue
            table = df.to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
            writer.write_table(table)
            total += df.height
            if total % 500_000 < df.height:
                logger.info(f"  wrote {total:,} EB-NeRD behavior rows")
    finally:
        if writer is not None:
            writer.close()
    return total


def _stream_large_behaviors(raw_dir: Path, splits: list[str], output_path: Path, batch_size: int = 50_000) -> tuple[int, dict[str, int]]:
    join_stats = {}

    def batches():
        for split in splits:
            path = raw_dir / split / "behaviors.parquet"
            if not path.exists():
                raise FileNotFoundError(path)
            parquet = pq.ParquetFile(path)
            join_stats[split] = parquet.metadata.num_rows
            logger.info(
                f"Streaming EB-NeRD {split} behaviors: "
                f"{parquet.metadata.num_rows:,} rows (batch={batch_size})"
            )
            columns = ["impression_id", "impression_time", "article_ids_inview", "article_ids_clicked", "user_id"]
            for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
                # Large mode deliberately leaves clicked_history empty. The native
                # history file is indexed once by feature_store/history_store.py.
                yield _parse_behavior_batch(batch, include_history=False)

    return _write_batches(batches(), output_path), join_stats


def parse_ebnerd_behaviors(raw_dir: Path, splits: list[str] | None = None) -> tuple[pl.DataFrame, dict[str, int]]:
    splits = splits or list(_SPLITS)
    all_frames = []
    stats = {}
    for split in splits:
        beh_path = raw_dir / split / "behaviors.parquet"
        hist_path = raw_dir / split / "history.parquet"
        history_lookup = _history_lookup_small(hist_path)
        beh = pl.read_parquet(beh_path)
        stats[split] = beh.height
        # Convert the entire small/demo split only; this function is not used for EB-large.
        cols = {name: beh[name].to_list() for name in ["impression_id", "impression_time", "article_ids_inview", "article_ids_clicked", "user_id"]}
        rows = []
        for i in range(beh.height):
            uid = cols["user_id"][i]
            inview = cols["article_ids_inview"][i] or []
            clicked = set(cols["article_ids_clicked"][i] or [])
            rows.append({
                "impression_id": str(cols["impression_id"][i]),
                "dataset": "ebnerd", "user_id": str(uid), "timestamp": cols["impression_time"][i],
                "clicked_history": json.dumps(history_lookup.get(int(uid), []), separators=(",", ":")),
                "candidates": json.dumps([str(x) for x in inview], separators=(",", ":")),
                "labels": json.dumps([1 if x in clicked else 0 for x in inview], separators=(",", ":")),
            })
        all_frames.append(pl.DataFrame(rows, schema=_OUTPUT_SCHEMA))
    return pl.concat(all_frames), stats


def _write_large_user_summary(history_path: Path, users_path: Path) -> None:
    """Write one row per user from native history, streamed in Arrow batches."""
    parquet = pq.ParquetFile(history_path)
    writer = None
    total = 0
    try:
        for batch in parquet.iter_batches(batch_size=20_000, columns=["user_id", "article_id_fixed", "impression_time_fixed"]):
            users, datasets, histories, lengths, lasts = [], [], [], [], []
            user_col = batch.column(0).to_pylist()
            article_col = batch.column(1).to_pylist()
            time_col = batch.column(2).to_pylist()
            for uid, aids, times in zip(user_col, article_col, time_col):
                aids = aids or []
                times = times or []
                entries = []
                last = None
                for j, aid in enumerate(aids):
                    ts = times[j] if j < len(times) else None
                    ts_str = ts.isoformat() if hasattr(ts, "isoformat") else None
                    entries.append({"article_id": str(aid), "clicked_at": ts_str})
                    if ts is not None and (last is None or ts > last):
                        last = ts
                users.append(str(uid)); datasets.append("ebnerd")
                histories.append(json.dumps(entries, separators=(",", ":")))
                lengths.append(len(entries)); lasts.append(last)
            df = pl.DataFrame({
                "user_id": users, "dataset": datasets, "all_history": histories,
                "history_len": lengths, "last_active_at": lasts,
            })
            table = df.to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(users_path, table.schema, compression="zstd")
            writer.write_table(table)
            total += df.height
    finally:
        if writer is not None:
            writer.close()
    logger.info(f"EB-NeRD large user summary written: {total:,} users")


def _copy_history_sources(raw_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in _SPLITS:
        src = raw_dir / split / "history.parquet"
        if src.exists():
            shutil.copy2(src, out_dir / f"history_{split}.parquet")


def main(splits: list[str] | None = None, scale: str = "small"):
    from src.common.paths import interim_dir
    raw_dir = _raw_dir_for_scale(scale)
    out_dir = interim_dir("ebnerd", scale)
    out_dir.mkdir(parents=True, exist_ok=True)
    if scale == "large":
        (out_dir / ".parse_complete").unlink(missing_ok=True)
    splits = splits or list(_SPLITS)

    articles = parse_ebnerd_articles(raw_dir)
    articles.write_parquet(out_dir / "articles.parquet", compression="zstd")

    if scale == "large":
        count, stats = _stream_large_behaviors(raw_dir, splits, out_dir / "behaviors.parquet", batch_size=50_000)
        logger.info(f"EB-NeRD-large behaviors written: {count:,} rows")
        _write_large_user_summary(raw_dir / "train" / "history.parquet", out_dir / "users.parquet")
        _copy_history_sources(raw_dir, out_dir)
        (out_dir / ".parse_complete").write_text("ok\n")
        return articles, None, None, stats

    behaviors, stats = parse_ebnerd_behaviors(raw_dir, splits=splits)
    # Backward-compatible user derivation for small/demo.
    users = []
    for r in behaviors.iter_rows(named=True):
        users.append(r)
    user_map = {}
    for r in users:
        state = user_map.setdefault(r["user_id"], {"user_id": r["user_id"], "dataset": "ebnerd", "history": {}, "last": r["timestamp"]})
        for e in json.loads(r["clicked_history"]):
            state["history"].setdefault(e["article_id"], e)
        state["last"] = max(state["last"], r["timestamp"])
    user_rows = [
        {
            "user_id": v["user_id"], "dataset": "ebnerd",
            "all_history": json.dumps(list(v["history"].values())),
            "history_len": len(v["history"]), "last_active_at": v["last"],
        }
        for v in user_map.values()
    ]
    pl.DataFrame(user_rows, schema={
        "user_id": pl.Utf8, "dataset": pl.Utf8, "all_history": pl.Utf8,
        "history_len": pl.Int64, "last_active_at": pl.Datetime("us"),
    }).write_parquet(out_dir / "users.parquet", compression="zstd")
    behaviors.write_parquet(out_dir / "behaviors.parquet", compression="zstd")
    return articles, behaviors, users, stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Parse EB-NeRD into unified Parquet")
    parser.add_argument("--split", action="append", choices=["train", "validation"])
    parser.add_argument("--scale", choices=["small", "large"], default="small")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
    articles, behaviors, users, stats = main(args.split, args.scale)
    print(f"EB-NeRD parsing complete: articles={len(articles):,}; behaviors={'streamed' if behaviors is None else len(behaviors)}")
    for split, n in stats.items():
        print(f"  {split}: {n:,} input rows")
