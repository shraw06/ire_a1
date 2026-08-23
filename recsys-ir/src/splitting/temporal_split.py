"""Temporal train/validation/test splitting with a streaming large-data path."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)

SPLIT_CONFIG = {
    "mind": {
        "val_cutoff": datetime(2019, 11, 14, 0, 0, 0),
        "native_test_start": datetime(2019, 11, 15, 0, 0, 0),
    },
    "ebnerd": {
        "val_cutoff": datetime(2023, 5, 24, 7, 0, 0),
        "native_test_start": datetime(2023, 5, 25, 7, 0, 0),
    },
}


def assign_temporal_split(behaviors: pl.DataFrame, dataset: str) -> pl.DataFrame:
    """Assign train/val/test labels to an in-memory behaviors DataFrame."""
    cfg = SPLIT_CONFIG[dataset]

    return behaviors.with_columns(
        pl.when(pl.col("timestamp") >= cfg["native_test_start"])
        .then(pl.lit("test"))
        .when(pl.col("timestamp") >= cfg["val_cutoff"])
        .then(pl.lit("val"))
        .otherwise(pl.lit("train"))
        .alias("split")
    )


def _escape_sql_path(path: Path) -> str:
    """Escape a filesystem path for use in a DuckDB SQL string literal."""
    return str(path).replace("'", "''")


def _split_large_duckdb(beh_path: Path, dataset: str) -> dict[str, int]:
    """Add a temporal split column without loading millions of rows into RAM."""
    import duckdb

    cfg = SPLIT_CONFIG[dataset]
    tmp_path = beh_path.with_suffix(".split.tmp.parquet")

    input_sql = _escape_sql_path(beh_path)
    output_sql = _escape_sql_path(tmp_path)
    val_cutoff = cfg["val_cutoff"].strftime("%Y-%m-%d %H:%M:%S")
    native_test_start = cfg["native_test_start"].strftime("%Y-%m-%d %H:%M:%S")

    con = duckdb.connect()
    try:
        # DuckDB 1.0.0 does not accept Python parameters for these file
        # arguments inside read_parquet()/COPY. Embed the escaped paths.
        con.execute(
            f"""
            COPY (
                SELECT *,
                       CASE
                           WHEN timestamp >= TIMESTAMP '{native_test_start}' THEN 'test'
                           WHEN timestamp >= TIMESTAMP '{val_cutoff}' THEN 'val'
                           ELSE 'train'
                       END AS split
                FROM read_parquet('{input_sql}')
            )
            TO '{output_sql}'
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )

        counts_rows = con.execute(
            f"""
            SELECT split, COUNT(*)
            FROM read_parquet('{output_sql}')
            GROUP BY split
            ORDER BY split
            """
        ).fetchall()

    finally:
        con.close()

    # Only replace the source after the streamed write and validation query
    # succeeded, so a failed/interrupted operation does not destroy the input.
    tmp_path.replace(beh_path)

    return {str(split): int(count) for split, count in counts_rows}


def split_and_persist(
    dataset: str,
    interim_dir: Path | None = None,
    scale: str = "small",
):
    """Split one dataset and persist the result.

    For large scale, use DuckDB to stream through the Parquet file and return
    only compact split counts. For small scale, preserve the existing in-memory
    Polars behavior.
    """
    from src.common.paths import interim_dir as scale_interim_dir

    if interim_dir is None:
        interim_dir = scale_interim_dir(dataset, scale)

    beh_path = interim_dir / "behaviors.parquet"
    if not beh_path.exists():
        raise FileNotFoundError(beh_path)

    if scale == "large":
        counts = _split_large_duckdb(beh_path, dataset)
        logger.info("%s large split counts: %s", dataset, counts)

        return pl.DataFrame(
            {
                "split": list(counts.keys()),
                "count": list(counts.values()),
            }
        )

    behaviors = pl.read_parquet(beh_path)

    if "split" in behaviors.columns:
        behaviors = behaviors.drop("split")

    behaviors = assign_temporal_split(behaviors, dataset)
    behaviors.write_parquet(beh_path, compression="zstd")

    for name in ("train", "val", "test"):
        logger.info(
            "%s %s: %d rows",
            dataset,
            name,
            behaviors.filter(pl.col("split") == name).height,
        )

    return behaviors


def split_all(scale: str = "small"):
    """Run the temporal split for both datasets."""
    return {
        ds: split_and_persist(ds, scale=scale)
        for ds in ("mind", "ebnerd")
    }


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["mind", "ebnerd", "all"],
        default="all",
    )
    parser.add_argument(
        "--scale",
        choices=["small", "large"],
        default="small",
    )
    args = parser.parse_args()

    results = (
        split_all(args.scale)
        if args.dataset == "all"
        else {
            args.dataset: split_and_persist(
                args.dataset,
                scale=args.scale,
            )
        }
    )

    print("\nTemporal split complete:")

    for ds, result in results.items():
        if args.scale == "large":
            print(
                f"  {ds}: "
                f"{dict(zip(result['split'].to_list(), result['count'].to_list()))}"
            )
        else:
            total = result.height
            print(f"  {ds}: {total:,} rows")
