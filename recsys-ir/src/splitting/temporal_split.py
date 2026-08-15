"""Dataset-agnostic temporal train/val/test split operating on the unified schema.

Strategy (identical logic applied per dataset):
  - TEST split  = the dataset's native held-out file, used AS-IS:
      * MIND dev  (Nov 15, 2019)
      * EB-NeRD validation  (May 25 07:00 – Jun 1, 2023)
    These are already cleanly time-separated from the native train file
    with no overlap.  No further cutting needed.

  - Within each dataset's native TRAIN file only, we carve our own internal
    train/val by time:
      * VAL  = the LAST 1 day of native train
      * TRAIN = everything before that
    The 1-day choice is deliberate: it leaves most of the already-small native
    train file intact for training (MIND train is only 6 days, EB-NeRD train
    is only 7) while still giving a meaningful validation chunk (~14-19% of
    native train impressions).

  - The split is persisted as a ``split`` column (``train|val|test``) added
    directly to the behaviors DataFrame / Parquet - no file duplication.

Cutoff boundaries (derived from EDA_SUMMARY.md §2):
  MIND:
    native_test_start = 2019-11-15 00:00:00  (first dev impression is 00:00:01)
    val_cutoff        = 2019-11-14 00:00:00  (midnight → val = all of Nov 14)
  EB-NeRD:
    native_test_start = 2023-05-25 07:00:00  (first validation impression is 07:00:15)
    val_cutoff        = 2023-05-24 07:00:00  (val = May 24 07:00 - May 25 06:59:52)
    NOTE: EB-NeRD's "days" start at ~07:00 local time, not midnight.

IMPORTANT - comparability caveat:
  Our local val/test numbers will NOT be directly comparable to any published
  MIND or EB-NeRD baseline numbers that use the datasets' native splits
  differently.  This is expected and intentional: our methodology needs its
  own tunable val set separate from the Codabench-facing test set.
  Do NOT use our val or test sets for hyperparameter tuning against published
  leaderboard numbers - those require submitting to the official evaluation
  server with the official held-out labels.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Per-dataset temporal boundaries
#
# Each entry defines:
#   val_cutoff:        timestamp where our internal val begins (within native train)
#   native_test_start: timestamp where the native held-out file begins (= our test)
#
# Both are INCLUSIVE lower-bounds (>=).

SPLIT_CONFIG = {
    "mind": {
        # MIND train covers Nov 9-14 (6 days), dev is Nov 15 (1 day).
        # Our val  = last 1 day of train = Nov 14 (midnight onward).
        # Our test = native dev = Nov 15 onward.
        "val_cutoff": datetime(2019, 11, 14, 0, 0, 0),
        "native_test_start": datetime(2019, 11, 15, 0, 0, 0),
    },
    "ebnerd": {
        # EB-NeRD train covers May 18 07:00 - May 25 06:59 (7 days).
        # validation covers May 25 07:00- Jun 1 06:59 (7 days).
        # EB-NeRD "days" start at ~07:00 local time, NOT midnight.
        # Our val  = last 1 day of train = May 24 07:00 - May 25 06:59:52.
        # Our test = native validation = May 25 07:00 onward.
        "val_cutoff": datetime(2023, 5, 24, 7, 0, 0),
        "native_test_start": datetime(2023, 5, 25, 7, 0, 0),
    },
}


def assign_temporal_split(behaviors: pl.DataFrame, dataset: str) -> pl.DataFrame:
    """Add a ``split`` column (train|val|test) to *behaviors* based on timestamp.

    Parameters
    ----------
    behaviors : pl.DataFrame
        Unified behaviors table (must have ``timestamp`` column of Datetime type).
    dataset : str
        One of ``"mind"`` or ``"ebnerd"`` - selects the split boundaries.

    Returns
    -------
    pl.DataFrame
        The input DataFrame with an added ``split`` column of type Utf8.
    """
    if dataset not in SPLIT_CONFIG:
        raise ValueError(f"Unknown dataset '{dataset}'. Expected one of {list(SPLIT_CONFIG)}")

    cfg = SPLIT_CONFIG[dataset]
    val_cutoff = cfg["val_cutoff"]
    native_test_start = cfg["native_test_start"]

    logger.info(
        "Assigning temporal splits for %s: val_cutoff=%s, native_test_start=%s",
        dataset, val_cutoff, native_test_start,
    )

    # Assign split label using Polars expressions (no row-by-row loop).
    # Order matters: test first (>= native_test_start), then val (>= val_cutoff),
    # then train (everything else).
    result = behaviors.with_columns(
        pl.when(pl.col("timestamp") >= native_test_start)
        .then(pl.lit("test"))
        .when(pl.col("timestamp") >= val_cutoff)
        .then(pl.lit("val"))
        .otherwise(pl.lit("train"))
        .alias("split")
    )

    # Log split sizes
    for split_name in ("train", "val", "test"):
        count = result.filter(pl.col("split") == split_name).height
        logger.info("  %s %s: %d rows", dataset, split_name, count)

    return result


def split_and_persist(dataset: str, interim_dir: Path | None = None) -> pl.DataFrame:
    """Load interim behaviors for *dataset*, assign splits, and overwrite the Parquet.

    Parameters
    ----------
    dataset : str
        ``"mind"`` or ``"ebnerd"``.
    interim_dir : Path, optional
        Override for the interim directory. Defaults to ``data/interim/{dataset}/``.

    Returns
    -------
    pl.DataFrame
        The behaviors DataFrame with the ``split`` column.
    """
    if interim_dir is None:
        interim_dir = _PROJECT_ROOT / "data" / "interim" / dataset

    beh_path = interim_dir / "behaviors.parquet"
    if not beh_path.exists():
        raise FileNotFoundError(
            f"Interim behaviors not found at {beh_path}. "
            f"Run the parse_{dataset} adapter first."
        )

    behaviors = pl.read_parquet(beh_path)
    logger.info("Loaded %s behaviors: %d rows", dataset, len(behaviors))

    # Drop any pre-existing split column (idempotent re-runs)
    if "split" in behaviors.columns:
        behaviors = behaviors.drop("split")

    behaviors = assign_temporal_split(behaviors, dataset)

    # Overwrite the interim Parquet with the new split column
    behaviors.write_parquet(beh_path)
    logger.info("Wrote %s behaviors with split column → %s", dataset, beh_path)

    return behaviors


def split_all() -> dict[str, pl.DataFrame]:
    """Run temporal splitting for both MIND and EB-NeRD. Returns {dataset: behaviors_df}."""
    results = {}
    for dataset in ("mind", "ebnerd"):
        results[dataset] = split_and_persist(dataset)
    return results


# CLI entry point 

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Assign temporal train/val/test splits to interim behaviors.",
    )
    parser.add_argument(
        "--dataset",
        choices=["mind", "ebnerd", "all"],
        default="all",
        help="Which dataset to split (default: all).",
    )
    args = parser.parse_args()

    if args.dataset == "all":
        results = split_all()
    else:
        results = {args.dataset: split_and_persist(args.dataset)}

    print(f"\n{'='*60}")
    for ds, beh in results.items():
        cfg = SPLIT_CONFIG[ds]
        print(f"\n  {ds.upper()}:")
        print(f"    val_cutoff:        {cfg['val_cutoff']}")
        print(f"    native_test_start: {cfg['native_test_start']}")
        for split_name in ("train", "val", "test"):
            count = beh.filter(pl.col("split") == split_name).height
            total = len(beh)
            pct = count / total * 100
            print(f"    {split_name:>5}: {count:>8,d} rows ({pct:5.1f}%)")
    print(f"\n{'='*60}")
