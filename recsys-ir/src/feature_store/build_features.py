"""Build feature-store Parquet files from interim data.

CLI entry point for the ``make features`` target.

Usage:
    python -m src.feature_store.build_features [--dataset mind|ebnerd|all]

Reads interim articles + behaviors for each dataset, computes per-article
and per-user features, and writes to ``data/processed/{dataset}/``.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_all(datasets: list[str] | None = None) -> dict[str, dict[str, Path]]:
    """Build article + user features for the requested datasets.

    Returns
    -------
    dict mapping dataset → {"articles": Path, "users": Path}
    """
    # Deferred imports so the module can be imported without triggering
    # heavy loads when only the CLI is needed.
    from src.feature_store.article_store import ArticleFeatureStore
    from src.feature_store.user_store import UserFeatureStore

    if datasets is None:
        datasets = ["mind", "ebnerd"]

    results: dict[str, dict[str, Path]] = {}
    for dataset in datasets:
        interim_dir = _PROJECT_ROOT / "data" / "interim" / dataset
        if not interim_dir.exists():
            logger.warning("Skipping %s — interim directory not found at %s", dataset, interim_dir)
            continue

        logger.info("Building features for %s ...", dataset)
        t0 = time.time()

        article_path = ArticleFeatureStore.build_features(dataset)
        user_path = UserFeatureStore.build_features(dataset)

        elapsed = time.time() - t0
        logger.info("  %s features built in %.1fs", dataset, elapsed)

        results[dataset] = {"articles": article_path, "users": user_path}

    return results


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Build feature-store Parquet files from interim data.",
    )
    parser.add_argument(
        "--dataset",
        choices=["mind", "ebnerd", "all"],
        default="all",
        help="Which dataset to build features for (default: all).",
    )
    args = parser.parse_args()

    if args.dataset == "all":
        datasets = ["mind", "ebnerd"]
    else:
        datasets = [args.dataset]

    t0 = time.time()
    results = build_all(datasets)
    total = time.time() - t0

    print(f"\n{'='*60}")
    print("Feature build complete.")
    for ds, paths in results.items():
        print(f"\n  {ds.upper()}:")
        for kind, path in paths.items():
            size_mb = path.stat().st_size / 1024**2
            print(f"    {kind:>10}: {path.name} ({size_mb:.1f} MB)")
    print(f"\n  Total wall-clock: {total:.1f}s")
    print(f"{'='*60}")

    sys.exit(0)


if __name__ == "__main__":
    main()
