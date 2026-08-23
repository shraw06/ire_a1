"""Build memory-safe feature stores for small and large pipelines."""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def build_all(datasets: list[str] | None = None, scale: str = "small") -> dict[str, dict[str, Path]]:
    from src.common.paths import interim_dir, processed_dir
    from src.feature_store.article_store import ArticleFeatureStore
    from src.feature_store.user_store import UserFeatureStore
    from src.feature_store.history_store import MemoryMappedHistoryStore

    datasets = datasets or ["mind", "ebnerd"]
    results: dict[str, dict[str, Path]] = {}

    for dataset in datasets:
        i_dir = interim_dir(dataset, scale)
        p_dir = processed_dir(dataset, scale)
        if not i_dir.exists():
            logger.warning("Skipping %s: %s does not exist", dataset, i_dir)
            continue
        p_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()

        article_path = ArticleFeatureStore.build_features(
            dataset, interim_dir=i_dir, processed_dir=p_dir
        )
        result = {"articles": article_path}

        if scale == "large" and dataset == "ebnerd":
            raw_root = Path(__file__).resolve().parents[2] / "data" / "raw" / "ebnerd" / "ebnerd_large"
            # Validation-time history must remain separate from training history.
            for split in ("train", "validation"):
                source = raw_root / split / "history.parquet"
                if source.exists():
                    index_dir = p_dir / f"history_index_{split}"
                    MemoryMappedHistoryStore.build(source, index_dir)
                    result[f"history_{split}"] = index_dir
            # A user_features file is intentionally not built from 25M behavior
            # rows. The large EB history index is the serving-time user feature store.
            logger.info("Large EB-NeRD user feature store: memory-mapped history indexes")
        elif scale == "large" and dataset == "mind":
            # MIND-large has per-impression history snapshots. Retrieval reads
            # them directly from behaviors.parquet; no giant per-user aggregate is
            # needed and no future interactions are introduced by aggregation.
            logger.info("Large MIND user features: per-impression history snapshots")
        else:
            result["users"] = UserFeatureStore.build_features(
                dataset, interim_dir=i_dir, processed_dir=p_dir
            )

        logger.info("Built %s features in %.1fs", dataset, time.time() - t0)
        results[dataset] = result

    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mind", "ebnerd", "all"], default="all")
    parser.add_argument("--scale", choices=["small", "large"], default="small")
    args = parser.parse_args()
    datasets = ["mind", "ebnerd"] if args.dataset == "all" else [args.dataset]
    results = build_all(datasets, scale=args.scale)
    for ds, paths in results.items():
        print(f"\n{ds.upper()}:")
        for name, path in paths.items():
            if path.is_file():
                print(f"  {name}: {path.name} ({path.stat().st_size / 1024**2:.1f} MB)")
            else:
                print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
