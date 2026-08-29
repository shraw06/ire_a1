"""Step 1: Precompute article click-counts from MIND training data.

Streams through the train split of behaviors.parquet and accumulates
per-article click frequencies. These popularity scores are a strong
feature for both LightGBM ranking and popularity re-ranking.

Output: data/processed/large/mind/article_popularity.json
        {article_id: click_count}

Usage:
    .venv/bin/python -m scripts.build_article_popularity
"""
from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")

    behaviors_path = _PROJECT_ROOT / "data" / "interim" / "large" / "mind" / "behaviors.parquet"
    out_path = _PROJECT_ROOT / "data" / "processed" / "large" / "mind" / "article_popularity.json"

    if out_path.exists():
        logger.info("Already exists: %s — loading", out_path)
        counts = json.loads(out_path.read_text())
        logger.info("Loaded %d article click counts", len(counts))
        return counts

    parquet = pq.ParquetFile(behaviors_path)
    columns = ["candidates", "labels", "split"]

    click_counts: dict[str, int] = defaultdict(int)
    impression_count = 0
    t0 = time.time()

    for batch in parquet.iter_batches(batch_size=50_000, columns=columns):
        data = batch.to_pydict()
        n = batch.num_rows
        for i in range(n):
            if data["split"][i] != "train":
                continue
            candidates_raw = data["candidates"][i]
            labels_raw = data["labels"][i]

            import json as _json
            candidates = _json.loads(candidates_raw) if isinstance(candidates_raw, str) else candidates_raw
            labels = _json.loads(labels_raw) if isinstance(labels_raw, str) else labels_raw

            for cid, lbl in zip(candidates, labels):
                if int(lbl) == 1:
                    click_counts[str(cid)] += 1

            impression_count += 1

        if impression_count % 100_000 < 50_000:
            logger.info("  %d train impressions processed (%.1f min)",
                        impression_count, (time.time() - t0) / 60)

    elapsed = time.time() - t0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dict(click_counts)))

    logger.info("Done: %d train impressions, %d unique clicked articles in %.1f min",
                impression_count, len(click_counts), elapsed / 60)
    logger.info("Saved: %s", out_path)

    # Print top-10 most clicked articles
    top = sorted(click_counts.items(), key=lambda x: -x[1])[:10]
    print("\nTop 10 clicked articles:")
    for aid, cnt in top:
        print(f"  {aid}: {cnt} clicks")

    return click_counts


if __name__ == "__main__":
    main()
