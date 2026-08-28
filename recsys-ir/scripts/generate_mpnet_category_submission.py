"""Generate MIND submission: mpnet-base-v2 embeddings + category-affinity blend.

Exp G: combines the strongest embedding model (mpnet, 768-D) with category
signal from actual click history. Uses actual history during inference (not
the top-K proxy used in offline tuning), so the signal is maximally accurate.

Usage:
    .venv/bin/python -m scripts.generate_mpnet_category_submission --beta 0.80 --history-cap 50
    .venv/bin/python -m scripts.generate_mpnet_category_submission --beta 0.80 --history-cap 100
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl

from src.parsing.submission_readers import (
    find_mind_test_behaviors,
    iter_mind_test,
    Impression,
)
from src.retrieval.ann import ArticleIndex
from src.submission.package_submission import package_prediction
from src.submission.writers import write_ranked_impression

logger = logging.getLogger(__name__)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_EMBED_DIR = _PROJECT_ROOT / "data" / "processed" / "embeddings"


def _load_mpnet_index() -> ArticleIndex:
    npy_path = _EMBED_DIR / "mind_mpnet_large.npy"
    ids_path = _EMBED_DIR / "mind_mpnet_large_ids.json"
    embeddings = np.load(npy_path)
    id_to_row: dict[str, int] = json.loads(ids_path.read_text())
    ordered = [""] * len(id_to_row)
    for aid, idx in id_to_row.items():
        ordered[int(idx)] = aid
    return ArticleIndex(embeddings, ordered, build_full_index=False), embeddings, id_to_row


def _load_categories() -> tuple[dict[str, str], dict[str, str]]:
    article_path = (_PROJECT_ROOT / "data" / "processed" / "large" / "mind"
                    / "article_features.parquet")
    df = pl.read_parquet(article_path, columns=["article_id", "category", "subcategory"])
    cat, subcat = {}, {}
    for row in df.iter_rows(named=True):
        aid = str(row["article_id"])
        cat[aid] = row["category"] or ""
        subcat[aid] = row["subcategory"] or ""
    return cat, subcat


def _minmax(v: np.ndarray) -> np.ndarray:
    lo, hi = v.min(), v.max()
    return np.zeros_like(v) if hi - lo < 1e-12 else (v - lo) / (hi - lo)


def _process_batch(
    batch: list[Impression],
    index: ArticleIndex,
    embeddings: np.ndarray,
    id_to_row: dict[str, int],
    article_category: dict[str, str],
    article_subcategory: dict[str, str],
    history_cap: int,
    beta: float,
    handle,
) -> None:
    for item in batch:
        history_ids = [str(e["article_id"]) for e in item.history[-history_cap:]]
        candidates = item.candidates

        # ── User vector (mean-pool mpnet) ──
        if history_ids:
            h_embs, _ = index.get_embeddings_batch(history_ids)
            if h_embs.shape[0] > 0:
                vec = h_embs.mean(axis=0, dtype=np.float32)
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec /= norm
            else:
                vec = np.zeros(index.dim, dtype=np.float32)
        else:
            vec = np.zeros(index.dim, dtype=np.float32)

        # ── Embedding similarity ──
        results = index.search_restricted(vec, candidates, k=len(candidates))
        sim_lookup = dict(results)
        sim_vals = np.array([sim_lookup.get(c, 0.0) for c in candidates], dtype=np.float64)
        sim_norm = _minmax(sim_vals)

        # ── Category affinity (from ACTUAL history) ──
        hist_cats = Counter(article_category.get(aid, "") for aid in history_ids)
        hist_subcats = Counter(article_subcategory.get(aid, "") for aid in history_ids)
        total_h = sum(hist_cats.values()) or 1

        cat_vals = np.array(
            [hist_cats.get(article_category.get(c, ""), 0) / total_h for c in candidates],
            dtype=np.float64,
        )
        subcat_vals = np.array(
            [hist_subcats.get(article_subcategory.get(c, ""), 0) / total_h for c in candidates],
            dtype=np.float64,
        )
        cat_signal = 0.5 * _minmax(cat_vals) + 0.5 * _minmax(subcat_vals)

        # ── Blend & rank ──
        blended = beta * sim_norm + (1.0 - beta) * cat_signal
        order = np.argsort(-blended, kind="stable")
        ordered = [candidates[i] for i in order]

        write_ranked_impression(handle, item.impression_id, item.candidates, ordered)


def main():
    parser = argparse.ArgumentParser(
        description="Generate MIND submission: mpnet + category-affinity blend"
    )
    parser.add_argument("--beta", type=float, default=0.80,
                        help="Embedding weight (1-beta goes to category). Best: 0.80")
    parser.add_argument("--history-cap", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=50_000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")

    started = time.time()
    index, embeddings, id_to_row = _load_mpnet_index()
    logger.info("Loaded mpnet index: %s", index)

    article_category, article_subcategory = _load_categories()
    logger.info("Loaded categories: %d articles", len(article_category))

    tag = f"beta{args.beta}_cap{args.history_cap}"
    output_dir = _PROJECT_ROOT / "submissions" / f"mind_mpnet_cat_{tag}"
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "prediction.txt"
    zip_path = output_dir / f"mind_mpnet_cat_{tag}_submission.zip"

    test_path = find_mind_test_behaviors(_PROJECT_ROOT / "data" / "raw" / "mind")
    batches = iter_mind_test(test_path, batch_size=args.batch_size)

    row_count = 0
    with prediction_path.open("w", encoding="utf-8") as handle:
        for batch in batches:
            _process_batch(
                batch, index, embeddings, id_to_row,
                article_category, article_subcategory,
                args.history_cap, args.beta, handle,
            )
            row_count += len(batch)
            if row_count % max(args.batch_size * 5, 200_000) < len(batch):
                logger.info("Generated %d predictions", row_count)

    package_prediction(prediction_path, zip_path)
    elapsed = time.time() - started

    print(f"\nMIND MPNET+CAT (beta={args.beta}, cap={args.history_cap}): "
          f"{row_count:,} rows, {elapsed/60:.1f} min")
    print(f"  prediction: {prediction_path}")
    print(f"  submission: {zip_path}")


if __name__ == "__main__":
    main()
