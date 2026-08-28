"""Generate a MIND Codabench submission blending embedding similarity with
category-affinity signal.

The category score for each candidate is computed from:
  - The user's click history (MIND: from the impression-level snapshot)
  - P(category|history) × 0.5 + P(subcategory|history) × 0.5

Blended score = beta * cosine_sim + (1-beta) * category_affinity

Writes to submissions/mind_category_<tag>/ so the baseline is never touched.

Usage:
    .venv/bin/python -m scripts.generate_category_submission --beta 0.85 --history-cap 50
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from src.parsing.submission_readers import (
    find_mind_test_behaviors,
    iter_mind_test,
    Impression,
)
from src.retrieval.ann import ArticleIndex
from src.retrieval.embeddings import load_embeddings
from src.submission.make_submission import _find_mind_catalog
from src.submission.package_submission import package_prediction
from src.submission.writers import write_ranked_impression

logger = logging.getLogger(__name__)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_article_categories(
    processed_dir: Path,
) -> tuple[dict[str, str], dict[str, str]]:
    """Return {article_id: category} and {article_id: subcategory}."""
    import polars as pl
    article_path = processed_dir / "article_features.parquet"
    df = pl.read_parquet(article_path, columns=["article_id", "category", "subcategory"])
    category: dict[str, str] = {}
    subcategory: dict[str, str] = {}
    for row in df.iter_rows(named=True):
        aid = str(row["article_id"])
        category[aid] = row["category"] or ""
        subcategory[aid] = row["subcategory"] or ""
    return category, subcategory


def _minmax(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def _process_batch_category(
    batch: list[Impression],
    index: ArticleIndex,
    article_category: dict[str, str],
    article_subcategory: dict[str, str],
    history_cap: int,
    beta: float,
    handle,
) -> None:
    for item in batch:
        history = item.history
        candidates = item.candidates

        # ── Build user vector ──
        history_ids = [str(e["article_id"]) for e in history[-history_cap:]]
        if history_ids:
            embs, _ = index.get_embeddings_batch(history_ids)
            if embs.shape[0] > 0:
                vec = embs.mean(axis=0, dtype=np.float32)
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec /= norm
            else:
                vec = np.zeros(index.dim, dtype=np.float32)
        else:
            vec = np.zeros(index.dim, dtype=np.float32)

        # ── Cosine similarity scores ──
        results = index.search_restricted(vec, candidates, k=len(candidates))
        sim_lookup = dict(results)
        sim_vals = np.array(
            [sim_lookup.get(cid, 0.0) for cid in candidates], dtype=np.float64
        )
        sim_norm = _minmax(sim_vals)

        # ── Category affinity ──
        # Use user's full history categories (not proxy), since we have history here
        all_hist_ids = [str(e["article_id"]) for e in history[-history_cap:]]
        hist_cats = Counter(article_category.get(aid, "") for aid in all_hist_ids)
        hist_subcats = Counter(article_subcategory.get(aid, "") for aid in all_hist_ids)
        total = sum(hist_cats.values()) or 1

        cat_vals = np.array(
            [hist_cats.get(article_category.get(cid, ""), 0) / total for cid in candidates],
            dtype=np.float64,
        )
        subcat_vals = np.array(
            [hist_subcats.get(article_subcategory.get(cid, ""), 0) / total for cid in candidates],
            dtype=np.float64,
        )
        cat_signal = 0.5 * _minmax(cat_vals) + 0.5 * _minmax(subcat_vals)

        # ── Blend ──
        blended = beta * sim_norm + (1.0 - beta) * cat_signal
        order = np.argsort(-blended, kind="stable")
        ordered = [candidates[i] for i in order]

        write_ranked_impression(handle, item.impression_id, item.candidates, ordered)


def main():
    parser = argparse.ArgumentParser(
        description="Generate MIND submission with category-affinity blend"
    )
    parser.add_argument("--beta", type=float, default=0.85,
                        help="Weight on embedding sim (1.0=pure embed). Try 0.85.")
    parser.add_argument("--history-cap", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument("--embedding-batch-size", type=int, default=256)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    started = time.time()

    # ── Load index ──
    article_ids, texts = _find_mind_catalog(_PROJECT_ROOT / "data" / "raw" / "mind")
    embeddings, id_to_row, _ = load_embeddings(
        "mind", "minilm",
        article_ids=article_ids, article_texts=texts,
        cache_tag="large",
        batch_size=args.embedding_batch_size, device=args.device,
    )
    article_ids_ordered = [""] * len(id_to_row)
    for aid, idx in id_to_row.items():
        article_ids_ordered[int(idx)] = aid
    index = ArticleIndex(embeddings, article_ids_ordered, build_full_index=False)
    logger.info("Loaded index: %s", index)

    # ── Load categories ──
    processed_dir = _PROJECT_ROOT / "data" / "processed" / "large" / "mind"
    article_category, article_subcategory = _load_article_categories(processed_dir)
    logger.info(
        "Loaded categories: %d articles, %d categories, %d subcategories",
        len(article_category),
        len(set(article_category.values())),
        len(set(article_subcategory.values())),
    )

    # ── Output paths ──
    tag = f"beta{args.beta}_cap{args.history_cap}"
    output_dir = _PROJECT_ROOT / "submissions" / f"mind_category_{tag}"
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "prediction.txt"
    zip_path = output_dir / f"mind_category_{tag}_submission.zip"

    # ── Stream test ──
    test_path = find_mind_test_behaviors(_PROJECT_ROOT / "data" / "raw" / "mind")
    batches = iter_mind_test(test_path, batch_size=args.batch_size)

    row_count = 0
    with prediction_path.open("w", encoding="utf-8") as handle:
        for batch in batches:
            _process_batch_category(
                batch, index,
                article_category, article_subcategory,
                args.history_cap, args.beta, handle,
            )
            row_count += len(batch)
            if row_count % max(args.batch_size * 5, 100_000) < len(batch):
                logger.info("Generated %d predictions", row_count)

    package_prediction(prediction_path, zip_path)
    elapsed = time.time() - started
    logger.info("Generated %d predictions in %.1fs", row_count, elapsed)

    print(f"\nMIND CATEGORY (beta={args.beta}, cap={args.history_cap}): "
          f"{row_count:,} rows, {elapsed/60:.1f} min")
    print(f"  prediction: {prediction_path}")
    print(f"  submission: {zip_path}")


if __name__ == "__main__":
    main()
