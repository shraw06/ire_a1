"""Generate a MIND Codabench submission using all-mpnet-base-v2 (768-D) embeddings.

Requires mpnet embeddings to be precomputed first:
    .venv/bin/python -m scripts.compute_mpnet_embeddings --model mpnet --device cuda

Writes to submissions/mind_mpnet_cap<N>/ so baseline is never overwritten.

Usage:
    .venv/bin/python -m scripts.generate_mpnet_submission --history-cap 50
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from src.parsing.submission_readers import (
    find_mind_test_behaviors,
    iter_mind_test,
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
    if not npy_path.exists():
        raise FileNotFoundError(
            "mpnet embeddings not found. Run:\n"
            "  .venv/bin/python -m scripts.compute_mpnet_embeddings --model mpnet --device cuda"
        )
    embeddings = np.load(npy_path)
    id_to_row: dict[str, int] = json.loads(ids_path.read_text())
    article_ids_ordered = [""] * len(id_to_row)
    for aid, idx in id_to_row.items():
        article_ids_ordered[int(idx)] = aid
    return ArticleIndex(embeddings, article_ids_ordered, build_full_index=False)


def _build_user_vector(
    history: list[dict],
    index: ArticleIndex,
    history_cap: int,
) -> np.ndarray:
    """Build L2-normalized mean-pool user vector from history."""
    ids = [str(e["article_id"]) for e in history[-history_cap:]]
    if not ids:
        return np.zeros(index.dim, dtype=np.float32)
    embeddings_batch, _ = index.get_embeddings_batch(ids)
    if embeddings_batch.shape[0] == 0:
        return np.zeros(index.dim, dtype=np.float32)
    vec = embeddings_batch.mean(axis=0, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def _process_batch(batch, index, history_cap, handle):
    for item in batch:
        user_vec = _build_user_vector(item.history, index, history_cap)
        results = index.search_restricted(user_vec, item.candidates, k=len(item.candidates))
        ordered = [aid for aid, _ in results]
        write_ranked_impression(handle, item.impression_id, item.candidates, ordered)


def main():
    parser = argparse.ArgumentParser(
        description="Generate MIND submission with all-mpnet-base-v2 embeddings"
    )
    parser.add_argument("--history-cap", type=int, default=50,
                        help="Max history articles for user vector (default: 50)")
    parser.add_argument("--batch-size", type=int, default=50_000)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    started = time.time()

    # ── Load index ──
    index = _load_mpnet_index()
    logger.info("Loaded mpnet index: %s", index)

    # ── Output paths ──
    tag = f"cap{args.history_cap}"
    output_dir = _PROJECT_ROOT / "submissions" / f"mind_mpnet_{tag}"
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "prediction.txt"
    zip_path = output_dir / f"mind_mpnet_{tag}_submission.zip"

    # ── Stream test ──
    test_path = find_mind_test_behaviors(_PROJECT_ROOT / "data" / "raw" / "mind")
    batches = iter_mind_test(test_path, batch_size=args.batch_size)

    row_count = 0
    with prediction_path.open("w", encoding="utf-8") as handle:
        for batch in batches:
            _process_batch(batch, index, args.history_cap, handle)
            row_count += len(batch)
            if row_count % max(args.batch_size * 5, 200_000) < len(batch):
                logger.info("Generated %d predictions", row_count)

    package_prediction(prediction_path, zip_path)
    elapsed = time.time() - started

    print(f"\nMIND MPNET (cap={args.history_cap}): {row_count:,} rows, {elapsed/60:.1f} min")
    print(f"  prediction: {prediction_path}")
    print(f"  submission: {zip_path}")


if __name__ == "__main__":
    main()
