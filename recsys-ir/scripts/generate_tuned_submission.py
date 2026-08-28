"""Generate a MIND Codabench submission with tuned user-vector parameters.

Uses the same infrastructure as make_submission.py but with configurable
history_cap and recency decay. Writes to submissions/mind_tuned/ so the
existing baseline is never overwritten.

Usage:
    .venv/bin/python -m scripts.generate_tuned_submission --history-cap 50 --decay 1.0
    .venv/bin/python -m scripts.generate_tuned_submission --history-cap 50 --decay 0.95
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np

from src.parsing.submission_readers import (
    find_mind_test_behaviors,
    iter_mind_test,
)
from src.retrieval.ann import ArticleIndex
from src.retrieval.embeddings import load_embeddings
from src.submission.make_submission import _find_mind_catalog
from src.submission.package_submission import package_prediction
from src.submission.writers import write_ranked_impression

logger = logging.getLogger(__name__)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _build_user_vectors_tuned(
    histories: list[list[dict]],
    index: ArticleIndex,
    history_cap: int,
    decay: float,
) -> np.ndarray:
    """Batch user vector construction with tuned parameters."""
    # Gather all unique article IDs first
    unique_ids: list[str] = []
    seen: set[str] = set()
    truncated: list[list[str]] = []
    for history in histories:
        ids = [str(entry["article_id"]) for entry in history[-history_cap:]]
        truncated.append(ids)
        for aid in ids:
            if aid not in seen:
                seen.add(aid)
                unique_ids.append(aid)

    vectors = np.zeros((len(histories), index.dim), dtype=np.float32)
    if not unique_ids:
        return vectors

    embeddings_batch, found_ids = index.get_embeddings_batch(unique_ids)
    row_by_id = {aid: i for i, aid in enumerate(found_ids)}

    for i, ids in enumerate(truncated):
        rows = [row_by_id[aid] for aid in ids if aid in row_by_id]
        if not rows:
            continue
        h_emb = embeddings_batch[rows]  # [H, D]
        H = h_emb.shape[0]

        if decay >= 1.0 - 1e-9:
            vec = h_emb.mean(axis=0, dtype=np.float32)
        else:
            weights = np.array(
                [decay ** (H - 1 - j) for j in range(H)],
                dtype=np.float32,
            )
            weights /= weights.sum()
            vec = (h_emb * weights[:, None]).sum(axis=0, dtype=np.float32)

        norm = np.linalg.norm(vec)
        if norm > 0:
            vectors[i] = vec / norm

    return vectors


def _process_batch(batch, index, history_cap, decay, handle):
    histories = [item.history for item in batch]
    candidates = [item.candidates for item in batch]
    vectors = _build_user_vectors_tuned(histories, index, history_cap, decay)

    for item, user_vec, cands in zip(batch, vectors, candidates):
        results = index.search_restricted(user_vec, cands, k=len(cands))
        ordered = [aid for aid, _ in results]
        write_ranked_impression(handle, item.impression_id, item.candidates, ordered)


def main():
    parser = argparse.ArgumentParser(description="Generate MIND submission with tuned user vector params")
    parser.add_argument("--history-cap", type=int, default=50,
                        help="Maximum history articles for user vector")
    parser.add_argument("--decay", type=float, default=1.0,
                        help="Recency decay (1.0 = uniform mean)")
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument("--embedding-batch-size", type=int, default=256)
    parser.add_argument("--device", default=None)
    parser.add_argument("--tag", default=None,
                        help="Custom tag for output directory (default: auto)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")

    started = time.time()

    # Load embedding index
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

    # Output paths
    tag = args.tag or f"cap{args.history_cap}_decay{args.decay}"
    output_dir = _PROJECT_ROOT / "submissions" / f"mind_tuned_{tag}"
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "prediction.txt"
    zip_path = output_dir / f"mind_tuned_{tag}_submission.zip"

    # Stream test data
    test_path = find_mind_test_behaviors(_PROJECT_ROOT / "data" / "raw" / "mind")
    batches = iter_mind_test(test_path, batch_size=args.batch_size)

    row_count = 0
    with prediction_path.open("w", encoding="utf-8") as handle:
        for batch in batches:
            _process_batch(batch, index, args.history_cap, args.decay, handle)
            row_count += len(batch)
            if row_count % max(args.batch_size * 5, 100_000) < len(batch):
                logger.info("Generated %d predictions", row_count)

    package_prediction(prediction_path, zip_path)
    elapsed = time.time() - started
    logger.info("Generated %d predictions in %.1fs", row_count, elapsed)

    print(f"\nMIND TUNED (cap={args.history_cap}, decay={args.decay}): "
          f"{row_count:,} rows, {elapsed/60:.1f} min")
    print(f"  prediction: {prediction_path}")
    print(f"  submission: {zip_path}")


if __name__ == "__main__":
    main()
