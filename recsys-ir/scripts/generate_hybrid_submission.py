"""Generate a hybrid (recency-weighted embedding + train-split popularity)
Codabench submission for the MIND or EB-NeRD large test set.

Reuses the existing, checkpointed submission machinery (index loading,
history streaming, prediction/ZIP writers) from src.submission.make_submission
and src.parsing.submission_readers; only the per-impression scoring step is
replaced with the hybrid reranker from src.retrieval.hybrid_rerank.

Writes to submissions/<dataset>_hybrid/ so the existing baseline
submissions/<dataset>/prediction.txt + zip (including the already-uploaded
MIND baseline) are never touched or overwritten.

Usage:
    .venv/bin/python -m scripts.generate_hybrid_submission --dataset mind --alpha 0.8
    .venv/bin/python -m scripts.generate_hybrid_submission --dataset ebnerd --alpha 0.8 --ebnerd-model w2v

Set --alpha from results/large/hybrid_alpha_tuning.csv (produced by
scripts/tune_hybrid_alpha.py) rather than guessing.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from src.feature_store.history_store import MemoryMappedHistoryStore
from src.parsing.submission_readers import (
    find_ebnerd_test_files,
    find_mind_test_behaviors,
    iter_ebnerd_test,
    iter_mind_test,
)
from src.retrieval.hybrid_rerank import (
    hybrid_rank_candidate_batch,
    load_train_popularity,
    recency_weighted_user_vectors,
)
from src.submission.make_submission import _history_batches, _load_index
from src.submission.package_submission import package_prediction
from src.submission.writers import write_ranked_impression

logger = logging.getLogger(__name__)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _process_batch_hybrid(
    batch,
    dataset: str,
    index,
    eb_history,
    history_cap: int,
    decay: float,
    popularity: dict[str, int],
    alpha: float,
    handle,
) -> None:
    histories = _history_batches(dataset, batch, eb_history)
    candidates = [item.candidates for item in batch]
    vectors = recency_weighted_user_vectors(histories, index, history_cap=history_cap, decay=decay)
    ranked = hybrid_rank_candidate_batch(vectors, candidates, index, popularity, alpha=alpha)
    for item, ordered in zip(batch, ranked):
        write_ranked_impression(handle, item.impression_id, item.candidates, ordered)


def generate_hybrid_submission(dataset: str, args: argparse.Namespace) -> tuple[Path, Path, int, float]:
    started = time.time()
    index = _load_index(dataset, args)
    logger.info("Loaded submission index: %s", index)

    popularity = load_train_popularity(dataset, "large")
    logger.info("Loaded train-split popularity: %d articles", len(popularity))

    output_dir = _PROJECT_ROOT / "submissions" / f"{dataset}_hybrid"
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "prediction.txt"
    zip_path = output_dir / f"{dataset}_hybrid_submission.zip"

    eb_history = None
    if dataset == "ebnerd":
        _, history_path = find_ebnerd_test_files(_PROJECT_ROOT / "data" / "raw" / "ebnerd")
        history_dir = _PROJECT_ROOT / "data" / "processed" / "submission" / "ebnerd_history"
        eb_history = MemoryMappedHistoryStore.build(
            history_path, history_dir, force=args.rebuild_history_index
        )
        batches = iter_ebnerd_test(
            find_ebnerd_test_files(_PROJECT_ROOT / "data" / "raw" / "ebnerd")[0],
            batch_size=args.batch_size,
        )
    else:
        test_path = find_mind_test_behaviors(_PROJECT_ROOT / "data" / "raw" / "mind")
        batches = iter_mind_test(test_path, batch_size=args.batch_size)

    row_count = 0
    with prediction_path.open("w", encoding="utf-8") as handle:
        for batch in batches:
            _process_batch_hybrid(
                batch, dataset, index, eb_history,
                args.history_cap, args.decay, popularity, args.alpha, handle,
            )
            row_count += len(batch)
            if row_count % max(args.batch_size * 5, 100_000) < len(batch):
                logger.info("Generated %d predictions", row_count)

    package_prediction(prediction_path, zip_path)
    elapsed = time.time() - started
    logger.info("Generated %d hybrid predictions in %.1fs", row_count, elapsed)
    return prediction_path, zip_path, row_count, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a hybrid Codabench submission ZIP")
    parser.add_argument("--dataset", choices=["mind", "ebnerd"], required=True)
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument("--history-cap", type=int, default=20)
    parser.add_argument("--ebnerd-model", choices=["w2v", "bert"], default="w2v")
    parser.add_argument("--embedding-batch-size", type=int, default=256)
    parser.add_argument("--device", default=None, help="sentence-transformers device, e.g. cuda or cpu")
    parser.add_argument("--rebuild-history-index", action="store_true")
    parser.add_argument(
        "--alpha", type=float, default=0.8,
        help="Weight on embedding similarity vs. popularity prior (1.0 = pure embedding baseline). "
             "Set from results/large/hybrid_alpha_tuning.csv, don't guess.",
    )
    parser.add_argument(
        "--decay", type=float, default=0.85,
        help="Recency decay for history weighting; 1.0 = uniform mean (current baseline).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    prediction, zip_path, rows, elapsed = generate_hybrid_submission(args.dataset, args)
    print(
        f"\n{args.dataset.upper()} HYBRID (alpha={args.alpha}, decay={args.decay}): "
        f"{rows:,} rows, {elapsed/60:.1f} min"
    )
    print(f"  prediction: {prediction}")
    print(f"  submission: {zip_path}")


if __name__ == "__main__":
    main()