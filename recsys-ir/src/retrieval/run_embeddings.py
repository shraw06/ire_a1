"""Embedding-based retrieval pipeline runner — load/compute embeddings, build index,
score candidates, compute recall@K.

Usage:
    python -m src.retrieval.run_embeddings --dataset mind
    python -m src.retrieval.run_embeddings --dataset ebnerd
    python -m src.retrieval.run_embeddings --dataset all

For each dataset, runs embedding retrieval on the val split using the
same protocol as BM25 (candidate-restricted, recall@{50,100,200}).

EB-NeRD: runs both BERT (primary) and Word2Vec (baseline).
MIND: runs all-MiniLM-L6-v2.

Results are written to:
  - ``results/embed_recall.csv``
  - ``data/processed/embed_scores_{dataset}.parquet``
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from src.evaluation.ranking_metrics import recall_at_k
from src.feature_store.article_store import ArticleFeatureStore
from src.feature_store.user_store import UserFeatureStore
from src.retrieval.ann import ArticleIndex
from src.retrieval.embeddings import load_embeddings

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Recall@K cutoffs — same as BM25
RECALL_KS = [50, 100, 200]

# Maximum number of history articles for user representation
DEFAULT_HISTORY_CAP = 20


def _build_user_vector(
    user_history: list[dict[str, Any]],
    index: ArticleIndex,
    history_cap: int = DEFAULT_HISTORY_CAP,
) -> np.ndarray:
    """Mean-pool embeddings of the user's recent history articles.

    Mean-pooling is chosen over recency-weighted pooling or attention-based
    encoders purely for compute/time budget — it's a single matrix operation
    and gives an interpretable baseline.

    Parameters
    ----------
    user_history : list[dict]
        From ``UserFeatureStore.get_user_history()`` — list of
        ``{article_id, clicked_at}`` dicts, oldest first.
    index : ArticleIndex
        The article embedding index.
    history_cap : int
        Maximum number of recent history articles to include.

    Returns
    -------
    np.ndarray
        L2-normalized user vector, shape ``(dim,)``.
        Zero vector if no history articles have embeddings.
    """
    # Take the most recent N (history is oldest-first → take from the end)
    recent = user_history[-history_cap:]
    history_ids = [entry["article_id"] for entry in recent]

    if not history_ids:
        return np.zeros(index.dim, dtype=np.float32)

    # Get embeddings for history articles that exist in the index
    embeds, found_ids = index.get_embeddings_batch(history_ids)

    if len(found_ids) == 0:
        # No history articles have embeddings - return zero vector
        return np.zeros(index.dim, dtype=np.float32)

    # Mean-pool
    user_vec = embeds.mean(axis=0)

    # L2-normalize
    norm = np.linalg.norm(user_vec)
    if norm > 0:
        user_vec = user_vec / norm

    return user_vec.astype(np.float32)


def run_embedding_retrieval(
    dataset: str,
    model: str = "default",
    history_cap: int = DEFAULT_HISTORY_CAP,
    spot_check_users: int = 5,
) -> list[dict[str, Any]]:
    """Run embedding retrieval on the val split of a dataset.

    Returns a list of result dicts for results/embed_recall.csv.
    """
    model_label = f"{dataset}_{model}" if model != "default" else dataset
    logger.info("Running embedding retrieval on %s (model=%s)", dataset, model)
    t0 = time.time()

    # 1. Load embeddings
    embeddings, id_to_row, coverage = load_embeddings(dataset, model)
    logger.info(
        "  Loaded embeddings: %d articles, dim=%d",
        embeddings.shape[0], embeddings.shape[1],
    )
    if coverage is not None:
        logger.info("  EB-NeRD embedding coverage: %.2f%%", coverage * 100)

    # 2. Build ANN index
    # Convert id_to_row to ordered article_ids list
    article_ids_ordered = [""] * len(id_to_row)
    for aid, idx in id_to_row.items():
        article_ids_ordered[idx] = aid

    index = ArticleIndex(embeddings, article_ids_ordered)
    logger.info("  Built index: %s", index)

    # 3. Self-similarity sanity check 
    index.self_similarity_check(sample_size=5)

    # 4. Dimensionality validation
    dim = embeddings.shape[1]
    assert all(
        embeddings[i].shape[0] == dim for i in range(len(embeddings))
    ), "Embedding dimensionality inconsistency detected!"
    logger.info("  Dimensionality validation passed: all %d rows have dim=%d", len(embeddings), dim)

    # 5. Load val-split impressions 
    behaviors_path = _PROJECT_ROOT / "data" / "interim" / dataset / "behaviors.parquet"
    behaviors_df = pl.read_parquet(behaviors_path)
    val_df = behaviors_df.filter(pl.col("split") == "val")
    logger.info("  Val impressions: %d", len(val_df))

    # ── 6. Load user store ────────────────────────────────────────
    user_store = UserFeatureStore(dataset)

    # ── 7. Score each impression ──────────────────────────────────
    all_scored: list[dict[str, Any]] = []
    per_impression_recalls: dict[int, list[float]] = {k: [] for k in RECALL_KS}
    spot_check_done = 0

    for i, row in enumerate(val_df.iter_rows(named=True)):
        imp_id = row["impression_id"]
        user_id = row["user_id"]
        timestamp = row["timestamp"]
        candidates = json.loads(row["candidates"])
        labels = json.loads(row["labels"])

        # Ground truth: clicked articles
        ground_truth = set()
        for cid, label in zip(candidates, labels):
            if label == 1:
                ground_truth.add(cid)

        # Get user history (leakage-safe) — SAME accessor as BM25
        history = user_store.get_user_history(user_id, timestamp, dataset)

        # Build user vector from history embeddings
        user_vec = _build_user_vector(history, index, history_cap)

        # Score candidates via embedding similarity (restricted to impression's candidates)
        max_k = max(RECALL_KS)
        results = index.search_restricted(user_vec, candidates, k=max_k)
        ranked_ids = [r[0] for r in results]

        # Ensure all candidates appear in results
        scored_ids = {r[0] for r in results}
        for cid in candidates:
            if cid not in scored_ids:
                results.append((cid, 0.0))
                ranked_ids.append(cid)

        # Compute recall@K
        for k_val in RECALL_KS:
            r = recall_at_k(ranked_ids, ground_truth, k_val)
            per_impression_recalls[k_val].append(r)

        # Store scored candidates
        all_scored.append({
            "impression_id": imp_id,
            "user_id": user_id,
            "timestamp": str(timestamp),
            "candidates": json.dumps(candidates),
            "labels": json.dumps(labels),
            "ranked_ids": json.dumps(ranked_ids),
            "scores": json.dumps([r[1] for r in results]),
            "ground_truth": json.dumps(list(ground_truth)),
        })

        # Spot-check
        if spot_check_done < spot_check_users and ground_truth:
            user_vec_norm = np.linalg.norm(user_vec)
            print(f"\n  📋 Spot-check [{dataset}/{model}] user={user_id} imp={imp_id}")
            print(f"     History len: {len(history)}, User vector norm: {user_vec_norm:.4f}")
            print(f"     Ground truth: {ground_truth}")
            print(f"     Candidates: {len(candidates)}")
            print(f"     Top-5 embedding hits:")
            for rank, (rid, rscore) in enumerate(results[:5], 1):
                marker = " ✓" if rid in ground_truth else ""
                print(f"       {rank}. {rid} (sim={rscore:.4f}){marker}")
            spot_check_done += 1

        if (i + 1) % 2000 == 0:
            logger.info("  Processed %d/%d impressions", i + 1, len(val_df))

    elapsed = time.time() - t0

    # ── 8. Compute average recall ─────────────────────────────────
    result_rows = []
    for k_val in RECALL_KS:
        recalls = per_impression_recalls[k_val]
        avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
        result_rows.append({
            "dataset": dataset,
            "model": model,
            "retriever": "embedding",
            "K": k_val,
            "recall_at_K": round(avg_recall, 6),
            "num_impressions": len(recalls),
            "wall_clock_s": round(elapsed, 1),
        })
        logger.info("  recall@%d = %.4f (avg over %d impressions)", k_val, avg_recall, len(recalls))

    # ── 9. Validate recall monotonicity ───────────────────────────
    recall_values = [r["recall_at_K"] for r in result_rows]
    for i in range(1, len(recall_values)):
        assert recall_values[i] >= recall_values[i - 1], (
            f"Recall monotonicity violated: recall@{RECALL_KS[i-1]}={recall_values[i-1]} > "
            f"recall@{RECALL_KS[i]}={recall_values[i]}"
        )
    logger.info("  Recall monotonicity check passed")

    # ── 10. Persist scored candidates ─────────────────────────────
    out_dir = _PROJECT_ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_path = out_dir / f"embed_scores_{dataset}_{model}.parquet"
    scores_df = pl.DataFrame(all_scored)
    scores_df.write_parquet(scores_path)
    logger.info("  Saved scored candidates: %s (%.1f MB)", scores_path, scores_path.stat().st_size / 1024**2)
    logger.info("  Total wall-clock: %.1fs", elapsed)

    return result_rows


def write_results_csv(
    results: list[dict[str, Any]],
    path: Path,
    keys_to_replace: list[tuple[str, str]] | None = None,
) -> None:
    """Write results to CSV, merging with existing rows.

    Parameters
    ----------
    results : list[dict]
        New result rows.
    path : Path
        Output CSV path.
    keys_to_replace : list[tuple[str, str]], optional
        ``[(dataset, model), ...]`` — existing rows matching these
        (dataset, model) pairs are replaced.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset", "model", "retriever", "K", "recall_at_K",
        "num_impressions", "wall_clock_s",
    ]
    existing_rows: list[dict[str, Any]] = []
    if keys_to_replace and path.exists():
        replace_set = set(keys_to_replace)
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row["dataset"], row["model"])
                if key not in replace_set:
                    existing_rows.append(row)

    merged = existing_rows + results
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged)
    logger.info("Wrote results to %s (%d rows)", path, len(merged))


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Run embedding-based retrieval + recall@K evaluation on val split.",
    )
    parser.add_argument(
        "--dataset",
        choices=["mind", "ebnerd", "all"],
        default="all",
        help="Which dataset to run on (default: all).",
    )
    parser.add_argument(
        "--history-cap",
        type=int,
        default=DEFAULT_HISTORY_CAP,
        help=f"Max number of recent history articles for user vector (default: {DEFAULT_HISTORY_CAP}).",
    )
    args = parser.parse_args()

    datasets = ["mind", "ebnerd"] if args.dataset == "all" else [args.dataset]
    all_results: list[dict[str, Any]] = []
    keys_to_replace: list[tuple[str, str]] = []

    for ds in datasets:
        # Check that feature stores exist
        processed_dir = _PROJECT_ROOT / "data" / "processed" / ds
        if not (processed_dir / "article_features.parquet").exists():
            logger.error("Article features not found for %s — run `make features` first", ds)
            sys.exit(1)
        if not (processed_dir / "user_features.parquet").exists():
            logger.error("User features not found for %s — run `make features` first", ds)
            sys.exit(1)

        if ds == "ebnerd":
            # Run both BERT (primary) and Word2Vec (baseline)
            for model in ["bert", "w2v"]:
                try:
                    results = run_embedding_retrieval(ds, model=model, history_cap=args.history_cap)
                    all_results.extend(results)
                    keys_to_replace.append((ds, model))
                except FileNotFoundError as e:
                    logger.warning("Skipping %s/%s: %s", ds, model, e)
        else:
            # MIND: single model
            results = run_embedding_retrieval(ds, model="minilm", history_cap=args.history_cap)
            all_results.extend(results)
            keys_to_replace.append((ds, "minilm"))

    # Write results CSV
    results_path = _PROJECT_ROOT / "results" / "embed_recall.csv"
    write_results_csv(all_results, results_path, keys_to_replace=keys_to_replace)

    # Print summary table
    print(f"\n{'='*80}")
    print("Embedding Retrieval Recall Results")
    print(f"{'='*80}")
    print(f"{'Dataset':<10} {'Model':<10} {'K':<6} {'Recall@K':<10} {'Impressions':<12}")
    print(f"{'-'*80}")
    for r in all_results:
        print(
            f"{r['dataset']:<10} {r['model']:<10} "
            f"{r['K']:<6} {r['recall_at_K']:<10.4f} {r['num_impressions']:<12}"
        )
    print(f"{'='*80}")
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
