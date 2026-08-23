"""Memory-safe embedding retrieval on labeled validation data.

Small scale keeps the original in-memory implementation.
Large scale uses the already-built large user-feature stores and streams the
validation Parquet file in bounded batches. No full validation table or list of
all scored impressions is kept in RAM.
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
from typing import Any, Iterator

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from src.evaluation.ranking_metrics import recall_at_k
from src.feature_store.article_store import ArticleFeatureStore
from src.retrieval.ann import ArticleIndex
from src.retrieval.embeddings import load_embeddings
from src.retrieval.user_representation import build_mean_user_vector, build_mean_user_vectors

logger = logging.getLogger(__name__)

RECALL_KS = [50, 100, 200]
DEFAULT_HISTORY_CAP = 20
LARGE_BATCH_SIZE = 2000


def _build_user_vector(
    user_history: list[dict[str, Any]],
    index: ArticleIndex,
    history_cap: int = DEFAULT_HISTORY_CAP,
) -> np.ndarray:
    return build_mean_user_vector(user_history, index, history_cap=history_cap)


def _parse_candidates_labels(row: dict[str, Any]) -> tuple[list[str], list[int]]:
    candidates_raw = row["candidates"]
    labels_raw = row["labels"]
    candidates = json.loads(candidates_raw) if isinstance(candidates_raw, str) else candidates_raw
    labels = json.loads(labels_raw) if isinstance(labels_raw, str) else labels_raw
    return [str(x) for x in candidates], [int(x) for x in labels]


def _parse_mind_history(row: dict[str, Any]) -> list[dict[str, Any]]:
    raw = row.get("clicked_history")
    if raw is None:
        return []
    if isinstance(raw, str):
        return json.loads(raw) if raw else []
    return list(raw)


def _normalize_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value.replace(tzinfo=None) if value.tzinfo else value
    return datetime.fromisoformat(str(value))


def _iter_validation_batches(
    behaviors_path: Path,
    batch_size: int,
) -> Iterator[list[dict[str, Any]]]:
    parquet = pq.ParquetFile(behaviors_path)
    columns = [
        "impression_id",
        "user_id",
        "timestamp",
        "clicked_history",
        "candidates",
        "labels",
        "split",
    ]

    buffer: list[dict[str, Any]] = []
    for record_batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        rows = pa.Table.from_batches([record_batch]).to_pylist()
        for row in rows:
            if row["split"] != "val":
                continue
            buffer.append(row)
            if len(buffer) >= batch_size:
                yield buffer
                buffer = []
    if buffer:
        yield buffer


def _score_rows(
    rows: list[dict[str, Any]],
    dataset: str,
    index: ArticleIndex,
    user_store,
    history_cap: int,
    model: str,
    score_writer: pq.ParquetWriter,
    spot_check_remaining: int,
) -> tuple[dict[int, float], int, int]:
    histories: list[list[dict[str, Any]]] = []
    parsed: list[tuple[dict[str, Any], list[str], list[int], set[str]]] = []

    for row in rows:
        candidates, labels = _parse_candidates_labels(row)
        ground_truth = {cid for cid, label in zip(candidates, labels) if label == 1}
        if dataset == "mind":
            features = user_store.from_behavior_row(row)
        else:
            features = user_store.get_features(
                str(row["user_id"]),
                _normalize_timestamp(row["timestamp"]),
            )
        history = features["history"]
        histories.append(history)
        parsed.append((row, candidates, labels, ground_truth))

    user_vectors = build_mean_user_vectors(
        histories,
        index,
        history_cap=history_cap,
    )

    recall_sums = {k: 0.0 for k in RECALL_KS}
    output_rows: list[dict[str, Any]] = []
    spot_checks = 0

    for i, (row, candidates, labels, ground_truth) in enumerate(parsed):
        results = index.search_restricted(
            user_vectors[i],
            candidates,
            k=max(RECALL_KS),
        )
        scored_ids = {rid for rid, _ in results}
        if len(results) < len(candidates):
            results.extend((cid, 0.0) for cid in candidates if cid not in scored_ids)
        ranked_ids = [rid for rid, _ in results]

        for k in RECALL_KS:
            recall_sums[k] += recall_at_k(ranked_ids, ground_truth, k)

        output_rows.append({
            "impression_id": str(row["impression_id"]),
            "user_id": str(row["user_id"]),
            "timestamp": _normalize_timestamp(row["timestamp"]).isoformat(),
            "candidates": json.dumps(candidates),
            "labels": json.dumps(labels),
            "ranked_ids": json.dumps(ranked_ids),
            "scores": json.dumps([float(score) for _, score in results]),
            "ground_truth": json.dumps(sorted(ground_truth)),
        })

        if spot_checks < spot_check_remaining and ground_truth:
            print(f"\n  Spot-check [{dataset}/{model}] user={row['user_id']} imp={row['impression_id']}")
            print(f"     History len: {len(histories[i])}, User vector norm: {np.linalg.norm(user_vectors[i]):.4f}")
            print(f"     Ground truth: {ground_truth}")
            print(f"     Candidates: {len(candidates)}")
            print("     Top-5 embedding hits:")
            for rank, (rid, score) in enumerate(results[:5], 1):
                marker = " ✓" if rid in ground_truth else ""
                print(f"       {rank}. {rid} (sim={score:.4f}){marker}")
            spot_checks += 1

    score_writer.write_table(pa.Table.from_pylist(output_rows))
    return recall_sums, len(parsed), spot_checks


def _run_large(
    dataset: str,
    model: str,
    history_cap: int = DEFAULT_HISTORY_CAP,
    spot_check_users: int = 5,
) -> list[dict[str, Any]]:
    from src.common.paths import interim_dir, processed_dir, results_dir
    from src.feature_store.large_user_store import LargeUserFeatureStore

    t0 = time.time()
    embeddings, id_to_row, coverage = load_embeddings(dataset, model, scale="large")
    article_ids_ordered = [""] * len(id_to_row)
    for aid, idx in id_to_row.items():
        article_ids_ordered[idx] = aid

    # Candidate-restricted ranking does not need a full FAISS catalog index.
    index = ArticleIndex(embeddings, article_ids_ordered, build_full_index=False)
    index.self_similarity_check(sample_size=min(5, index.n_articles))

    processed = processed_dir(dataset, "large")
    user_store = LargeUserFeatureStore(dataset, processed)
    behaviors_path = interim_dir(dataset, "large") / "behaviors.parquet"
    scores_path = processed / f"embed_scores_{dataset}_{model}.parquet"
    scores_path.parent.mkdir(parents=True, exist_ok=True)

    schema = pa.schema([
        ("impression_id", pa.string()),
        ("user_id", pa.string()),
        ("timestamp", pa.string()),
        ("candidates", pa.string()),
        ("labels", pa.string()),
        ("ranked_ids", pa.string()),
        ("scores", pa.string()),
        ("ground_truth", pa.string()),
    ])

    recall_totals = {k: 0.0 for k in RECALL_KS}
    count = 0
    spot_remaining = spot_check_users

    logger.info("Streaming large validation data from %s", behaviors_path)
    if coverage is not None:
        logger.info("  Embedding coverage: %.2f%%", coverage * 100)

    with pq.ParquetWriter(scores_path, schema, compression="zstd") as writer:
        for batch_idx, rows in enumerate(_iter_validation_batches(behaviors_path, LARGE_BATCH_SIZE), 1):
            batch_sums, batch_count, used_spots = _score_rows(
                rows,
                dataset,
                index,
                user_store,
                history_cap,
                model,
                writer,
                spot_remaining,
            )
            for k in RECALL_KS:
                recall_totals[k] += batch_sums[k]
            count += batch_count
            spot_remaining -= used_spots

            if count % 20_000 < batch_count:
                elapsed_min = (time.time() - t0) / 60.0
                logger.info("  Processed %d validation impressions (%.1f min)", count, elapsed_min)

    elapsed = time.time() - t0
    results = []
    for k in RECALL_KS:
        value = recall_totals[k] / count if count else 0.0
        results.append({
            "dataset": dataset,
            "model": model,
            "retriever": "embedding",
            "K": k,
            "recall_at_K": round(value, 6),
            "num_impressions": count,
            "wall_clock_s": round(elapsed, 1),
        })
        logger.info("  recall@%d = %.4f (avg over %d impressions)", k, value, count)

    logger.info("  Saved scored candidates: %s (%.1f MB)", scores_path, scores_path.stat().st_size / 1024**2)
    logger.info("  Total wall-clock: %.1fs", elapsed)
    return results


def _run_small(
    dataset: str,
    model: str,
    history_cap: int,
) -> list[dict[str, Any]]:
    from src.common.paths import interim_dir, processed_root
    from src.feature_store.user_store import UserFeatureStore

    t0 = time.time()
    embeddings, id_to_row, _ = load_embeddings(dataset, model, scale="small")
    article_ids_ordered = [""] * len(id_to_row)
    for aid, idx in id_to_row.items():
        article_ids_ordered[idx] = aid
    index = ArticleIndex(embeddings, article_ids_ordered)

    behaviors_path = interim_dir(dataset, "small") / "behaviors.parquet"
    val_df = pl.read_parquet(behaviors_path).filter(pl.col("split") == "val")
    user_store = UserFeatureStore(dataset, scale="small")

    recalls = {k: [] for k in RECALL_KS}
    scored: list[dict[str, Any]] = []
    for row in val_df.iter_rows(named=True):
        candidates, labels = _parse_candidates_labels(row)
        truth = {cid for cid, label in zip(candidates, labels) if label == 1}
        history = user_store.get_user_history(row["user_id"], row["timestamp"], dataset)
        user_vec = _build_user_vector(history, index, history_cap)
        results = index.search_restricted(user_vec, candidates, k=max(RECALL_KS))
        if len(results) < len(candidates):
            seen = {x[0] for x in results}
            results.extend((cid, 0.0) for cid in candidates if cid not in seen)
        ranked = [x[0] for x in results]
        for k in RECALL_KS:
            recalls[k].append(recall_at_k(ranked, truth, k))
        scored.append({
            "impression_id": str(row["impression_id"]),
            "user_id": str(row["user_id"]),
            "timestamp": str(row["timestamp"]),
            "candidates": json.dumps(candidates),
            "labels": json.dumps(labels),
            "ranked_ids": json.dumps(ranked),
            "scores": json.dumps([float(s) for _, s in results]),
            "ground_truth": json.dumps(sorted(truth)),
        })

    elapsed = time.time() - t0
    out_path = processed_root("small") / f"embed_scores_{dataset}_{model}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(scored).write_parquet(out_path)

    return [{
        "dataset": dataset,
        "model": model,
        "retriever": "embedding",
        "K": k,
        "recall_at_K": round(sum(recalls[k]) / len(recalls[k]), 6) if recalls[k] else 0.0,
        "num_impressions": len(recalls[k]),
        "wall_clock_s": round(elapsed, 1),
    } for k in RECALL_KS]


def run_embedding_retrieval(
    dataset: str,
    model: str = "default",
    history_cap: int = DEFAULT_HISTORY_CAP,
    spot_check_users: int = 5,
    scale: str = "small",
) -> list[dict[str, Any]]:
    if scale == "large":
        return _run_large(dataset, model, history_cap, spot_check_users)
    return _run_small(dataset, model, history_cap)


def write_results_csv(
    results: list[dict[str, Any]],
    path: Path,
    keys_to_replace: list[tuple[str, str]] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["dataset", "model", "retriever", "K", "recall_at_K", "num_impressions", "wall_clock_s"]
    existing: list[dict[str, Any]] = []
    if keys_to_replace and path.exists():
        replace = set(keys_to_replace)
        with open(path, "r", newline="") as f:
            for row in csv.DictReader(f):
                if (row["dataset"], row["model"]) not in replace:
                    existing.append(row)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing + results)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
    parser = argparse.ArgumentParser(description="Run embedding retrieval + recall@K evaluation on val split.")
    parser.add_argument(
        "--dataset",
        choices=["mind", "ebnerd", "all"],
        default="all",
    )

    parser.add_argument(
        "--scale",
        choices=["small", "large"],
        default="small",
    )

    parser.add_argument(
        "--history-cap",
        type=int,
        default=DEFAULT_HISTORY_CAP,
    )

    parser.add_argument(
        "--model",
        choices=["minilm", "bert", "w2v"],
        default=None,
        help=(
            "Run only the specified model. "
            "For EB-NeRD use 'bert' or 'w2v'; "
            "for MIND use 'minilm'. "
            "When omitted, run the default model set."
        ),
    )
    args = parser.parse_args()

    datasets = ["mind", "ebnerd"] if args.dataset == "all" else [args.dataset]
    all_results: list[dict[str, Any]] = []
    replace: list[tuple[str, str]] = []

    from src.common.paths import processed_dir, results_dir

    for dataset in datasets:
        pdir = processed_dir(dataset, args.scale)
        if not (pdir / "article_features.parquet").exists():
            logger.error("Article features not found for %s/%s", dataset, args.scale)
            sys.exit(1)

        if args.scale == "large" and dataset == "ebnerd":
            if not (pdir / "history_index_validation" / "metadata.json").exists():
                logger.error("EB-NeRD validation history index missing")
                sys.exit(1)
        if args.scale == "small" and not (pdir / "user_features.parquet").exists():
            logger.error("Small-scale user features missing; run `make features DATA_SCALE=small`")
            sys.exit(1)

        if args.model is not None:
            if dataset == "mind" and args.model != "minilm":
                logger.error(
                    "MIND supports only --model minilm, got %s",
                    args.model,
                )
                sys.exit(1)

            if dataset == "ebnerd" and args.model == "minilm":
                logger.error(
                    "EB-NeRD does not use --model minilm"
                )
                sys.exit(1)

            models = [args.model]
        else:
            models = (
                ["bert", "w2v"]
                if dataset == "ebnerd"
                else ["minilm"]
            )
        for model in models:
            try:
                results = run_embedding_retrieval(dataset, model=model, history_cap=args.history_cap, scale=args.scale)
            except FileNotFoundError as exc:
                if dataset == "ebnerd":
                    logger.warning("Skipping %s/%s: %s", dataset, model, exc)
                    continue
                raise
            all_results.extend(results)
            replace.append((dataset, model))

    results_path = results_dir(args.scale) / "embed_recall.csv"
    write_results_csv(all_results, results_path, keys_to_replace=replace)

    print("\n" + "=" * 80)
    print("Embedding Retrieval Recall Results")
    print("=" * 80)
    for result in all_results:
        print(
            f"{result['dataset']:<10} {result['model']:<10} "
            f"K={result['K']:<3} recall={result['recall_at_K']:.4f} "
            f"n={result['num_impressions']}"
        )
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
