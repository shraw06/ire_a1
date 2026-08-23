"""BM25 retrieval pipeline runner.

Small scale:
    Uses the existing UserFeatureStore and in-memory Polars path.

Large scale:
    Uses streaming Parquet batches and dataset-native history:
      - MIND: clicked_history stored directly in each impression.
      - EB-NeRD: MemoryMappedHistoryStore for leakage-safe as-of lookup.

Results are accumulated only as scalar recall statistics and written to
Parquet incrementally; the complete validation set is never loaded into RAM.
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

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from src.evaluation.ranking_metrics import recall_at_k
from src.feature_store.article_store import ArticleFeatureStore
from src.retrieval.bm25 import BM25Engine

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

RECALL_KS = [50, 100, 200]
DEFAULT_HISTORY_CAP = 20
DEFAULT_K1 = 1.5
DEFAULT_B = 0.75
BATCH_SIZE = 10_000


def _build_query_text(
    user_history: list[dict[str, Any]],
    article_texts: dict[str, str],
    history_cap: int = DEFAULT_HISTORY_CAP,
) -> str:
    """Build a query from the most recent history articles."""
    if not user_history:
        return ""

    recent = user_history[-history_cap:]
    parts: list[str] = []

    for entry in recent:
        aid = str(entry["article_id"])
        text = article_texts.get(aid)
        if text:
            parts.append(text)

    return " ".join(parts)


def _load_article_corpus(
    dataset: str,
    scale: str,
) -> tuple[list[tuple[str, str]], dict[str, str]]:
    """Load the relatively small article corpus."""
    article_store = ArticleFeatureStore(dataset, scale=scale)
    df = article_store.get_articles_for_bm25()

    corpus: list[tuple[str, str]] = []
    article_texts: dict[str, str] = {}

    for row in df.iter_rows(named=True):
        aid = str(row["article_id"])
        text = row["cleaned_text"] or ""
        article_texts[aid] = text
        corpus.append((aid, text))

    logger.info("  Loaded %d articles for indexing", len(corpus))
    return corpus, article_texts


def _parse_timestamp(value: Any) -> datetime:
    """Normalize a Parquet timestamp to a naive datetime."""
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            return value.replace(tzinfo=None)
        return value

    return datetime.fromisoformat(str(value))


def _parse_mind_history(value: Any) -> list[dict[str, Any]]:
    """Parse the already leakage-safe MIND impression history."""
    if value is None:
        return []

    if isinstance(value, str):
        if not value:
            return []
        return json.loads(value)

    return value


def _parse_candidates_and_labels(
    candidates_value: Any,
    labels_value: Any,
) -> tuple[list[str], list[int]]:
    """Parse candidate and label JSON fields."""
    candidates = json.loads(candidates_value) if isinstance(candidates_value, str) else candidates_value
    labels = json.loads(labels_value) if isinstance(labels_value, str) else labels_value

    candidates = [str(x) for x in candidates]
    labels = [int(x) for x in labels]

    return candidates, labels


def _load_large_history_store(
    dataset: str,
    processed_dir: Path,
):
    """Return the appropriate large-scale history implementation."""
    if dataset == "ebnerd":
        from src.feature_store.history_store import MemoryMappedHistoryStore

        # Validation impressions must use the validation history index.
        index_dir = processed_dir / "history_index_validation"

        if not index_dir.exists():
            raise FileNotFoundError(
                f"EB-NeRD validation history index not found at {index_dir}. "
                "Run the large feature-store build first."
            )

        return MemoryMappedHistoryStore(index_dir)

    return None


def _iter_validation_batches(
    behaviors_path: Path,
    batch_size: int = BATCH_SIZE,
):
    """Yield only validation impressions without loading the whole file."""
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

    for batch in parquet.iter_batches(
        batch_size=batch_size,
        columns=columns,
    ):
        table = pa.Table.from_batches([batch])

        # Convert only this bounded batch to Python.
        rows = table.to_pylist()

        for row in rows:
            if row["split"] == "val":
                yield row


def _make_scores_writer(path: Path) -> pq.ParquetWriter:
    """Create an incremental Parquet writer for scored impressions."""
    schema = pa.schema(
        [
            ("impression_id", pa.string()),
            ("user_id", pa.string()),
            ("timestamp", pa.string()),
            ("candidates", pa.string()),
            ("labels", pa.string()),
            ("ranked_ids", pa.string()),
            ("scores", pa.string()),
            ("ground_truth", pa.string()),
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)

    return pq.ParquetWriter(
        path,
        schema,
        compression="zstd",
    )


def _write_score_batch(
    writer: pq.ParquetWriter,
    rows: list[dict[str, Any]],
) -> None:
    """Write a bounded result batch."""
    if not rows:
        return

    table = pa.Table.from_pylist(rows)
    writer.write_table(table)


def _large_output_name(
    dataset: str,
    text_fields: str,
    use_stopwords: bool,
    use_stemming: bool,
) -> str:
    """Produce unique output names for the four BM25 ablations."""
    sw = "sw" if use_stopwords else "nosw"
    stem = "stem" if use_stemming else "nostem"

    return f"bm25_scores_{dataset}_{text_fields}_{sw}_{stem}.parquet"


def run_bm25_large(
    dataset: str,
    use_stopwords: bool = True,
    use_stemming: bool = False,
    text_fields: str = "title_abstract",
    history_cap: int = DEFAULT_HISTORY_CAP,
    k1: float = DEFAULT_K1,
    b: float = DEFAULT_B,
    spot_check_users: int = 5,
) -> list[dict[str, Any]]:
    """Memory-bounded BM25 evaluation for a large dataset."""

    from src.common.paths import interim_dir, processed_dir, results_dir

    logger.info(
        "Running LARGE BM25 on %s "
        "(stopwords=%s, stemming=%s, text=%s, history_cap=%d)",
        dataset,
        use_stopwords,
        use_stemming,
        text_fields,
        history_cap,
    )

    t0 = time.time()

    # ---------------------------------------------------------------
    # 1. Article corpus
    # ---------------------------------------------------------------
    corpus, article_texts = _load_article_corpus(
        dataset,
        scale="large",
    )

    # ---------------------------------------------------------------
    # 2. BM25 index
    # ---------------------------------------------------------------
    t_idx = time.time()

    engine = BM25Engine.from_corpus(
        corpus,
        dataset=dataset,
        use_stopwords=use_stopwords,
        use_stemming=use_stemming,
        k1=k1,
        b=b,
    )

    logger.info(
        "  Index built in %.1fs: %s",
        time.time() - t_idx,
        engine,
    )

    # ---------------------------------------------------------------
    # 3. Large-scale history source
    # ---------------------------------------------------------------
    processed = processed_dir(dataset, "large")

    history_store = _load_large_history_store(
        dataset,
        processed,
    )

    # ---------------------------------------------------------------
    # 4. Streaming validation
    # ---------------------------------------------------------------
    behaviors_path = interim_dir(dataset, "large") / "behaviors.parquet"

    if not behaviors_path.exists():
        raise FileNotFoundError(behaviors_path)

    logger.info(
        "  Streaming validation impressions from %s",
        behaviors_path,
    )

    per_impression_sum: dict[int, float] = {
        k: 0.0 for k in RECALL_KS
    }

    num_impressions = 0
    spot_check_done = 0

    scores_path = (
        processed
        / _large_output_name(
            dataset,
            text_fields,
            use_stopwords,
            use_stemming,
        )
    )

    writer = _make_scores_writer(scores_path)
    score_buffer: list[dict[str, Any]] = []

    try:
        for row in _iter_validation_batches(
            behaviors_path,
            batch_size=BATCH_SIZE,
        ):
            imp_id = str(row["impression_id"])
            user_id = str(row["user_id"])
            timestamp = _parse_timestamp(row["timestamp"])

            candidates, labels = _parse_candidates_and_labels(
                row["candidates"],
                row["labels"],
            )

            ground_truth = {
                cid
                for cid, label in zip(candidates, labels)
                if label == 1
            }

            # -------------------------------------------------------
            # Dataset-specific leakage-safe history
            # -------------------------------------------------------
            if dataset == "mind":
                history = _parse_mind_history(
                    row["clicked_history"]
                )
            else:
                history = history_store.get_history(
                    user_id,
                    timestamp,
                )

            query_text = _build_query_text(
                history,
                article_texts,
                history_cap,
            )

            # -------------------------------------------------------
            # Candidate-restricted BM25
            # -------------------------------------------------------
            results = engine.rank(
                query_text,
                candidate_ids=candidates,
                top_k=max(RECALL_KS),
            )

            ranked_ids = [r[0] for r in results]

            # -------------------------------------------------------
            # Recall
            # -------------------------------------------------------
            for k in RECALL_KS:
                per_impression_sum[k] += recall_at_k(
                    ranked_ids,
                    ground_truth,
                    k,
                )

            num_impressions += 1

            # -------------------------------------------------------
            # Incremental score output
            # -------------------------------------------------------
            score_buffer.append(
                {
                    "impression_id": imp_id,
                    "user_id": user_id,
                    "timestamp": timestamp.isoformat(),
                    "candidates": json.dumps(candidates),
                    "labels": json.dumps(labels),
                    "ranked_ids": json.dumps(ranked_ids),
                    "scores": json.dumps([r[1] for r in results]),
                    "ground_truth": json.dumps(
                        sorted(ground_truth)
                    ),
                }
            )

            if len(score_buffer) >= BATCH_SIZE:
                _write_score_batch(writer, score_buffer)
                score_buffer.clear()

            # -------------------------------------------------------
            # Spot check
            # -------------------------------------------------------
            if (
                spot_check_done < spot_check_users
                and ground_truth
            ):
                print(
                    f"\n  Spot-check [{dataset}] "
                    f"user={user_id} imp={imp_id}"
                )
                print(
                    f"     History len: {len(history)}, "
                    f"Query tokens: {len(query_text.split())}"
                )
                print(
                    f"     Ground truth: {ground_truth}"
                )
                print(
                    f"     Candidates: {len(candidates)}"
                )
                print("     Top-5 BM25 hits:")

                for rank, (rid, score) in enumerate(
                    results[:5],
                    1,
                ):
                    marker = (
                        " ✓"
                        if rid in ground_truth
                        else ""
                    )
                    snippet = article_texts.get(
                        str(rid),
                        "",
                    )[:80]

                    if snippet:
                        snippet += "..."

                    print(
                        f"       {rank}. {rid} "
                        f"(score={score:.4f})"
                        f"{marker} — {snippet}"
                    )

                spot_check_done += 1

            if num_impressions % 2_000 == 0:
                elapsed = time.time() - t0

                logger.info(
                    "  Processed %d validation impressions "
                    "(%.1f min)",
                    num_impressions,
                    elapsed / 60.0,
                )

    finally:
        if score_buffer:
            _write_score_batch(
                writer,
                score_buffer,
            )

        writer.close()

    # ---------------------------------------------------------------
    # 5. Final recall
    # ---------------------------------------------------------------
    elapsed = time.time() - t0

    result_rows: list[dict[str, Any]] = []

    for k in RECALL_KS:
        avg_recall = (
            per_impression_sum[k] / num_impressions
            if num_impressions
            else 0.0
        )

        result_rows.append(
            {
                "dataset": dataset,
                "text_fields": text_fields,
                "use_stopwords": use_stopwords,
                "use_stemming": use_stemming,
                "K": k,
                "recall_at_K": round(
                    avg_recall,
                    6,
                ),
                "num_impressions": num_impressions,
                "wall_clock_s": round(
                    elapsed,
                    1,
                ),
            }
        )

        logger.info(
            "  recall@%d = %.4f "
            "(avg over %d impressions)",
            k,
            avg_recall,
            num_impressions,
        )

    logger.info(
        "  Saved scored candidates: %s (%.1f MB)",
        scores_path,
        scores_path.stat().st_size / 1024**2,
    )

    logger.info(
        "  Total wall-clock: %.1fs",
        elapsed,
    )

    return result_rows


def _run_bm25_small(
    dataset: str,
    use_stopwords: bool,
    use_stemming: bool,
    text_fields: str,
    history_cap: int,
) -> list[dict[str, Any]]:
    """Preserve the existing small-data implementation."""

    from src.common.paths import (
        interim_dir,
        processed_root,
        results_dir,
    )
    from src.feature_store.user_store import UserFeatureStore

    t0 = time.time()

    corpus, article_texts = _load_article_corpus(
        dataset,
        scale="small",
    )

    engine = BM25Engine.from_corpus(
        corpus,
        dataset=dataset,
        use_stopwords=use_stopwords,
        use_stemming=use_stemming,
        k1=DEFAULT_K1,
        b=DEFAULT_B,
    )

    behaviors_path = (
        interim_dir(dataset, "small")
        / "behaviors.parquet"
    )

    behaviors_df = pl.read_parquet(
        behaviors_path
    )
    val_df = behaviors_df.filter(
        pl.col("split") == "val"
    )

    user_store = UserFeatureStore(
        dataset,
        scale="small",
    )

    per_impression_recalls = {
        k: [] for k in RECALL_KS
    }

    all_scored: list[dict[str, Any]] = []

    for row in val_df.iter_rows(named=True):
        imp_id = row["impression_id"]
        user_id = row["user_id"]
        timestamp = row["timestamp"]

        candidates, labels = (
            _parse_candidates_and_labels(
                row["candidates"],
                row["labels"],
            )
        )

        ground_truth = {
            cid
            for cid, label in zip(candidates, labels)
            if label == 1
        }

        history = user_store.get_user_history(
            user_id,
            timestamp,
            dataset,
        )

        query_text = _build_query_text(
            history,
            article_texts,
            history_cap,
        )

        results = engine.rank(
            query_text,
            candidate_ids=candidates,
            top_k=max(RECALL_KS),
        )

        ranked_ids = [r[0] for r in results]

        for k in RECALL_KS:
            per_impression_recalls[k].append(
                recall_at_k(
                    ranked_ids,
                    ground_truth,
                    k,
                )
            )

        all_scored.append(
            {
                "impression_id": imp_id,
                "user_id": user_id,
                "timestamp": str(timestamp),
                "candidates": json.dumps(candidates),
                "labels": json.dumps(labels),
                "ranked_ids": json.dumps(ranked_ids),
                "scores": json.dumps(
                    [r[1] for r in results]
                ),
                "ground_truth": json.dumps(
                    sorted(ground_truth)
                ),
            }
        )

    elapsed = time.time() - t0

    results = []

    for k in RECALL_KS:
        recalls = per_impression_recalls[k]

        avg_recall = (
            sum(recalls) / len(recalls)
            if recalls
            else 0.0
        )

        results.append(
            {
                "dataset": dataset,
                "text_fields": text_fields,
                "use_stopwords": use_stopwords,
                "use_stemming": use_stemming,
                "K": k,
                "recall_at_K": round(
                    avg_recall,
                    6,
                ),
                "num_impressions": len(recalls),
                "wall_clock_s": round(
                    elapsed,
                    1,
                ),
            }
        )

    out_dir = processed_root("small")
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    scores_path = (
        out_dir
        / f"bm25_scores_{dataset}_{text_fields}.parquet"
    )

    pl.DataFrame(all_scored).write_parquet(
        scores_path
    )

    return results


def run_bm25_for_dataset(
    dataset: str,
    use_stopwords: bool = True,
    use_stemming: bool = False,
    text_fields: str = "title_abstract",
    history_cap: int = DEFAULT_HISTORY_CAP,
    k1: float = DEFAULT_K1,
    b: float = DEFAULT_B,
    spot_check_users: int = 5,
    scale: str = "small",
) -> list[dict[str, Any]]:
    """Dispatch to the appropriate scale-specific implementation."""

    if scale == "large":
        return run_bm25_large(
            dataset=dataset,
            use_stopwords=use_stopwords,
            use_stemming=use_stemming,
            text_fields=text_fields,
            history_cap=history_cap,
            k1=k1,
            b=b,
            spot_check_users=spot_check_users,
        )

    return _run_bm25_small(
        dataset=dataset,
        use_stopwords=use_stopwords,
        use_stemming=use_stemming,
        text_fields=text_fields,
        history_cap=history_cap,
    )


def run_ablation(
    dataset: str,
    history_cap: int = DEFAULT_HISTORY_CAP,
    scale: str = "small",
) -> list[dict[str, Any]]:
    """Run the four stopword/stemming configurations."""

    configs = [
        (True, False),
        (False, False),
        (True, True),
        (False, True),
    ]

    all_results: list[dict[str, Any]] = []

    for use_sw, use_stem in configs:
        logger.info(
            "=" * 70
        )
        logger.info(
            "BM25 configuration: "
            "stopwords=%s, stemming=%s",
            use_sw,
            use_stem,
        )

        results = run_bm25_for_dataset(
            dataset,
            use_stopwords=use_sw,
            use_stemming=use_stem,
            history_cap=history_cap,
            scale=scale,
        )

        all_results.extend(results)

    return all_results


def write_results_csv(
    results: list[dict[str, Any]],
    path: Path,
    datasets_to_replace: list[str] | None = None,
) -> None:
    """Write recall results while preserving other datasets."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "dataset",
        "text_fields",
        "use_stopwords",
        "use_stemming",
        "K",
        "recall_at_K",
        "num_impressions",
        "wall_clock_s",
    ]

    existing_rows: list[dict[str, Any]] = []

    if datasets_to_replace and path.exists():
        with open(
            path,
            "r",
            newline="",
        ) as f:
            reader = csv.DictReader(f)

            for row in reader:
                if (
                    row["dataset"]
                    not in datasets_to_replace
                ):
                    existing_rows.append(row)

    merged = existing_rows + results

    with open(
        path,
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(merged)

    logger.info(
        "Wrote results to %s (%d rows)",
        path,
        len(merged),
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s [%(levelname)s] "
            "%(name)s — %(message)s"
        ),
    )

    parser = argparse.ArgumentParser(
        description=(
            "Run BM25 retrieval + recall@K "
            "evaluation on val split."
        )
    )

    parser.add_argument(
        "--dataset",
        choices=["mind", "ebnerd", "all"],
        default="all",
    )

    parser.add_argument(
        "--history-cap",
        type=int,
        default=DEFAULT_HISTORY_CAP,
    )

    parser.add_argument(
        "--scale",
        choices=["small", "large"],
        default="small",
    )

    parser.add_argument(
        "--no-ablation",
        action="store_true",
        help=(
            "Run only the default configuration "
            "(stopwords=True, stemming=False)."
        ),
    )

    args = parser.parse_args()

    datasets = (
        ["mind", "ebnerd"]
        if args.dataset == "all"
        else [args.dataset]
    )

    all_results: list[dict[str, Any]] = []

    for dataset in datasets:
        from src.common.paths import (
            processed_dir as scale_processed_dir,
            results_dir,
        )

        processed = scale_processed_dir(
            dataset,
            args.scale,
        )

        if not (
            processed
            / "article_features.parquet"
        ).exists():
            logger.error(
                "Article features not found for %s. "
                "Run the feature-store build first.",
                dataset,
            )
            sys.exit(1)

        if args.scale == "large":
            if dataset == "ebnerd":
                if not (
                    processed
                    / "history_index_validation"
                    / "metadata.json"
                ).exists():
                    logger.error(
                        "EB-NeRD validation history index "
                        "not found for large scale."
                    )
                    sys.exit(1)
        else:
            if not (
                processed
                / "user_features.parquet"
            ).exists():
                logger.error(
                    "User features not found for %s. "
                    "Run `make features` first.",
                    dataset,
                )
                sys.exit(1)

        if args.no_ablation:
            results = run_bm25_for_dataset(
                dataset,
                history_cap=args.history_cap,
                scale=args.scale,
            )
        else:
            results = run_ablation(
                dataset,
                history_cap=args.history_cap,
                scale=args.scale,
            )

        all_results.extend(results)

    results_path = (
        results_dir(args.scale)
        / "bm25_recall.csv"
    )

    write_results_csv(
        all_results,
        results_path,
        datasets_to_replace=datasets,
    )

    print("\n" + "=" * 80)
    print("BM25 Recall Results")
    print("=" * 80)

    print(
        f"{'Dataset':<10} "
        f"{'Text Fields':<18} "
        f"{'Stopwords':<10} "
        f"{'Stemming':<10} "
        f"{'K':<6} "
        f"{'Recall@K':<10}"
    )

    print("-" * 80)

    for result in all_results:
        print(
            f"{result['dataset']:<10} "
            f"{result['text_fields']:<18} "
            f"{str(result['use_stopwords']):<10} "
            f"{str(result['use_stemming']):<10} "
            f"{result['K']:<6} "
            f"{result['recall_at_K']:<10.4f}"
        )

    print("=" * 80)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()