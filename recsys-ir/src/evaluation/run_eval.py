"""Offline evaluation harness for labeled validation retrieval results.

Small scale keeps the existing in-memory behavior.

Large scale:
    * streams large behavior/score Parquet files;
    * never constructs a legacy large user_features.parquet;
    * uses MIND impression-level histories and EB-NeRD validation history indexes;
    * reuses already-generated retrieval score files;
    * keeps metric arrays rather than millions of Python record dictionaries;
    * writes the required offline summaries for the labeled validation data only.

The unlabeled Codabench test sets are not read by this module.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from src.evaluation.ranking_metrics import auc_score, mrr, ndcg_at_k
from src.evaluation.beyond_accuracy import (
    compute_coverage,
    compute_intra_list_diversity,
    compute_novelty,
)
from src.evaluation.slicing import get_article_slice, get_user_slice
from src.evaluation.bootstrap import compute_bootstrap_ci

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

STREAM_BATCH_SIZE = 20_000
RECALL_KS = [5, 10, 50, 100, 200]
METRICS = ["AUC", "MRR", "nDCG@5", "nDCG@10", "ILD", "Novelty"]

# The completed large BM25 baseline selected for comparison/evaluation.
BM25_CONFIG_TAG = "sw_nostem"


def _parse_json(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


def _load_popularity(
    dataset: str,
    split: str | None,
    scale: str,
) -> dict[str, int]:
    """Count candidate appearances from the specified labeled split.

    Large scale is streamed with PyArrow to avoid materializing the behavior
    table in RAM.
    """
    from src.common.paths import interim_dir

    path = interim_dir(dataset, scale) / "behaviors.parquet"

    if scale == "small":
        df = pl.read_parquet(path)

        if split is not None:
            df = df.filter(pl.col("split") == split)

        popularity: dict[str, int] = {}
        for row in df.iter_rows(named=True):
            for cid in _parse_json(row["candidates"]):
                cid = str(cid)
                popularity[cid] = popularity.get(cid, 0) + 1
        return popularity

    popularity: dict[str, int] = {}
    pf = pq.ParquetFile(path)

    columns = ["candidates", "split"] if split is not None else ["candidates"]

    for batch in pf.iter_batches(
        batch_size=STREAM_BATCH_SIZE,
        columns=columns,
    ):
        candidates_col = batch.column(0).to_pylist()
        split_col = batch.column(1).to_pylist() if split is not None else None

        for i, raw_candidates in enumerate(candidates_col):
            if split_col is not None and split_col[i] != split:
                continue
            if raw_candidates is None:
                continue

            for cid in _parse_json(raw_candidates):
                cid = str(cid)
                popularity[cid] = popularity.get(cid, 0) + 1

    return popularity


def _load_user_history_lens(
    dataset: str,
    scale: str,
) -> dict[str, int]:
    """Return user history lengths for cold/warm slicing."""
    if scale == "small":
        from src.feature_store.user_store import UserFeatureStore

        store = UserFeatureStore(dataset, scale=scale)
        rows = store._store.query_sql(
            f"SELECT user_id, history_len FROM {store._store.alias}"
        )
        return {
            str(r["user_id"]): int(r["history_len"])
            for r in rows
        }

    if dataset == "ebnerd":
        from src.common.paths import processed_dir

        index_dir = (
            processed_dir(dataset, scale)
            / "history_index_validation"
        )
        user_ids = np.load(
            index_dir / "user_ids.npy",
            mmap_mode="r",
        )
        offsets = np.load(
            index_dir / "offsets.npy",
            mmap_mode="r",
        )

        return {
            str(int(uid)): int(offsets[i + 1] - offsets[i])
            for i, uid in enumerate(user_ids)
        }

    # MIND-large: derive the largest observed as-of history snapshot per user
    # from the existing impression-level feature already present in behaviors.
    from src.common.paths import interim_dir

    path = interim_dir(dataset, scale) / "behaviors.parquet"
    out: dict[str, int] = {}

    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(
        batch_size=STREAM_BATCH_SIZE,
        columns=["user_id", "clicked_history"],
    ):
        users = batch.column(0).to_pylist()
        histories = batch.column(1).to_pylist()

        for uid, raw_history in zip(users, histories):
            uid = str(uid)
            history = _parse_json(raw_history) if raw_history else []
            n = len(history)
            if n > out.get(uid, 0):
                out[uid] = n

    return out


def _load_retrieval_embeddings(
    dataset: str,
    model: str,
    scale: str,
) -> dict[str, np.ndarray]:
    """Load the same article embedding catalog used during retrieval."""
    from src.retrieval.embeddings import load_embeddings

    embeddings, id_to_row, _ = load_embeddings(
        dataset,
        model,
        scale=scale,
    )

    return {
        str(article_id): embeddings[row]
        for article_id, row in id_to_row.items()
    }


def _score_path_candidates(
    dataset: str,
    retriever: str,
    scale: str,
) -> list[Path]:
    """Return valid score-file candidates, newest/most specific first."""
    from src.common.paths import processed_dir, processed_root

    if scale == "large":
        root = processed_dir(dataset, scale)
        if retriever == "bm25":
            return [
                root
                / f"bm25_scores_{dataset}_title_abstract_{BM25_CONFIG_TAG}.parquet",
                root
                / f"bm25_scores_{dataset}_title_abstract.parquet",
            ]
        return [
            root / f"embed_scores_{dataset}_{retriever}.parquet"
        ]

    root = processed_root(scale)
    if retriever == "bm25":
        return [
            root
            / f"bm25_scores_{dataset}_title_abstract.parquet"
        ]
    return [
        root / f"embed_scores_{dataset}_{retriever}.parquet"
    ]


def _find_score_path(
    dataset: str,
    retriever: str,
    scale: str,
) -> Path | None:
    for path in _score_path_candidates(
        dataset,
        retriever,
        scale,
    ):
        if path.exists():
            return path
    return None


def _iter_score_rows(
    score_path: Path,
) -> Iterator[dict[str, Any]]:
    """Stream only fields needed by the evaluation harness."""
    pf = pq.ParquetFile(score_path)

    columns = [
        "impression_id",
        "user_id",
        "candidates",
        "labels",
        "ranked_ids",
        "scores",
    ]

    for batch in pf.iter_batches(
        batch_size=STREAM_BATCH_SIZE,
        columns=columns,
    ):
        table = pa.Table.from_batches([batch])
        yield from table.to_pylist()


class _MetricAccumulator:
    """Memory-bounded numeric accumulator for sliced offline evaluation.

    Lists contain only numeric metric values and therefore remain much smaller
    than a Python dict per impression.
    """

    SLICES = (
        "all",
        "cold_fixed",
        "cold_data",
        "warm_fixed",
        "warm_data",
        "tail_fixed",
        "tail_data",
        "head_fixed",
        "head_data",
    )

    def __init__(self) -> None:
        self.values: dict[
            str,
            dict[str, list[float]]
        ] = {
            s: {m: [] for m in METRICS}
            for s in self.SLICES
        }
        self.top10: dict[str, set[str]] = {
            s: set() for s in self.SLICES
        }
        self.total_impressions = 0

    def add(
        self,
        metrics: dict[str, float],
        top10: list[str],
        user_slice_fixed: str,
        user_slice_data: str,
        article_slice_fixed: str,
        article_slice_data: str,
    ) -> None:
        slices = {
            "all",
            f"{user_slice_fixed}_fixed",
            f"{user_slice_data}_data",
            f"{article_slice_fixed}_fixed",
            f"{article_slice_data}_data",
        }

        for s in slices:
            for metric_name in METRICS:
                self.values[s][metric_name].append(
                    float(metrics[metric_name])
                )
            self.top10[s].update(top10)

        self.total_impressions += 1


def _evaluate_score_file(
    score_path: Path,
    dataset: str,
    retriever: str,
    train_pop: dict[str, int],
    full_pop: dict[str, int],
    user_hist: dict[str, int],
    embeddings: dict[str, np.ndarray],
) -> _MetricAccumulator:
    """Evaluate one retrieval score file as a stream."""
    accumulator = _MetricAccumulator()

    total_train_items = sum(train_pop.values())
    total_full_items = sum(full_pop.values())

    for i, row in enumerate(_iter_score_rows(score_path), start=1):
        if i % 20_000 == 0:
            logger.info(
                "  %s: evaluated %d impressions",
                retriever,
                i,
            )

        candidates = [
            str(x) for x in _parse_json(row["candidates"])
        ]
        labels = [
            int(x) for x in _parse_json(row["labels"])
        ]
        ranked_ids = [
            str(x) for x in _parse_json(row["ranked_ids"])
        ]
        scores = [
            float(x) for x in _parse_json(row["scores"])
        ]

        if not candidates:
            continue

        cand_to_label = dict(
            zip(candidates, labels)
        )
        ranked_labels = [
            cand_to_label.get(rid, 0)
            for rid in ranked_ids
        ]

        auc = auc_score(
            ranked_labels,
            scores,
        )
        mrr_val = mrr(
            ranked_labels,
            scores,
        )
        ndcg5 = ndcg_at_k(
            ranked_labels,
            scores,
            5,
        )
        ndcg10 = ndcg_at_k(
            ranked_labels,
            scores,
            10,
        )

        top10 = ranked_ids[:10]

        novelty_train = compute_novelty(
            top10,
            train_pop,
            total_train_items,
        )
        novelty_full = compute_novelty(
            top10,
            full_pop,
            total_full_items,
        )

        ild = 0.0
        if embeddings:
            top_embs = [
                embeddings.get(rid)
                for rid in top10
                if rid in embeddings
            ]
            if len(top_embs) > 1:
                ild = compute_intra_list_diversity(
                    np.vstack(top_embs)
                )

        user_id = str(row["user_id"])
        history_len = user_hist.get(
            user_id,
            0,
        )

        gt_ids = [
            cid
            for cid, lbl in cand_to_label.items()
            if lbl == 1
        ]

        avg_pop_train = (
            float(
                np.mean(
                    [
                        train_pop.get(
                            aid,
                            0,
                        )
                        for aid in gt_ids
                    ]
                )
            )
            if gt_ids
            else 0.0
        )

        user_slice_fixed = get_user_slice(
            history_len,
            dataset,
            "fixed",
        )
        user_slice_data = get_user_slice(
            history_len,
            dataset,
            "data-driven",
        )
        article_slice_fixed = get_article_slice(
            avg_pop_train,
            dataset,
            "fixed",
        )
        article_slice_data = get_article_slice(
            avg_pop_train,
            dataset,
            "data-driven",
        )

        base_metrics = {
            "AUC": float(auc),
            "MRR": float(mrr_val),
            "nDCG@5": float(ndcg5),
            "nDCG@10": float(ndcg10),
            "ILD": float(ild),
            # Store train-based novelty as the default. The full-pop novelty
            # is retained separately below when summaries are generated.
            "Novelty": float(novelty_train),
        }

        accumulator.add(
            base_metrics,
            top10,
            user_slice_fixed,
            user_slice_data,
            article_slice_fixed,
            article_slice_data,
        )

        # Store the full-pop novelty as a parallel hidden metric list. This is
        # needed for the "leaked" summary while keeping the same slice counts.
        for s in (
            "all",
            f"{user_slice_fixed}_fixed",
            f"{user_slice_data}_data",
            f"{article_slice_fixed}_fixed",
            f"{article_slice_data}_data",
        ):
            # Lazily create the parallel array dictionary.
            full_store = getattr(
                accumulator,
                "_full_novelty",
                None,
            )
            if full_store is None:
                full_store = {
                    slice_name: []
                    for slice_name in _MetricAccumulator.SLICES
                }
                accumulator._full_novelty = full_store

            full_store[s].append(
                float(novelty_full)
            )

    return accumulator


def _bootstrap_summary_value(
    values: list[float],
) -> tuple[float, Any, Any]:
    """Compute the requested mean and bootstrap CI."""
    if not values:
        return 0.0, "insufficient_n", "insufficient_n"

    arr = np.asarray(
        values,
        dtype=np.float64,
    )

    if arr.size < 2:
        return (
            float(arr.mean()),
            "insufficient_n",
            "insufficient_n",
        )

    return compute_bootstrap_ci(
        arr,
        b=1000,
    )


def _summary_from_accumulator(
    accumulator: _MetricAccumulator,
    dataset: str,
    retriever: str,
    catalog_size: int,
    use_full_novelty: bool,
) -> pl.DataFrame:
    """Convert numeric accumulators into the assignment summary table."""
    rows: list[dict[str, Any]] = []

    total_n = accumulator.total_impressions

    for slice_name in _MetricAccumulator.SLICES:
        counts = len(
            accumulator.values[slice_name]["AUC"]
        )

        if counts == 0:
            continue

        frac = (
            counts / total_n
            if total_n
            else 0.0
        )

        flagged_small_slice = (
            slice_name != "all"
            and (frac < 0.01 or frac > 0.99)
        )

        top10 = accumulator.top10[slice_name]
        coverage = compute_coverage(
            top10,
            catalog_size,
        )

        row: dict[str, Any] = {
            "dataset": dataset,
            "retriever": retriever,
            "slice": slice_name,
            "n_impressions": counts,
            "frac_population": frac,
            "Coverage": coverage,
            "flagged_small_slice": flagged_small_slice,
        }

        for metric_name in METRICS:
            if metric_name == "Novelty" and use_full_novelty:
                values = accumulator._full_novelty[slice_name]
            else:
                values = accumulator.values[slice_name][
                    metric_name
                ]

            if (
                flagged_small_slice
                or len(values) < 2
            ):
                mean_val = (
                    float(np.mean(values))
                    if values
                    else 0.0
                )
                ci_low = "insufficient_n"
                ci_high = "insufficient_n"
            else:
                mean_val, ci_low, ci_high = (
                    _bootstrap_summary_value(values)
                )

            row[metric_name] = mean_val
            row[f"{metric_name}_CI_low"] = ci_low
            row[f"{metric_name}_CI_high"] = ci_high

        rows.append(row)

    return pl.DataFrame(rows)


def _run_large(scale: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Run memory-safe evaluation for large labeled validation results."""
    from src.common.paths import results_dir

    datasets = ["mind", "ebnerd"]

    all_unleaked: list[pl.DataFrame] = []
    all_leaked: list[pl.DataFrame] = []

    for dataset in datasets:
        logger.info(
            "Processing large labeled validation: %s",
            dataset,
        )

        train_pop = _load_popularity(
            dataset,
            "train",
            scale,
        )
        full_pop = _load_popularity(
            dataset,
            None,
            scale,
        )

        user_hist = _load_user_history_lens(
            dataset,
            scale,
        )

        # Use the same primary semantic model as the comparison:
        # MIND MiniLM, EB-NeRD Word2Vec.
        semantic_model = (
            "minilm"
            if dataset == "mind"
            else "w2v"
        )

        embeddings = _load_retrieval_embeddings(
            dataset,
            semantic_model,
            scale,
        )

        if dataset == "mind":
            retrievers = [
                ("bm25", "bm25"),
                ("embed_minilm", "minilm"),
            ]
        else:
            retrievers = [
                ("bm25", "bm25"),
                ("embed_w2v", "w2v"),
            ]

        catalog_size = len(full_pop)

        for retriever_label, score_model in retrievers:
            score_path = _find_score_path(
                dataset,
                score_model,
                scale,
            )

            if score_path is None:
                logger.warning(
                    "Skipping %s/%s: score file missing",
                    dataset,
                    retriever_label,
                )
                continue

            logger.info(
                "Evaluating %s/%s from %s",
                dataset,
                retriever_label,
                score_path,
            )

            accumulator = _evaluate_score_file(
                score_path,
                dataset,
                retriever_label,
                train_pop,
                full_pop,
                user_hist,
                embeddings,
            )

            all_unleaked.append(
                _summary_from_accumulator(
                    accumulator,
                    dataset,
                    retriever_label,
                    catalog_size,
                    use_full_novelty=False,
                )
            )

            all_leaked.append(
                _summary_from_accumulator(
                    accumulator,
                    dataset,
                    retriever_label,
                    catalog_size,
                    use_full_novelty=True,
                )
            )

            logger.info(
                "Completed %s/%s: %d impressions",
                dataset,
                retriever_label,
                accumulator.total_impressions,
            )

    if all_unleaked:
        unleaked = pl.concat(
            all_unleaked,
            how="vertical",
        )
    else:
        unleaked = pl.DataFrame()

    if all_leaked:
        leaked = pl.concat(
            all_leaked,
            how="vertical",
        )
    else:
        leaked = pl.DataFrame()

    out_dir = results_dir(scale)
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not unleaked.is_empty():
        unleaked.write_csv(
            out_dir / "eval_summary_stripped.csv"
        )

    if not leaked.is_empty():
        leaked.write_csv(
            out_dir / "eval_summary.csv"
        )

    return unleaked, leaked


def _run_small(scale: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Preserve the original small-scale evaluation behavior."""
    from src.common.paths import processed_root, results_dir

    datasets = ["mind", "ebnerd"]
    all_unleaked: list[dict[str, Any]] = []
    all_leaked: list[dict[str, Any]] = []

    catalog_sizes: dict[str, int] = {}

    for dataset in datasets:
        logger.info(
            "Processing %s",
            dataset,
        )

        train_pop = _load_popularity(
            dataset,
            "train",
            scale,
        )
        full_pop = _load_popularity(
            dataset,
            None,
            scale,
        )
        user_hist = _load_user_history_lens(
            dataset,
            scale,
        )

        catalog_sizes[dataset] = len(full_pop)

        model = (
            "minilm"
            if dataset == "mind"
            else "w2v"
        )

        embeddings = _load_retrieval_embeddings(
            dataset,
            model,
            scale,
        )

        root = processed_root(scale)

        if dataset == "mind":
            retrievers = [
                ("bm25", "bm25_scores_mind_title_abstract.parquet"),
                ("embed_minilm", "embed_scores_mind_minilm.parquet"),
            ]
        else:
            retrievers = [
                ("bm25", "bm25_scores_ebnerd_title_abstract.parquet"),
                ("embed_w2v", "embed_scores_ebnerd_w2v.parquet"),
            ]

        for retriever, filename in retrievers:
            score_path = root / filename
            if not score_path.exists():
                continue

            df = pl.read_parquet(score_path)

            train_total = sum(train_pop.values())
            full_total = sum(full_pop.values())

            for row in df.iter_rows(named=True):
                candidates = _parse_json(row["candidates"])
                labels = _parse_json(row["labels"])
                ranked_ids = _parse_json(row["ranked_ids"])
                scores = _parse_json(row["scores"])

                if not candidates:
                    continue

                cand_to_label = dict(
                    zip(candidates, labels)
                )
                ranked_labels = [
                    cand_to_label.get(rid, 0)
                    for rid in ranked_ids
                ]

                top10 = ranked_ids[:10]

                gt_ids = [
                    cid
                    for cid, lbl in cand_to_label.items()
                    if lbl == 1
                ]

                history_len = user_hist.get(
                    str(row["user_id"]),
                    0,
                )

                avg_pop_train = (
                    np.mean(
                        [
                            train_pop.get(aid, 0)
                            for aid in gt_ids
                        ]
                    )
                    if gt_ids
                    else 0.0
                )

                common = {
                    "dataset": dataset,
                    "retriever": retriever,
                    "AUC": auc_score(
                        ranked_labels,
                        scores,
                    ),
                    "MRR": mrr(
                        ranked_labels,
                        scores,
                    ),
                    "nDCG@5": ndcg_at_k(
                        ranked_labels,
                        scores,
                        5,
                    ),
                    "nDCG@10": ndcg_at_k(
                        ranked_labels,
                        scores,
                        10,
                    ),
                    "ILD": (
                        compute_intra_list_diversity(
                            np.vstack(
                                [
                                    embeddings[rid]
                                    for rid in top10
                                    if rid in embeddings
                                ]
                            )
                        )
                        if len(
                            [
                                rid
                                for rid in top10
                                if rid in embeddings
                            ]
                        ) > 1
                        else 0.0
                    ),
                    "slice_cold_fixed": get_user_slice(
                        history_len,
                        dataset,
                        "fixed",
                    ),
                    "slice_cold_data": get_user_slice(
                        history_len,
                        dataset,
                        "data-driven",
                    ),
                    "slice_tail_fixed": get_article_slice(
                        avg_pop_train,
                        dataset,
                        "fixed",
                    ),
                    "slice_tail_data": get_article_slice(
                        avg_pop_train,
                        dataset,
                        "data-driven",
                    ),
                    "top_10": top10,
                }

                common_unleaked = dict(common)
                common_unleaked["Novelty"] = compute_novelty(
                    top10,
                    train_pop,
                    train_total,
                )

                common_leaked = dict(common)
                common_leaked["Novelty"] = compute_novelty(
                    top10,
                    full_pop,
                    full_total,
                )

                all_unleaked.append(
                    common_unleaked
                )
                all_leaked.append(
                    common_leaked
                )

    # Preserve the original grouping/summary semantics.
    def summarize(
        records: list[dict[str, Any]],
    ) -> pl.DataFrame:
        if not records:
            return pl.DataFrame()

        slices = [
            ("all", "all"),
            ("cold_fixed", "cold"),
            ("cold_data", "cold"),
            ("warm_fixed", "warm"),
            ("warm_data", "warm"),
            ("tail_fixed", "tail"),
            ("tail_data", "tail"),
            ("head_fixed", "head"),
            ("head_data", "head"),
        ]

        groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            groups[
                (
                    record["dataset"],
                    record["retriever"],
                )
            ].append(record)

        summary_rows = []

        for (dataset, retriever), group in groups.items():
            total_n = len(group)
            catalog_size = catalog_sizes.get(
                dataset,
                1,
            )

            for slice_desc, _ in slices:
                if slice_desc == "all":
                    sub = group
                elif slice_desc == "cold_fixed":
                    sub = [
                        r for r in group
                        if r["slice_cold_fixed"] == "cold"
                    ]
                elif slice_desc == "cold_data":
                    sub = [
                        r for r in group
                        if r["slice_cold_data"] == "cold"
                    ]
                elif slice_desc == "warm_fixed":
                    sub = [
                        r for r in group
                        if r["slice_cold_fixed"] == "warm"
                    ]
                elif slice_desc == "warm_data":
                    sub = [
                        r for r in group
                        if r["slice_cold_data"] == "warm"
                    ]
                elif slice_desc == "tail_fixed":
                    sub = [
                        r for r in group
                        if r["slice_tail_fixed"] == "tail"
                    ]
                elif slice_desc == "tail_data":
                    sub = [
                        r for r in group
                        if r["slice_tail_data"] == "tail"
                    ]
                elif slice_desc == "head_fixed":
                    sub = [
                        r for r in group
                        if r["slice_tail_fixed"] == "head"
                    ]
                else:
                    sub = [
                        r for r in group
                        if r["slice_tail_data"] == "head"
                    ]

                if not sub:
                    continue

                frac = len(sub) / total_n
                flagged = (
                    slice_desc != "all"
                    and (frac < 0.01 or frac > 0.99)
                )

                coverage = compute_coverage(
                    {
                        aid
                        for r in sub
                        for aid in r["top_10"]
                    },
                    catalog_size,
                )

                row_out = {
                    "dataset": dataset,
                    "retriever": retriever,
                    "slice": slice_desc,
                    "n_impressions": len(sub),
                    "frac_population": frac,
                    "Coverage": coverage,
                    "flagged_small_slice": flagged,
                }

                for metric in METRICS:
                    values = np.asarray(
                        [r[metric] for r in sub],
                        dtype=np.float64,
                    )

                    if flagged or values.size < 2:
                        mean_val = (
                            float(values.mean())
                            if values.size
                            else 0.0
                        )
                        low = "insufficient_n"
                        high = "insufficient_n"
                    else:
                        mean_val, low, high = compute_bootstrap_ci(
                            values,
                            b=1000,
                        )

                    row_out[metric] = mean_val
                    row_out[f"{metric}_CI_low"] = low
                    row_out[f"{metric}_CI_high"] = high

                summary_rows.append(row_out)

        return pl.DataFrame(summary_rows)

    out_dir = results_dir(scale)
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    unleaked = summarize(all_unleaked)
    leaked = summarize(all_leaked)

    if not unleaked.is_empty():
        unleaked.write_csv(
            out_dir / "eval_summary_stripped.csv"
        )
    if not leaked.is_empty():
        leaked.write_csv(
            out_dir / "eval_summary.csv"
        )

    return unleaked, leaked


def main(scale: str = "small") -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if scale == "large":
        _run_large(scale)
    else:
        _run_small(scale)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scale",
        choices=["small", "large"],
        default="small",
    )
    args = parser.parse_args()

    main(scale=args.scale)
