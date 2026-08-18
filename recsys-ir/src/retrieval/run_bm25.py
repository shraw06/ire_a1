"""BM25 retrieval pipeline runner — build index, score candidates, compute recall@K.

Usage:
    python -m src.retrieval.run_bm25 --dataset mind
    python -m src.retrieval.run_bm25 --dataset ebnerd
    python -m src.retrieval.run_bm25 --dataset all

For each dataset, runs a 4-way ablation (±stopwords × ±stemming) and reports
recall@{50, 100, 200} on the val split.  Results are written to:
  - ``results/bm25_recall.csv``
  - ``data/processed/bm25_scores_{dataset}_{text_fields}.parquet``
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

from src.evaluation.ranking_metrics import recall_at_k
from src.feature_store.article_store import ArticleFeatureStore
from src.feature_store.user_store import UserFeatureStore
from src.retrieval.bm25 import BM25Engine, lang_for_dataset

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Recall@K cutoffs
RECALL_KS = [50, 100, 200]

# Maximum number of history articles to use for query construction
DEFAULT_HISTORY_CAP = 20

# BM25 parameters
DEFAULT_K1 = 1.5
DEFAULT_B = 0.75


def _build_query_text(
    user_history: list[dict[str, Any]],
    article_texts: dict[str, str],
    history_cap: int = DEFAULT_HISTORY_CAP,
) -> str:
    """Construct a BM25 query from the user's recent click history.

    Concatenates the cleaned_text (title + abstract) of the most recent
    *history_cap* articles in the user's history.

    Parameters
    ----------
    user_history : list[dict]
        From ``UserFeatureStore.get_user_history()`` — list of
        ``{article_id, clicked_at}`` dicts, oldest first.
    article_texts : dict[str, str]
        Mapping ``article_id → cleaned_text`` for all articles.
    history_cap : int
        Maximum number of recent history articles to include.

    Returns
    -------
    str
        Concatenated text of the most recent *history_cap* history articles.
    """
    # Take the most recent N (history is oldest-first → take from the end)
    recent = user_history[-history_cap:]
    parts = []
    for entry in recent:
        aid = entry["article_id"]
        text = article_texts.get(aid)
        if text:
            parts.append(text)
    return " ".join(parts)


def run_bm25_for_dataset(
    dataset: str,
    use_stopwords: bool = True,
    use_stemming: bool = False,
    text_fields: str = "title_abstract",
    history_cap: int = DEFAULT_HISTORY_CAP,
    k1: float = DEFAULT_K1,
    b: float = DEFAULT_B,
    spot_check_users: int = 5,
) -> list[dict[str, Any]]:
    """Run BM25 retrieval on the val split of a dataset.

    Returns a list of result dicts for results/bm25_recall.csv.
    """
    logger.info(
        "Running BM25 on %s (stopwords=%s, stemming=%s, text=%s, history_cap=%d)",
        dataset, use_stopwords, use_stemming, text_fields, history_cap,
    )
    t0 = time.time()

    # ── 1. Load article texts ─────────────────────────────────────
    article_store = ArticleFeatureStore(dataset)
    bm25_df = article_store.get_articles_for_bm25()
    article_texts: dict[str, str] = {}
    corpus: list[tuple[str, str]] = []
    for row in bm25_df.iter_rows(named=True):
        aid = row["article_id"]
        text = row["cleaned_text"] or ""
        article_texts[aid] = text
        corpus.append((aid, text))

    logger.info("  Loaded %d articles for indexing", len(corpus))

    # ── 2. Build inverted index ───────────────────────────────────
    t_idx = time.time()
    engine = BM25Engine.from_corpus(
        corpus, dataset=dataset,
        use_stopwords=use_stopwords, use_stemming=use_stemming,
        k1=k1, b=b,
    )
    logger.info("  Index built in %.1fs: %s", time.time() - t_idx, engine)

    # ── 3. Load val-split impressions ─────────────────────────────
    behaviors_path = _PROJECT_ROOT / "data" / "interim" / dataset / "behaviors.parquet"
    behaviors_df = pl.read_parquet(behaviors_path)
    val_df = behaviors_df.filter(pl.col("split") == "val")
    logger.info("  Val impressions: %d", len(val_df))

    # ── 4. Load user store ────────────────────────────────────────
    user_store = UserFeatureStore(dataset)

    # ── 5. Score each impression ──────────────────────────────────
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

        # Get user history (leakage-safe)
        history = user_store.get_user_history(user_id, timestamp, dataset)

        # Build query from history
        query_text = _build_query_text(history, article_texts, history_cap)

        # Score candidates via BM25
        max_k = max(RECALL_KS)
        results = engine.rank(query_text, candidate_ids=candidates, top_k=max_k)
        ranked_ids = [r[0] for r in results]

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

        # Spot-check: print top-5 for first N users
        if spot_check_done < spot_check_users and ground_truth:
            print(f"\n  📋 Spot-check [{dataset}] user={user_id} imp={imp_id}")
            print(f"     History len: {len(history)}, Query tokens (approx): {len(query_text.split())}")
            print(f"     Ground truth: {ground_truth}")
            print(f"     Candidates: {len(candidates)}")
            print(f"     Top-5 BM25 hits:")
            for rank, (rid, rscore) in enumerate(results[:5], 1):
                marker = " ✓" if rid in ground_truth else ""
                # Show a snippet of the article text
                snippet = (article_texts.get(rid, "")[:80] + "...") if article_texts.get(rid) else "(no text)"
                print(f"       {rank}. {rid} (score={rscore:.4f}){marker}  — {snippet}")
            spot_check_done += 1

        if (i + 1) % 2000 == 0:
            logger.info("  Processed %d/%d impressions", i + 1, len(val_df))

    elapsed = time.time() - t0

    # ── 6. Compute average recall ─────────────────────────────────
    result_rows = []
    for k_val in RECALL_KS:
        recalls = per_impression_recalls[k_val]
        avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
        result_rows.append({
            "dataset": dataset,
            "text_fields": text_fields,
            "use_stopwords": use_stopwords,
            "use_stemming": use_stemming,
            "K": k_val,
            "recall_at_K": round(avg_recall, 6),
            "num_impressions": len(recalls),
            "wall_clock_s": round(elapsed, 1),
        })
        logger.info("  recall@%d = %.4f (avg over %d impressions)", k_val, avg_recall, len(recalls))

    # ── 7. Persist scored candidates ──────────────────────────────
    out_dir = _PROJECT_ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_path = out_dir / f"bm25_scores_{dataset}_{text_fields}.parquet"
    scores_df = pl.DataFrame(all_scored)
    scores_df.write_parquet(scores_path)
    logger.info("  Saved scored candidates: %s (%.1f MB)", scores_path, scores_path.stat().st_size / 1024**2)
    logger.info("  Total wall-clock: %.1fs", elapsed)

    return result_rows


def run_ablation(
    dataset: str,
    history_cap: int = DEFAULT_HISTORY_CAP,
) -> list[dict[str, Any]]:
    """Run 4-way ablation (±stopwords × ±stemming) for a dataset."""
    all_results = []
    configs = [
        (True, False),   # stopwords ON, stemming OFF
        (False, False),  # stopwords OFF, stemming OFF
        (True, True),    # stopwords ON, stemming ON
        (False, True),   # stopwords OFF, stemming ON
    ]
    for use_sw, use_stem in configs:
        results = run_bm25_for_dataset(
            dataset,
            use_stopwords=use_sw,
            use_stemming=use_stem,
            history_cap=history_cap,
        )
        all_results.extend(results)
    return all_results


def write_results_csv(
    results: list[dict[str, Any]],
    path: Path,
    datasets_to_replace: list[str] | None = None,
) -> None:
    """Write results to CSV, merging with existing rows for other datasets.

    If *datasets_to_replace* is given, existing rows for those datasets are
    removed and replaced with *results*.  Rows for other datasets are kept.
    If None, the entire CSV is overwritten.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset", "text_fields", "use_stopwords", "use_stemming",
        "K", "recall_at_K", "num_impressions", "wall_clock_s",
    ]
    existing_rows: list[dict[str, Any]] = []
    if datasets_to_replace and path.exists():
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["dataset"] not in datasets_to_replace:
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
        description="Run BM25 retrieval + recall@K evaluation on val split.",
    )
    parser.add_argument(
        "--dataset",
        choices=["mind", "ebnerd", "all"],
        default="all",
        help="Which dataset to run BM25 on (default: all).",
    )
    parser.add_argument(
        "--history-cap",
        type=int,
        default=DEFAULT_HISTORY_CAP,
        help=f"Max number of recent history articles for query (default: {DEFAULT_HISTORY_CAP}).",
    )
    parser.add_argument(
        "--no-ablation",
        action="store_true",
        help="Run only the default config (stopwords=True, stemming=False) instead of full ablation.",
    )
    args = parser.parse_args()

    datasets = ["mind", "ebnerd"] if args.dataset == "all" else [args.dataset]
    all_results: list[dict[str, Any]] = []

    for ds in datasets:
        # Check that feature stores exist
        processed_dir = _PROJECT_ROOT / "data" / "processed" / ds
        if not (processed_dir / "article_features.parquet").exists():
            logger.error("Article features not found for %s — run `make features` first", ds)
            sys.exit(1)
        if not (processed_dir / "user_features.parquet").exists():
            logger.error("User features not found for %s — run `make features` first", ds)
            sys.exit(1)

        if args.no_ablation:
            results = run_bm25_for_dataset(ds, history_cap=args.history_cap)
        else:
            results = run_ablation(ds, history_cap=args.history_cap)
        all_results.extend(results)

    # Write results CSV (merge with existing rows from other datasets)
    results_path = _PROJECT_ROOT / "results" / "bm25_recall.csv"
    write_results_csv(all_results, results_path, datasets_to_replace=datasets)

    # Print summary table
    print(f"\n{'='*80}")
    print("BM25 Recall Results")
    print(f"{'='*80}")
    print(f"{'Dataset':<10} {'Text Fields':<18} {'Stopwords':<10} {'Stemming':<10} {'K':<6} {'Recall@K':<10}")
    print(f"{'-'*80}")
    for r in all_results:
        print(
            f"{r['dataset']:<10} {r['text_fields']:<18} "
            f"{str(r['use_stopwords']):<10} {str(r['use_stemming']):<10} "
            f"{r['K']:<6} {r['recall_at_K']:<10.4f}"
        )
    print(f"{'='*80}")
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
