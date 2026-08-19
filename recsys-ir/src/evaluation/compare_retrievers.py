"""Lexical vs. Semantic retriever comparison — merges BM25 and embedding
recall results, slices by cold/warm users and head/tail articles, produces
a comparison CSV and plot.

Usage:
    python -m src.evaluation.compare_retrievers

Reads:
  - ``results/bm25_recall.csv``   (from run_bm25.py)
  - ``results/embed_recall.csv``  (from run_embeddings.py)
  - ``data/processed/bm25_scores_{dataset}_title_abstract.parquet``
  - ``data/processed/embed_scores_{dataset}_{model}.parquet``
  - ``data/interim/{dataset}/behaviors.parquet``

Writes:
  - ``results/lexical_vs_semantic.csv``
  - ``results/lexical_vs_semantic.png``
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from src.evaluation.ranking_metrics import recall_at_k
from src.feature_store.user_store import UserFeatureStore

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Slicing thresholds (from EDA_SUMMARY.md §4-5)
# Cold-start user: ≤5 clicks (covers ~50-75% of MIND users, bottom quartile of EB-NeRD)
COLD_USER_THRESHOLD = 5
# Tail article: ≤10 impressions (median is ~7-15 per EDA §5)
TAIL_ARTICLE_THRESHOLD = 10

RECALL_KS = [5, 10, 50, 100, 200]


def _load_article_popularity(dataset: str) -> dict[str, int]:
    """Count total impressions per article from the full behaviors table.

    Returns ``{article_id: impression_count}`` for all articles.
    """
    behaviors_path = _PROJECT_ROOT / "data" / "interim" / dataset / "behaviors.parquet"
    df = pl.read_parquet(behaviors_path)

    popularity: dict[str, int] = {}
    for row in df.iter_rows(named=True):
        candidates = json.loads(row["candidates"])
        for cid in candidates:
            popularity[cid] = popularity.get(cid, 0) + 1

    return popularity


def _load_user_history_lens(dataset: str) -> dict[str, int]:
    """Load history lengths for all users from the user feature store.

    Returns ``{user_id: history_len}``.
    """
    user_store = UserFeatureStore(dataset)
    # Read all user rows via the store's backend
    rows = user_store._store.query_sql(
        f"SELECT user_id, history_len FROM {user_store._store.alias}"
    )
    return {r["user_id"]: r["history_len"] for r in rows}


def _compute_per_impression_recall(
    scores_path: Path,
    ks: list[int] = RECALL_KS,
) -> list[dict[str, Any]]:
    """Re-compute per-impression recall@K from a scored-candidates parquet.

    Returns a list of dicts with keys:
      impression_id, user_id, K, recall_at_K, ground_truth_ids, candidates
    """
    df = pl.read_parquet(scores_path)
    results = []

    for row in df.iter_rows(named=True):
        imp_id = row["impression_id"]
        user_id = row["user_id"]
        ranked_ids = json.loads(row["ranked_ids"])
        ground_truth = set(json.loads(row["ground_truth"]))
        candidates = json.loads(row["candidates"])

        for k in ks:
            r = recall_at_k(ranked_ids, ground_truth, k)
            results.append({
                "impression_id": imp_id,
                "user_id": user_id,
                "K": k,
                "recall_at_K": r,
                "ground_truth_ids": list(ground_truth),
                "candidates": candidates,
            })

    return results


def _classify_impressions(
    per_imp_recall: list[dict[str, Any]],
    user_hist_lens: dict[str, int],
    article_pop: dict[str, int],
) -> list[dict[str, Any]]:
    """Add user_warmth and article_popularity slice labels to each impression.

    User slicing:
      - cold: history_len ≤ COLD_USER_THRESHOLD
      - warm: history_len > COLD_USER_THRESHOLD

    Article slicing (based on ground-truth clicked articles):
      - tail: avg impression count of clicked articles ≤ TAIL_ARTICLE_THRESHOLD
      - head: avg impression count > TAIL_ARTICLE_THRESHOLD
      - If no ground truth clicks, classify as tail (conservative).
    """
    for entry in per_imp_recall:
        uid = entry["user_id"]
        hist_len = user_hist_lens.get(uid, 0)
        entry["history_len"] = hist_len
        entry["user_slice"] = "cold" if hist_len <= COLD_USER_THRESHOLD else "warm"

        # Article popularity: average over ground-truth clicked articles
        gt_ids = entry["ground_truth_ids"]
        if gt_ids:
            avg_pop = np.mean([article_pop.get(aid, 0) for aid in gt_ids])
            entry["article_slice"] = "tail" if avg_pop <= TAIL_ARTICLE_THRESHOLD else "head"
        else:
            entry["article_slice"] = "tail"

    return per_imp_recall


def build_comparison(datasets: list[str] | None = None) -> Path:
    """Build the full lexical-vs-semantic comparison.

    Returns the path to ``results/lexical_vs_semantic.csv``.
    """
    if datasets is None:
        datasets = ["mind", "ebnerd"]

    all_rows: list[dict[str, Any]] = []

    for dataset in datasets:
        logger.info("Building comparison for %s", dataset)

        # Load article popularity and user history lens
        article_pop = _load_article_popularity(dataset)
        user_hist_lens = _load_user_history_lens(dataset)
        logger.info(
            "  %s: %d articles, %d users loaded for slicing",
            dataset, len(article_pop), len(user_hist_lens),
        )

        # ── BM25 scores ──────────────────────────────────────────
        bm25_path = _PROJECT_ROOT / "data" / "processed" / f"bm25_scores_{dataset}_title_abstract.parquet"
        if not bm25_path.exists():
            logger.warning("  BM25 scores not found at %s — skipping", bm25_path)
            continue

        bm25_per_imp = _compute_per_impression_recall(bm25_path)
        bm25_per_imp = _classify_impressions(bm25_per_imp, user_hist_lens, article_pop)

        # ── Embedding scores ──────────────────────────────────────
        # Find the primary embedding scores file
        if dataset == "ebnerd":
            embed_models = ["bert", "w2v"]
        else:
            embed_models = ["minilm"]

        for model in embed_models:
            embed_path = _PROJECT_ROOT / "data" / "processed" / f"embed_scores_{dataset}_{model}.parquet"
            if not embed_path.exists():
                logger.warning("  Embedding scores not found at %s — skipping", embed_path)
                continue

            embed_per_imp = _compute_per_impression_recall(embed_path)
            embed_per_imp = _classify_impressions(embed_per_imp, user_hist_lens, article_pop)

            # ── Aggregate by slice ────────────────────────────────
            slices = ["all", "cold", "warm", "tail", "head"]
            for k in RECALL_KS:
                for slice_name in slices:
                    # Filter BM25
                    if slice_name == "all":
                        bm25_k = [e for e in bm25_per_imp if e["K"] == k]
                        embed_k = [e for e in embed_per_imp if e["K"] == k]
                    elif slice_name in ("cold", "warm"):
                        bm25_k = [e for e in bm25_per_imp if e["K"] == k and e["user_slice"] == slice_name]
                        embed_k = [e for e in embed_per_imp if e["K"] == k and e["user_slice"] == slice_name]
                    elif slice_name in ("tail", "head"):
                        bm25_k = [e for e in bm25_per_imp if e["K"] == k and e["article_slice"] == slice_name]
                        embed_k = [e for e in embed_per_imp if e["K"] == k and e["article_slice"] == slice_name]

                    bm25_recall = np.mean([e["recall_at_K"] for e in bm25_k]) if bm25_k else 0.0
                    embed_recall = np.mean([e["recall_at_K"] for e in embed_k]) if embed_k else 0.0
                    n_bm25 = len(bm25_k)
                    n_embed = len(embed_k)

                    all_rows.append({
                        "dataset": dataset,
                        "K": k,
                        "slice": slice_name,
                        "bm25_recall": round(float(bm25_recall), 6),
                        "embed_model": model,
                        "embed_recall": round(float(embed_recall), 6),
                        "bm25_n": n_bm25,
                        "embed_n": n_embed,
                        "bm25_wins": 1 if bm25_recall > embed_recall else 0,
                    })

    # Write CSV
    out_path = _PROJECT_ROOT / "results" / "lexical_vs_semantic.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset", "K", "slice", "bm25_recall", "embed_model",
        "embed_recall", "bm25_n", "embed_n", "bm25_wins",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    logger.info("Wrote comparison to %s (%d rows)", out_path, len(all_rows))
    return out_path


def plot_comparison(csv_path: Path | None = None) -> Path:
    """Generate the lexical vs. semantic comparison plot.

    Plots recall@100 grouped by slice for each dataset × retriever.

    Returns the path to ``results/lexical_vs_semantic.png``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    rcParams["font.family"] = "sans-serif"
    rcParams["font.size"] = 11

    if csv_path is None:
        csv_path = _PROJECT_ROOT / "results" / "lexical_vs_semantic.csv"

    # Read CSV
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    # Filter to K=10 (the most informative due to saturation at higher Ks)
    target_k = 10
    k_rows = [r for r in rows if int(r["K"]) == target_k]

    # Group by dataset × embed_model
    groups: dict[str, list[dict]] = {}
    for r in k_rows:
        key = f"{r['dataset']}_{r['embed_model']}"
        if key not in groups:
            groups[key] = []
        groups[key].append(r)

    n_groups = len(groups)
    if n_groups == 0:
        logger.warning("No data to plot")
        return csv_path

    fig, axes = plt.subplots(1, n_groups, figsize=(6 * n_groups, 5), squeeze=False)
    fig.suptitle(f"Lexical (BM25) vs. Semantic (Embedding) Retrieval — Recall@{target_k}", fontsize=14, fontweight="bold")

    slice_order = ["all", "cold", "warm", "tail", "head"]
    slice_labels = ["All", "Cold\nUsers", "Warm\nUsers", "Tail\nArticles", "Head\nArticles"]
    bar_width = 0.35
    colors_bm25 = "#4A90D9"
    colors_embed = "#E8734A"

    for ax_idx, (group_key, group_rows) in enumerate(sorted(groups.items())):
        ax = axes[0][ax_idx]

        # Build data arrays
        bm25_vals = []
        embed_vals = []
        for sl in slice_order:
            match = [r for r in group_rows if r["slice"] == sl]
            if match:
                bm25_vals.append(float(match[0]["bm25_recall"]))
                embed_vals.append(float(match[0]["embed_recall"]))
            else:
                bm25_vals.append(0.0)
                embed_vals.append(0.0)

        x = np.arange(len(slice_order))
        ax.bar(x - bar_width / 2, bm25_vals, bar_width, label="BM25 (lexical)",
               color=colors_bm25, alpha=0.85, edgecolor="white", linewidth=0.5)
        ax.bar(x + bar_width / 2, embed_vals, bar_width, label="Embedding (semantic)",
               color=colors_embed, alpha=0.85, edgecolor="white", linewidth=0.5)

        # Labels
        dataset_name, model_name = group_key.split("_", 1)
        ax.set_title(f"{dataset_name.upper()} — {model_name}", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(slice_labels, fontsize=9)
        ax.set_ylabel(f"Recall@{target_k}")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)

        # Annotate values on bars
        for xi, (bv, ev) in enumerate(zip(bm25_vals, embed_vals)):
            ax.text(xi - bar_width / 2, bv + 0.01, f"{bv:.3f}", ha="center", va="bottom", fontsize=7)
            ax.text(xi + bar_width / 2, ev + 0.01, f"{ev:.3f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    out_path = _PROJECT_ROOT / "results" / "lexical_vs_semantic.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved comparison plot to %s", out_path)
    return out_path


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    csv_path = build_comparison()
    plot_path = plot_comparison(csv_path)

    # Print headline numbers
    print(f"\n{'='*90}")
    print("Lexical vs. Semantic Comparison — Headline Numbers")
    print(f"{'='*90}")

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Print table for K=10 and K=100
    target_ks_print = [10, 100]
    for target_k_p in target_ks_print:
        print(f"\nRecall@{target_k_p}")
        k_rows = [r for r in rows if int(r["K"]) == target_k_p]
        print(f"{'Dataset':<10} {'Model':<10} {'Slice':<12} {'BM25':<10} {'Embed':<10} {'Winner':<10}")
        print(f"{'-'*62}")
        for r in k_rows:
            winner = "BM25" if int(r["bm25_wins"]) == 1 else "Embed"
            print(
                f"{r['dataset']:<10} {r['embed_model']:<10} {r['slice']:<12} "
                f"{float(r['bm25_recall']):<10.4f} {float(r['embed_recall']):<10.4f} {winner:<10}"
            )

    print(f"\n{'='*90}")
    print(f"CSV: {csv_path}")
    print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
