"""Offline alpha tuning for the embedding+popularity hybrid reranker.

Reads the ALREADY-PERSISTED large-scale validation embedding scores
(data/processed/large/{dataset}/embed_scores_{dataset}_{model}.parquet) --
no embeddings are recomputed and no new model inference happens here.
Fast: a few minutes for both datasets combined, pure CPU.

Usage:
    .venv/bin/python -m scripts.tune_hybrid_alpha
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import polars as pl

from src.common.paths import processed_dir, results_dir
from src.evaluation.ranking_metrics import auc_score, mrr, ndcg_at_k
from src.retrieval.hybrid_rerank import load_train_popularity

ALPHAS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]  # 1.0 = pure embedding baseline (control)

DATASET_MODEL = {
    "mind": "minilm",
    "ebnerd": "w2v",
}


def _minmax(values: list[float]) -> list[float]:
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def _pop_score(article_id: str, popularity: dict[str, int]) -> float:
    return math.log1p(popularity.get(article_id, 0))


def evaluate_alpha(df: pl.DataFrame, popularity: dict[str, int], alpha: float) -> dict[str, float]:
    aucs, mrrs, ndcg5s, ndcg10s = [], [], [], []
    for row in df.iter_rows(named=True):
        candidates = json.loads(row["candidates"])
        labels = json.loads(row["labels"])
        ranked_ids = json.loads(row["ranked_ids"])
        sims = json.loads(row["scores"])
        sim_lookup = dict(zip(ranked_ids, sims))

        if sum(labels) == 0 or sum(labels) == len(labels):
            continue  # AUC/MRR/nDCG undefined for degenerate impressions

        sim_vals = [sim_lookup.get(cid, 0.0) for cid in candidates]
        pop_vals = [_pop_score(cid, popularity) for cid in candidates]
        sim_norm = _minmax(sim_vals)
        pop_norm = _minmax(pop_vals)
        blended = [alpha * s + (1 - alpha) * p for s, p in zip(sim_norm, pop_norm)]

        aucs.append(auc_score(labels, blended))
        mrrs.append(mrr(labels, blended))
        ndcg5s.append(ndcg_at_k(labels, blended, 5))
        ndcg10s.append(ndcg_at_k(labels, blended, 10))

    n = len(aucs)
    return {
        "n_impressions": n,
        "AUC": sum(aucs) / n if n else 0.0,
        "MRR": sum(mrrs) / n if n else 0.0,
        "nDCG@5": sum(ndcg5s) / n if n else 0.0,
        "nDCG@10": sum(ndcg10s) / n if n else 0.0,
    }


def main() -> None:
    rows = []
    for dataset, model in DATASET_MODEL.items():
        path = processed_dir(dataset, "large") / f"embed_scores_{dataset}_{model}.parquet"
        if not path.exists():
            print(f"  ! Missing {path}, skipping {dataset} "
                  f"(run `make embed DATA_SCALE=large` first)")
            continue

        df = pl.read_parquet(path)
        popularity = load_train_popularity(dataset, "large")
        print(f"\n{dataset.upper()} ({len(df):,} validation impressions, "
              f"{len(popularity):,} train-split candidate popularity entries)")

        for alpha in ALPHAS:
            metrics = evaluate_alpha(df, popularity, alpha)
            rows.append({"dataset": dataset, "alpha": alpha, **metrics})
            print(f"  alpha={alpha:.1f}  AUC={metrics['AUC']:.4f}  "
                  f"MRR={metrics['MRR']:.4f}  nDCG@5={metrics['nDCG@5']:.4f}  "
                  f"nDCG@10={metrics['nDCG@10']:.4f}  (n={metrics['n_impressions']:,})")

    if not rows:
        print("\nNo embed_scores files found -- nothing to tune. "
              "Run `make embed DATA_SCALE=large` for at least one dataset first.")
        return

    out = pl.DataFrame(rows)
    out_path = results_dir("large") / "hybrid_alpha_tuning.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(out_path)
    print(f"\nSaved: {out_path}")

    print("\nBest alpha by AUC per dataset:")
    for dataset in DATASET_MODEL:
        sub = out.filter(pl.col("dataset") == dataset)
        if sub.is_empty():
            continue
        best = sub.sort("AUC", descending=True).row(0, named=True)
        print(f"  {dataset}: alpha={best['alpha']:.1f}  AUC={best['AUC']:.4f}")


if __name__ == "__main__":
    main()