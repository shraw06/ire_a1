"""Offline blend tuning: BM25 + embedding similarity (no popularity).
Reads ALREADY-PERSISTED validation scores for both -- nothing recomputed.
Usage:
  .venv/bin/python -m scripts.tune_bm25_embed_blend --sample 5000   # fast sanity pass
  .venv/bin/python -m scripts.tune_bm25_embed_blend                 # full run
"""
from __future__ import annotations
import argparse, json, time
import numpy as np
import polars as pl
from src.common.paths import processed_dir, results_dir

BETAS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.0]
DATASET_CONFIG = {
    "mind": {"embed_model": "minilm", "bm25_suffix": "title_abstract_sw_nostem"},
    "ebnerd": {"embed_model": "w2v", "bm25_suffix": "title_abstract_sw_nostem"},
}


def _minmax(v: np.ndarray) -> np.ndarray:
    lo, hi = v.min(), v.max()
    return np.zeros_like(v) if hi - lo < 1e-12 else (v - lo) / (hi - lo)


def _auc_fast(labels: np.ndarray, scores: np.ndarray) -> float:
    """Rank-sum AUC, equivalent to sklearn.roc_auc_score for binary labels,
    without the per-call overhead of importing/validating through sklearn."""
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = scores.argsort(kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _mrr(labels: np.ndarray, scores: np.ndarray) -> float:
    if labels.sum() == 0:
        return 0.0
    order = np.argsort(-scores, kind="mergesort")
    rank = int(np.argmax(labels[order])) + 1 if labels[order].any() else 0
    return 1.0 / rank if rank else 0.0


def _ndcg_at_k(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    n_pos = labels.sum()
    if n_pos == 0:
        return 0.0
    order = np.argsort(-scores, kind="mergesort")[:k]
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float((labels[order] * discounts[: len(order)]).sum())
    ideal_hits = min(n_pos, k)
    idcg = float(discounts[:ideal_hits].sum())
    return dcg / idcg if idcg else 0.0


def _prep_row(row: dict):
    """Parse JSON + compute normalized score arrays ONCE per row (shared across all betas)."""
    labels = np.array(json.loads(row["labels"]), dtype=np.int64)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return None
    cand = json.loads(row["candidates"])
    e_lookup = dict(zip(json.loads(row["embed_ranked_ids"]), json.loads(row["embed_scores"])))
    b_lookup = dict(zip(json.loads(row["bm25_ranked_ids"]), json.loads(row["bm25_scores"])))
    e_vals = _minmax(np.array([e_lookup.get(c, 0.0) for c in cand], dtype=np.float64))
    b_vals = _minmax(np.array([b_lookup.get(c, 0.0) for c in cand], dtype=np.float64))
    return labels, e_vals, b_vals


def evaluate_all_betas(joined: pl.DataFrame, betas: list[float], log_every: int = 20_000):
    # Single pass over rows: parse once, reuse across every beta.
    prepped = []
    t0 = time.time()
    for i, row in enumerate(joined.iter_rows(named=True)):
        r = _prep_row(row)
        if r is not None:
            prepped.append(r)
        if (i + 1) % log_every == 0:
            print(f"    prepped {i+1:,}/{len(joined):,} rows ({time.time()-t0:.1f}s elapsed)")
    print(f"    usable impressions: {len(prepped):,} (of {len(joined):,}); prep took {time.time()-t0:.1f}s")

    results = []
    for beta in betas:
        aucs, mrrs, n5, n10 = [], [], [], []
        for labels, e_vals, b_vals in prepped:
            blended = beta * e_vals + (1 - beta) * b_vals
            aucs.append(_auc_fast(labels, blended))
            mrrs.append(_mrr(labels, blended))
            n5.append(_ndcg_at_k(labels, blended, 5))
            n10.append(_ndcg_at_k(labels, blended, 10))
        n = len(aucs)
        results.append({
            "beta_embed": beta, "n_impressions": n,
            "AUC": sum(aucs) / n if n else 0.0, "MRR": sum(mrrs) / n if n else 0.0,
            "nDCG@5": sum(n5) / n if n else 0.0, "nDCG@10": sum(n10) / n if n else 0.0,
        })
        print(f"  beta_embed={beta:.1f}  AUC={results[-1]['AUC']:.4f}  MRR={results[-1]['MRR']:.4f}  "
              f"nDCG@5={results[-1]['nDCG@5']:.4f}  nDCG@10={results[-1]['nDCG@10']:.4f}  (n={n:,})")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=None, help="subsample N impressions per dataset for a fast pass")
    args = ap.parse_args()

    out_path = results_dir("large") / "bm25_embed_blend_tuning.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_rows = []

    for dataset, cfg in DATASET_CONFIG.items():
        embed_path = processed_dir(dataset, "large") / f"embed_scores_{dataset}_{cfg['embed_model']}.parquet"
        bm25_path = processed_dir(dataset, "large") / f"bm25_scores_{dataset}_{cfg['bm25_suffix']}.parquet"
        if not embed_path.exists() or not bm25_path.exists():
            print(f"skip {dataset}: missing {embed_path if not embed_path.exists() else bm25_path}")
            continue

        embed_df = pl.read_parquet(embed_path).select(
            ["impression_id", "candidates", "labels", "ranked_ids", "scores"]
        ).rename({"ranked_ids": "embed_ranked_ids", "scores": "embed_scores"})
        bm25_df = pl.read_parquet(bm25_path).select(
            ["impression_id", "ranked_ids", "scores"]
        ).rename({"ranked_ids": "bm25_ranked_ids", "scores": "bm25_scores"})

        # Hard guard against the join-explosion failure mode.
        if embed_df["impression_id"].n_unique() != len(embed_df):
            raise RuntimeError(f"{dataset}: embed_scores parquet has duplicate impression_id -- fix upstream before tuning.")
        if bm25_df["impression_id"].n_unique() != len(bm25_df):
            raise RuntimeError(f"{dataset}: bm25_scores parquet has duplicate impression_id -- fix upstream before tuning.")

        joined = embed_df.join(bm25_df, on="impression_id", how="inner")
        if len(joined) > 2 * min(len(embed_df), len(bm25_df)):
            raise RuntimeError(f"{dataset}: join produced {len(joined):,} rows from "
                                f"{len(embed_df):,}/{len(bm25_df):,} inputs -- explosion, aborting.")

        if args.sample and len(joined) > args.sample:
            joined = joined.sample(n=args.sample, seed=42)

        print(f"\n{dataset.upper()}: {len(joined):,} impressions{' (sampled)' if args.sample else ''}")
        results = evaluate_all_betas(joined, BETAS)
        for r in results:
            all_rows.append({"dataset": dataset, **r})

        # checkpoint after EVERY dataset so a later crash doesn't lose earlier work
        pl.DataFrame(all_rows).write_csv(out_path)
        print(f"  checkpointed -> {out_path}")

    if not all_rows:
        print("nothing to tune"); return
    out = pl.DataFrame(all_rows)
    print(f"\nSaved: {out_path}")
    for dataset in DATASET_CONFIG:
        sub = out.filter(pl.col("dataset") == dataset)
        if not sub.is_empty():
            best = sub.sort("AUC", descending=True).row(0, named=True)
            print(f"best {dataset}: beta_embed={best['beta_embed']:.1f} AUC={best['AUC']:.4f}")


if __name__ == "__main__":
    main()