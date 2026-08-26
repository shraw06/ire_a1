"""Train a small gradient-boosted combiner over embed+BM25 features (validation only).
Usage: .venv/bin/python -m scripts.train_blend_combiner
"""
from __future__ import annotations
import json
import joblib
import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit
from src.common.paths import processed_dir, results_dir

DATASET_CONFIG = {
    "mind": {"embed_model": "minilm", "bm25_suffix": "title_abstract_sw_nostem"},
    "ebnerd": {"embed_model": "w2v", "bm25_suffix": "title_abstract_sw_nostem"},
}


def _minmax(v: np.ndarray) -> np.ndarray:
    lo, hi = v.min(), v.max()
    return np.zeros_like(v) if hi - lo < 1e-12 else (v - lo) / (hi - lo)


def _rank_pct(v: np.ndarray) -> np.ndarray:
    # 1.0 = best score in this candidate list, 0.0 = worst
    order = v.argsort()
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(v))
    return ranks / max(len(v) - 1, 1)


def build_examples(joined: pl.DataFrame):
    """Expand impression-level rows into candidate-level (features, label, group) rows."""
    X, y, groups = [], [], []
    for gid, row in enumerate(joined.iter_rows(named=True)):
        labels = np.array(json.loads(row["labels"]), dtype=np.int64)
        if labels.sum() == 0 or labels.sum() == len(labels):
            continue
        cand = json.loads(row["candidates"])
        e_lookup = dict(zip(json.loads(row["embed_ranked_ids"]), json.loads(row["embed_scores"])))
        b_lookup = dict(zip(json.loads(row["bm25_ranked_ids"]), json.loads(row["bm25_scores"])))
        e_raw = np.array([e_lookup.get(c, 0.0) for c in cand], dtype=np.float64)
        b_raw = np.array([b_lookup.get(c, 0.0) for c in cand], dtype=np.float64)
        e_norm, b_norm = _minmax(e_raw), _minmax(b_raw)
        e_rank, b_rank = _rank_pct(e_raw), _rank_pct(b_raw)
        for i in range(len(cand)):
            X.append([e_norm[i], b_norm[i], e_rank[i], b_rank[i], e_norm[i] - b_norm[i]])
            y.append(labels[i])
            groups.append(gid)
    return np.array(X), np.array(y), np.array(groups)


def _auc_fast(labels: np.ndarray, scores: np.ndarray) -> float:
    n_pos, n_neg = labels.sum(), len(labels) - labels.sum()
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = scores.argsort(kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def group_auc(model, X, y, groups):
    preds = model.predict_proba(X)[:, 1]
    aucs = []
    for g in np.unique(groups):
        mask = groups == g
        aucs.append(_auc_fast(y[mask], preds[mask]))
    return float(np.mean(aucs))


def main():
    for dataset, cfg in DATASET_CONFIG.items():
        embed_path = processed_dir(dataset, "large") / f"embed_scores_{dataset}_{cfg['embed_model']}.parquet"
        bm25_path = processed_dir(dataset, "large") / f"bm25_scores_{dataset}_{cfg['bm25_suffix']}.parquet"
        if not embed_path.exists() or not bm25_path.exists():
            print(f"skip {dataset}: missing cached scores"); continue

        embed_df = pl.read_parquet(embed_path).select(
            ["impression_id", "candidates", "labels", "ranked_ids", "scores"]
        ).rename({"ranked_ids": "embed_ranked_ids", "scores": "embed_scores"})
        bm25_df = pl.read_parquet(bm25_path).select(
            ["impression_id", "ranked_ids", "scores"]
        ).rename({"ranked_ids": "bm25_ranked_ids", "scores": "bm25_scores"})
        joined = embed_df.join(bm25_df, on="impression_id", how="inner")
        if len(joined) > 2 * min(len(embed_df), len(bm25_df)):
            raise RuntimeError(f"{dataset}: join explosion, aborting.")

        print(f"\n{dataset.upper()}: building features from {len(joined):,} impressions")
        X, y, groups = build_examples(joined)
        print(f"  {len(X):,} candidate rows, {len(np.unique(groups)):,} impressions")

        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, val_idx = next(gss.split(X, y, groups))

        model = HistGradientBoostingClassifier(max_depth=4, max_iter=150, random_state=42)
        model.fit(X[train_idx], y[train_idx])

        held_out_auc = group_auc(model, X[val_idx], y[val_idx], groups[val_idx])
        print(f"  combiner held-out AUC: {held_out_auc:.4f}")

        model_path = results_dir("large") / f"combiner_{dataset}.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        print(f"  saved -> {model_path}")


if __name__ == "__main__":
    main()