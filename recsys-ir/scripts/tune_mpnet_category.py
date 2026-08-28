"""Tune category-affinity blend on top of mpnet-base-v2 embeddings.

Unlike tune_category_blend.py (which reads pre-scored parquets and uses a 
top-K proxy for user interest), this script streams validation behaviors 
directly — giving access to actual click history categories.

This is a fairer evaluation of the category signal.

Usage:
    .venv/bin/python -m scripts.tune_mpnet_category
    .venv/bin/python -m scripts.tune_mpnet_category --sample 20000
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from src.common.paths import interim_dir, processed_dir, results_dir
from src.retrieval.ann import ArticleIndex

logger = logging.getLogger(__name__)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_EMBED_DIR = _PROJECT_ROOT / "data" / "processed" / "embeddings"

HISTORY_CAP = 50
BETAS = [1.0, 0.95, 0.90, 0.85, 0.80, 0.70, 0.60, 0.50]
BATCH_SIZE = 5000


def _auc_fast(labels: np.ndarray, scores: np.ndarray) -> float:
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = scores.argsort(kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _mrr(labels: np.ndarray, scores: np.ndarray) -> float:
    if labels.sum() == 0:
        return 0.0
    for rank, idx in enumerate(np.argsort(-scores, kind="mergesort"), 1):
        if labels[idx] == 1:
            return 1.0 / rank
    return 0.0


def _ndcg(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    n_pos = int(labels.sum())
    if n_pos == 0:
        return 0.0
    top = np.argsort(-scores, kind="mergesort")[:k]
    dcg = sum(labels[i] / np.log2(r + 2) for r, i in enumerate(top))
    idcg = sum(1.0 / np.log2(r + 2) for r in range(min(n_pos, k)))
    return float(dcg / idcg) if idcg > 0 else 0.0


def _minmax(v: np.ndarray) -> np.ndarray:
    lo, hi = v.min(), v.max()
    return np.zeros_like(v) if hi - lo < 1e-12 else (v - lo) / (hi - lo)


def _parse_mind_history(value: Any) -> list[dict]:
    if value is None:
        return []
    if isinstance(value, str):
        return json.loads(value) if value else []
    return list(value)


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")

    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=None)
    args = ap.parse_args()

    # Load mpnet index
    t0 = time.time()
    embeddings = np.load(_EMBED_DIR / "mind_mpnet_large.npy")
    id_to_row: dict[str, int] = json.loads(
        (_EMBED_DIR / "mind_mpnet_large_ids.json").read_text()
    )
    article_ids_ordered = [""] * len(id_to_row)
    for aid, idx in id_to_row.items():
        article_ids_ordered[int(idx)] = aid
    index = ArticleIndex(embeddings, article_ids_ordered, build_full_index=False)
    logger.info("Loaded mpnet index: %s (%.1fs)", index, time.time() - t0)

    # Load categories
    article_path = processed_dir("mind", "large") / "article_features.parquet"
    df_cats = pl.read_parquet(article_path, columns=["article_id", "category", "subcategory"])
    article_category: dict[str, str] = {}
    article_subcategory: dict[str, str] = {}
    for row in df_cats.iter_rows(named=True):
        aid = str(row["article_id"])
        article_category[aid] = row["category"] or ""
        article_subcategory[aid] = row["subcategory"] or ""
    logger.info("Loaded categories for %d articles", len(article_category))

    # Accumulators per beta
    accum = {
        beta: {"auc_sum": 0.0, "mrr_sum": 0.0, "n5_sum": 0.0, "n10_sum": 0.0, "n": 0}
        for beta in BETAS
    }

    behaviors_path = interim_dir("mind", "large") / "behaviors.parquet"
    parquet = pq.ParquetFile(behaviors_path)
    columns = ["clicked_history", "candidates", "labels", "split"]

    total = 0
    for batch in parquet.iter_batches(batch_size=BATCH_SIZE, columns=columns):
        for row in pa.Table.from_batches([batch]).to_pylist():
            if row["split"] != "val":
                continue

            candidates_raw = row["candidates"]
            labels_raw = row["labels"]
            candidates = json.loads(candidates_raw) if isinstance(candidates_raw, str) else candidates_raw
            labels = json.loads(labels_raw) if isinstance(labels_raw, str) else labels_raw
            candidates = [str(c) for c in candidates]
            labels_arr = np.array([int(l) for l in labels], dtype=np.int64)

            if labels_arr.sum() == 0 or labels_arr.sum() == len(labels_arr):
                continue

            history = _parse_mind_history(row.get("clicked_history"))
            history_ids = [str(e["article_id"]) for e in history[-HISTORY_CAP:]]

            # Build user vector
            if history_ids:
                h_embs, _ = index.get_embeddings_batch(history_ids)
                if h_embs.shape[0] > 0:
                    vec = h_embs.mean(axis=0, dtype=np.float32)
                    norm = np.linalg.norm(vec)
                    if norm > 0:
                        vec /= norm
                else:
                    vec = np.zeros(index.dim, dtype=np.float32)
            else:
                vec = np.zeros(index.dim, dtype=np.float32)

            # Embedding similarity scores
            cand_idxs = [id_to_row.get(c) for c in candidates]
            valid = [(i, c) for i, (idx_val, c) in enumerate(zip(cand_idxs, candidates)) if idx_val is not None]
            sim_vals = np.zeros(len(candidates), dtype=np.float64)
            if valid:
                v_idxs = [id_to_row[c] for _, c in valid]
                sims = embeddings[v_idxs] @ vec
                for (i, _), s in zip(valid, sims):
                    sim_vals[i] = float(s)
            sim_norm = _minmax(sim_vals)

            # Category affinity from ACTUAL click history (not proxy)
            hist_cats = Counter(article_category.get(aid, "") for aid in history_ids)
            hist_subcats = Counter(article_subcategory.get(aid, "") for aid in history_ids)
            total_h = sum(hist_cats.values()) or 1

            cat_vals = np.array(
                [hist_cats.get(article_category.get(c, ""), 0) / total_h for c in candidates],
                dtype=np.float64,
            )
            subcat_vals = np.array(
                [hist_subcats.get(article_subcategory.get(c, ""), 0) / total_h for c in candidates],
                dtype=np.float64,
            )
            cat_signal = 0.5 * _minmax(cat_vals) + 0.5 * _minmax(subcat_vals)

            for beta in BETAS:
                blended = beta * sim_norm + (1.0 - beta) * cat_signal
                a = accum[beta]
                a["auc_sum"] += _auc_fast(labels_arr, blended)
                a["mrr_sum"] += _mrr(labels_arr, blended)
                a["n5_sum"] += _ndcg(labels_arr, blended, 5)
                a["n10_sum"] += _ndcg(labels_arr, blended, 10)
                a["n"] += 1

            total += 1

        if total % 10_000 < BATCH_SIZE:
            logger.info("  %d impressions (%.1f min)", total, (time.time() - t0) / 60)
        if args.sample and total >= args.sample:
            logger.info("Sample limit (%d) reached", args.sample)
            break

    elapsed = time.time() - t0
    logger.info("Done: %d impressions in %.1f min", total, elapsed / 60)

    rows = []
    print(f"\n{'='*70}")
    print(f"mpnet + Category Blend (cap={HISTORY_CAP}, {total:,} impressions, {elapsed/60:.1f} min)")
    print(f"{'='*70}")
    print(f"{'beta':>6}  {'AUC':>8}  {'MRR':>8}  {'nDCG@5':>8}  {'nDCG@10':>8}")
    print("-" * 52)

    best_auc, best_beta = 0.0, 1.0
    baseline_auc = accum[1.0]["auc_sum"] / max(accum[1.0]["n"], 1)
    for beta in BETAS:
        a = accum[beta]
        n = a["n"]
        if n == 0:
            continue
        auc = a["auc_sum"] / n
        if auc > best_auc:
            best_auc, best_beta = auc, beta
        print(f"{beta:>6.2f}  {auc:>8.4f}  {a['mrr_sum']/n:>8.4f}  "
              f"{a['n5_sum']/n:>8.4f}  {a['n10_sum']/n:>8.4f}")
        rows.append({
            "model": "mpnet",
            "history_cap": HISTORY_CAP,
            "beta_embed": beta,
            "n_impressions": n,
            "AUC": auc,
            "MRR": a["mrr_sum"] / n,
            "nDCG@5": a["n5_sum"] / n,
            "nDCG@10": a["n10_sum"] / n,
        })

    print(f"\nBest: beta={best_beta:.2f}, AUC={best_auc:.4f}")
    print(f"Pure mpnet (beta=1.0): AUC={baseline_auc:.4f}")
    print(f"Delta: {best_auc - baseline_auc:+.4f}")

    out_path = results_dir("large") / "mpnet_category_tuning.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
