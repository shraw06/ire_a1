"""Tune candidate-conditioned history attention (query-key attention, no training).

For each candidate c, the user representation is computed as:
    α_i(c) = softmax( h_i · c / T )   ← history item relevance to this candidate
    u(c)    = Σ α_i(c) h_i             ← candidate-specific user vector
    score   = u(c) · c

This is the NRMS attention read operation WITHOUT any trained parameters —
it uses cosine similarity between history items and the candidate as the
attention weight. This is strictly better than uniform mean-pool because:
  - For each candidate, it focuses on the most relevant history items
  - No training required → no overfitting to temporal patterns
  - Temperature T controls sharpness (T→∞ = mean-pool, T→0 = argmax)

This should outperform the trained attention head (0.5217 test AUC) because
the query conditioning per candidate is fundamentally a stronger inductive bias.

Sweep: T ∈ {0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0} × history_cap ∈ {50, 100}
       on a 30K-impression sample from val split.

Usage:
    .venv/bin/python -m scripts.tune_query_key_attention
    .venv/bin/python -m scripts.tune_query_key_attention --sample 100000
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_EMBED_DIR = _PROJECT_ROOT / "data" / "processed" / "embeddings"


def _parse_history(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parsed = json.loads(value) if value else []
    else:
        parsed = list(value)
    return [str(e["article_id"]) if isinstance(e, dict) else str(e) for e in parsed]


def _auc(scores: list[float], labels: list[int]) -> float:
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return float("nan")
    total = sum(1.0 if p > q else 0.5 for p in pos for q in neg)
    return total / (len(pos) * len(neg))


def _mrr(scores: list[float], labels: list[int]) -> float:
    for rank, (_, lbl) in enumerate(sorted(zip(scores, labels), key=lambda x: -x[0]), 1):
        if lbl == 1:
            return 1.0 / rank
    return 0.0


def _ndcg(scores: list[float], labels: list[int], k: int) -> float:
    ranked = [l for _, l in sorted(zip(scores, labels), key=lambda x: -x[0])]
    ideal = sorted(labels, reverse=True)
    dcg = sum(l / math.log2(i + 2) for i, l in enumerate(ranked[:k]))
    idcg = sum(l / math.log2(i + 2) for i, l in enumerate(ideal[:k]))
    return dcg / idcg if idcg > 0 else 0.0


def score_impression_qk(
    history_embs: np.ndarray,  # [H, D]
    cand_embs: np.ndarray,     # [C, D]
    temperature: float,
    mean_pool_alpha: float = 0.0,   # blend with mean-pool: score = (1-a)*qk + a*mean
) -> np.ndarray:
    """Candidate-conditioned attention scoring. Returns [C] scores."""
    C, D = cand_embs.shape
    H = history_embs.shape[0]

    # Mean-pool baseline for blending
    if H == 0:
        return np.zeros(C)

    mean_vec = history_embs.mean(axis=0)  # [D]
    mean_scores = cand_embs @ mean_vec     # [C]

    if mean_pool_alpha == 1.0:
        return mean_scores

    # Query-key attention: for each candidate, attend over history
    # sim[c, h] = candidate_emb[c] · history_emb[h]
    sim = cand_embs @ history_embs.T  # [C, H]
    weights = sim / temperature        # [C, H]

    # Numerically stable softmax per row
    weights -= weights.max(axis=1, keepdims=True)
    weights = np.exp(weights)
    weights /= weights.sum(axis=1, keepdims=True)  # [C, H]

    # Candidate-conditioned user vectors
    user_vecs = weights @ history_embs  # [C, D]
    user_norms = np.linalg.norm(user_vecs, axis=1, keepdims=True).clip(min=1e-8)
    user_vecs /= user_norms            # [C, D] — normalize

    # Score: u(c) · c  (query-key style)
    qk_scores = (user_vecs * cand_embs).sum(axis=1)  # [C]

    return (1 - mean_pool_alpha) * qk_scores + mean_pool_alpha * mean_scores


def eval_config(
    embeddings: np.ndarray,
    id_to_row: dict[str, int],
    val_rows: list[dict],
    temperature: float,
    history_cap: int,
    mean_pool_alpha: float = 0.0,
) -> tuple[float, float, float, float]:
    """Evaluate one config on sampled val rows. Returns (AUC, MRR, nDCG@5, nDCG@10)."""
    D = embeddings.shape[1]
    auc_sum = mrr_sum = ndcg5_sum = ndcg10_sum = 0.0
    n = 0

    for row in val_rows:
        history_ids = _parse_history(row["clicked_history"])[-history_cap:]
        candidates_raw = row["candidates"]
        labels_raw = row["labels"]
        candidates = [str(c) for c in (json.loads(candidates_raw) if isinstance(candidates_raw, str) else candidates_raw)]
        labels = [int(l) for l in (json.loads(labels_raw) if isinstance(labels_raw, str) else labels_raw)]

        n_pos = sum(labels)
        if n_pos == 0 or n_pos == len(labels):
            continue

        hist_rows = [id_to_row[aid] for aid in history_ids if aid in id_to_row]
        cand_rows = [id_to_row.get(c) for c in candidates]

        if not hist_rows:
            continue

        hist_embs = embeddings[hist_rows]                      # [H, D]
        valid_cands = [(j, int(r)) for j, r in enumerate(cand_rows) if r is not None]
        if not valid_cands:
            continue

        cand_embs = embeddings[[r for _, r in valid_cands]]   # [C_valid, D]
        cand_indices = [j for j, _ in valid_cands]

        qk_scores_valid = score_impression_qk(hist_embs, cand_embs, temperature, mean_pool_alpha)

        scores = [0.0] * len(candidates)
        for (j, _), s in zip(valid_cands, qk_scores_valid):
            scores[j] = float(s)

        auc_sum += _auc(scores, labels)
        mrr_sum += _mrr(scores, labels)
        ndcg5_sum += _ndcg(scores, labels, 5)
        ndcg10_sum += _ndcg(scores, labels, 10)
        n += 1

    if n == 0:
        return 0.5, 0.0, 0.0, 0.0
    return auc_sum / n, mrr_sum / n, ndcg5_sum / n, ndcg10_sum / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=50000,
                        help="Number of val impressions to sample for speed")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")

    # Load embeddings
    t0 = time.time()
    embeddings = np.load(_EMBED_DIR / "mind_mpnet_large.npy")
    id_to_row: dict[str, int] = json.loads(
        (_EMBED_DIR / "mind_mpnet_large_ids.json").read_text()
    )
    logger.info("Loaded embeddings: %s (%.1fs)", embeddings.shape, time.time() - t0)

    # Load val rows
    behaviors_path = _PROJECT_ROOT / "data" / "interim" / "large" / "mind" / "behaviors.parquet"
    parquet = pq.ParquetFile(behaviors_path)
    columns = ["clicked_history", "candidates", "labels", "split"]

    val_rows = []
    for batch in parquet.iter_batches(batch_size=50_000, columns=columns):
        data = batch.to_pydict()
        for i in range(batch.num_rows):
            if data["split"][i] != "val":
                continue
            val_rows.append({
                "clicked_history": data["clicked_history"][i],
                "candidates": data["candidates"][i],
                "labels": data["labels"][i],
            })

    if args.sample and len(val_rows) > args.sample:
        import random; random.seed(42)
        val_rows = random.sample(val_rows, args.sample)

    logger.info("Val rows sampled: %d", len(val_rows))

    # Configs to sweep
    temperatures = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    history_caps = [50, 100]

    # First compute mean-pool baseline
    logger.info("Computing mean-pool baseline (alpha=1.0) ...")
    mean_auc, mean_mrr, mean_n5, mean_n10 = eval_config(
        embeddings, id_to_row, val_rows,
        temperature=1.0, history_cap=50, mean_pool_alpha=1.0
    )
    logger.info("Mean-pool (cap=50): AUC=%.4f  MRR=%.4f  nDCG@10=%.4f",
                mean_auc, mean_mrr, mean_n10)

    mean_auc100, _, _, mean_n10_100 = eval_config(
        embeddings, id_to_row, val_rows,
        temperature=1.0, history_cap=100, mean_pool_alpha=1.0
    )
    logger.info("Mean-pool (cap=100): AUC=%.4f  nDCG@10=%.4f", mean_auc100, mean_n10_100)

    print(f"\n{'='*70}")
    print(f"Query-Key Attention sweep ({len(val_rows):,} val impressions)")
    print(f"{'='*70}")
    print(f"{'cap':>4} {'T':>6} {'AUC':>8} {'Δ AUC':>8} {'MRR':>8} {'nDCG@10':>9}")
    print(f"{'-'*70}")

    best_auc = 0.0
    best_cfg = {}

    for cap in history_caps:
        base_auc = mean_auc if cap == 50 else mean_auc100
        for T in temperatures:
            t_eval = time.time()
            auc, mrr, n5, n10 = eval_config(
                embeddings, id_to_row, val_rows,
                temperature=T, history_cap=cap, mean_pool_alpha=0.0
            )
            delta = auc - base_auc
            print(f"{cap:>4} {T:>6.2f}   {auc:.4f}   {delta:+.4f}   {mrr:.4f}   {n10:.4f}  "
                  f"  ({time.time()-t_eval:.1f}s)")

            if auc > best_auc:
                best_auc = auc
                best_cfg = {"cap": cap, "T": T, "AUC": auc, "MRR": mrr, "nDCG10": n10}

    print(f"\nBest: cap={best_cfg['cap']}, T={best_cfg['T']:.2f}, AUC={best_cfg['AUC']:.4f}")
    print(f"Mean-pool baselines: cap=50 AUC={mean_auc:.4f}, cap=100 AUC={mean_auc100:.4f}")

    # Save to CSV
    out_path = _PROJECT_ROOT / "results" / "large" / "qk_attention_tuning.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        f.write("history_cap,temperature,AUC,MRR,nDCG5,nDCG10\n")
        for cap in history_caps:
            base_auc = mean_auc if cap == 50 else mean_auc100
            for T in temperatures:
                auc, mrr, n5, n10 = eval_config(
                    embeddings, id_to_row, val_rows,
                    temperature=T, history_cap=cap, mean_pool_alpha=0.0
                )
                f.write(f"{cap},{T},{auc:.6f},{mrr:.6f},{n5:.6f},{n10:.6f}\n")
    logger.info("Saved: %s", out_path)


if __name__ == "__main__":
    main()
