"""Sweep recency-decay and history-cap hyperparameters for MIND embedding retrieval.

Streams through the MIND large validation set once, and for each impression
evaluates ALL (history_cap, decay) configurations. This avoids N full passes
over the 431K-impression file.

Usage:
    .venv/bin/python -m scripts.tune_user_vector
    .venv/bin/python -m scripts.tune_user_vector --sample 10000   # fast sanity pass
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from src.common.paths import interim_dir, processed_dir, results_dir
from src.retrieval.ann import ArticleIndex
from src.retrieval.embeddings import load_embeddings

HISTORY_CAPS = [5, 10, 15, 20, 30, 50]
DECAYS = [0.7, 0.8, 0.85, 0.9, 0.95, 1.0]
BATCH_SIZE = 5000


def _auc_impression(labels: list[int], scores: list[float]) -> float:
    """Per-impression AUC via rank-sum formula."""
    n = len(labels)
    if n == 0:
        return 0.5
    pos_scores = [s for s, l in zip(scores, labels) if l == 1]
    neg_scores = [s for s, l in zip(scores, labels) if l == 0]
    n_pos, n_neg = len(pos_scores), len(neg_scores)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    total = 0.0
    for ps in pos_scores:
        for ns in neg_scores:
            if ps > ns:
                total += 1.0
            elif ps == ns:
                total += 0.5
    return total / (n_pos * n_neg)


def _mrr_impression(labels: list[int], scores: list[float]) -> float:
    """Per-impression MRR."""
    if sum(labels) == 0:
        return 0.0
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    for rank, idx in enumerate(order, 1):
        if labels[idx] == 1:
            return 1.0 / rank
    return 0.0


def _ndcg_at_k_impression(labels: list[int], scores: list[float], k: int) -> float:
    """Per-impression nDCG@K."""
    n_pos = sum(labels)
    if n_pos == 0:
        return 0.0
    order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
    dcg = sum(labels[idx] / np.log2(r + 2) for r, idx in enumerate(order))
    ideal_hits = min(n_pos, k)
    idcg = sum(1.0 / np.log2(r + 2) for r in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def _build_user_vector(
    history_embeddings: np.ndarray,
    decay: float,
) -> np.ndarray:
    """Build recency-weighted, L2-normalized user vector.
    
    history_embeddings: [H, D] — oldest first, most recent last.
    decay: exponential decay factor. 1.0 = uniform mean (baseline).
    """
    H = history_embeddings.shape[0]
    if H == 0:
        return np.zeros(history_embeddings.shape[1], dtype=np.float32)
    
    if decay >= 1.0 - 1e-9:
        # Uniform mean — fast path
        vec = history_embeddings.mean(axis=0, dtype=np.float32)
    else:
        weights = np.array(
            [decay ** (H - 1 - i) for i in range(H)],
            dtype=np.float32,
        )
        weights /= weights.sum()
        vec = (history_embeddings * weights[:, None]).sum(axis=0, dtype=np.float32)
    
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.astype(np.float32, copy=False)


def _parse_mind_history(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, str):
        return json.loads(value) if value else []
    return list(value)


def _iter_mind_val(behaviors_path: Path, batch_size: int):
    """Yield validation rows in batches."""
    parquet = pq.ParquetFile(behaviors_path)
    columns = ["impression_id", "user_id", "timestamp", "clicked_history",
               "candidates", "labels", "split"]
    buffer = []
    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        for row in pa.Table.from_batches([batch]).to_pylist():
            if row["split"] == "val":
                buffer.append(row)
                if len(buffer) >= batch_size:
                    yield buffer
                    buffer = []
    if buffer:
        yield buffer


def main():
    import logging
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
    logger = logging.getLogger(__name__)

    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=None,
                    help="Subsample N impressions for a fast pass")
    args = ap.parse_args()

    dataset = "mind"
    scale = "large"
    model = "minilm"

    # ── Load embeddings & build index ──
    t0 = time.time()
    embeddings, id_to_row, _ = load_embeddings(dataset, model, scale=scale)
    article_ids_ordered = [""] * len(id_to_row)
    for aid, idx in id_to_row.items():
        article_ids_ordered[int(idx)] = aid
    index = ArticleIndex(embeddings, article_ids_ordered, build_full_index=False)
    logger.info("Loaded index: %s (%.1fs)", index, time.time() - t0)

    # ── Build config grid ──
    configs = [(cap, decay) for cap in HISTORY_CAPS for decay in DECAYS]
    logger.info("Sweeping %d configs: caps=%s, decays=%s",
                len(configs), HISTORY_CAPS, DECAYS)

    # Running accumulators: {(cap, decay): {"auc_sum": ..., "mrr_sum": ..., "n": ...}}
    accum = {
        cfg: {"auc_sum": 0.0, "mrr_sum": 0.0, "ndcg5_sum": 0.0, "ndcg10_sum": 0.0, "n": 0}
        for cfg in configs
    }

    # ── Stream through validation ──
    behaviors_path = interim_dir(dataset, scale) / "behaviors.parquet"
    total_processed = 0
    max_cap = max(HISTORY_CAPS)

    for batch_rows in _iter_mind_val(behaviors_path, BATCH_SIZE):
        for row in batch_rows:
            candidates_raw = row["candidates"]
            labels_raw = row["labels"]
            candidates = json.loads(candidates_raw) if isinstance(candidates_raw, str) else candidates_raw
            labels = json.loads(labels_raw) if isinstance(labels_raw, str) else labels_raw
            candidates = [str(c) for c in candidates]
            labels = [int(l) for l in labels]

            # Skip degenerate impressions
            n_pos = sum(labels)
            if n_pos == 0 or n_pos == len(labels):
                continue

            # Get full history (truncated to max_cap)
            history = _parse_mind_history(row.get("clicked_history"))
            full_history = history[-max_cap:]
            
            # Get history article embeddings (once, for the full set)
            history_ids = [str(entry["article_id"]) for entry in full_history]
            if not history_ids:
                # No history: user vector is zero for ALL configs
                for cfg in configs:
                    accum[cfg]["n"] += 1
                    accum[cfg]["auc_sum"] += 0.5
                total_processed += 1
                continue

            hist_embeddings, found_ids = index.get_embeddings_batch(history_ids)
            if hist_embeddings.shape[0] == 0:
                for cfg in configs:
                    accum[cfg]["n"] += 1
                    accum[cfg]["auc_sum"] += 0.5
                total_processed += 1
                continue

            # Build id→embedding position map for this history
            found_set = set(found_ids)
            # Reconstruct ordered embeddings matching history order
            id_to_emb_idx = {aid: i for i, aid in enumerate(found_ids)}
            ordered_emb_indices = [id_to_emb_idx[aid] for aid in history_ids if aid in found_set]
            ordered_embeddings = hist_embeddings[ordered_emb_indices]  # [H_found, D]

            # Get candidate embeddings for scoring
            valid_cand_indices = []
            valid_cand_ids = []
            for cid in candidates:
                idx_val = id_to_row.get(cid)
                if idx_val is not None:
                    valid_cand_indices.append(int(idx_val))
                    valid_cand_ids.append(cid)
            
            if not valid_cand_indices:
                for cfg in configs:
                    accum[cfg]["n"] += 1
                    accum[cfg]["auc_sum"] += 0.5
                total_processed += 1
                continue

            cand_embeddings = embeddings[valid_cand_indices]  # [C_found, D]
            cand_id_set = set(valid_cand_ids)

            # For each config, build user vector and score
            for cap, decay in configs:
                # Truncate history to this cap
                n_to_use = min(cap, ordered_embeddings.shape[0])
                # Take the LAST n_to_use (most recent)
                h_emb = ordered_embeddings[-n_to_use:]
                
                user_vec = _build_user_vector(h_emb, decay)
                
                # Score candidates via cosine similarity
                sim_scores = cand_embeddings @ user_vec  # [C_found]
                
                # Build full score list (0.0 for missing candidates)
                score_lookup = dict(zip(valid_cand_ids, sim_scores.tolist()))
                scores = [score_lookup.get(cid, 0.0) for cid in candidates]
                
                auc = _auc_impression(labels, scores)
                mrr = _mrr_impression(labels, scores)
                ndcg5 = _ndcg_at_k_impression(labels, scores, 5)
                ndcg10 = _ndcg_at_k_impression(labels, scores, 10)
                
                a = accum[(cap, decay)]
                a["auc_sum"] += auc
                a["mrr_sum"] += mrr
                a["ndcg5_sum"] += ndcg5
                a["ndcg10_sum"] += ndcg10
                a["n"] += 1

            total_processed += 1

        if total_processed % 10_000 < BATCH_SIZE:
            elapsed = time.time() - t0
            logger.info("  Processed %d impressions (%.1f min)", total_processed, elapsed / 60)

        if args.sample and total_processed >= args.sample:
            logger.info("  Sample limit reached (%d)", args.sample)
            break

    # ── Results ──
    elapsed = time.time() - t0
    logger.info("Done: %d impressions in %.1f min", total_processed, elapsed / 60)

    rows = []
    for (cap, decay), a in sorted(accum.items()):
        n = a["n"]
        if n == 0:
            continue
        rows.append({
            "history_cap": cap,
            "decay": decay,
            "n_impressions": n,
            "AUC": a["auc_sum"] / n,
            "MRR": a["mrr_sum"] / n,
            "nDCG@5": a["ndcg5_sum"] / n,
            "nDCG@10": a["ndcg10_sum"] / n,
        })

    out = pl.DataFrame(rows)
    out_path = results_dir(scale) / "user_vector_tuning.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(out_path)

    print(f"\n{'='*90}")
    print(f"User Vector Tuning Results ({total_processed:,} impressions, {elapsed/60:.1f} min)")
    print(f"{'='*90}")
    print(f"{'cap':>5} {'decay':>6} {'AUC':>8} {'MRR':>8} {'nDCG@5':>8} {'nDCG@10':>8}")
    print("-" * 50)

    best_auc = 0.0
    best_cfg = None
    for r in rows:
        marker = ""
        if r["AUC"] > best_auc:
            best_auc = r["AUC"]
            best_cfg = (r["history_cap"], r["decay"])
        print(f"{r['history_cap']:>5} {r['decay']:>6.2f} {r['AUC']:>8.4f} {r['MRR']:>8.4f} "
              f"{r['nDCG@5']:>8.4f} {r['nDCG@10']:>8.4f}")

    print(f"\nBest config: history_cap={best_cfg[0]}, decay={best_cfg[1]}, AUC={best_auc:.4f}")
    print(f"Baseline (cap=20, decay=1.0): AUC={accum[(20, 1.0)]['auc_sum']/accum[(20, 1.0)]['n']:.4f}")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
