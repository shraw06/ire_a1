"""Sweep history_cap for the mpnet-base-v2 embedding model on MIND validation.

Since mpnet (768-dim) may have different optimal history depth than MiniLM (384-dim),
we run a targeted cap sweep using the cached mpnet embeddings.

Usage:
    .venv/bin/python -m scripts.tune_mpnet_cap
    .venv/bin/python -m scripts.tune_mpnet_cap --sample 20000
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.common.paths import interim_dir, results_dir
from src.retrieval.ann import ArticleIndex

logger = logging.getLogger(__name__)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_EMBED_DIR = _PROJECT_ROOT / "data" / "processed" / "embeddings"

HISTORY_CAPS = [10, 20, 30, 50, 75, 100]
BATCH_SIZE = 5000


def _auc_impression(labels: list[int], scores: list[float]) -> float:
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return 0.5
    total = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return total / (len(pos) * len(neg))


def _mrr_impression(labels: list[int], scores: list[float]) -> float:
    if sum(labels) == 0:
        return 0.0
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    for rank, idx in enumerate(order, 1):
        if labels[idx] == 1:
            return 1.0 / rank
    return 0.0


def _ndcg_at_k(labels: list[int], scores: list[float], k: int) -> float:
    n_pos = sum(labels)
    if n_pos == 0:
        return 0.0
    order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
    dcg = sum(labels[idx] / np.log2(r + 2) for r, idx in enumerate(order))
    idcg = sum(1.0 / np.log2(r + 2) for r in range(min(n_pos, k)))
    return dcg / idcg if idcg > 0 else 0.0


def _parse_mind_history(value: Any) -> list[dict[str, Any]]:
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

    # Load mpnet embeddings
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

    max_cap = max(HISTORY_CAPS)
    accum = {
        cap: {"auc_sum": 0.0, "mrr_sum": 0.0, "ndcg5_sum": 0.0, "ndcg10_sum": 0.0, "n": 0}
        for cap in HISTORY_CAPS
    }

    behaviors_path = interim_dir("mind", "large") / "behaviors.parquet"
    parquet = pq.ParquetFile(behaviors_path)
    columns = ["impression_id", "user_id", "clicked_history", "candidates", "labels", "split"]

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
            labels = [int(l) for l in labels]

            if sum(labels) == 0 or sum(labels) == len(labels):
                continue

            history = _parse_mind_history(row.get("clicked_history"))
            full_history_ids = [str(e["article_id"]) for e in history[-max_cap:]]
            if not full_history_ids:
                total += 1
                continue

            full_embs, found_ids = index.get_embeddings_batch(full_history_ids)
            if full_embs.shape[0] == 0:
                total += 1
                continue

            # Get candidate embeddings once
            cand_idxs = [id_to_row.get(c) for c in candidates]
            valid_mask = [idx is not None for idx in cand_idxs]
            valid_cand_emb_idxs = [int(idx) for idx, valid in zip(cand_idxs, valid_mask) if valid]
            valid_cands = [c for c, valid in zip(candidates, valid_mask) if valid]

            if not valid_cand_emb_idxs:
                total += 1
                continue
            cand_embs = embeddings[valid_cand_emb_idxs]  # [C, D]

            for cap in HISTORY_CAPS:
                # Use last `cap` articles from what we have
                h_emb = full_embs[-cap:]
                vec = h_emb.mean(axis=0, dtype=np.float32)
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec /= norm

                sims = cand_embs @ vec  # [C]
                score_lookup = dict(zip(valid_cands, sims.tolist()))
                scores = [score_lookup.get(c, 0.0) for c in candidates]

                a = accum[cap]
                a["auc_sum"] += _auc_impression(labels, scores)
                a["mrr_sum"] += _mrr_impression(labels, scores)
                a["ndcg5_sum"] += _ndcg_at_k(labels, scores, 5)
                a["ndcg10_sum"] += _ndcg_at_k(labels, scores, 10)
                a["n"] += 1

            total += 1

        if total % 10_000 < BATCH_SIZE:
            logger.info("  %d impressions (%.1f min)", total, (time.time() - t0) / 60)
        if args.sample and total >= args.sample:
            logger.info("Sample limit (%d) reached", args.sample)
            break

    elapsed = time.time() - t0
    logger.info("Done: %d impressions in %.1f min", total, elapsed / 60)

    import polars as pl
    rows = []
    print(f"\n{'='*65}")
    print(f"mpnet Cap Sweep ({total:,} impressions, {elapsed/60:.1f} min)")
    print(f"{'='*65}")
    print(f"{'cap':>5}  {'AUC':>8}  {'MRR':>8}  {'nDCG@5':>8}  {'nDCG@10':>8}")
    print("-" * 50)

    best_auc, best_cap = 0.0, None
    for cap in HISTORY_CAPS:
        a = accum[cap]
        n = a["n"]
        if n == 0:
            continue
        auc = a["auc_sum"] / n
        if auc > best_auc:
            best_auc, best_cap = auc, cap
        print(f"{cap:>5}  {auc:>8.4f}  {a['mrr_sum']/n:>8.4f}  "
              f"{a['ndcg5_sum']/n:>8.4f}  {a['ndcg10_sum']/n:>8.4f}")
        rows.append({
            "model": "mpnet",
            "history_cap": cap,
            "n_impressions": n,
            "AUC": auc,
            "MRR": a["mrr_sum"] / n,
            "nDCG@5": a["ndcg5_sum"] / n,
            "nDCG@10": a["ndcg10_sum"] / n,
        })

    print(f"\nBest: cap={best_cap}, AUC={best_auc:.4f}")
    out = pl.DataFrame(rows)
    out_path = results_dir("large") / "mpnet_cap_tuning.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
