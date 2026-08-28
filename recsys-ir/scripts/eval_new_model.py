"""Evaluate an alternate embedding model on MIND large validation.

Streams through behaviors, constructs user vectors from the specified model's
embeddings, scores candidates, computes AUC/MRR/nDCG. Supports the --model
flag to select between cached embedding sets.

Usage:
    .venv/bin/python -m scripts.eval_new_model --model mpnet
    .venv/bin/python -m scripts.eval_new_model --model mpnet --sample 10000
    .venv/bin/python -m scripts.eval_new_model --model mpnet --history-cap 50
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

BATCH_SIZE = 5000


def _load_model_embeddings(model: str) -> tuple[np.ndarray, dict[str, int]]:
    """Load cached embeddings for the specified model."""
    prefix_map = {
        "minilm": "mind_minilm",
        "mpnet": "mind_mpnet",
        "minilm12": "mind_minilm12",
        "bge": "mind_bge",
    }
    prefix = prefix_map.get(model)
    if prefix is None:
        raise ValueError(f"Unknown model: {model}. Options: {list(prefix_map.keys())}")

    npy_path = _EMBED_DIR / f"{prefix}_large.npy"
    ids_path = _EMBED_DIR / f"{prefix}_large_ids.json"

    if not npy_path.exists():
        raise FileNotFoundError(
            f"Embeddings not cached for {model}. Run:\n"
            f"  .venv/bin/python -m scripts.compute_mpnet_embeddings --model {model}"
        )

    embeddings = np.load(npy_path)
    id_to_row = json.loads(ids_path.read_text())
    return embeddings, id_to_row


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


def _build_user_vector(
    history_ids: list[str],
    index: ArticleIndex,
    history_cap: int,
    decay: float,
) -> np.ndarray:
    ids = history_ids[-history_cap:]
    if not ids:
        return np.zeros(index.dim, dtype=np.float32)

    embeddings_batch, found_ids = index.get_embeddings_batch(ids)
    if embeddings_batch.shape[0] == 0:
        return np.zeros(index.dim, dtype=np.float32)

    H = embeddings_batch.shape[0]
    if decay >= 1.0 - 1e-9:
        vec = embeddings_batch.mean(axis=0, dtype=np.float32)
    else:
        weights = np.array([decay ** (H - 1 - i) for i in range(H)], dtype=np.float32)
        weights /= weights.sum()
        vec = (embeddings_batch * weights[:, None]).sum(axis=0, dtype=np.float32)

    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec.astype(np.float32, copy=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["minilm", "mpnet", "minilm12", "bge"])
    parser.add_argument("--history-cap", type=int, default=20)
    parser.add_argument("--decay", type=float, default=1.0)
    parser.add_argument("--sample", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")

    # Load embeddings
    t0 = time.time()
    embeddings, id_to_row = _load_model_embeddings(args.model)
    article_ids_ordered = [""] * len(id_to_row)
    for aid, idx in id_to_row.items():
        article_ids_ordered[int(idx)] = aid
    index = ArticleIndex(embeddings, article_ids_ordered, build_full_index=False)
    logger.info("Loaded %s index: %s (%.1fs)", args.model, index, time.time() - t0)

    # Stream validation
    behaviors_path = interim_dir("mind", "large") / "behaviors.parquet"
    parquet = pq.ParquetFile(behaviors_path)
    columns = ["impression_id", "user_id", "timestamp", "clicked_history",
               "candidates", "labels", "split"]

    auc_sum, mrr_sum, ndcg5_sum, ndcg10_sum = 0.0, 0.0, 0.0, 0.0
    count = 0

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

            n_pos = sum(labels)
            if n_pos == 0 or n_pos == len(labels):
                continue

            history = _parse_mind_history(row.get("clicked_history"))
            history_ids = [str(entry["article_id"]) for entry in history]

            user_vec = _build_user_vector(history_ids, index, args.history_cap, args.decay)

            # Score candidates
            results = index.search_restricted(user_vec, candidates, k=len(candidates))
            score_lookup = dict(results)
            scores = [score_lookup.get(cid, 0.0) for cid in candidates]

            auc_sum += _auc_impression(labels, scores)
            mrr_sum += _mrr_impression(labels, scores)
            ndcg5_sum += _ndcg_at_k(labels, scores, 5)
            ndcg10_sum += _ndcg_at_k(labels, scores, 10)
            count += 1

        if count % 10_000 < BATCH_SIZE:
            elapsed = time.time() - t0
            logger.info("  %d impressions (%.1f min), running AUC=%.4f",
                        count, elapsed / 60, auc_sum / count if count else 0)

        if args.sample and count >= args.sample:
            logger.info("Sample limit reached (%d)", args.sample)
            break

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Model: {args.model} | cap={args.history_cap} | decay={args.decay}")
    print(f"{'='*70}")
    if count > 0:
        print(f"  AUC:     {auc_sum / count:.4f}")
        print(f"  MRR:     {mrr_sum / count:.4f}")
        print(f"  nDCG@5:  {ndcg5_sum / count:.4f}")
        print(f"  nDCG@10: {ndcg10_sum / count:.4f}")
    print(f"  Impressions: {count:,}")
    print(f"  Time: {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
