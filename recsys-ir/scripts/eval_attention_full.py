"""Evaluate attention user encoder on the full MIND validation set (431K impressions).

Uses the same streaming approach as the submission generator but computes
AUC, MRR, nDCG@5, nDCG@10 on all labeled impressions.

Usage:
    .venv/bin/python -m scripts.eval_attention_full
"""
from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch
import pyarrow.parquet as pq

from scripts.train_attention_fast import AdditiveAttentionUserEncoder, HISTORY_CAP

logger = logging.getLogger(__name__)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_EMBED_DIR = _PROJECT_ROOT / "data" / "processed" / "embeddings"
_MODELS_DIR = _PROJECT_ROOT / "models"


def _parse_history(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parsed = json.loads(value) if value else []
    else:
        parsed = list(value)
    return [str(e["article_id"]) if isinstance(e, dict) else str(e) for e in parsed]


def _dcg(ranked_labels: list[int], k: int) -> float:
    return sum(
        lbl / math.log2(i + 2)
        for i, lbl in enumerate(ranked_labels[:k])
    )


def _ndcg(scores: list[float], labels: list[int], k: int) -> float:
    paired = sorted(zip(scores, labels), key=lambda x: -x[0])
    ranked = [l for _, l in paired]
    ideal = sorted(labels, reverse=True)
    dcg_val = _dcg(ranked, k)
    idcg_val = _dcg(ideal, k)
    return dcg_val / idcg_val if idcg_val > 0 else 0.0


def _mrr(scores: list[float], labels: list[int]) -> float:
    paired = sorted(zip(scores, labels), key=lambda x: -x[0])
    for rank, (_, lbl) in enumerate(paired, 1):
        if lbl == 1:
            return 1.0 / rank
    return 0.0


def _auc(scores: list[float], labels: list[int]) -> float:
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return 0.5
    total = sum(1.0 if p > q else 0.5 for p in pos for q in neg)
    return total / (len(pos) * len(neg))


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Load model
    ckpt = torch.load(_MODELS_DIR / "attention_user_encoder.pt", map_location=device)
    model = AdditiveAttentionUserEncoder(
        embed_dim=ckpt["embed_dim"], attn_dim=ckpt["attn_dim"]
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    logger.info("Loaded model (epoch=%d, val_AUC_20K=%.4f)", ckpt["epoch"], ckpt["val_auc"])

    # Load embeddings
    embeddings = torch.from_numpy(
        np.load(_EMBED_DIR / "mind_mpnet_large.npy")
    )  # CPU
    id_to_row: dict[str, int] = json.loads(
        (_EMBED_DIR / "mind_mpnet_large_ids.json").read_text()
    )
    D = embeddings.shape[1]
    logger.info("Embeddings: %s", embeddings.shape)

    behaviors_path = _PROJECT_ROOT / "data" / "interim" / "large" / "mind" / "behaviors.parquet"
    parquet = pq.ParquetFile(behaviors_path)
    columns = ["clicked_history", "candidates", "labels", "split"]

    auc_sum = mrr_sum = ndcg5_sum = ndcg10_sum = 0.0
    n = 0
    t0 = time.time()

    with torch.no_grad():
        for batch in parquet.iter_batches(batch_size=2000, columns=columns):
            data = batch.to_pydict()
            for i in range(batch.num_rows):
                if data["split"][i] != "val":
                    continue

                history_ids = _parse_history(data["clicked_history"][i])[-HISTORY_CAP:]
                candidates_raw = data["candidates"][i]
                labels_raw = data["labels"][i]
                candidates = [str(c) for c in (json.loads(candidates_raw) if isinstance(candidates_raw, str) else candidates_raw)]
                labels = [int(l) for l in (json.loads(labels_raw) if isinstance(labels_raw, str) else labels_raw)]

                n_pos = sum(labels)
                if n_pos == 0 or n_pos == len(labels):
                    continue

                hist_rows = [id_to_row[aid] for aid in history_ids if aid in id_to_row]
                H = min(len(hist_rows), HISTORY_CAP)

                if H == 0:
                    # Fall back to mean-pool (no history) — all scores 0
                    scores = [0.0] * len(candidates)
                else:
                    h_idx = np.zeros(HISTORY_CAP, dtype=np.int32)
                    h_idx[:H] = hist_rows[:H]
                    h_mask = np.zeros(HISTORY_CAP, dtype=bool)
                    h_mask[:H] = True

                    h_emb_t = embeddings[h_idx].unsqueeze(0).to(device)  # [1, H, D]
                    h_mask_t = torch.from_numpy(h_mask).unsqueeze(0).to(device)
                    user_vec = model(h_emb_t, h_mask_t)[0]  # [D]

                    scores = []
                    for cid in candidates:
                        r = id_to_row.get(cid)
                        if r is not None:
                            scores.append(float(user_vec @ embeddings[r].to(device)))
                        else:
                            scores.append(0.0)

                auc_sum += _auc(scores, labels)
                mrr_sum += _mrr(scores, labels)
                ndcg5_sum += _ndcg(scores, labels, 5)
                ndcg10_sum += _ndcg(scores, labels, 10)
                n += 1

            if n % 50000 == 0 and n > 0:
                elapsed = time.time() - t0
                logger.info("  %d impressions | AUC=%.4f | MRR=%.4f | nDCG@10=%.4f | %.1f min",
                            n, auc_sum / n, mrr_sum / n, ndcg10_sum / n, elapsed / 60)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Full-val evaluation: {n:,} impressions in {elapsed/60:.1f} min")
    print(f"{'='*60}")
    print(f"  AUC    = {auc_sum/n:.4f}")
    print(f"  MRR    = {mrr_sum/n:.4f}")
    print(f"  nDCG@5 = {ndcg5_sum/n:.4f}")
    print(f"  nDCG@10= {ndcg10_sum/n:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
