"""Trainable attention-based user encoder on frozen mpnet embeddings.

Architecture (NRMS-lite):
  - News encoder: FROZEN mpnet-base-v2 embeddings (768-D)
  - User encoder: Additive attention over history embeddings (trainable)
    α_i = softmax(v^T tanh(W h_i))
    u   = Σ α_i h_i
  - Click predictor: dot(u, candidate_emb) → softmax over candidates
  - Loss: cross-entropy with in-impression negatives (1 positive per impression)

By training only the small attention head (~768*200 + 200 = 154K params),
we adapt the user representation to MIND's click patterns without
fine-tuning the news encoder — fast training (~30-40 min) with
large gains expected over mean-pool.

Usage:
    .venv/bin/python -m scripts.train_attention_ranker
    .venv/bin/python -m scripts.train_attention_ranker --epochs 5 --sample 500000
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_EMBED_DIR = _PROJECT_ROOT / "data" / "processed" / "embeddings"
_MODELS_DIR = _PROJECT_ROOT / "models"

HISTORY_CAP = 50
MAX_CANDS = 50       # max candidates per impression to avoid OOM
BATCH_IMPRESSIONS = 128   # impressions per gradient step


class AdditiveAttentionUserEncoder(nn.Module):
    """Trainable attention-weighted user representation.
    
    Given H history embeddings of dim D, computes a single user vector
    via learned additive attention. Params: W (D×A) + v (A×1) = D*A + A.
    """

    def __init__(self, embed_dim: int = 768, attn_dim: int = 200) -> None:
        super().__init__()
        self.W = nn.Linear(embed_dim, attn_dim, bias=True)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(
        self,
        history_embs: torch.Tensor,  # [B, H, D]
        history_mask: torch.Tensor,  # [B, H] — True where VALID (not padding)
    ) -> torch.Tensor:               # [B, D]
        """Compute attention-weighted user vector."""
        # [B, H, A]
        attn = torch.tanh(self.W(history_embs))
        # [B, H, 1] → [B, H]
        scores = self.v(attn).squeeze(-1)
        # Mask padding positions
        scores = scores.masked_fill(~history_mask, -1e9)
        weights = F.softmax(scores, dim=-1)  # [B, H]
        # Weighted sum → [B, D]
        user_vec = (history_embs * weights.unsqueeze(-1)).sum(dim=1)
        return F.normalize(user_vec, dim=-1)


def _parse_mind_history(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parsed = json.loads(value) if value else []
        return [str(e["article_id"]) if isinstance(e, dict) else str(e) for e in parsed]
    return [str(e["article_id"]) if isinstance(e, dict) else str(e) for e in value]


def _iter_train_impressions(behaviors_path: Path, sample: int | None = None):
    """Yield raw training impression dicts."""
    parquet = pq.ParquetFile(behaviors_path)
    columns = ["clicked_history", "candidates", "labels", "split"]
    rows = []
    for batch in parquet.iter_batches(batch_size=50_000, columns=columns):
        data = batch.to_pydict()
        for i in range(batch.num_rows):
            if data["split"][i] != "train":
                continue
            rows.append({
                "history_raw": data["clicked_history"][i],
                "candidates_raw": data["candidates"][i],
                "labels_raw": data["labels"][i],
            })
    
    if sample and len(rows) > sample:
        random.seed(42)
        rows = random.sample(rows, sample)
    
    logger.info("Loaded %d training impressions", len(rows))
    return rows


def _make_batch(
    impression_rows: list[dict],
    embeddings: np.ndarray,
    id_to_row: dict[str, int],
    history_cap: int,
    max_cands: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    """Pack a list of impression dicts into padded tensors."""
    D = embeddings.shape[1]
    B = len(impression_rows)

    hist_embs_list = []
    hist_masks_list = []
    cand_embs_list = []
    cand_masks_list = []
    pos_indices = []
    valid = []  # impression indices with at least 1 positive

    for row in impression_rows:
        history_ids = _parse_mind_history(row["history_raw"])[-history_cap:]
        candidates_raw = row["candidates_raw"]
        labels_raw = row["labels_raw"]
        candidates = json.loads(candidates_raw) if isinstance(candidates_raw, str) else candidates_raw
        labels = json.loads(labels_raw) if isinstance(labels_raw, str) else labels_raw
        candidates = [str(c) for c in candidates][:max_cands]
        labels = [int(l) for l in labels][:max_cands]

        # Need exactly 1 positive
        pos_list = [j for j, l in enumerate(labels) if l == 1]
        if not pos_list:
            continue
        pos_idx = pos_list[0]  # first positive
        pos_indices.append(pos_idx)
        valid.append(True)

        # History embeddings
        hist_rows = [id_to_row[aid] for aid in history_ids if aid in id_to_row]
        H = min(len(hist_rows), history_cap)
        hist_emb = np.zeros((history_cap, D), dtype=np.float32)
        hist_mask = np.zeros(history_cap, dtype=bool)
        if H > 0:
            hist_emb[:H] = embeddings[hist_rows[:H]]
            hist_mask[:H] = True
        hist_embs_list.append(hist_emb)
        hist_masks_list.append(hist_mask)

        # Candidate embeddings
        C = len(candidates)
        cand_emb = np.zeros((max_cands, D), dtype=np.float32)
        cand_mask = np.zeros(max_cands, dtype=bool)
        for j, cid in enumerate(candidates):
            r = id_to_row.get(cid)
            if r is not None:
                cand_emb[j] = embeddings[int(r)]
                cand_mask[j] = True
        cand_embs_list.append(cand_emb)
        cand_masks_list.append(cand_mask)

    if not pos_indices:
        return None, None, None, None, []

    hist_t = torch.tensor(np.stack(hist_embs_list), dtype=torch.float32, device=device)
    hist_m = torch.tensor(np.stack(hist_masks_list), dtype=torch.bool, device=device)
    cand_t = torch.tensor(np.stack(cand_embs_list), dtype=torch.float32, device=device)
    cand_m = torch.tensor(np.stack(cand_masks_list), dtype=torch.bool, device=device)
    pos_t = torch.tensor(pos_indices, dtype=torch.long, device=device)

    return hist_t, hist_m, cand_t, cand_m, pos_t


def _eval_val(
    model: AdditiveAttentionUserEncoder,
    embeddings: np.ndarray,
    id_to_row: dict[str, int],
    behaviors_path: Path,
    device: torch.device,
    max_impressions: int = 20000,
) -> float:
    """Quick AUC estimate on val impressions."""
    model.eval()
    D = embeddings.shape[1]
    parquet = pq.ParquetFile(behaviors_path)
    columns = ["clicked_history", "candidates", "labels", "split"]

    auc_sum = 0.0
    n = 0

    with torch.no_grad():
        for batch in parquet.iter_batches(batch_size=1000, columns=columns):
            data = batch.to_pydict()
            for i in range(batch.num_rows):
                if data["split"][i] != "val":
                    continue
                if n >= max_impressions:
                    break

                history_ids = _parse_mind_history(data["clicked_history"][i])[-HISTORY_CAP:]
                candidates_raw = data["candidates"][i]
                labels_raw = data["labels"][i]
                candidates = json.loads(candidates_raw) if isinstance(candidates_raw, str) else candidates_raw
                labels = [int(l) for l in (json.loads(labels_raw) if isinstance(labels_raw, str) else labels_raw)]
                candidates = [str(c) for c in candidates]

                n_pos = sum(labels)
                if n_pos == 0 or n_pos == len(labels):
                    continue

                hist_rows = [id_to_row[aid] for aid in history_ids if aid in id_to_row]
                H = min(len(hist_rows), HISTORY_CAP)
                hist_emb = np.zeros((1, HISTORY_CAP, D), dtype=np.float32)
                hist_mask = np.zeros((1, HISTORY_CAP), dtype=bool)
                if H > 0:
                    hist_emb[0, :H] = embeddings[hist_rows[:H]]
                    hist_mask[0, :H] = True

                ht = torch.tensor(hist_emb, dtype=torch.float32, device=device)
                hm = torch.tensor(hist_mask, dtype=torch.bool, device=device)
                user_vec = model(ht, hm)[0]  # [D]

                cand_rows = [id_to_row.get(c) for c in candidates]
                valid_cands = [(j, int(r)) for j, r in enumerate(cand_rows) if r is not None]
                if not valid_cands:
                    continue

                scores = [0.0] * len(candidates)
                for j, r in valid_cands:
                    cand_emb_t = torch.tensor(embeddings[r], dtype=torch.float32, device=device)
                    scores[j] = float(user_vec @ cand_emb_t)

                # AUC
                pos_s = [s for s, l in zip(scores, labels) if l == 1]
                neg_s = [s for s, l in zip(scores, labels) if l == 0]
                total = sum(1.0 if p > q else 0.5 for p in pos_s for q in neg_s)
                auc_sum += total / (len(pos_s) * len(neg_s))
                n += 1

            if n >= max_impressions:
                break

    return auc_sum / n if n > 0 else 0.5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--attn-dim", type=int, default=200)
    parser.add_argument("--sample", type=int, default=None,
                        help="Sample N training impressions. None = all 1.8M")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info("Device: %s", device)

    # Load mpnet embeddings
    t0 = time.time()
    embeddings = np.load(_EMBED_DIR / "mind_mpnet_large.npy")
    id_to_row: dict[str, int] = json.loads(
        (_EMBED_DIR / "mind_mpnet_large_ids.json").read_text()
    )
    D = embeddings.shape[1]
    logger.info("Loaded embeddings: %s (%.1fs)", embeddings.shape, time.time() - t0)

    # Build model
    model = AdditiveAttentionUserEncoder(embed_dim=D, attn_dim=args.attn_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model: %d params", n_params)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.1
    )

    # Load training data
    behaviors_path = _PROJECT_ROOT / "data" / "interim" / "large" / "mind" / "behaviors.parquet"
    train_rows = _iter_train_impressions(behaviors_path, sample=args.sample)

    best_auc = 0.0
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = _MODELS_DIR / "attention_user_encoder.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(train_rows)

        epoch_loss = 0.0
        epoch_batches = 0
        t_epoch = time.time()

        for batch_start in range(0, len(train_rows), args.batch):
            batch_rows = train_rows[batch_start: batch_start + args.batch]

            hist_t, hist_m, cand_t, cand_m, pos_t = _make_batch(
                batch_rows, embeddings, id_to_row,
                HISTORY_CAP, MAX_CANDS, device,
            )
            if hist_t is None or len(pos_t) == 0:
                continue

            B = hist_t.shape[0]

            # Forward
            user_vecs = model(hist_t, hist_m)  # [B, D]

            # Scores: [B, MAX_CANDS]
            scores = torch.bmm(cand_t, user_vecs.unsqueeze(-1)).squeeze(-1)
            # Mask invalid candidate positions with -inf
            scores = scores.masked_fill(~cand_m, -1e9)

            # Softmax cross-entropy: treat as classification over candidates
            loss = F.cross_entropy(scores, pos_t)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_batches += 1

            if epoch_batches % 500 == 0:
                elapsed = time.time() - t_epoch
                logger.info("  Epoch %d | batch %d/%d | loss=%.4f | %.1f min",
                            epoch, epoch_batches,
                            math.ceil(len(train_rows) / args.batch),
                            epoch_loss / epoch_batches, elapsed / 60)

        scheduler.step()

        # Validation
        val_auc = _eval_val(model, embeddings, id_to_row, behaviors_path, device)
        avg_loss = epoch_loss / max(epoch_batches, 1)
        elapsed = time.time() - t_epoch
        logger.info("Epoch %d done | loss=%.4f | val_AUC(20K)=%.4f | %.1f min",
                    epoch, avg_loss, val_auc, elapsed / 60)

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({
                "model_state": model.state_dict(),
                "attn_dim": args.attn_dim,
                "embed_dim": D,
                "epoch": epoch,
                "val_auc": val_auc,
            }, save_path)
            logger.info("  → Saved best model (AUC=%.4f)", val_auc)

    logger.info("Training complete. Best val AUC=%.4f. Model at %s", best_auc, save_path)


if __name__ == "__main__":
    main()
