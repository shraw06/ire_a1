"""Fast attention ranker training via pre-extracted numpy arrays.

Strategy: First extract all training data into compact numpy arrays
(indices only, not embeddings), then train with pure GPU batching.
This avoids per-impression Python overhead during training.

Phase A (~5 min): Extract all 1.8M training impressions into:
  - history_idx.npy:   int32[N, H] — embedding row indices for history
  - history_len.npy:   int16[N]    — number of valid history entries
  - cand_idx.npy:      int32[N, C] — embedding row indices for candidates
  - cand_len.npy:      int16[N]    — number of valid candidates
  - pos_idx.npy:       int16[N]    — index of clicked candidate within impression
  Save to data/processed/large/mind/train_features/

Phase B (~20 min/epoch): Load embedding matrix onto GPU (or CPU),
  stream batches of pre-extracted indices, compute attention,
  compute softmax loss, backprop.

Usage:
    .venv/bin/python -m scripts.train_attention_fast
    .venv/bin/python -m scripts.train_attention_fast --skip-extract  # if arrays exist
    .venv/bin/python -m scripts.train_attention_fast --epochs 5 --batch 256
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
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_EMBED_DIR = _PROJECT_ROOT / "data" / "processed" / "embeddings"
_FEAT_DIR = _PROJECT_ROOT / "data" / "processed" / "large" / "mind" / "train_features"
_MODELS_DIR = _PROJECT_ROOT / "models"

HISTORY_CAP = 50
MAX_CANDS = 60  # 99th percentile candidates per MIND impression


class AdditiveAttentionUserEncoder(nn.Module):
    def __init__(self, embed_dim: int = 768, attn_dim: int = 200) -> None:
        super().__init__()
        self.W = nn.Linear(embed_dim, attn_dim, bias=True)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, history_embs: torch.Tensor, history_mask: torch.Tensor) -> torch.Tensor:
        """history_embs: [B, H, D], history_mask: [B, H] bool True=valid → [B, D]"""
        attn = torch.tanh(self.W(history_embs))          # [B, H, A]
        scores = self.v(attn).squeeze(-1)                 # [B, H]
        scores = scores.masked_fill(~history_mask, -1e9)
        weights = F.softmax(scores, dim=-1)               # [B, H]
        user_vec = (history_embs * weights.unsqueeze(-1)).sum(dim=1)  # [B, D]
        return F.normalize(user_vec, dim=-1)


def _parse_history(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parsed = json.loads(value) if value else []
    else:
        parsed = list(value)
    return [str(e["article_id"]) if isinstance(e, dict) else str(e) for e in parsed]


# ──────────────────────────────────────────────────────────────────────────────
# Phase A: Feature extraction
# ──────────────────────────────────────────────────────────────────────────────

def extract_train_features(id_to_row: dict[str, int]) -> None:
    _FEAT_DIR.mkdir(parents=True, exist_ok=True)
    behaviors_path = _PROJECT_ROOT / "data" / "interim" / "large" / "mind" / "behaviors.parquet"

    t0 = time.time()
    logger.info("Extracting training features → %s", _FEAT_DIR)

    hist_idx_list: list[np.ndarray] = []   # [H] int32
    hist_len_list: list[int] = []
    cand_idx_list: list[np.ndarray] = []   # [C] int32
    cand_len_list: list[int] = []
    pos_idx_list: list[int] = []

    parquet = pq.ParquetFile(behaviors_path)
    columns = ["clicked_history", "candidates", "labels", "split"]
    n_train = 0
    n_skip = 0

    for batch in parquet.iter_batches(batch_size=50_000, columns=columns):
        data = batch.to_pydict()
        for i in range(batch.num_rows):
            if data["split"][i] != "train":
                continue

            candidates_raw = data["candidates"][i]
            labels_raw = data["labels"][i]
            candidates = json.loads(candidates_raw) if isinstance(candidates_raw, str) else candidates_raw
            labels = json.loads(labels_raw) if isinstance(labels_raw, str) else labels_raw
            candidates = [str(c) for c in candidates][:MAX_CANDS]
            labels = [int(l) for l in labels][:MAX_CANDS]

            pos_list = [j for j, l in enumerate(labels) if l == 1]
            if not pos_list:
                n_skip += 1
                continue
            pos_idx = pos_list[0]

            history_ids = _parse_history(data["clicked_history"][i])[-HISTORY_CAP:]
            hist_rows = np.array(
                [id_to_row[aid] for aid in history_ids if aid in id_to_row],
                dtype=np.int32,
            )
            H = len(hist_rows)
            hist_buf = np.full(HISTORY_CAP, -1, dtype=np.int32)
            if H > 0:
                hist_buf[:H] = hist_rows

            cand_rows = np.array(
                [id_to_row.get(c, -1) for c in candidates],
                dtype=np.int32,
            )
            C = len(cand_rows)
            cand_buf = np.full(MAX_CANDS, -1, dtype=np.int32)
            cand_buf[:C] = cand_rows

            hist_idx_list.append(hist_buf)
            hist_len_list.append(min(H, HISTORY_CAP))
            cand_idx_list.append(cand_buf)
            cand_len_list.append(C)
            pos_idx_list.append(pos_idx)
            n_train += 1

        if n_train % 100_000 < 50_000 and n_train > 0:
            logger.info("  Extracted %d train impressions (%.1f min)", n_train, (time.time() - t0) / 60)

    hist_idx = np.stack(hist_idx_list)       # [N, H]
    cand_idx = np.stack(cand_idx_list)       # [N, C]
    hist_len = np.array(hist_len_list, dtype=np.int16)
    cand_len = np.array(cand_len_list, dtype=np.int16)
    pos_idx = np.array(pos_idx_list, dtype=np.int16)

    np.save(_FEAT_DIR / "hist_idx.npy", hist_idx)
    np.save(_FEAT_DIR / "cand_idx.npy", cand_idx)
    np.save(_FEAT_DIR / "hist_len.npy", hist_len)
    np.save(_FEAT_DIR / "cand_len.npy", cand_len)
    np.save(_FEAT_DIR / "pos_idx.npy", pos_idx)

    elapsed = time.time() - t0
    logger.info("Extracted %d impressions (skipped %d) in %.1f min", n_train, n_skip, elapsed / 60)
    logger.info("Sizes: hist_idx=%s, cand_idx=%s", hist_idx.shape, cand_idx.shape)


# ──────────────────────────────────────────────────────────────────────────────
# Phase B: Training
# ──────────────────────────────────────────────────────────────────────────────

def train(args) -> None:
    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("Device: %s", device)

    # Load embeddings as torch tensor — keep on CPU, index on GPU
    t0 = time.time()
    embeddings_np = np.load(_EMBED_DIR / "mind_mpnet_large.npy")
    # Float16 saves memory, still accurate enough for dot products
    embeddings = torch.from_numpy(embeddings_np.astype(np.float32))
    D = embeddings.shape[1]
    logger.info("Embeddings: %s (%.1fs)", embeddings.shape, time.time() - t0)

    id_to_row: dict[str, int] = json.loads(
        (_EMBED_DIR / "mind_mpnet_large_ids.json").read_text()
    )

    # Feature extraction
    if not args.skip_extract and not (_FEAT_DIR / "hist_idx.npy").exists():
        logger.info("Extracting features …")
        extract_train_features(id_to_row)
    else:
        logger.info("Loading pre-extracted features from %s", _FEAT_DIR)

    hist_idx = np.load(_FEAT_DIR / "hist_idx.npy")   # [N, H]
    cand_idx = np.load(_FEAT_DIR / "cand_idx.npy")   # [N, C]
    hist_len = np.load(_FEAT_DIR / "hist_len.npy")   # [N]
    cand_len = np.load(_FEAT_DIR / "cand_len.npy")   # [N]
    pos_idx  = np.load(_FEAT_DIR / "pos_idx.npy")    # [N]
    N = len(pos_idx)
    logger.info("Training data: %d impressions", N)

    model = AdditiveAttentionUserEncoder(embed_dim=D, attn_dim=args.attn_dim).to(device)
    logger.info("Model: %d params", sum(p.numel() for p in model.parameters()))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.1)

    # Load or init best checkpoint
    save_path = _MODELS_DIR / "attention_user_encoder.pt"
    best_auc = 0.0
    if save_path.exists():
        ckpt = torch.load(save_path, map_location="cpu")
        best_auc = ckpt.get("val_auc", 0.0)
        logger.info("Existing best val AUC: %.4f", best_auc)

    indices = np.arange(N, dtype=np.int32)

    for epoch in range(1, args.epochs + 1):
        model.train()
        np.random.shuffle(indices)
        epoch_loss = 0.0
        n_batches = 0
        t_epoch = time.time()

        for batch_start in range(0, N, args.batch):
            bidx = indices[batch_start: batch_start + args.batch]
            B = len(bidx)

            # ── Gather history embeddings ──
            h_idx = hist_idx[bidx]           # [B, H], int32, -1=pad
            h_len = hist_len[bidx]           # [B]
            h_mask = (h_idx >= 0)            # [B, H] bool
            # Clamp -1 → 0 for safe embedding lookup, then zero out via mask
            h_idx_safe = np.where(h_mask, h_idx, 0)
            h_embs = embeddings[h_idx_safe]  # [B, H, D] — CPU
            h_embs_t = h_embs.to(device)
            h_mask_t = torch.from_numpy(h_mask).to(device)

            # ── Gather candidate embeddings ──
            c_idx = cand_idx[bidx]           # [B, C], int32, -1=pad
            c_len = cand_len[bidx]           # [B]
            c_mask = (c_idx >= 0)            # [B, C] bool
            c_idx_safe = np.where(c_mask, c_idx, 0)
            c_embs = embeddings[c_idx_safe]  # [B, C, D] — CPU
            c_embs_t = c_embs.to(device)

            p_idx_t = torch.from_numpy(pos_idx[bidx].astype(np.int64)).to(device)

            # ── Forward ──
            user_vecs = model(h_embs_t, h_mask_t)      # [B, D]
            scores = torch.bmm(c_embs_t, user_vecs.unsqueeze(-1)).squeeze(-1)  # [B, C]

            # Mask padding candidates
            c_mask_t = torch.from_numpy(c_mask).to(device)
            scores = scores.masked_fill(~c_mask_t, -1e9)

            loss = F.cross_entropy(scores, p_idx_t)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

            if n_batches % 1000 == 0:
                elapsed = time.time() - t_epoch
                eta = elapsed / n_batches * (math.ceil(N / args.batch) - n_batches)
                logger.info("  Epoch %d | %d/%d batches | loss=%.4f | %.1f min elapsed, %.1f min ETA",
                            epoch, n_batches, math.ceil(N / args.batch),
                            epoch_loss / n_batches, elapsed / 60, eta / 60)

        scheduler.step()

        # ── Quick validation (20K impressions) ──
        val_auc = _eval_val(model, embeddings, id_to_row, device)
        avg_loss = epoch_loss / max(n_batches, 1)
        elapsed = time.time() - t_epoch
        logger.info("Epoch %d | loss=%.4f | val_AUC(20K)=%.4f | %.1f min",
                    epoch, avg_loss, val_auc, elapsed / 60)

        _MODELS_DIR.mkdir(parents=True, exist_ok=True)
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
        else:
            logger.info("  (No improvement; best=%.4f)", best_auc)

    logger.info("Done. Best val AUC=%.4f", best_auc)


def _eval_val(
    model: AdditiveAttentionUserEncoder,
    embeddings: torch.Tensor,
    id_to_row: dict[str, int],
    device: torch.device,
    max_impressions: int = 20000,
) -> float:
    model.eval()
    D = embeddings.shape[1]
    behaviors_path = _PROJECT_ROOT / "data" / "interim" / "large" / "mind" / "behaviors.parquet"
    parquet = pq.ParquetFile(behaviors_path)
    columns = ["clicked_history", "candidates", "labels", "split"]

    auc_sum = 0.0
    n = 0

    with torch.no_grad():
        for batch in parquet.iter_batches(batch_size=2000, columns=columns):
            data = batch.to_pydict()
            for i in range(batch.num_rows):
                if n >= max_impressions:
                    break
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
                    n += 1
                    continue

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
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--attn-dim", type=int, default=200)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-extract", action="store_true",
                        help="Skip feature extraction (use pre-extracted arrays)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")

    train(args)


if __name__ == "__main__":
    main()
