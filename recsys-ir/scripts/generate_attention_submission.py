"""Generate MIND submission using trained attention user encoder.

Loads the best saved model from models/attention_user_encoder.pt and uses
it for test-set inference. At inference time, the user representation is:
    u = Σ α_i h_i   where α_i = softmax(v^T tanh(W h_i))
instead of the uniform mean used in all previous submissions.

Usage:
    .venv/bin/python -m scripts.generate_attention_submission
    .venv/bin/python -m scripts.generate_attention_submission --history-cap 50
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from scripts.train_attention_ranker import AdditiveAttentionUserEncoder
from src.parsing.submission_readers import find_mind_test_behaviors, iter_mind_test
from src.retrieval.ann import ArticleIndex
from src.submission.package_submission import package_prediction
from src.submission.writers import write_ranked_impression

logger = logging.getLogger(__name__)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_EMBED_DIR = _PROJECT_ROOT / "data" / "processed" / "embeddings"
_MODELS_DIR = _PROJECT_ROOT / "models"


def _load_attention_model(device: torch.device) -> tuple[AdditiveAttentionUserEncoder, dict]:
    checkpoint = torch.load(_MODELS_DIR / "attention_user_encoder.pt",
                            map_location=device)
    model = AdditiveAttentionUserEncoder(
        embed_dim=checkpoint["embed_dim"],
        attn_dim=checkpoint["attn_dim"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    logger.info("Loaded attention model (epoch=%d, val_AUC=%.4f)",
                checkpoint["epoch"], checkpoint["val_auc"])
    return model, checkpoint


@torch.no_grad()
def _compute_user_vec_attention(
    history_ids: list[str],
    embeddings: np.ndarray,
    id_to_row: dict[str, int],
    model: AdditiveAttentionUserEncoder,
    history_cap: int,
    device: torch.device,
    D: int,
) -> np.ndarray:
    """Compute attention user vector for a single impression."""
    ids = history_ids[-history_cap:]
    hist_rows = [id_to_row[aid] for aid in ids if aid in id_to_row]
    H = min(len(hist_rows), history_cap)

    hist_emb = np.zeros((1, history_cap, D), dtype=np.float32)
    hist_mask = np.zeros((1, history_cap), dtype=bool)
    if H > 0:
        hist_emb[0, :H] = embeddings[hist_rows[:H]]
        hist_mask[0, :H] = True

    ht = torch.tensor(hist_emb, dtype=torch.float32, device=device)
    hm = torch.tensor(hist_mask, dtype=torch.bool, device=device)
    user_vec = model(ht, hm)[0].cpu().numpy()
    return user_vec


def _process_batch(
    batch,
    embeddings: np.ndarray,
    id_to_row: dict[str, int],
    model: AdditiveAttentionUserEncoder,
    index: ArticleIndex,
    history_cap: int,
    device: torch.device,
    D: int,
    handle,
) -> None:
    for item in batch:
        history_ids = [str(e["article_id"]) for e in item.history]
        user_vec = _compute_user_vec_attention(
            history_ids, embeddings, id_to_row, model,
            history_cap, device, D,
        )
        results = index.search_restricted(user_vec, item.candidates, k=len(item.candidates))
        ordered = [aid for aid, _ in results]
        write_ranked_impression(handle, item.impression_id, item.candidates, ordered)


def main():
    parser = argparse.ArgumentParser(
        description="Generate MIND submission using trained attention user encoder"
    )
    parser.add_argument("--history-cap", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    started = time.time()

    # Load model
    model, ckpt = _load_attention_model(device)
    val_auc = ckpt["val_auc"]
    epoch = ckpt["epoch"]

    # Load embeddings & index
    embeddings = np.load(_EMBED_DIR / "mind_mpnet_large.npy")
    id_to_row: dict[str, int] = json.loads(
        (_EMBED_DIR / "mind_mpnet_large_ids.json").read_text()
    )
    D = embeddings.shape[1]
    article_ids_ordered = [""] * len(id_to_row)
    for aid, idx in id_to_row.items():
        article_ids_ordered[int(idx)] = aid
    index = ArticleIndex(embeddings, article_ids_ordered, build_full_index=False)
    logger.info("Loaded index: %s", index)

    # Output paths
    tag = f"attn_ep{epoch}_vauc{val_auc:.4f}_cap{args.history_cap}"
    output_dir = _PROJECT_ROOT / "submissions" / f"mind_attention_{tag}"
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "prediction.txt"
    zip_path = output_dir / f"mind_attention_{tag}_submission.zip"

    # Stream test
    test_path = find_mind_test_behaviors(_PROJECT_ROOT / "data" / "raw" / "mind")
    batches = iter_mind_test(test_path, batch_size=args.batch_size)

    row_count = 0
    with prediction_path.open("w", encoding="utf-8") as handle:
        for batch in batches:
            _process_batch(
                batch, embeddings, id_to_row,
                model, index, args.history_cap, device, D, handle,
            )
            row_count += len(batch)
            if row_count % max(args.batch_size * 5, 200_000) < len(batch):
                logger.info("Generated %d predictions", row_count)

    package_prediction(prediction_path, zip_path)
    elapsed = time.time() - started

    print(f"\nMIND ATTENTION (epoch={epoch}, val_AUC={val_auc:.4f}, cap={args.history_cap}): "
          f"{row_count:,} rows, {elapsed/60:.1f} min")
    print(f"  prediction: {prediction_path}")
    print(f"  submission: {zip_path}")


if __name__ == "__main__":
    main()
