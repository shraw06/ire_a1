"""Generate MIND test submission using candidate-conditioned (query-key) attention.

For each impression, each candidate is scored using a candidate-specific
user vector computed by soft-weighting the user's history:

    α_i(c) = softmax( (h_i · c) / T )    ← how relevant is history item i to candidate c?
    u(c)   = Σ α_i(c) · h_i              ← candidate-conditioned user vector
    score  = u(c) · c                     ← candidate-specific ranking score

This is equivalent to the NRMS read-attention mechanism with zero trainable
parameters — it uses frozen mpnet embeddings only.

Key advantage over trained additive attention:
  - No training → no temporal overfitting
  - Inductive bias is directly grounded in embedding geometry
  - Validated val AUC=0.8248 (cap=100, T=0.20) vs 0.8205 for mean-pool

Tuning result: best T=0.20, history_cap=100 (from tune_query_key_attention.py)
Full table: results/large/qk_attention_tuning.csv

Usage:
    .venv/bin/python -m scripts.generate_qk_submission
    .venv/bin/python -m scripts.generate_qk_submission --temperature 0.20 --history-cap 100
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from src.parsing.submission_readers import find_mind_test_behaviors, iter_mind_test
from src.retrieval.ann import ArticleIndex
from src.submission.package_submission import package_prediction
from src.submission.writers import write_ranked_impression

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


def _qk_score_impression(
    history_embs: np.ndarray,   # [H, D], already looked up and stacked
    cand_embs: np.ndarray,      # [C, D], already looked up
    temperature: float,
) -> np.ndarray:
    """Candidate-conditioned attention scores. Returns [C] scores."""
    if len(history_embs) == 0:
        return (cand_embs @ cand_embs.mean(axis=0)).flatten()

    # sim[c, h] = cand_emb[c] · hist_emb[h]
    sim = cand_embs @ history_embs.T           # [C, H]
    weights = sim / temperature                # [C, H]
    weights -= weights.max(axis=1, keepdims=True)   # stable softmax
    weights = np.exp(weights)
    weights /= weights.sum(axis=1, keepdims=True)   # [C, H]

    # Candidate-conditioned user vectors: u(c) = Σ α_i(c) h_i
    user_vecs = weights @ history_embs         # [C, D]
    user_norms = np.linalg.norm(user_vecs, axis=1, keepdims=True).clip(min=1e-8)
    user_vecs /= user_norms

    return (user_vecs * cand_embs).sum(axis=1)  # [C]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", type=float, default=0.20)
    parser.add_argument("--history-cap", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=100_000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")

    t0 = time.time()

    # Load embeddings
    embeddings = np.load(_EMBED_DIR / "mind_mpnet_large.npy")
    id_to_row: dict[str, int] = json.loads(
        (_EMBED_DIR / "mind_mpnet_large_ids.json").read_text()
    )
    logger.info("Loaded embeddings: %s", embeddings.shape)

    # Output paths
    tag = f"qk_T{args.temperature:.2f}_cap{args.history_cap}"
    output_dir = _PROJECT_ROOT / "submissions" / f"mind_{tag}"
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "prediction.txt"
    zip_path = output_dir / f"mind_{tag}_submission.zip"

    test_path = find_mind_test_behaviors(_PROJECT_ROOT / "data" / "raw" / "mind")
    logger.info("Streaming test from: %s", test_path)

    row_count = 0
    with prediction_path.open("w", encoding="utf-8") as handle:
        for batch in iter_mind_test(test_path, batch_size=args.batch_size):
            for item in batch:
                history_ids = [str(e["article_id"]) for e in item.history][-args.history_cap:]
                candidates = item.candidates

                # Look up history embeddings
                hist_rows = [id_to_row[aid] for aid in history_ids if aid in id_to_row]
                if hist_rows:
                    hist_embs = embeddings[hist_rows]   # [H, D]
                else:
                    hist_embs = np.zeros((0, embeddings.shape[1]), dtype=np.float32)

                # Look up candidate embeddings
                cand_ids = [str(c) for c in candidates]
                cand_rows = [id_to_row.get(c) for c in cand_ids]
                valid_idx = [j for j, r in enumerate(cand_rows) if r is not None]
                valid_embs = embeddings[[cand_rows[j] for j in valid_idx]]   # [C_valid, D]

                if len(valid_idx) == 0 or len(hist_embs) == 0:
                    # Fallback: identity ranking
                    ordered = candidates
                else:
                    qk_scores_valid = _qk_score_impression(hist_embs, valid_embs, args.temperature)

                    # Build per-candidate score lookup by original index
                    full_scores = np.zeros(len(candidates), dtype=np.float32)
                    for j, s in zip(valid_idx, qk_scores_valid):
                        full_scores[j] = float(s)
                    order = np.argsort(-full_scores)
                    ordered = [candidates[j] for j in order]

                write_ranked_impression(handle, item.impression_id, candidates, ordered)

            row_count += len(batch)
            if row_count % 500_000 < len(batch):
                logger.info("Generated %d predictions", row_count)

    package_prediction(prediction_path, zip_path)
    elapsed = time.time() - t0

    print(f"\nMIND QK-Attention (T={args.temperature}, cap={args.history_cap}): "
          f"{row_count:,} rows, {elapsed/60:.1f} min")
    print(f"  prediction : {prediction_path}")
    print(f"  submission : {zip_path}")


if __name__ == "__main__":
    main()
