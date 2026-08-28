"""Compute MIND article embeddings with a stronger model (all-mpnet-base-v2).

Caches the result alongside the existing MiniLM embeddings. Does NOT
overwrite any existing cached files.

Usage:
    .venv/bin/python -m scripts.compute_mpnet_embeddings
    .venv/bin/python -m scripts.compute_mpnet_embeddings --device cuda
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from pathlib import Path

import numpy as np
import polars as pl

from src.submission.make_submission import _find_mind_catalog

logger = logging.getLogger(__name__)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_EMBED_DIR = _PROJECT_ROOT / "data" / "processed" / "embeddings"

MODEL_CONFIGS = {
    "mpnet": {
        "hf_name": "sentence-transformers/all-mpnet-base-v2",
        "dim": 768,
        "cache_prefix": "mind_mpnet",
    },
    "minilm12": {
        "hf_name": "sentence-transformers/all-MiniLM-L12-v2",
        "dim": 384,
        "cache_prefix": "mind_minilm12",
    },
    "bge": {
        "hf_name": "BAAI/bge-base-en-v1.5",
        "dim": 768,
        "cache_prefix": "mind_bge",
    },
}


def compute_embeddings(
    article_ids: list[str],
    texts: list[str],
    model_name: str,
    cache_prefix: str,
    cache_tag: str = "large",
    batch_size: int = 64,
    device: str | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    """Compute and cache sentence-transformer embeddings."""
    _EMBED_DIR.mkdir(parents=True, exist_ok=True)
    cache_npy = _EMBED_DIR / f"{cache_prefix}_{cache_tag}.npy"
    cache_ids = _EMBED_DIR / f"{cache_prefix}_{cache_tag}_ids.json"

    if cache_npy.exists() and cache_ids.exists():
        logger.info("Loading cached embeddings from %s", cache_npy)
        embeddings = np.load(cache_npy)
        id_to_row = json.loads(cache_ids.read_text())
        logger.info("Loaded %d embeddings (%d-D)", len(id_to_row), embeddings.shape[1])
        return embeddings, id_to_row

    from sentence_transformers import SentenceTransformer

    logger.info("Computing embeddings for %d articles using %s", len(article_ids), model_name)
    model = SentenceTransformer(model_name, device=device)

    clean_texts = [text or "" for text in texts]
    
    t0 = time.time()
    embeddings = model.encode(
        clean_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    elapsed = time.time() - t0

    id_to_row = {aid: i for i, aid in enumerate(article_ids)}

    np.save(cache_npy, embeddings)
    cache_ids.write_text(json.dumps(id_to_row))

    logger.info("Computed %d %d-D embeddings in %.1fs (%.1f articles/s)",
                len(article_ids), embeddings.shape[1], elapsed,
                len(article_ids) / elapsed)
    logger.info("Cached at %s (%.1f MB)", cache_npy,
                cache_npy.stat().st_size / 1024**2)

    return embeddings, id_to_row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_CONFIGS.keys()),
                        default="mpnet", help="Model to compute embeddings for")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for encoding (reduce for low VRAM)")
    parser.add_argument("--device", default=None,
                        help="Device: 'cuda', 'cpu', or None for auto")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")

    cfg = MODEL_CONFIGS[args.model]
    
    # Load MIND article catalog (same as submission pipeline)
    article_ids, texts = _find_mind_catalog(_PROJECT_ROOT / "data" / "raw" / "mind")
    logger.info("MIND catalog: %d articles", len(article_ids))

    embeddings, id_to_row = compute_embeddings(
        article_ids, texts,
        model_name=cfg["hf_name"],
        cache_prefix=cfg["cache_prefix"],
        batch_size=args.batch_size,
        device=args.device,
    )

    print(f"\n✓ {args.model}: {embeddings.shape[0]} articles × {embeddings.shape[1]} dims")
    print(f"  Cache: {_EMBED_DIR / cfg['cache_prefix']}_large.npy")


if __name__ == "__main__":
    main()
