"""Load/cache article embeddings for small experiments and large submission catalogs."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_EMBED_DIR = _PROJECT_ROOT / "data" / "processed" / "embeddings"


def _catalog_tag(article_ids: list[str] | None, explicit: str | None) -> str:
    if explicit:
        return explicit
    if not article_ids:
        return "all"
    digest = hashlib.sha1("\n".join(article_ids).encode()).hexdigest()[:10]
    return f"cat_{digest}"


def _cache_path(dataset: str, model: str, tag: str, suffix: str) -> Path:
    return _EMBED_DIR / f"{dataset}_{model}_{tag}{suffix}"


def _l2_normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vecs / norms


def _find_ebnerd_embedding_parquet(model_name: str) -> Path | None:
    """Find a provided EB-NeRD embedding parquet file.

    Handles the official artifact naming used by the assignment, including
    the BERT file whose filename does not match the logical model directory.
    """
    raw_dir = _PROJECT_ROOT / "data" / "raw" / "ebnerd"

    candidates = [
        # Conventional layouts.
        raw_dir / model_name / f"{model_name}.parquet",
        raw_dir / f"{model_name}.parquet",
        raw_dir / model_name / "embeddings.parquet",
        raw_dir / model_name / "document_vector.parquet",

        # Actual assignment artifact names.
        raw_dir
        / "google_bert_base_multilingual_cased"
        / "google_bert_base_multilingual_cased"
        / "bert_base_multilingual_cased.parquet",

        raw_dir
        / "Ekstra_Bladet_word2vec"
        / "Ekstra_Bladet_word2vec"
        / "document_vector.parquet",
    ]

    for path in candidates:
        if path.exists():
            return path

    # Generic recursive fallback.
    model_dir = raw_dir / model_name
    if model_dir.exists():
        matches = sorted(model_dir.rglob("*.parquet"))
        if matches:
            return matches[0]

    # Handle model aliases whose directory/file names differ.
    if model_name == "google_bert_base_multilingual_cased":
        matches = sorted(
            raw_dir.rglob("bert_base_multilingual_cased.parquet")
        )
        if matches:
            return matches[0]

    if model_name == "Ekstra_Bladet_word2vec":
        matches = sorted(
            raw_dir.rglob("document_vector.parquet")
        )
        if matches:
            return matches[0]

    return None

def _ordered_mapping(article_ids: list[str]) -> dict[str, int]:
    return {aid: i for i, aid in enumerate(article_ids)}


def load_ebnerd_embeddings(
    model_name: str,
    article_ids: list[str] | None = None,
    cache_tag: str | None = None,
) -> tuple[np.ndarray, dict[str, int], float]:
    """Load EB-NeRD provided embeddings, optionally restricted to a catalog."""
    tag = _catalog_tag(article_ids, cache_tag)
    cache_npy = _cache_path("ebnerd", model_name, tag, ".npy")
    cache_ids = _cache_path("ebnerd", model_name, tag, "_ids.json")
    target_ids = set(article_ids) if article_ids is not None else None

    if cache_npy.exists() and cache_ids.exists():
        embeddings = np.load(cache_npy)
        id_to_row = json.loads(cache_ids.read_text())
        coverage = len(id_to_row) / len(article_ids) if article_ids else 1.0
        return embeddings, id_to_row, coverage

    parquet_path = _find_ebnerd_embedding_parquet(model_name)
    if parquet_path is None:
        raise FileNotFoundError(f"EB-NeRD embedding parquet not found for {model_name}")

    logger.info("Loading EB-NeRD %s embeddings from %s", model_name, parquet_path)
    emb_df = pl.read_parquet(parquet_path)
    id_col = "article_id"
    if id_col not in emb_df.columns:
        raise ValueError(f"Embedding file has no {id_col} column: {emb_df.columns}")
    emb_df = emb_df.with_columns(pl.col(id_col).cast(pl.Utf8))
    if target_ids is not None:
        emb_df = emb_df.filter(pl.col(id_col).is_in(list(target_ids)))

    emb_cols = [c for c in emb_df.columns if c != id_col]
    if len(emb_cols) == 1 and str(emb_df[emb_cols[0]].dtype).lower().startswith("list"):
        embeddings = np.asarray(emb_df[emb_cols[0]].to_list(), dtype=np.float32)
    else:
        embeddings = emb_df.select(emb_cols).to_numpy().astype(np.float32)
    article_ids_out = emb_df[id_col].to_list()
    embeddings = _l2_normalize(embeddings)
    id_to_row = _ordered_mapping(article_ids_out)
    coverage = len(id_to_row) / len(article_ids) if article_ids else 1.0

    _EMBED_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache_npy, embeddings)
    cache_ids.write_text(json.dumps(id_to_row))
    logger.info(
        "Cached %d %d-D EB-NeRD embeddings at %s (coverage %.2f%%)",
        len(article_ids_out), embeddings.shape[1], cache_npy, coverage * 100,
    )
    return embeddings, id_to_row, coverage


def compute_mind_embeddings(
    article_ids: list[str],
    texts: list[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 256,
    device: str | None = None,
    cache_tag: str | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    """Compute or load MIND sentence-transformer embeddings."""
    tag = _catalog_tag(article_ids, cache_tag)
    cache_npy = _cache_path("mind", "minilm", tag, ".npy")
    cache_ids = _cache_path("mind", "minilm", tag, "_ids.json")
    if cache_npy.exists() and cache_ids.exists():
        return np.load(cache_npy), json.loads(cache_ids.read_text())

    from sentence_transformers import SentenceTransformer

    logger.info("Computing MIND embeddings for %d articles", len(article_ids))
    model = SentenceTransformer(model_name, device=device)
    clean_texts = [text or "" for text in texts]
    embeddings = model.encode(
        clean_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    id_to_row = _ordered_mapping(article_ids)
    _EMBED_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache_npy, embeddings)
    cache_ids.write_text(json.dumps(id_to_row))
    return embeddings, id_to_row


def load_embeddings(
    dataset: str,
    model: str = "default",
    *,
    article_ids: list[str] | None = None,
    article_texts: list[str] | None = None,
    cache_tag: str | None = None,
    batch_size: int = 256,
    device: str | None = None,
    scale: str = "small",
) -> tuple[np.ndarray, dict[str, int], float | None]:
    """Unified loader.

    For MIND, article IDs + texts are required when a large submission catalog is
    requested. For EB-NeRD the provided embedding parquet is used directly.
    """
    if dataset == "ebnerd":
        model_map = {
            "default": "Ekstra_Bladet_word2vec",
            "w2v": "Ekstra_Bladet_word2vec",
            "bert": "google_bert_base_multilingual_cased",
        }
        model_name = model_map.get(model, model)
        if article_ids is None and cache_tag is None:
            # Preserve development behavior when called by the existing runner.
            from src.feature_store.article_store import ArticleFeatureStore
            article_ids = ArticleFeatureStore(dataset, scale=scale).get_articles_for_bm25()["article_id"].to_list()
            cache_tag = "demo"
        return load_ebnerd_embeddings(model_name, article_ids, cache_tag)

    if dataset == "mind":
        if article_ids is None or article_texts is None:
            from src.feature_store.article_store import ArticleFeatureStore
            articles = ArticleFeatureStore(dataset, scale=scale).get_articles_for_bm25()
            article_ids = articles["article_id"].to_list()
            article_texts = articles["cleaned_text"].to_list()
        embeddings, id_to_row = compute_mind_embeddings(
            article_ids, article_texts, batch_size=batch_size, device=device, cache_tag=cache_tag
        )
        return embeddings, id_to_row, None

    raise ValueError(f"Unknown dataset: {dataset}")
