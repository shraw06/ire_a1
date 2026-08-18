"""Load pretrained article embeddings (EB-NeRD) or compute BERT embeddings (MIND).

This module handles two distinct embedding sources:

EB-NeRD:
    Uses the DATASET-PROVIDED embeddings (Word2Vec and multilingual BERT)
    downloaded from the EB-NeRD artifact repository.  These were trained by
    the dataset authors over their own text fields — not something we control.
    Using them directly avoids burning free-tier GPU budget and keeps results
    comparable to the benchmark repo.

MIND:
    No embeddings are provided, so we compute them here via
    ``sentence-transformers/all-MiniLM-L6-v2`` — a small, CPU-friendly,
    English-tuned model with strong semantic-similarity quality.  A different
    model family per dataset is fine because the comparison axis is
    lexical-vs-semantic WITHIN each dataset, not a claim that the two
    datasets' embedding spaces are directly comparable.

All embeddings are L2-normalized before caching (so dot-product == cosine similarity).
Cached to ``data/processed/embeddings/{dataset}_{model}.npy`` plus an
``{dataset}_{model}_ids.json`` (article_id → row index mapping).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ── Cache paths ────────────────────────────────────────────────────

_EMBED_DIR = _PROJECT_ROOT / "data" / "processed" / "embeddings"


def _cache_path(dataset: str, model: str, suffix: str) -> Path:
    """Return the cache file path for a given dataset/model/suffix."""
    return _EMBED_DIR / f"{dataset}_{model}{suffix}"


def _l2_normalize(vecs: np.ndarray) -> np.ndarray:
    """L2-normalize rows of a 2-D array (in-place safe)."""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    # Avoid division by zero for zero vectors
    norms = np.where(norms == 0, 1.0, norms)
    return vecs / norms


# ── EB-NeRD: Load provided embeddings ─────────────────────────────

def _find_ebnerd_embedding_parquet(model_name: str) -> Path | None:
    """Locate the unzipped EB-NeRD embedding parquet file.

    After unzipping the EB-NeRD artifact zips, the files are typically at:
      data/raw/ebnerd/Ekstra_Bladet_word2vec/Ekstra_Bladet_word2vec.parquet
      data/raw/ebnerd/google_bert_base_multilingual_cased/google_bert_base_multilingual_cased.parquet

    Also handles the case where they are directly in data/raw/ebnerd/.
    """
    raw_dir = _PROJECT_ROOT / "data" / "raw" / "ebnerd"
    # Common patterns after unzipping
    candidates = [
        raw_dir / model_name / f"{model_name}.parquet",
        raw_dir / f"{model_name}.parquet",
        raw_dir / model_name / "embeddings.parquet",
        raw_dir / model_name / "document_vector.parquet",
    ]
    for p in candidates:
        if p.exists():
            return p

    # Fallback: search recursively for any parquet in the model dir
    model_dir = raw_dir / model_name
    if model_dir.exists():
        parquets = list(model_dir.glob("*.parquet"))
        if parquets:
            return parquets[0]

    return None


def load_ebnerd_embeddings(
    model_name: str,
    demo_article_ids: list[str],
) -> tuple[np.ndarray, dict[str, int], float]:
    """Load EB-NeRD provided embeddings and join to demo bundle articles.

    Parameters
    ----------
    model_name : str
        ``"Ekstra_Bladet_word2vec"`` or ``"google_bert_base_multilingual_cased"``.
    demo_article_ids : list[str]
        Article IDs from the demo bundle (string format).

    Returns
    -------
    embeddings : np.ndarray
        L2-normalized embeddings, shape ``(n_matched, dim)``.
    id_to_row : dict[str, int]
        Mapping ``article_id → row index`` in the embeddings array.
    coverage : float
        Fraction of demo_article_ids that had a matching embedding.
    """
    import polars as pl

    cache_npy = _cache_path("ebnerd", model_name, ".npy")
    cache_ids = _cache_path("ebnerd", model_name, "_ids.json")

    if cache_npy.exists() and cache_ids.exists():
        logger.info("Loading cached EB-NeRD %s embeddings", model_name)
        embeddings = np.load(cache_npy)
        with open(cache_ids) as f:
            id_to_row = json.load(f)
        coverage = len(id_to_row) / len(demo_article_ids) if demo_article_ids else 0.0
        logger.info(
            "  Loaded %d embeddings (dim=%d), coverage=%.2f%%",
            len(id_to_row), embeddings.shape[1], coverage * 100,
        )
        # Validation: all rows have the same dimensionality
        assert embeddings.ndim == 2, f"Expected 2-D array, got {embeddings.ndim}-D"
        assert len(set(embeddings.shape[1:])) == 1, "Inconsistent embedding dimensionality"
        return embeddings, id_to_row, coverage

    # Find the parquet file
    pq_path = _find_ebnerd_embedding_parquet(model_name)
    if pq_path is None:
        raise FileNotFoundError(
            f"EB-NeRD embedding parquet not found for model '{model_name}'. "
            f"Download and unzip the embedding zip first."
        )

    logger.info("Loading EB-NeRD %s embeddings from %s", model_name, pq_path)
    emb_df = pl.read_parquet(pq_path)
    logger.info("  Embedding file: %d rows, columns=%s", len(emb_df), emb_df.columns)

    # Identify the article_id column and embedding column
    # The EB-NeRD embedding files typically have article_id (Int32/Int64) and an embedding column
    id_col = "article_id"
    emb_cols = [c for c in emb_df.columns if c != id_col]

    # Check if it's a single list/array column or many float columns
    if len(emb_cols) == 1:
        emb_col = emb_cols[0]
        dtype = emb_df[emb_col].dtype
        logger.info("  Single embedding column: %s (dtype=%s)", emb_col, dtype)

        # Cast article_id to string for joining
        emb_df = emb_df.with_columns(pl.col(id_col).cast(pl.Utf8).alias(id_col))

        # Filter to demo articles
        demo_set = set(demo_article_ids)
        emb_df = emb_df.filter(pl.col(id_col).is_in(list(demo_set)))
        logger.info("  Matched %d / %d demo articles", len(emb_df), len(demo_set))

        # Extract embeddings as numpy array
        # Handle list column (each cell is a list of floats)
        if str(dtype).startswith("List") or str(dtype).startswith("list"):
            embeddings = np.array(emb_df[emb_col].to_list(), dtype=np.float32)
        else:
            # Single float column — shouldn't happen but handle gracefully
            embeddings = emb_df[emb_col].to_numpy().reshape(-1, 1).astype(np.float32)

        article_ids = emb_df[id_col].to_list()
    else:
        # Multiple float columns = one embedding dimension per column
        logger.info("  Multiple embedding columns: %d dimensions", len(emb_cols))

        # Cast article_id to string
        emb_df = emb_df.with_columns(pl.col(id_col).cast(pl.Utf8).alias(id_col))

        # Filter to demo articles
        demo_set = set(demo_article_ids)
        emb_df = emb_df.filter(pl.col(id_col).is_in(list(demo_set)))
        logger.info("  Matched %d / %d demo articles", len(emb_df), len(demo_set))

        article_ids = emb_df[id_col].to_list()
        embeddings = emb_df.select(emb_cols).to_numpy().astype(np.float32)

    # L2-normalize
    embeddings = _l2_normalize(embeddings)

    # Build id → row mapping
    id_to_row = {aid: i for i, aid in enumerate(article_ids)}

    # Coverage
    coverage = len(id_to_row) / len(demo_article_ids) if demo_article_ids else 0.0
    logger.info(
        "  Final: %d embeddings (dim=%d), coverage=%.2f%%",
        embeddings.shape[0], embeddings.shape[1], coverage * 100,
    )

    # Validate dimensionality consistency
    assert embeddings.ndim == 2, f"Expected 2-D array, got {embeddings.ndim}-D"
    dim = embeddings.shape[1]
    assert all(
        embeddings[i].shape[0] == dim for i in range(min(10, len(embeddings)))
    ), "Inconsistent embedding dimensionality detected"

    # Cache
    _EMBED_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache_npy, embeddings)
    with open(cache_ids, "w") as f:
        json.dump(id_to_row, f)
    logger.info("  Cached to %s", cache_npy)

    return embeddings, id_to_row, coverage


# ── MIND: Compute embeddings via sentence-transformers ─────────────

def compute_mind_embeddings(
    article_ids: list[str],
    texts: list[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 256,
) -> tuple[np.ndarray, dict[str, int]]:
    """Compute embeddings for MIND articles using sentence-transformers.

    Parameters
    ----------
    article_ids : list[str]
        Article IDs corresponding to texts.
    texts : list[str]
        Article texts (title + abstract).
    model_name : str
        HuggingFace model name.
    batch_size : int
        Encoding batch size.

    Returns
    -------
    embeddings : np.ndarray
        L2-normalized embeddings, shape ``(len(article_ids), dim)``.
    id_to_row : dict[str, int]
        Mapping ``article_id → row index``.
    """
    cache_label = "minilm"
    cache_npy = _cache_path("mind", cache_label, ".npy")
    cache_ids = _cache_path("mind", cache_label, "_ids.json")

    if cache_npy.exists() and cache_ids.exists():
        logger.info("Loading cached MIND embeddings")
        embeddings = np.load(cache_npy)
        with open(cache_ids) as f:
            id_to_row = json.load(f)
        logger.info(
            "  Loaded %d embeddings (dim=%d)", len(id_to_row), embeddings.shape[1]
        )
        # Validation
        assert embeddings.ndim == 2
        return embeddings, id_to_row

    logger.info("Computing MIND embeddings with %s (%d articles)", model_name, len(article_ids))

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    # Replace empty/None texts with a placeholder
    clean_texts = [t if t else "" for t in texts]

    # Encode in batches with progress
    logger.info("  Encoding %d articles (batch_size=%d)...", len(clean_texts), batch_size)
    embeddings = model.encode(
        clean_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # L2-normalize built into sentence-transformers
        convert_to_numpy=True,
    )
    embeddings = embeddings.astype(np.float32)

    logger.info("  Computed embeddings: shape=%s", embeddings.shape)

    # Build id → row mapping
    id_to_row = {aid: i for i, aid in enumerate(article_ids)}

    # Validate dimensionality consistency
    assert embeddings.ndim == 2, f"Expected 2-D array, got {embeddings.ndim}-D"
    dim = embeddings.shape[1]
    logger.info("  Embedding dimension: %d", dim)

    # Cache
    _EMBED_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache_npy, embeddings)
    with open(cache_ids, "w") as f:
        json.dump(id_to_row, f)
    logger.info("  Cached to %s", cache_npy)

    return embeddings, id_to_row


# ── Unified loader ─────────────────────────────────────────────────

def load_embeddings(
    dataset: str,
    model: str = "default",
) -> tuple[np.ndarray, dict[str, int], float | None]:
    """Load or compute embeddings for a dataset.

    Parameters
    ----------
    dataset : str
        ``"mind"`` or ``"ebnerd"``.
    model : str
        For EB-NeRD: ``"bert"`` (default) or ``"w2v"``.
        For MIND: ``"minilm"`` (only option, used as default).

    Returns
    -------
    embeddings : np.ndarray
        L2-normalized embeddings.
    id_to_row : dict[str, int]
        article_id → row index.
    coverage : float | None
        Fraction of articles covered (only for EB-NeRD; None for MIND).
    """
    import polars as pl
    from src.feature_store.article_store import ArticleFeatureStore

    if dataset == "ebnerd":
        model_map = {
            "default": "google_bert_base_multilingual_cased",
            "bert": "google_bert_base_multilingual_cased",
            "w2v": "Ekstra_Bladet_word2vec",
        }
        model_name = model_map.get(model, model)

        # Get demo bundle article IDs
        store = ArticleFeatureStore(dataset)
        all_articles = store.get_articles_for_bm25()
        demo_ids = all_articles["article_id"].to_list()

        embeddings, id_to_row, coverage = load_ebnerd_embeddings(model_name, demo_ids)
        return embeddings, id_to_row, coverage

    elif dataset == "mind":
        # Load article texts from feature store
        store = ArticleFeatureStore(dataset)
        articles_df = store.get_articles_for_bm25()
        article_ids = articles_df["article_id"].to_list()
        texts = articles_df["cleaned_text"].to_list()

        embeddings, id_to_row = compute_mind_embeddings(article_ids, texts)
        return embeddings, id_to_row, None  # 100% coverage by construction

    else:
        raise ValueError(f"Unknown dataset: {dataset}")
