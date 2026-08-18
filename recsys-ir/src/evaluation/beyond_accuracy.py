"""Beyond-accuracy evaluation - diversity, novelty, catalog coverage."""

from __future__ import annotations

import math
import numpy as np


def compute_intra_list_diversity(
    recommended_embeddings: np.ndarray,
) -> float:
    """Compute Intra-List Diversity (ILD) for a top-K recommendation list.
    
    Formula: mean pairwise (1 - cosine similarity) over the top-K embeddings.
    
    Parameters
    ----------
    recommended_embeddings : np.ndarray
        Array of shape (K, D) containing the embeddings of the top-K items.
        
    Returns
    -------
    float
        The average pairwise diversity. Returns 0.0 if K < 2.
    """
    k = recommended_embeddings.shape[0]
    if k < 2:
        return 0.0
        
    # Normalize embeddings for cosine similarity
    norms = np.linalg.norm(recommended_embeddings, axis=1, keepdims=True)
    # Handle zero norms to avoid division by zero
    norms = np.where(norms == 0, 1.0, norms)
    normed_emb = recommended_embeddings / norms
    
    # Cosine similarity matrix (K x K)
    sim_matrix = np.dot(normed_emb, normed_emb.T)
    
    # We want upper triangle without diagonal
    i_indices, j_indices = np.triu_indices(k, k=1)
    
    pairwise_sims = sim_matrix[i_indices, j_indices]
    pairwise_distances = 1.0 - pairwise_sims
    
    return float(np.mean(pairwise_distances))


def compute_novelty(
    recommended_ids: list[str],
    train_popularity: dict[str, int],
    total_train_items: int,
) -> float:
    """Compute Novelty for a top-K recommendation list.
    
    Formula: mean self-information -log2(popularity(article)/N) over recommended items.
    
    Parameters
    ----------
    recommended_ids : list[str]
        List of recommended item IDs.
    train_popularity : dict[str, int]
        Dictionary mapping item ID to its popularity count in the TRAIN split ONLY.
    total_train_items : int
        Total number of impressions/interactions in the TRAIN split (N).
        
    Returns
    -------
    float
        The average novelty.
    """
    if not recommended_ids or total_train_items == 0:
        return 0.0
        
    novelty_sum = 0.0
    for item_id in recommended_ids:
        # Default to 1 (unseen items) to avoid log(0) if item wasn't in train
        count = train_popularity.get(item_id, 0)
        # We can cap at 1 to prevent log(0) but also we can use Laplace smoothing
        # Or just max(1, count)
        count = max(1, count)
        prob = count / total_train_items
        novelty_sum += -math.log2(prob)
        
    return novelty_sum / len(recommended_ids)


def compute_coverage(
    all_recommended_ids: set[str],
    full_catalog_size: int,
) -> float:
    """Compute Catalog Coverage.
    
    Formula: fraction of the full catalog appearing at least once across all users' top-K.
    
    Parameters
    ----------
    all_recommended_ids : set[str]
        Set of unique item IDs recommended across all users in top-K.
    full_catalog_size : int
        Total number of unique items in the catalog.
        
    Returns
    -------
    float
        The coverage fraction.
    """
    if full_catalog_size == 0:
        return 0.0
    return len(all_recommended_ids) / full_catalog_size
