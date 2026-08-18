"""Bootstrap confidence interval estimation for evaluation metrics."""

from __future__ import annotations

import numpy as np

def compute_bootstrap_ci(
    metric_values: np.ndarray,
    b: int = 1000,
    random_state: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval for a metric.
    
    Vectorized computation over pre-calculated per-impression metrics.
    
    Parameters
    ----------
    metric_values : np.ndarray
        1D array of metric values (one per impression).
    b : int
        Number of bootstrap iterations.
    random_state : int
        Random seed for reproducibility.
        
    Returns
    -------
    tuple[float, float, float]
        (mean, ci_low, ci_high) representing the sample mean and the
        [2.5th, 97.5th] percentiles of the bootstrap distribution.
    """
    if len(metric_values) == 0:
        return 0.0, 0.0, 0.0
        
    rng = np.random.default_rng(random_state)
    n = len(metric_values)
    
    # Generate B bootstrap samples of size N (indices with replacement)
    # Shape: (B, N)
    indices = rng.integers(0, n, size=(b, n))
    
    # Vectorized indexing: grab metric values for all bootstrap samples
    # Shape: (B, N)
    samples = metric_values[indices]
    
    # Compute mean for each bootstrap sample
    # Shape: (B,)
    sample_means = np.mean(samples, axis=1)
    
    # Calculate 2.5th and 97.5th percentiles
    ci_low = float(np.percentile(sample_means, 2.5))
    ci_high = float(np.percentile(sample_means, 97.5))
    mean_val = float(np.mean(metric_values))
    
    return mean_val, ci_low, ci_high
