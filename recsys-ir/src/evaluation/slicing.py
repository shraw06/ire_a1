"""Evaluation slicing - cold/warm users, head/tail articles."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Fixed thresholds
FIXED_COLD_USER_THRESHOLD = 5
FIXED_TAIL_ARTICLE_THRESHOLD = 10

# Data-driven thresholds
DATA_DRIVEN_COLD_USER_THRESHOLDS = {
    "mind": 3,
    "ebnerd": 9,
}

DATA_DRIVEN_TAIL_ARTICLE_THRESHOLDS = {
    "mind": 7,
    "ebnerd": 15,
}

def get_user_slice(history_len: int, dataset: str, slice_type: str = "fixed") -> str:
    """Classify user as cold or warm.
    
    slice_type: 'fixed' or 'data-driven'
    """
    if slice_type == "fixed":
        threshold = FIXED_COLD_USER_THRESHOLD
    else:
        threshold = DATA_DRIVEN_COLD_USER_THRESHOLDS[dataset]
        
    return "cold" if history_len <= threshold else "warm"


def get_article_slice(avg_popularity: float, dataset: str, slice_type: str = "fixed") -> str:
    """Classify article (or group of articles) as tail or head.
    
    slice_type: 'fixed' or 'data-driven'
    """
    if slice_type == "fixed":
        threshold = FIXED_TAIL_ARTICLE_THRESHOLD
    else:
        threshold = DATA_DRIVEN_TAIL_ARTICLE_THRESHOLDS[dataset]
        
    return "tail" if avg_popularity <= threshold else "head"
