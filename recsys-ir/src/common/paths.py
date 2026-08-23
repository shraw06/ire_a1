"""Centralized scale-aware paths for the reproducible pipeline.

Development data keeps the historical layout for backwards compatibility::
    data/interim/<dataset>/
    data/processed/<dataset>/

Large-scale artifacts live separately::
    data/interim/large/<dataset>/
    data/processed/large/<dataset>/

This prevents a large build from overwriting the small/demo artifacts used by
fast unit tests and local iteration.
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VALID_SCALES = {"small", "large"}


def normalize_scale(scale: str | None) -> str:
    value = (scale or "small").lower()
    if value not in VALID_SCALES:
        raise ValueError(f"Unknown data scale '{scale}'. Expected one of {sorted(VALID_SCALES)}")
    return value


def interim_dir(dataset: str, scale: str = "small") -> Path:
    scale = normalize_scale(scale)
    return PROJECT_ROOT / "data" / "interim" / (Path(".") if scale == "small" else Path("large")) / dataset


def processed_dir(dataset: str, scale: str = "small") -> Path:
    scale = normalize_scale(scale)
    return PROJECT_ROOT / "data" / "processed" / (Path(".") if scale == "small" else Path("large")) / dataset


def processed_root(scale: str = "small") -> Path:
    scale = normalize_scale(scale)
    return PROJECT_ROOT / "data" / "processed" / ("" if scale == "small" else "large")


def results_dir(scale: str = "small") -> Path:
    scale = normalize_scale(scale)
    return PROJECT_ROOT / "results" / ("" if scale == "small" else "large")
