"""SHA-256 checksum utilities for dataset archive integrity.

Provides three operations used by the download scripts:

  1. **compute** - hash a local archive file (streaming, low memory).
  2. **verify_or_populate** - if checksums.yaml has a stored digest for a file,
     compare it against the local archive; if the stored digest is null,
     compute from the existing archive and persist it.  Returns whether the
     archive is valid (True = skip re-download).
  3. **load / save** - read and write configs/checksums.yaml.

Usage from download scripts:
    from src.ingestion.checksums import verify_or_populate
    if verify_or_populate("mind", "MINDsmall_train.zip", dest_path):
        print("Checksum OK, skipping download")
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CHECKSUMS_PATH = _PROJECT_ROOT / "configs" / "checksums.yaml"

_CHUNK_SIZE = 256 * 1024  # 256 KB for hashing


def _load_checksums() -> dict:
    """Load configs/checksums.yaml. Returns empty dict if file is missing."""
    if not _CHECKSUMS_PATH.exists():
        return {}
    with open(_CHECKSUMS_PATH, "r") as f:
        data = yaml.safe_load(f) or {}
    return data


def _save_checksums(data: dict) -> None:
    """Write data back to configs/checksums.yaml, preserving the header comment."""
    header = (
        "# Dataset file checksums\n"
        "# Used by download scripts to verify integrity and skip re-download.\n"
        "# Format: SHA-256 hex digest of the downloaded zip file.\n"
        "#\n"
        "# To regenerate:\n"
        "#   sha256sum data/raw/mind/MINDsmall_train.zip\n"
        "#   sha256sum data/raw/ebnerd/ebnerd_demo.zip\n"
        "\n"
    )
    with open(_CHECKSUMS_PATH, "w") as f:
        f.write(header)
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 hex digest of a file (streaming, low memory)."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(_CHUNK_SIZE)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def verify_or_populate(dataset_key: str, filename: str, filepath: Path) -> bool:
    """Check or populate the SHA-256 checksum for a dataset archive.

    Args:
        dataset_key: Top-level key in checksums.yaml (e.g. "mind", "ebnerd").
        filename: Archive filename (e.g. "MINDsmall_train.zip").
        filepath: Absolute path to the local archive file.

    Returns:
        True  - archive exists and checksum is verified (safe to skip download).
        False - archive is missing, or checksum mismatch (need to re-download).

    Side effects:
        If the archive exists locally and the stored checksum is null, computes
        the checksum and persists it to checksums.yaml.
    """
    if not filepath.exists():
        return False

    data = _load_checksums()
    stored = data.get(dataset_key, {}).get(filename)

    actual = compute_sha256(filepath)

    if stored is None:
        # First time seeing this archive - compute and persist
        if dataset_key not in data:
            data[dataset_key] = {}
        data[dataset_key][filename] = actual
        _save_checksums(data)
        print(f"  # Checksum computed and saved for {filename}: {actual[:16]}...")
        return True

    if stored == actual:
        print(f"  # Checksum verified for {filename}: {actual[:16]}...")
        return True

    # Mismatch - archive may be corrupted or a different version
    print(
        f"  ✗ Checksum MISMATCH for {filename}!\n"
        f"    Expected: {stored}\n"
        f"    Got:      {actual}\n"
        f"    The archive will be re-downloaded."
    )
    return False
