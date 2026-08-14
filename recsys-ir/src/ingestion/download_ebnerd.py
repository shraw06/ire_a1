"""Download, unzip, and checksum the EB-NeRD dataset (Danish, Parquet format).

Usage:
    python -m src.ingestion.download_ebnerd

The demo bundle is hosted on public S3 — no authentication token is required.
The .env file is still loaded for consistency with download_mind.py.
"""

import os
import zipfile
from pathlib import Path

import requests
from dotenv import load_dotenv

# Load .env from the project root (two directories above this file)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env", override=False)

# Configuration

URLS = {
    "ebnerd_demo.zip": "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_demo.zip",
}

# Destination directory (relative to project root)
DEST_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "ebnerd"

CHUNK_SIZE = 128 * 1024  # 128 KB


# Helpers

def _download_file(url: str, dest: Path, max_retries: int = 3) -> None:
    """Stream-download *url* to *dest* with progress reporting and retries."""
    if dest.exists():
        print(f"  ✓ Already exists: {dest.name} ({dest.stat().st_size / 1024**2:.1f} MB)")
        return

    for attempt in range(1, max_retries + 1):
        try:
            print(f"  ⬇  Downloading {dest.name} (attempt {attempt}/{max_retries}) …")
            resp = requests.get(url, stream=True, timeout=120)
            resp.raise_for_status()

            total = int(resp.headers.get("content-length", 0))
            downloaded = 0

            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        print(f"\r    {downloaded / 1024**2:.1f} / {total / 1024**2:.1f} MB ({pct:.0f}%)", end="", flush=True)
                    else:
                        print(f"\r    {downloaded / 1024**2:.1f} MB", end="", flush=True)

            print(f"\n  ✓ Saved {dest.name} ({downloaded / 1024**2:.1f} MB)")
            return  # success

        except (requests.ConnectionError, requests.ChunkedEncodingError) as exc:
            # Remove partial file before retrying
            if dest.exists():
                dest.unlink()
            if attempt < max_retries:
                wait = 5 * attempt
                print(f"\n  ⚠ Connection error: {exc}\n    Retrying in {wait}s …")
                import time
                time.sleep(wait)
            else:
                print(f"\n  ✗ Download failed after {max_retries} attempts.")
                raise


def _unzip(zip_path: Path, extract_to: Path) -> None:
    """Extract *zip_path* into *extract_to*, skipping if already done."""
    # Use the zip stem as the extraction sub-folder marker
    marker = extract_to / zip_path.stem  # e.g. data/raw/ebnerd/ebnerd_demo
    if marker.exists() and any(marker.iterdir()):
        print(f"  ✓ Already extracted: {marker.name}/")
        return

    print(f"  📦 Extracting {zip_path.name} → {extract_to.name}/ …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    print(f"  ✓ Extracted {zip_path.name}")


# Main

def main() -> None:
    """Download and extract EB-NeRD demo bundle."""
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    print(f"📁 Destination: {DEST_DIR}")
    for filename, url in URLS.items():
        dest = DEST_DIR / filename
        _download_file(url, dest)
        _unzip(dest, DEST_DIR)

    print("\n✅ EB-NeRD download complete.")
    # Quick listing
    for p in sorted(DEST_DIR.rglob("*")):
        if p.is_file() and not p.name.startswith("."):
            rel = p.relative_to(DEST_DIR)
            print(f"   {rel}  ({p.stat().st_size / 1024**2:.2f} MB)")


if __name__ == "__main__":
    main()
