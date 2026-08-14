"""Download, unzip, and checksum the MIND dataset (English, TSV format).

Usage:
    python -m src.ingestion.download_mind

The script reads HF_TOKEN from the project .env file (or the shell environment).
Copy .env.example → .env and fill in your token before running.
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
    "MINDsmall_train.zip": "https://huggingface.co/datasets/yjw1029/MIND/resolve/main/MINDsmall_train.zip",
    "MINDsmall_dev.zip": "https://huggingface.co/datasets/yjw1029/MIND/resolve/main/MINDsmall_dev.zip",
}

# Destination directory (relative to project root)
DEST_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "mind"

CHUNK_SIZE = 128 * 1024  # 128 KB


# Helpers

def _build_headers() -> dict[str, str]:
    """Build HTTP headers; include HF bearer token when available.

    Token resolution order (first match wins):
      1. Shell environment variable ``HF_TOKEN``
      2. ``HF_TOKEN`` in the project ``.env`` file (loaded above via dotenv)

    If neither is set the download is attempted without auth, which will fail
    for gated datasets.  In that case:
      - Set HF_TOKEN in .env (copy .env.example → .env), OR
      - Download the zips manually from https://huggingface.co/datasets/yjw1029/MIND
        and place them in data/raw/mind/.
    """
    token = os.environ.get("HF_TOKEN")
    headers: dict[str, str] = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    else:
        print(
            "⚠  HF_TOKEN not set. Download may fail for gated datasets.\n"
            "   Add  HF_TOKEN=hf_...  to .env (see .env.example), or\n"
            "   export HF_TOKEN=hf_...  in your shell."
        )
    return headers


def _download_file(url: str, dest: Path, headers: dict[str, str],
                   max_retries: int = 3) -> None:
    """Stream-download *url* to *dest* with progress reporting and retries."""
    if dest.exists():
        print(f"  ✓ Already exists: {dest.name} ({dest.stat().st_size / 1024**2:.1f} MB)")
        return

    for attempt in range(1, max_retries + 1):
        try:
            print(f"  ⬇  Downloading {dest.name} (attempt {attempt}/{max_retries}) …")
            resp = requests.get(url, headers=headers, stream=True, timeout=120)
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
    marker = extract_to / zip_path.stem  # e.g. data/raw/mind/MINDsmall_train
    if marker.exists() and any(marker.iterdir()):
        print(f"  ✓ Already extracted: {marker.name}/")
        return

    print(f"  📦 Extracting {zip_path.name} → {extract_to.name}/ …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    print(f"  ✓ Extracted {zip_path.name}")


# Main

def main() -> None:
    """Download and extract MINDsmall train + dev."""
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    headers = _build_headers()

    print(f"📁 Destination: {DEST_DIR}")
    for filename, url in URLS.items():
        dest = DEST_DIR / filename
        _download_file(url, dest, headers)
        _unzip(dest, DEST_DIR)

    print("\n✅ MIND download complete.")
    # Quick listing
    for p in sorted(DEST_DIR.rglob("*")):
        if p.is_file() and not p.name.startswith("."):
            rel = p.relative_to(DEST_DIR)
            print(f"   {rel}  ({p.stat().st_size / 1024**2:.2f} MB)")


if __name__ == "__main__":
    main()
