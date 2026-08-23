"""Download the large datasets/artifacts required for Codabench submission.

This is intentionally separate from the development `make data` pipeline because the
large files are multi-GB and should never be downloaded accidentally during ordinary
experimentation.

The current EB-NeRD large bundle already contains the article catalog
(`articles.parquet`). Therefore, there is deliberately no separate
`articles_large_only.zip` download here; that historical URL is no longer valid.
"""

from __future__ import annotations

import argparse
import logging
import time
import zipfile
from pathlib import Path

import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env", override=False)

MIND_URLS = {
    "MINDlarge_train.zip": "https://huggingface.co/datasets/yjw1029/MIND/resolve/main/MINDlarge_train.zip",
    "MINDlarge_dev.zip": "https://huggingface.co/datasets/yjw1029/MIND/resolve/main/MINDlarge_dev.zip",
    "MINDlarge_test.zip": "https://huggingface.co/datasets/yjw1029/MIND/resolve/main/MINDlarge_test.zip",
}

# Do not add articles_large_only.zip here. The separate S3 artifact URL is obsolete
# and currently returns 404. articles.parquet is included in ebnerd_large.zip.
EBNERD_URLS = {
    "ebnerd_large.zip": "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_large.zip",
    "ebnerd_testset.zip": "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_testset.zip",
    "Ekstra_Bladet_word2vec.zip": "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/artifacts/Ekstra_Bladet_word2vec.zip",
}


def _stream_download(url: str, path: Path, retries: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        logger.info("Already present: %s (%.2f GB)", path, path.stat().st_size / 1024**3)
        return

    tmp = path.with_suffix(path.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            logger.info("Downloading %s (attempt %d/%d)", url, attempt, retries)
            with requests.get(url, stream=True, timeout=120) as response:
                response.raise_for_status()
                total = int(response.headers.get("content-length", 0))
                done = 0
                started = time.time()
                with tmp.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                        if chunk:
                            handle.write(chunk)
                            done += len(chunk)
                            if total:
                                elapsed = max(time.time() - started, 1e-6)
                                rate = done / elapsed / 1024**2
                                logger.info(
                                    "  %.1f%% | %.1f MB/s",
                                    100 * done / total,
                                    rate,
                                )
            tmp.replace(path)
            logger.info("Saved %s (%.2f GB)", path, path.stat().st_size / 1024**3)
            return
        except Exception:
            if tmp.exists():
                tmp.unlink()
            if attempt == retries:
                raise
            time.sleep(5 * attempt)


def _extract(path: Path, dest: Path) -> None:
    # Use a per-archive marker so repeated calls do not re-extract.
    done_marker = dest / f".{path.stem}.extracted"
    if done_marker.exists():
        logger.info("Already extracted: %s", path.name)
        return
    logger.info("Extracting %s", path)
    with zipfile.ZipFile(path) as archive:
        archive.extractall(dest)
    done_marker.write_text("ok\n")


def download_group(name: str) -> None:
    if name in {"mind", "all"}:
        raw = _PROJECT_ROOT / "data" / "raw" / "mind"
        raw.mkdir(parents=True, exist_ok=True)
        for filename, url in MIND_URLS.items():
            path = raw / filename
            _stream_download(url, path)
            _extract(path, raw)

    if name in {"ebnerd", "all"}:
        raw = _PROJECT_ROOT / "data" / "raw" / "ebnerd"
        raw.mkdir(parents=True, exist_ok=True)
        for filename, url in EBNERD_URLS.items():
            path = raw / filename
            _stream_download(url, path)
            _extract(path, raw)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download large Codabench artifacts")
    parser.add_argument("--dataset", choices=["mind", "ebnerd", "all"], default="all")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    download_group(args.dataset)


if __name__ == "__main__":
    main()
