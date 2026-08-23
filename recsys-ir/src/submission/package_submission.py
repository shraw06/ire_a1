"""Package and validate a Codabench result submission."""

from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path

from src.submission.writers import validate_prediction_file


def package_prediction(prediction_path: Path, zip_path: Path) -> Path:
    prediction_path = prediction_path.resolve()
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    validate_prediction_file(prediction_path)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        zf.write(prediction_path, arcname=prediction_path.name)
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        if names != [prediction_path.name]:
            raise ValueError(f"Submission ZIP must contain exactly one file: {names}")
    return zip_path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prediction")
    parser.add_argument("zip")
    args = parser.parse_args()
    zip_path = package_prediction(Path(args.prediction), Path(args.zip))
    print(f"Created {zip_path}")
    print(f"SHA256 {sha256(zip_path)}")


if __name__ == "__main__":
    main()
