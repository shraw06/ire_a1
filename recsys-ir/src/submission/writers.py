"""Competition output writers and schema checks."""

from __future__ import annotations

from pathlib import Path
from typing import TextIO, Sequence


def ranked_ids_to_positions(candidate_ids: Sequence[str], ranked_ids: Sequence[str]) -> list[int]:
    """Convert article ranking into the official 1-based candidate positions."""
    position = {str(article_id): i + 1 for i, article_id in enumerate(candidate_ids)}
    if len(position) != len(candidate_ids):
        raise ValueError("Candidate list contains duplicate article IDs")
    ranks = [position[str(article_id)] for article_id in ranked_ids]
    if sorted(ranks) != list(range(1, len(candidate_ids) + 1)):
        raise ValueError("Ranked IDs are not a permutation of the candidate list")
    return ranks


def write_ranked_impression(
    handle: TextIO,
    impression_id: str,
    candidate_ids: Sequence[str],
    ranked_ids: Sequence[str],
) -> None:
    ranks = ranked_ids_to_positions(candidate_ids, ranked_ids)
    handle.write(f"{impression_id} [{','.join(map(str, ranks))}]\n")


def validate_prediction_file(path: Path, expected_rows: int | None = None) -> int:
    """Validate MIND/EB-NeRD-style position-list prediction file."""
    rows = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.rstrip("\n")
            if not line:
                raise ValueError(f"Blank prediction line at {line_no}")
            if " [" not in line or not line.endswith("]"):
                raise ValueError(f"Malformed prediction line {line_no}: {line[:100]}")
            impression_id, ranks_raw = line.split(" [", 1)
            ranks_raw = ranks_raw[:-1]
            if not impression_id:
                raise ValueError(f"Missing impression ID at line {line_no}")
            ranks = [] if not ranks_raw else [int(x) for x in ranks_raw.split(",")]
            if sorted(ranks) != list(range(1, len(ranks) + 1)):
                raise ValueError(f"Ranks at line {line_no} are not 1..N")
            rows += 1
    if expected_rows is not None and rows != expected_rows:
        raise ValueError(f"Prediction row count {rows} != expected {expected_rows}")
    return rows


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Validate a Codabench prediction file")
    parser.add_argument("prediction")
    parser.add_argument("--expected-rows", type=int, default=None)
    args = parser.parse_args()
    rows = validate_prediction_file(Path(args.prediction), args.expected_rows)
    print(f"VALID: {rows:,} prediction rows")


if __name__ == "__main__":
    main()
