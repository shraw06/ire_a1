from pathlib import Path
import zipfile

import pytest

from src.submission.make_submission import (
    _parse_mind_row,
    _stable_rank_positions,
    validate_prediction_file,
    validate_zip,
)


def test_stable_rank_positions_ties_keep_input_order():
    assert _stable_rank_positions([0.2, 0.9, 0.2, -1.0]) == [2, 1, 3, 4]


def test_mind_test_row_parser():
    row = _parse_mind_row(
        "123\tU1\t11/11/2019 9:05:58 AM\tN1 N2\tN10 N11 N12"
    )
    assert row.impression_id == "123"
    assert row.history_ids == ("N1", "N2")
    assert row.candidate_ids == ("N10", "N11", "N12")


def test_prediction_validation(tmp_path: Path):
    prediction = tmp_path / "prediction.txt"
    prediction.write_text("1 [2,1]\n2 [1,2]\n", encoding="utf-8")
    assert validate_prediction_file(prediction, "prediction.txt") == 2

    bad = tmp_path / "bad.txt"
    bad.write_text("1 [1,1]\n", encoding="utf-8")
    with pytest.raises(ValueError):
        validate_prediction_file(bad, "bad.txt")


def test_zip_validation(tmp_path: Path):
    prediction = tmp_path / "predictions.txt"
    prediction.write_text("1 [2,1]\n", encoding="utf-8")
    zip_path = tmp_path / "submission.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(prediction, arcname="predictions.txt")
    validate_zip(zip_path, "predictions.txt")
