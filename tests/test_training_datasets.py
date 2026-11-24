import hashlib
import json
from pathlib import Path

import pytest

from gravion_ai.cli import main
from gravion_ai.errors import TrainingError
from gravion_ai.training.datasets import (
    prepare_dataset,
    read_examples,
)
# Aliased: this file defines its own _write_jsonl fixture helper below,
# which would otherwise shadow the library function under test.
from gravion_ai.training.datasets import _write_jsonl as write_jsonl


def _write_jsonl(path: Path, records: list) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            r if isinstance(r, str) else json.dumps(r) for r in records
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _examples(n: int) -> list[dict]:
    return [
        {"prompt": f"question {i}", "completion": f"answer {i}"}
        for i in range(n)
    ]


def test_prepare_splits_dedupes_and_reports(tmp_path):
    source = _write_jsonl(
        tmp_path / "raw.jsonl",
        _examples(10) + [{"prompt": "question 0", "completion": "answer 0"}],
    )
    report = prepare_dataset(source, tmp_path / "out", validation_fraction=0.2)
    assert report["total_examples"] == 11
    assert report["duplicates_removed"] == 1
    assert report["unique_examples"] == 10
    assert report["train_count"] == 8
    assert report["validation_count"] == 2
    # The report's hashes match the files on disk.
    train_bytes = (tmp_path / "out" / "train.jsonl").read_bytes()
    assert report["train_sha256"] == hashlib.sha256(train_bytes).hexdigest()
    written = json.loads(
        (tmp_path / "out" / "dataset-report.json").read_text(encoding="utf-8")
    )
    assert written == report
    # No overlap between the splits.
    train = {json.loads(l)["prompt"] for l in train_bytes.decode().splitlines()}
    validation = {
        json.loads(l)["prompt"]
        for l in (tmp_path / "out" / "validation.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    }
    assert not (train & validation)
    assert len(train | validation) == 10


def test_prepare_is_deterministic(tmp_path):
    source = _write_jsonl(tmp_path / "raw.jsonl", _examples(25))
    first = prepare_dataset(source, tmp_path / "a")
    second = prepare_dataset(source, tmp_path / "b")
    assert first["train_sha256"] == second["train_sha256"]
    assert (tmp_path / "a" / "train.jsonl").read_bytes() == (
        tmp_path / "b" / "train.jsonl"
    ).read_bytes()


def test_prepare_reads_a_directory_of_files(tmp_path):
    data_dir = tmp_path / "data"
    _write_jsonl(data_dir / "one.jsonl", _examples(3))
    _write_jsonl(
        data_dir / "two.jsonl",
        [{"prompt": "extra", "completion": "example"}],
    )
    report = prepare_dataset(data_dir, tmp_path / "out")
    assert report["unique_examples"] == 4


def test_prepare_refuses_invalid_lines(tmp_path):
    source = _write_jsonl(
        tmp_path / "raw.jsonl",
        _examples(2) + ["{not json", {"prompt": "no completion"}],
    )
    with pytest.raises(TrainingError, match="invalid dataset"):
        prepare_dataset(source, tmp_path / "out")
    assert not (tmp_path / "out" / "train.jsonl").exists()


def test_prepare_refuses_an_empty_dataset(tmp_path):
    source = _write_jsonl(tmp_path / "raw.jsonl", [])
    with pytest.raises(TrainingError, match="no examples"):
        prepare_dataset(source, tmp_path / "out")


def test_prepare_refuses_a_missing_source(tmp_path):
    with pytest.raises(TrainingError, match="neither a .jsonl file"):
        prepare_dataset(tmp_path / "absent.jsonl", tmp_path / "out")


def test_single_example_goes_entirely_to_train(tmp_path):
    source = _write_jsonl(tmp_path / "raw.jsonl", _examples(1))
    report = prepare_dataset(source, tmp_path / "out")
    assert report["train_count"] == 1
    assert report["validation_count"] == 0


def test_read_examples_skips_blank_lines(tmp_path):
    path = tmp_path / "raw.jsonl"
    path.write_text(
        '\n{"prompt": "q", "completion": "a"}\n\n', encoding="utf-8"
    )
    assert read_examples(path) == [("q", "a")]


def test_train_prepare_cli(capsys, tmp_path):
    source = _write_jsonl(tmp_path / "raw.jsonl", _examples(5))
    out_dir = tmp_path / "dataset"
    assert main(
        ["train", "prepare", "--data", str(source), "--out", str(out_dir)]
    ) == 0
    out = capsys.readouterr().out
    assert "dataset prepared" in out
    assert (out_dir / "dataset-report.json").exists()


def test_write_jsonl_hash_describes_the_file_on_disk(tmp_path):
    """The recorded hash must match the bytes actually written.

    The dataset report records a SHA-256 for each output file so a dataset's
    identity can be verified before it is trained on.  That only holds if the
    bytes on disk are exactly the bytes that were hashed.  Writing the file in
    text mode translated LF to CRLF on Windows, so the recorded hash did not
    describe the file.
    """
    out = tmp_path / "train.jsonl"
    recorded = write_jsonl(out, [("question 0", "answer 0"), ("question 1", "answer 1")])

    on_disk = out.read_bytes()
    assert recorded == hashlib.sha256(on_disk).hexdigest()
    assert b"\r\n" not in on_disk
    assert on_disk == (
        b'{"completion": "answer 0", "prompt": "question 0"}\n'
        b'{"completion": "answer 1", "prompt": "question 1"}\n'
    )
