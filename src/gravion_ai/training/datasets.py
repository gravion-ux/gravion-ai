"""Dataset preparation for customer fine-tuning (Piece 11).

The training pipeline's local half: turning a deployment's raw
example files into a clean, split, *checkable* dataset that the
hosted training pipeline (see jobs.py) can consume.

Input is JSONL — one `{"prompt": ..., "completion": ...}` object
per line — from a single file or every `*.jsonl` file in a
directory. Preparation is strict and deterministic:

- **Validation.** Every line must be a JSON object with non-empty
  string `prompt` and `completion` fields, within the size limit.
  Any bad line refuses the whole preparation, with the problems
  named — data is never silently dropped or repaired.
- **Deduplication.** Exact duplicate (prompt, completion) pairs
  collapse; the report counts them.
- **Split.** Examples are ordered by the SHA-256 of their content
  and cut at the validation fraction — the same input always
  produces byte-identical outputs, on any machine.
- **Report.** `dataset-report.json` records the source, counts,
  and the SHA-256 of both output files, so a dataset's identity
  can be verified before it is trained on.

What preparation is *not*: training. Producing weights still
requires the hosted training pipeline; `run_training_job` in
jobs.py says so plainly.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..errors import TrainingError

MAX_EXAMPLE_CHARS = 32_000


def _source_files(data) -> list[Path]:
    path = Path(data)
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(path.glob("*.jsonl"))
        if files:
            return files
    raise TrainingError(
        f"dataset source {path} is neither a .jsonl file nor a "
        "directory containing .jsonl files"
    )


def read_examples(data) -> list[tuple[str, str]]:
    """Read and validate every example from the source.

    Raises TrainingError naming every problem found (file, line)
    — a dataset with bad lines is refused whole, never partially
    imported.
    """
    examples: list[tuple[str, str]] = []
    problems: list[str] = []
    for path in _source_files(data):
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            where = f"{path.name}:{lineno}"
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                problems.append(f"{where}: not valid JSON")
                continue
            if not isinstance(record, dict):
                problems.append(f"{where}: not a JSON object")
                continue
            prompt, completion = record.get("prompt"), record.get("completion")
            if not isinstance(prompt, str) or not prompt.strip():
                problems.append(f"{where}: 'prompt' must be a non-empty string")
                continue
            if not isinstance(completion, str) or not completion.strip():
                problems.append(
                    f"{where}: 'completion' must be a non-empty string"
                )
                continue
            if len(prompt) + len(completion) > MAX_EXAMPLE_CHARS:
                problems.append(
                    f"{where}: example exceeds {MAX_EXAMPLE_CHARS} characters"
                )
                continue
            examples.append((prompt, completion))
    if problems:
        shown = "; ".join(problems[:5])
        more = f" (+{len(problems) - 5} more)" if len(problems) > 5 else ""
        raise TrainingError(f"invalid dataset: {shown}{more}")
    if not examples:
        raise TrainingError("dataset contains no examples")
    return examples


def _content_hash(prompt: str, completion: str) -> str:
    return hashlib.sha256(f"{prompt}\n{completion}".encode("utf-8")).hexdigest()


def _write_jsonl(path: Path, examples: list[tuple[str, str]]) -> str:
    lines = [
        json.dumps({"prompt": p, "completion": c}, sort_keys=True)
        for p, c in examples
    ]
    text = "\n".join(lines) + ("\n" if lines else "")
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prepare_dataset(data, out_dir, *, validation_fraction: float = 0.2) -> dict:
    """Prepare a dataset: validate, dedupe, split, write, report.

    Writes `train.jsonl`, `validation.jsonl` and
    `dataset-report.json` into `out_dir` and returns the report.
    Deterministic: the same source always yields the same files.
    """
    if not 0.0 <= validation_fraction < 1.0:
        raise TrainingError(
            f"validation_fraction must be in [0, 1), got {validation_fraction}"
        )
    examples = read_examples(data)

    seen: set[tuple[str, str]] = set()
    unique: list[tuple[str, str]] = []
    for example in examples:
        if example not in seen:
            seen.add(example)
            unique.append(example)
    duplicates = len(examples) - len(unique)

    ordered = sorted(unique, key=lambda e: _content_hash(*e))
    if len(ordered) >= 2 and validation_fraction > 0:
        n_validation = max(1, round(len(ordered) * validation_fraction))
        n_validation = min(n_validation, len(ordered) - 1)
    else:
        n_validation = 0
    validation = ordered[:n_validation]
    train = ordered[n_validation:]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_hash = _write_jsonl(out_dir / "train.jsonl", train)
    validation_hash = _write_jsonl(out_dir / "validation.jsonl", validation)
    report = {
        "source": str(data),
        "total_examples": len(examples),
        "duplicates_removed": duplicates,
        "unique_examples": len(unique),
        "train_count": len(train),
        "validation_count": len(validation),
        "validation_fraction": validation_fraction,
        "train_sha256": train_hash,
        "validation_sha256": validation_hash,
    }
    (out_dir / "dataset-report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report
