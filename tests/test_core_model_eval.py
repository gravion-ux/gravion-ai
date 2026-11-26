"""Held-out eval: bundled v3 Gravion Core vs the 750-parameter v1 seed model."""

import json
from pathlib import Path

from gravion_ai.core_model import count_parameters, load_bundled_model, load_weights

FIXTURES = Path(__file__).parent / "fixtures"


def _eval_rows():
    return [
        json.loads(line)
        for line in (FIXTURES / "core-eval.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _accuracy(model, rows):
    correct = sum(1 for row in rows if model.predict(row["prompt"]) == row["completion"])
    return correct / len(rows)


def test_bundled_v3_beats_v1_seed_model_on_held_out_eval():
    rows = _eval_rows()
    assert len(rows) >= 1_000

    v3 = load_bundled_model()
    v1 = load_weights(FIXTURES / "gravion-core-v1-weights.json")

    assert v1.parameter_count == 750
    assert v3.parameter_count == count_parameters(v3.to_dict())
    assert v3.parameter_count >= 10_000_000
    assert v3.artifact["format"] == "gravion-core-local-v3"
    assert len(v3.artifact["features"]) >= 10_000

    v3_accuracy = _accuracy(v3, rows)
    v1_accuracy = _accuracy(v1, rows)
    assert v3_accuracy >= 0.75
    assert v3_accuracy > v1_accuracy


def test_bundled_v3_rejects_unrelated_prompts():
    v3 = load_bundled_model()
    assert v3.predict("unrelated zebra question") is None
    assert v3.predict("what is gravion core") == (
        "Gravion Core is the built-in local Gravion model for offline chat and support."
    )
