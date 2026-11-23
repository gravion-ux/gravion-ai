import json

import pytest

from gravion_ai.cli import main
from gravion_ai.errors import TrainingError
from gravion_ai.training.jobs import load_job, run_training_job, validate_job


def _job(**overrides):
    document = {"model": "gravion-chat", "data": "examples.jsonl"}
    document.update(overrides)
    return document


def test_validate_job_accepts_a_well_formed_job():
    job = validate_job(_job(base_tier="tier-3-intermediate", notes="weekly run"))
    assert job.model == "gravion-chat"
    assert job.data == "examples.jsonl"
    assert job.base_tier == "tier-3-intermediate"
    assert job.capabilities == ("chat", "support")


def test_validate_job_rejects_an_unknown_model():
    with pytest.raises(TrainingError, match="unknown model"):
        validate_job(_job(model="gravion-nope"))


def test_validate_job_rejects_a_planned_model():
    """A planned model has no bundle to fine-tune yet. (Every
    bundled model is available since Piece 8, so the case is
    exercised against a hand-built catalog.)"""
    from gravion_ai.catalog import ModelCatalog, ModelEntry

    catalog = ModelCatalog([
        ModelEntry(
            id="gravion-future", name="Gravion Future", status="planned",
            tiers=("tier-2-basic",), capabilities=("chat",), notes="",
        )
    ])
    with pytest.raises(TrainingError, match="no trained bundle"):
        validate_job(_job(model="gravion-future"), catalog=catalog)


def test_validate_job_rejects_a_missing_dataset():
    with pytest.raises(TrainingError, match="dataset"):
        validate_job({"model": "gravion-chat"})


def test_validate_job_rejects_unknown_fields_and_tiers():
    with pytest.raises(TrainingError, match="unknown job fields"):
        validate_job(_job(surprise=True))
    with pytest.raises(TrainingError, match="unknown base_tier"):
        validate_job(_job(base_tier="tier-9-ultimate"))


def test_load_job_round_trip_and_missing_file(tmp_path):
    path = tmp_path / "job.json"
    path.write_text(json.dumps(_job()), encoding="utf-8")
    assert load_job(path)["model"] == "gravion-chat"
    with pytest.raises(TrainingError, match="not found"):
        load_job(tmp_path / "absent.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(TrainingError, match="not valid JSON"):
        load_job(bad)


def test_run_training_job_is_honest_about_the_missing_backend():
    """A valid job still cannot train in this build — and must say
    so instead of faking weights or progress."""
    job = validate_job(_job())
    with pytest.raises(TrainingError, match="no training backend"):
        run_training_job(job)


def test_train_cli_validate_and_run(capsys, tmp_path):
    path = tmp_path / "job.json"
    path.write_text(json.dumps(_job()), encoding="utf-8")
    assert main(["train", "validate", "--job", str(path)]) == 0
    out = capsys.readouterr().out
    assert "training job valid" in out
    # Run validates, then refuses honestly: exit 1, nothing trained.
    assert main(["train", "run", "--job", str(path)]) == 1
    err = capsys.readouterr().err
    assert "no training backend" in err
