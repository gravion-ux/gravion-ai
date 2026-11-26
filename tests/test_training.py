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


def test_run_training_job_trains_specialist_weights(tmp_path):
    data = tmp_path / "examples.jsonl"
    data.write_text(
        json.dumps({"prompt": "greet a new user", "completion": "Hello and welcome."}) + "\n"
        + json.dumps({"prompt": "say goodbye politely", "completion": "Goodbye and take care."}) + "\n",
        encoding="utf-8",
    )
    job = validate_job(_job(data=str(data)))
    out_path = tmp_path / "gravion-chat-weights.json"
    report = run_training_job(job, out_path=out_path)
    assert report["model"] == "gravion-chat"
    assert report["parameter_count"] > 0
    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    assert artifact["model_id"] == "gravion-chat"
    assert artifact["parameter_count"] == report["parameter_count"]


def test_train_cli_validate_and_run(capsys, tmp_path):
    data = tmp_path / "examples.jsonl"
    data.write_text(
        json.dumps({"prompt": "greet a new user", "completion": "Hello and welcome."}) + "\n"
        + json.dumps({"prompt": "say goodbye politely", "completion": "Goodbye and take care."}) + "\n",
        encoding="utf-8",
    )
    path = tmp_path / "job.json"
    path.write_text(json.dumps(_job(data=str(data))), encoding="utf-8")
    home = tmp_path / "home"
    assert main(["--home", str(home), "train", "validate", "--job", str(path)]) == 0
    out = capsys.readouterr().out
    assert "training job valid" in out
    assert main(["--home", str(home), "train", "run", "--job", str(path)]) == 0
    out = capsys.readouterr().out
    assert "training complete: model gravion-chat" in out
    assert (home / "data" / "models" / "gravion-chat-weights.json").exists()
