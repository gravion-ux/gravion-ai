import json

import pytest

from gravion_ai.catalog import ModelCatalog
from gravion_ai.cli import main
from gravion_ai.core_model import (
    count_parameters,
    load_bundled_model,
    load_weights,
    save_weights,
    train_local_core,
)
from gravion_ai.errors import TrainingError
from gravion_ai.providers import GravionCoreProvider
from gravion_ai.training.jobs import run_training_job, validate_job


def _examples():
    return [
        ("gravion local status", "Gravion local model is ready."),
        ("how does gravion count parameters", "Gravion counts stored numeric weights."),
        ("what happens when gravion is unsure", "Gravion falls back offline."),
    ]


def test_train_save_load_and_parameter_count(tmp_path):
    model = train_local_core(_examples(), source="unit-test")
    assert model.parameter_count == count_parameters(model.to_dict())
    assert model.parameter_count > 0
    assert model.predict("gravion local status") == "Gravion local model is ready."
    assert model.predict("unrelated zebra question") is None
    assert model.continue_text("gravion", max_tokens=3)

    path = tmp_path / "models" / "gravion-core-weights.json"
    report = save_weights(model, path)
    assert report["parameter_count"] == model.parameter_count
    assert report["weights_sha256"] == model.weights_sha256
    assert path.exists()

    loaded = load_weights(path)
    assert loaded.parameter_count == model.parameter_count
    assert loaded.predict("how does gravion count parameters") == (
        "Gravion counts stored numeric weights."
    )

    artifact = json.loads(path.read_text(encoding="utf-8"))
    artifact["parameter_count"] += 1
    bad_path = tmp_path / "bad-weights.json"
    bad_path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(TrainingError, match="parameter_count"):
        load_weights(bad_path)


def test_bundled_core_weights_match_registry():
    model = load_bundled_model()
    assert model.parameter_count == count_parameters(model.to_dict())
    assert model.predict("what is gravion core") == (
        "Gravion Core is the built-in local Gravion model for offline chat and support."
    )

    core = ModelCatalog.load().get("gravion-core")
    assert core is not None
    assert core.weights_status == "included"
    assert core.weights_included is True
    assert core.parameter_count == model.parameter_count
    assert core.weights_sha256 == model.weights_sha256


def test_core_provider_trained_inference_and_fallback(tmp_path):
    model = train_local_core(_examples(), source="provider-test")
    weights_path = tmp_path / "gravion-core-weights.json"
    save_weights(model, weights_path)

    provider = GravionCoreProvider(weights_path=weights_path)
    assert provider.parameter_count == model.parameter_count
    assert provider.complete(
        [{"role": "user", "content": "gravion local status"}]
    ) == "Gravion local model is ready."
    assert provider.complete(
        [{"role": "user", "content": "hello"}]
    ) == "[gravion-core] You said: hello"

    missing = GravionCoreProvider(weights_path=tmp_path / "missing.json")
    assert missing.parameter_count == 0
    assert missing.complete(
        [{"role": "user", "content": "gravion local status"}]
    ) == "[gravion-core] You said: gravion local status"

    corrupt_path = tmp_path / "corrupt.json"
    corrupt_path.write_text("{not json", encoding="utf-8")
    corrupt = GravionCoreProvider(weights_path=corrupt_path)
    assert corrupt.complete(
        [{"role": "user", "content": "gravion local status"}]
    ) == "[gravion-core] You said: gravion local status"


def test_run_training_job_trains_gravion_core(tmp_path):
    data_path = tmp_path / "examples.jsonl"
    data_path.write_text(
        "\n".join(
            json.dumps({"prompt": prompt, "completion": completion})
            for prompt, completion in _examples()
        )
        + "\n",
        encoding="utf-8",
    )
    job = validate_job({"model": "gravion-core", "data": str(data_path)})
    out_path = tmp_path / "data" / "models" / "gravion-core-weights.json"
    report = run_training_job(job, out_path=out_path)
    assert report["parameter_count"] > 0
    assert out_path.exists()
    assert load_weights(out_path).predict("gravion local status") == (
        "Gravion local model is ready."
    )


def test_train_cli_runs_gravion_core_job(tmp_path, capsys):
    data_path = tmp_path / "examples.jsonl"
    data_path.write_text(
        json.dumps(
            {
                "prompt": "gravion local status",
                "completion": "Gravion local model is ready.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    job_path = tmp_path / "job.json"
    job_path.write_text(
        json.dumps({"model": "gravion-core", "data": str(data_path)}),
        encoding="utf-8",
    )
    home = tmp_path / "home"
    assert main(["--home", str(home), "train", "run", "--job", str(job_path)]) == 0
    out = capsys.readouterr().out
    assert "training complete" in out
    assert "parameters:" in out
    assert (home / "data" / "models" / "gravion-core-weights.json").exists()


def test_training_rejects_tokenless_prompts_and_completions():
    # Regression: a punctuation-only prompt tokenizes to nothing, so
    # it used to train an exact-match entry for the empty normalized
    # prompt — after which "", "!!!" and "???" all predicted that
    # completion at confidence 1.0.
    with pytest.raises(TrainingError, match="at least one word or number"):
        train_local_core(
            [("!!!", "Punctuation answer."), *_examples()], source="tokenless"
        )
    with pytest.raises(TrainingError, match="at least one word or number"):
        train_local_core(
            [("gravion local status", "!!!")], source="tokenless-completion"
        )


def test_training_rejects_conflicting_normalized_prompts():
    # Regression: "Hello!" and "hello" normalize to the same prompt,
    # so two different completions for them are contradictory data.
    # Training used to accept both and silently keep the last one.
    with pytest.raises(TrainingError, match="conflicting completions"):
        train_local_core(
            [
                ("Hello!", "Answer A."),
                ("hello", "Answer B."),
                *_examples(),
            ],
            source="conflicting",
        )
    # The same normalized prompt with the same completion is a
    # duplicate, not a conflict: it trains once, not twice.
    model = train_local_core(
        [
            ("Hello!", "Answer A."),
            ("hello", "Answer A."),
            *_examples(),
        ],
        source="duplicate-normalized",
    )
    assert model.artifact["trained_from"]["examples"] == len(_examples()) + 1
    assert model.predict("HELLO") == "Answer A."


def test_empty_prompt_is_never_a_confident_match(tmp_path):
    model = train_local_core(_examples(), source="empty-prompt")
    assert model.predict("") is None
    assert model.predict("!!!") is None
    assert model.score("???") == (None, 0.0)
    with pytest.raises(TypeError, match="text must be a string"):
        model.predict(None)

    # An artifact carrying an empty exact-prompt key is invalid:
    # it would make every empty prompt a confident match.
    path = tmp_path / "gravion-core-weights.json"
    save_weights(model, path)
    artifact = json.loads(path.read_text(encoding="utf-8"))
    artifact["exact_prompts"][""] = 0
    bad_path = tmp_path / "empty-exact-weights.json"
    bad_path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(TrainingError, match="exact prompt map is invalid"):
        load_weights(bad_path)
