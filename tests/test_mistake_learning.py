import pytest

from gravion_ai.errors import TrainingError
from gravion_ai.training.mistakes import (
    MistakeCase,
    MistakeLog,
    apply_preference_learning,
    build_preference_pairs,
    evaluate_preference_pairs,
    load_bundled_mistakes,
    render_regression_tests,
    run_regression_case,
    train_with_mistake_learning,
    write_regression_tests,
)


def _case(**overrides):
    values = {
        "kind": "weak_response",
        "prompt": "greet a new user",
        "rejected_output": "Go away.",
        "corrected_output": "Hello and welcome.",
        "model_id": "gravion-chat",
        "regression_target": "provider",
    }
    values.update(overrides)
    return MistakeCase(**values)


def test_mistake_log_records_dedupes_and_loads(tmp_path):
    log = MistakeLog(tmp_path / "mistakes.jsonl")
    first = log.record_failure(
        kind="failing_test",
        prompt="run the failing check",
        rejected_output="failure",
        corrected_output="pass",
        source="unit test",
    )
    second = log.append(first)

    assert second.id == first.id
    assert log.load() == [first]
    assert len((tmp_path / "mistakes.jsonl").read_text(encoding="utf-8").splitlines()) == 1


def test_mistake_case_rejects_unknown_kind():
    with pytest.raises(TrainingError, match="mistake kind"):
        _case(kind="mystery")


def test_preference_pairs_use_corrected_cases_only():
    corrected = _case()
    uncorrected = _case(
        prompt="another prompt", rejected_output="wrong", corrected_output=""
    )

    pairs = build_preference_pairs([corrected, uncorrected, corrected])

    assert len(pairs) == 1
    assert pairs[0].rejected_output == "Go away."
    assert pairs[0].chosen_output == "Hello and welcome."


def test_preference_learning_overrides_rejected_training_output():
    examples = apply_preference_learning(
        [("greet a new user", "Go away."), ("say goodbye", "Goodbye.")],
        [_case()],
        catalog_examples=[("route capability chat tier tier-2-basic", "gravion-chat")],
    )

    assert ("greet a new user", "Hello and welcome.") in examples
    assert ("greet a new user", "Go away.") not in examples
    assert ("route capability chat tier tier-2-basic", "gravion-chat") in examples


def test_train_with_mistake_learning_uses_verified_catalog_examples():
    model, report = train_with_mistake_learning(
        [("greet a new user", "Go away."), ("say goodbye", "Goodbye.")],
        [_case()],
        model_id="gravion-chat",
        source="mistake-learning-test",
    )

    assert model.predict("greet a new user") == "Hello and welcome."
    assert report["preference_pairs"] == 1
    assert report["catalog_examples"] == 56
    assert report["preference_evaluation"]["accuracy"] == 1.0
    assert evaluate_preference_pairs(model, build_preference_pairs([_case()]))["wins"] == 1


def test_regression_test_generation_covers_each_corrected_case(tmp_path):
    case = _case()
    source = render_regression_tests([case])
    path = write_regression_tests(tmp_path / "test_generated_regressions.py", [case])

    assert case.id in source
    assert "test_corrected_mistake_stays_fixed" in source
    assert path.read_text(encoding="utf-8") == source


def test_bundled_mistakes_are_corrected_and_regressed():
    cases = load_bundled_mistakes()

    assert len(cases) == 4
    assert all(case.corrected for case in cases)
    assert {case.kind for case in cases} == {
        "weak_response",
        "wrong_route",
        "catalog_failure",
    }
    assert all(run_regression_case(case) for case in cases)
