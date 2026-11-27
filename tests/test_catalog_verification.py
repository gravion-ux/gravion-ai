from dataclasses import replace

from gravion_ai.catalog import ModelCatalog
from gravion_ai.catalog_verification import (
    catalog_routing_examples,
    evaluate_catalog_routing,
    select_verified_model,
    verify_catalog,
)


def test_bundled_catalog_verifies_models_and_modules():
    report = verify_catalog()

    assert report.ok is True
    assert len(report.models) == 17
    assert len(report.modules) == 15
    assert len(report.verified_model_ids) == 17
    assert len(report.verified_module_names) == 15
    assert all(model.output_length > 0 for model in report.models)
    assert all(module.output_length > 0 for module in report.modules)


def test_verified_catalog_feeds_routing_training_and_evaluation():
    catalog = ModelCatalog.load()
    report = verify_catalog(catalog)
    examples = catalog_routing_examples(catalog, report)
    evaluation = evaluate_catalog_routing(catalog, report)

    assert len(examples) == 56
    assert ("route capability coder tier tier-2-basic", "gravion-code") in examples
    assert evaluation["cases"] == 56
    assert evaluation["correct"] == 56
    assert evaluation["accuracy"] == 1.0
    assert evaluation["training_examples"] == 56


def test_stale_model_entry_is_excluded_from_verified_selection():
    catalog = ModelCatalog.load()
    stale_core = replace(catalog.get("gravion-core"), parameter_count=1)
    stale_catalog = ModelCatalog(
        [stale_core if entry.id == "gravion-core" else entry for entry in catalog.list()]
    )

    report = verify_catalog(stale_catalog, run_smoke=False)

    assert report.ok is False
    assert "gravion-core" not in report.verified_model_ids
    selected = select_verified_model(stale_catalog, report, "chat", tier="tier-2-basic")
    assert selected is not None
    assert selected.id == "gravion-chat"
    assert any(
        issue.subject == "gravion-core" and issue.check == "parameter_count"
        for model in report.models
        for issue in model.issues
    )
