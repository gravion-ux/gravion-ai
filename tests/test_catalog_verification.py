import shutil
from dataclasses import replace
from pathlib import Path

from gravion_ai.catalog import ModelCatalog
from gravion_ai.catalog_verification import (
    _verify_module_manifest,
    catalog_routing_examples,
    evaluate_catalog_routing,
    select_verified_model,
    verify_catalog,
)
from gravion_ai.registry import ModuleManifest, ModuleType


def _chat_module_copy(tmp_path: Path) -> Path:
    """A copy of the bundled chat module, rewritten with CRLF endings."""
    import gravion_ai.modules.chat as chat_pkg

    module_dir = tmp_path / "chat-module"
    shutil.copytree(
        Path(chat_pkg.__file__).parent,
        module_dir,
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    for path in module_dir.rglob("*.py"):
        data = path.read_bytes().replace(b"\r\n", b"\n")
        path.write_bytes(data.replace(b"\n", b"\r\n"))
    return module_dir


def _chat_manifest(module_dir: Path) -> ModuleManifest:
    return ModuleManifest(
        name="chat",
        type=ModuleType.CHAT,
        version="0.1.0",
        entrypoint="gravion_ai.modules.chat.module:ChatModule",
        path=module_dir,
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


def test_module_manifest_verifies_for_a_crlf_checkout(tmp_path):
    # Git stores LF; a Windows checkout rewrites the working tree to
    # CRLF. The manifest records the LF hashes, so it must still verify.
    module_dir = _chat_module_copy(tmp_path)

    issues = _verify_module_manifest(_chat_manifest(module_dir))

    assert issues == []


def test_module_manifest_reports_a_changed_file_in_a_crlf_checkout(tmp_path):
    module_dir = _chat_module_copy(tmp_path)
    target = module_dir / "module.py"
    target.write_bytes(target.read_bytes() + b"# tampered\r\n")

    issues = _verify_module_manifest(_chat_manifest(module_dir))

    assert any(issue.check == "manifest_hashes" for issue in issues)
