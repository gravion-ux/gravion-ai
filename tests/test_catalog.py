import pytest

from gravion_ai.catalog import ModelCatalog
from gravion_ai.cli import main
from gravion_ai.errors import CatalogError

def test_bundled_registry_loads_and_validates():
    catalog = ModelCatalog.load()
    entries = catalog.list()
    assert len(entries) >= 10
    core = catalog.get("gravion-core")
    assert core is not None and core.status == "available"


def test_registry_contains_only_gravion_models():
    """The registry is Gravion's own model family — every entry is a
    Gravion model, named as one."""
    catalog = ModelCatalog.load()
    for entry in catalog.list():
        assert entry.id.startswith("gravion-"), entry.id
        assert entry.name.startswith("Gravion "), entry.name


def test_models_for_capability_returns_only_matching():
    catalog = ModelCatalog.load()
    for capability in ("chat", "image", "video", "music", "cyber", "coder"):
        matches = catalog.models_for(capability)
        assert matches, capability
        assert all(e.serves(capability) for e in matches)
    image_models = {e.id for e in catalog.models_for("image")}
    assert "gravion-vision" in image_models
    assert "gravion-chat" not in image_models


def test_models_for_tier_filter():
    catalog = ModelCatalog.load()
    omni = catalog.models_for("video", tier="tier-4-advanced")
    assert any(e.id == "gravion-omni" for e in omni)
    basic_video = catalog.models_for("video", tier="tier-2-basic")
    assert all(e.id != "gravion-omni" for e in basic_video)
    assert any(e.id == "gravion-video" for e in basic_video)


def test_unknown_capability_rejected():
    catalog = ModelCatalog.load()
    with pytest.raises(CatalogError):
        catalog.models_for("teleportation")


def _entry(**overrides):
    base = {
        "id": "gravion-test",
        "name": "Gravion Test",
        "capabilities": ["chat"],
        "tiers": ["tier-2-basic"],
        "status": "planned",
    }
    base.update(overrides)
    return base


def test_malformed_entry_missing_field_fails_clearly():
    data = {"models": [_entry()]}
    del data["models"][0]["capabilities"]
    with pytest.raises(CatalogError, match="capabilities"):
        ModelCatalog.from_data(data)


def test_malformed_entry_bad_capability_fails():
    with pytest.raises(CatalogError, match="unknown capabilities"):
        ModelCatalog.from_data({"models": [_entry(capabilities=["dreaming"])]})


def test_malformed_entry_bad_tier_and_status_fail():
    with pytest.raises(CatalogError, match="unknown tiers"):
        ModelCatalog.from_data({"models": [_entry(tiers=["tier-9-ultra"])]})
    with pytest.raises(CatalogError, match="status"):
        ModelCatalog.from_data({"models": [_entry(status="shipped")]})


def test_duplicate_id_fails():
    with pytest.raises(CatalogError, match="duplicate"):
        ModelCatalog.from_data({"models": [_entry(), _entry()]})


def test_deployment_override_registry(tmp_path):
    import json

    (tmp_path / "models.json").write_text(
        json.dumps({"models": [_entry(id="gravion-private", name="Gravion Private")]}),
        encoding="utf-8",
    )
    catalog = ModelCatalog.load_for_home(tmp_path)
    assert catalog.get("gravion-private") is not None
    assert catalog.get("gravion-core") is None  # override replaces, not merges


def test_cli_models_list(capsys):
    assert main(["models", "list"]) == 0
    out = capsys.readouterr().out
    assert "gravion-core" in out
    assert "gravion-omni" in out


def test_cli_models_list_capability_filter(capsys):
    assert main(["models", "list", "--capability", "video"]) == 0
    out = capsys.readouterr().out
    assert "gravion-video" in out
    assert "gravion-vision" not in out
