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
    assert core.weights_status == "included"
    assert core.weights_included is True
    assert core.parameter_count == 10586543
    assert core.weights_artifact == "data/gravion-core-weights.json"
    assert core.weights_sha256


def test_bundled_registry_entries_include_verified_local_weights():
    import json
    from importlib import resources

    from gravion_ai.core_model import count_parameters

    catalog = ModelCatalog.load()
    for entry in catalog.list():
        assert entry.weights_included is True, entry.id
        assert entry.weights_status == "included", entry.id
        assert entry.parameter_count and entry.parameter_count > 0, entry.id
        assert entry.weights_artifact.startswith("data/"), entry.id
        artifact = json.loads(
            resources.files("gravion_ai")
            .joinpath(entry.weights_artifact)
            .read_text(encoding="utf-8")
        )
        assert artifact["model_id"] == entry.id
        assert count_parameters(artifact) == entry.parameter_count
        assert artifact["weights_sha256"] == entry.weights_sha256


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


def test_gravion_chat_and_code_available_as_tier2_bundles():
    """Piece 2: the Gravion Chat and Gravion Code registry entries resolve at
    Tier 2 (Basic) with included local trained weights."""
    catalog = ModelCatalog.load()
    chat = catalog.get("gravion-chat")
    assert chat is not None and chat.status == "available"
    assert chat.tiers == ("tier-1-complete", "tier-2-basic", "tier-3-intermediate", "tier-4-advanced")
    code = catalog.get("gravion-code")
    assert code is not None and code.status == "available"
    assert code.tiers == ("tier-1-complete", "tier-2-basic", "tier-3-intermediate", "tier-4-advanced")
    assert code.capabilities == ("coder",)


def test_gravion_vision_and_audio_available_as_tier2_bundles():
    """Piece 3: the Gravion Vision and Gravion Audio registry entries resolve
    at Tier 2 (Basic) with included local trained weights."""
    catalog = ModelCatalog.load()
    vision = catalog.get("gravion-vision")
    assert vision is not None and vision.status == "available"
    assert vision.tiers == ("tier-1-complete", "tier-2-basic", "tier-3-intermediate", "tier-4-advanced")
    assert vision.capabilities == ("image",)
    audio = catalog.get("gravion-audio")
    assert audio is not None and audio.status == "available"
    assert audio.tiers == ("tier-1-complete", "tier-2-basic", "tier-3-intermediate", "tier-4-advanced")
    assert audio.capabilities == ("music",)


def test_gravion_video_available_as_tier2_bundle():
    """Piece 4: the Gravion Video registry entry resolves at Tier 2 (Basic)
    with included local trained weights or long-form stitching."""
    catalog = ModelCatalog.load()
    video = catalog.get("gravion-video")
    assert video is not None and video.status == "available"
    assert video.tiers == ("tier-1-complete", "tier-2-basic", "tier-3-intermediate", "tier-4-advanced")
    assert video.capabilities == ("video",)


def test_gravion_builder_available_as_tier2_bundle():
    """Piece 5: the Gravion Builder registry entry resolves at Tier 2 (Basic)
    with included local trained weights."""
    catalog = ModelCatalog.load()
    builder = catalog.get("gravion-builder")
    assert builder is not None and builder.status == "available"
    assert builder.tiers == ("tier-1-complete", "tier-2-basic", "tier-3-intermediate", "tier-4-advanced")
    assert builder.capabilities == ("website-builder",)


def test_gravion_shield_and_agent_available_as_tier2_bundles():
    """Piece 6: the Gravion Shield and Gravion Agent registry entries resolve
    at Tier 2 (Basic) with included local trained weights."""
    catalog = ModelCatalog.load()
    shield = catalog.get("gravion-shield")
    assert shield is not None and shield.status == "available"
    assert shield.tiers == ("tier-1-complete", "tier-2-basic", "tier-3-intermediate", "tier-4-advanced")
    assert shield.capabilities == ("cyber",)
    agent = catalog.get("gravion-agent")
    assert agent is not None and agent.status == "available"
    assert agent.tiers == ("tier-1-complete", "tier-2-basic", "tier-3-intermediate", "tier-4-advanced")
    assert agent.capabilities == ("agentic",)


def test_tier_bundles_resolve_by_capability_and_tier():
    """Since the tier-packaging piece the specialists resolve at
    Tiers 1–3, and since Piece 8 (Gravion Omni) at Tier 4 too."""
    catalog = ModelCatalog.load()
    basic_coder = {e.id for e in catalog.models_for("coder", tier="tier-2-basic")}
    assert "gravion-code" in basic_coder
    complete_coder = {e.id for e in catalog.models_for("coder", tier="tier-1-complete")}
    assert "gravion-code" in complete_coder
    intermediate_coder = {e.id for e in catalog.models_for("coder", tier="tier-3-intermediate")}
    assert "gravion-code" in intermediate_coder
    advanced_coder = {e.id for e in catalog.models_for("coder", tier="tier-4-advanced")}
    assert "gravion-code" in advanced_coder
    basic_chat = {e.id for e in catalog.models_for("chat", tier="tier-2-basic")}
    assert "gravion-chat" in basic_chat
    complete_chat = {e.id for e in catalog.models_for("chat", tier="tier-1-complete")}
    assert "gravion-chat" in complete_chat
    assert "gravion-core" in complete_chat  # Core serves every tier


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
    with pytest.raises(CatalogError, match="weights_status"):
        ModelCatalog.from_data(
            {"models": [_entry(weights_status="definitely-trained")]}
        )


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
    assert "weights=included" in out


def test_cli_models_list_capability_filter(capsys):
    assert main(["models", "list", "--capability", "video"]) == 0
    out = capsys.readouterr().out
    assert "gravion-video" in out
    assert "gravion-vision" not in out


def test_gravion_support_and_review_available_as_specialists():
    """Catalog expansion: Gravion Support is the dedicated support
    specialist and Gravion Review the code-review specialist, both
    with included local trained weights."""
    catalog = ModelCatalog.load()
    support = catalog.get("gravion-support")
    assert support is not None and support.status == "available"
    assert support.capabilities == ("support",)
    assert support.weights_included is True
    review = catalog.get("gravion-review")
    assert review is not None and review.status == "available"
    assert review.capabilities == ("coder",)
    assert review.weights_included is True
    assert "gravion-support" in {
        e.id for e in catalog.models_for("support", tier="tier-2-basic")
    }
    assert "gravion-review" in {
        e.id for e in catalog.models_for("coder", tier="tier-2-basic")
    }


def test_catalog_entries_resolve_to_real_models_and_modules():
    """Catalog integrity: every registry entry is a bundled local
    model, and every capability it claims maps to an implemented
    module type with at least one available model in every tier."""
    from gravion_ai.catalog import CAPABILITIES
    from gravion_ai.core_model import LOCAL_MODEL_IDS
    from gravion_ai.registry import ModuleStatus, default_registry
    from gravion_ai.tiers import TIERS

    catalog = ModelCatalog.load()
    assert {e.id for e in catalog.list()} == set(LOCAL_MODEL_IDS)
    implemented = {
        m.type.value
        for m in default_registry().list()
        if m.status is ModuleStatus.IMPLEMENTED
    }
    for entry in catalog.list():
        assert entry.status == "available", entry.id
        for capability in entry.capabilities:
            assert capability in implemented, (entry.id, capability)
    for capability in CAPABILITIES:
        for tier in TIERS:
            serving = catalog.models_for(capability, tier=tier)
            assert serving, (capability, tier)
            assert all(e.status == "available" for e in serving)


def test_gravion_omni_available_at_tier4():
    """Piece 8: Gravion Omni is the Tier 4 (Advanced) all-capability
    registry entry, with included local trained weights."""
    catalog = ModelCatalog.load()
    omni = catalog.get("gravion-omni")
    assert omni is not None and omni.status == "available"
    assert omni.tiers == ("tier-4-advanced",)
    from gravion_ai.catalog import CAPABILITIES
    assert set(omni.capabilities) == set(CAPABILITIES)
