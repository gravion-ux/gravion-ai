import json

import pytest

from gravion_ai.bundles import (
    all_bundle_specs,
    build_bundle,
    bundle_hash,
    bundle_spec,
)
from gravion_ai.cli import main
from gravion_ai.errors import BundleError

PACKAGED_TIERS = ("tier-1-complete", "tier-2-basic", "tier-3-intermediate", "tier-4-advanced")


@pytest.mark.parametrize("tier", PACKAGED_TIERS)
def test_packaged_tiers_are_complete(tier):
    spec = bundle_spec(tier)
    assert spec["complete"] is True
    assert spec["gaps"] == []
    by_type = {m["type"]: m for m in spec["modules"]}
    assert "gravion-code" in by_type["coder"]["models"]
    assert "gravion-chat" in by_type["chat"]["models"]
    assert "gravion-shield" in by_type["cyber"]["models"]
    assert "gravion-agent" in by_type["agentic"]["models"]


def test_tier4_complete_since_omni():
    """Tier 4 (Advanced) landed with Gravion Omni in Piece 8:
    the specialists offer Tier 4 bundles and Omni serves the
    all-in-one package."""
    spec = bundle_spec("tier-4-advanced")
    assert spec["complete"] is True
    assert spec["gaps"] == []
    by_type = {m["type"]: m for m in spec["modules"]}
    assert "gravion-code" in by_type["coder"]["models"]
    assert "gravion-core" in by_type["chat"]["models"]


def test_all_in_one_is_listed_with_its_serving_models():
    """The all-in-one module adds no capability, so it is never a
    gap; its serving models (entries covering every capability)
    are listed — Gravion Omni at Tier 4, none at Tier 2, where a
    deployment would use its own custom endpoint model instead."""
    spec = bundle_spec("tier-2-basic")
    by_name = {m["module"]: m for m in spec["modules"]}
    assert by_name["all-in-one"]["status"] == "implemented"
    assert by_name["all-in-one"]["models"] == []
    assert "all-in-one" not in spec["gaps"]
    spec4 = bundle_spec("tier-4-advanced")
    by_name4 = {m["module"]: m for m in spec4["modules"]}
    assert by_name4["all-in-one"]["models"] == ["gravion-omni"]


def test_unknown_tier_rejected():
    with pytest.raises(BundleError, match="unknown tier"):
        bundle_spec("tier-9-ultimate")


def test_build_bundle_writes_a_hashed_descriptor(tmp_path):
    out = tmp_path / "bundle.json"
    document = build_bundle("tier-1-complete", out)
    assert out.exists()
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written == document
    assert written["bundle"]["tier"] == "tier-1-complete"
    assert written["bundle_hash"] == bundle_hash(written["bundle"])
    assert written["built_by"].startswith("gravion-ai ")


def test_bundle_build_is_deterministic(tmp_path):
    first = build_bundle("tier-3-intermediate", tmp_path / "a.json")
    second = build_bundle("tier-3-intermediate", tmp_path / "b.json")
    assert first["bundle_hash"] == second["bundle_hash"]
    assert (tmp_path / "a.json").read_bytes() == (tmp_path / "b.json").read_bytes()


def test_build_bundle_refuses_an_incomplete_tier(tmp_path):
    """The refusal path itself: a hand-built registry/catalog pair
    with a genuine gap must still refuse, naming the gap."""
    from gravion_ai.catalog import ModelCatalog, ModelEntry
    from gravion_ai.registry import ModuleRegistry, ModuleManifest, ModuleType, ModuleStatus
    registry = ModuleRegistry()
    registry.register(ModuleManifest(
        name="coder", type=ModuleType.CODER, version="0.1.0",
        entrypoint="x:Y", status=ModuleStatus.IMPLEMENTED))
    catalog = ModelCatalog([ModelEntry(
        id="gravion-core", name="Gravion Core", status="available",
        tiers=("tier-2-basic",), capabilities=("chat",), notes="")])
    with pytest.raises(BundleError, match="incomplete"):
        build_bundle("tier-2-basic", tmp_path / "nope.json",
                     registry=registry, catalog=catalog)
    assert not (tmp_path / "nope.json").exists()


def test_all_bundle_specs_cover_every_tier():
    specs = all_bundle_specs()
    assert [s["tier"] for s in specs] == [
        "tier-1-complete",
        "tier-2-basic",
        "tier-3-intermediate",
        "tier-4-advanced",
    ]


def test_bundles_cli_list_and_build(capsys, tmp_path):
    assert main(["bundles", "list"]) == 0
    out = capsys.readouterr().out
    assert "Tier 1 — Complete: complete" in out
    assert "Tier 4 — Advanced: complete" in out
    target = tmp_path / "b.json"
    assert main(["bundles", "build", "--tier", "tier-2-basic",
                 "--out", str(target)]) == 0
    out = capsys.readouterr().out
    assert "bundle written to" in out
    assert target.exists()
    assert main(["bundles", "build", "--tier", "tier-4-advanced",
                 "--out", str(tmp_path / "x.json")]) == 0
