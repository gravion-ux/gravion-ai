import pytest

from gravion_ai.errors import ModuleError
from gravion_ai.providers import GravionCoreProvider
from gravion_ai.registry import (
    ModuleContext,
    ModuleManifest,
    ModuleRegistry,
    ModuleStatus,
    ModuleType,
    default_registry,
)


def test_catalogue_implemented_and_planned_modules():
    registry = default_registry()
    modules = {m.name: m for m in registry.list()}
    for name in ("chat", "support", "coder", "image", "music", "video",
                 "website-builder", "cyber", "agentic", "all-in-one"):
        assert modules[name].status.value == "implemented", name
    assert modules["video"].type is ModuleType.VIDEO
    assert modules["website-builder"].type is ModuleType.WEBSITE_BUILDER
    assert modules["all-in-one"].type is ModuleType.ALL_IN_ONE
    assert modules["cyber"].type is ModuleType.CYBER
    assert modules["agentic"].type is ModuleType.AGENTIC
    assert modules["support"].type is ModuleType.SUPPORT
    assert modules["coder"].type is ModuleType.CODER
    assert modules["image"].type is ModuleType.IMAGE
    assert modules["music"].type is ModuleType.MUSIC


def test_instantiate_chat_and_run():
    registry = default_registry()
    module = registry.instantiate("chat")
    ctx = ModuleContext("You are Gravion.", GravionCoreProvider(), model="gravion-core")
    assert "ping" in module.run(ctx, "ping")


def test_planned_module_refuses_to_instantiate():
    """Every bundled module is implemented since Piece 8, so the
    refusal is exercised on a hand-registered planned manifest."""
    registry = ModuleRegistry()
    registry.register(
        ModuleManifest(
            name="future-module",
            type=ModuleType.CHAT,
            version="0.0.0",
            entrypoint="",
            status=ModuleStatus.PLANNED,
        )
    )
    with pytest.raises(ModuleError):
        registry.instantiate("future-module")


def test_unknown_module_raises():
    registry = default_registry()
    with pytest.raises(ModuleError):
        registry.get("teleporter")
