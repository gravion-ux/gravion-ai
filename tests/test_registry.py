import pytest

from gravion_ai.errors import ModuleError
from gravion_ai.providers import GravionCoreProvider
from gravion_ai.registry import ModuleContext, ModuleType, default_registry


def test_catalogue_implemented_and_planned_modules():
    registry = default_registry()
    modules = {m.name: m for m in registry.list()}
    for name in ("chat", "support", "coder"):
        assert modules[name].status.value == "implemented", name
    for name in (
        "website-builder", "music", "image",
        "video", "cyber", "agentic", "all-in-one",
    ):
        assert modules[name].status.value == "planned", name
    assert modules["video"].type is ModuleType.VIDEO
    assert modules["support"].type is ModuleType.SUPPORT
    assert modules["coder"].type is ModuleType.CODER


def test_instantiate_chat_and_run():
    registry = default_registry()
    module = registry.instantiate("chat")
    ctx = ModuleContext("You are Gravion.", GravionCoreProvider(), model="gravion-core")
    assert "ping" in module.run(ctx, "ping")


def test_planned_module_refuses_to_instantiate():
    registry = default_registry()
    with pytest.raises(ModuleError):
        registry.instantiate("video")


def test_unknown_module_raises():
    registry = default_registry()
    with pytest.raises(ModuleError):
        registry.get("teleporter")
