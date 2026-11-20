import pytest

from gravion_ai.errors import ModuleError
from gravion_ai.providers import GravionCoreProvider
from gravion_ai.registry import ModuleContext, ModuleType, default_registry


def test_catalogue_has_chat_implemented_and_rest_planned():
    registry = default_registry()
    modules = {m.name: m for m in registry.list()}
    assert modules["chat"].status.value == "implemented"
    for name in (
        "support", "coder", "website-builder", "music", "image",
        "video", "cyber", "agentic", "all-in-one",
    ):
        assert modules[name].status.value == "planned", name
    assert modules["video"].type is ModuleType.VIDEO


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
