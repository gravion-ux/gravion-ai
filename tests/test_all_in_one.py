from pathlib import Path

import pytest

from gravion_ai.errors import ModuleError, ModuleTamperError
from gravion_ai.licensing.modules_sig import (
    build_manifest,
    sign_manifest,
    verify_module,
    write_signed_manifest,
)
from gravion_ai.modules.all_in_one.module import parse_route
from gravion_ai.providers import GravionCoreProvider
from gravion_ai.registry import ModuleContext, default_registry
from gravion_ai.runtime import GravionRuntime


class RecordingProvider:
    name = "recording"

    def __init__(self):
        self.calls = []

    def complete(self, messages, model=None):
        self.calls.append((list(messages), model))
        return "recorded reply"


def _ctx(provider, model="gravion-omni"):
    return ModuleContext("You are Gravion, an AI assistant for Acme.", provider, model=model)


@pytest.mark.parametrize(
    "prompt,expected",
    [
        ("coder: fix this function", ("coder", "fix this function")),
        ("image: a logo for the bakery", ("image", "a logo for the bakery")),
        ("cyber: triage: alert from 10.0.0.8",
         ("cyber", "triage: alert from 10.0.0.8")),
        ("website-builder: a site for our cafe",
         ("website-builder", "a site for our cafe")),
        ("just a question about pricing",
         ("chat", "just a question about pricing")),
        ("note: remember the milk", ("chat", "note: remember the milk")),
        ("http: not a directive", ("chat", "http: not a directive")),
    ],
)
def test_parse_route(prompt, expected):
    assert parse_route(prompt) == expected


def test_parse_route_refuses_self_routing():
    with pytest.raises(ModuleError, match="cannot route to itself"):
        parse_route("all-in-one: coder: fix this")


def test_all_in_one_defaults_to_chat():
    module = default_registry().instantiate("all-in-one")
    out = module.run(_ctx(GravionCoreProvider()), "what can you do?")
    assert "what can you do?" in out


def test_all_in_one_routes_with_the_target_modules_own_brief():
    """Routing is delegation, not reimplementation: a coder
    directive reaches the provider with the coder module's own
    task guidance and brief applied."""
    provider = RecordingProvider()
    module = default_registry().instantiate("all-in-one")
    out = module.run(_ctx(provider), "coder: explain: what a decorator is")
    assert out == "recorded reply"
    (messages, model), = provider.calls
    assert model == "gravion-omni"
    assert "Coder mode: explain" in messages[0]["content"]
    assert messages[1] == {"role": "user", "content": "explain: what a decorator is"}


def test_all_in_one_runtime_dev_end_to_end_with_gravion_omni(make_config):
    runtime = GravionRuntime(make_config(mode="development", model="gravion-omni"))
    assert runtime.resolve_model("all-in-one") == "gravion-omni"
    out = runtime.all_in_one("coder: write a haiku function")
    assert "write a haiku function" in out


def test_all_in_one_rejects_a_specialist_model(make_config):
    """A specialist behind the all-in-one package would silently
    downgrade most modules — refused, with the gaps named."""
    runtime = GravionRuntime(make_config(mode="development", model="gravion-chat"))
    with pytest.raises(ModuleError, match="cannot serve the all-in-one"):
        runtime.all_in_one("hello")


def test_all_in_one_accepts_a_custom_endpoint_model(make_config):
    """A custom model id (the deployment's own endpoint) passes
    through, as with every other capability."""
    runtime = GravionRuntime(make_config(mode="development", model="acme-llm-9000"))
    assert runtime.resolve_model("all-in-one") == "acme-llm-9000"


def test_all_in_one_manifest_signs_and_verifies(issuer, all_in_one_module_copy):
    payload = build_manifest(
        all_in_one_module_copy,
        name="all-in-one",
        module_type="all-in-one",
        version="0.1.0",
        entrypoint="gravion_ai.modules.all_in_one.module:AllInOneModule",
    )
    write_signed_manifest(all_in_one_module_copy, sign_manifest(payload, issuer))
    verified = verify_module(all_in_one_module_copy, issuer)
    assert verified["name"] == "all-in-one"
    assert verified["type"] == "all-in-one"
    assert "module.py" in verified["files"]


def test_bundled_all_in_one_manifest_is_unsigned(issuer):
    import gravion_ai.modules.all_in_one as all_in_one_pkg

    with pytest.raises(ModuleTamperError):
        verify_module(Path(all_in_one_pkg.__file__).parent, issuer)
