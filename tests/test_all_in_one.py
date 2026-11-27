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


@pytest.mark.parametrize(
    "prompt,expected",
    [
        # Regression: a directive with no space after the colon, or
        # with capitalised name, used to fall through to chat.
        ("coder:fix this function", ("coder", "fix this function")),
        ("Coder: fix this function", ("coder", "fix this function")),
        ("IMAGE:a logo for the bakery", ("image", "a logo for the bakery")),
    ],
)
def test_parse_route_directive_spelling_variants(prompt, expected):
    assert parse_route(prompt) == expected


def test_parse_route_refuses_capitalised_self_routing():
    with pytest.raises(ModuleError, match="cannot route to itself"):
        parse_route("All-In-One: coder: fix this")


@pytest.mark.parametrize(
    "prompt,expected_module",
    [
        ("write a python function to sort a list", "coder"),
        ("draw a logo for the bakery", "image"),
        ("compose a song about the rain", "music"),
        ("make a short video clip of the sea", "video"),
        ("build a website for our cafe", "website-builder"),
        ("triage this alert about CVE-2024-12345", "cyber"),
        ("i need a refund for my invoice", "support"),
        ("plan steps for an agent workflow", "agentic"),
        ("what is a function in mathematics", "chat"),
        ("make something nice for me", "chat"),
        # The five text modules used to be directive-only: plain
        # requests for them fell through to chat.
        ("translate thank you into hindi", "translation"),
        ("summarize this article about monsoon rains", "summarization"),
        ("rank these passages for my query", "retrieval"),
        ("analyze these numbers and compute the mean", "data-analysis"),
        ("answer this question from the document", "document-qa"),
        # A lone text-module verb is not enough on its own.
        ("translate this for me", "chat"),
        ("summarize it please", "chat"),
        # Regression: a negated verb still routed on its
        # verb+noun signal, so "do not draw a logo" produced
        # an image instead of the explanation asked for.
        ("do not draw a logo, just explain what a logo is", "chat"),
        ("don't write a python function, explain functions", "chat"),
        ("never make a video, tell me about videos", "chat"),
        # A negation aimed at one module does not cancel a
        # clear signal for another.
        ("do not draw anything, write a python function", "coder"),
    ],
)
def test_parse_route_infers_clear_intent(prompt, expected_module):
    name, routed = parse_route(prompt)
    assert name == expected_module
    assert routed == prompt


@pytest.mark.parametrize(
    "prompt,expected_module",
    [
        (
            "document: The library opens at nine.\nquestion: when does it open",
            "document-qa",
        ),
        (
            "query: how do cats groom\npassage: Cats groom with their tongues.",
            "retrieval",
        ),
        ("summarize: Cats sleep for long hours each day.", "summarization"),
        ("summarise: Cats sleep for long hours each day.", "summarization"),
        # Regression: the "summarize this:" form scored one
        # intent point and fell through to chat, so the
        # module's own input form could never reach it.
        ("summarize this: Cats sleep for long hours each day.", "summarization"),
        ("summarize the following: Cats sleep for long hours each day.", "summarization"),
        # Regression: CVE sequence numbers have no seven-digit
        # cap in the cyber module, but the router's capped
        # pattern sent an eight-digit id to chat instead.
        ("CVE-2026-12345678 details", "cyber"),
        # Regression: the translation module's own leading
        # forms scored one intent point and fell through to
        # chat, so its documented input format could never
        # reach it through the all-in-one package.
        ("translate: hello", "translation"),
        ("translate to hindi: hello", "translation"),
        ("translate into english: namaste", "translation"),
        # The say-in-language shape carries no translate verb,
        # so it scored one intent point and fell through to
        # chat; the translation module coverage-gates its text.
        ("how do you say thank you in hindi", "translation"),
        ("how do you say shukriya in english", "translation"),
    ],
)
def test_parse_route_structural_markers_are_decisive(prompt, expected_module):
    name, routed = parse_route(prompt)
    assert name == expected_module
    assert routed == prompt


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
    assert "haiku" in out.lower()
    assert "function" in out.lower()


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
