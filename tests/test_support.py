import pytest

from gravion_ai.errors import ModuleTamperError
from gravion_ai.licensing import MockAuthorityClient
from gravion_ai.licensing.modules_sig import (
    build_manifest,
    sign_manifest,
    verify_module,
    write_signed_manifest,
)
from gravion_ai.modules.support.module import (
    ESCALATION_RESPONSE,
    SUPPORT_ADDENDUM,
    needs_human,
)
from gravion_ai.providers import GravionCoreProvider
from gravion_ai.registry import (
    ModuleContext,
    ModuleManifest,
    ModuleRegistry,
    ModuleStatus,
    ModuleType,
    default_registry,
)
from gravion_ai.runtime import GravionRuntime


class RecordingProvider:
    name = "recording"

    def __init__(self):
        self.calls = []

    def complete(self, messages, model=None):
        self.calls.append((list(messages), model))
        return "recorded reply"


class ExplodingProvider:
    name = "exploding"

    def complete(self, messages, model=None):
        raise AssertionError("the model must not be called for escalations")


def _ctx(provider, model="gravion-core"):
    return ModuleContext("You are Gravion, an AI assistant for Acme.", provider, model=model)


def test_support_instantiates_and_answers():
    module = default_registry().instantiate("support")
    out = module.run(_ctx(GravionCoreProvider()), "where is my invoice?")
    assert "where is my invoice?" in out


def test_support_appends_grounding_addendum_and_passes_model():
    provider = RecordingProvider()
    module = default_registry().instantiate("support")
    out = module.run(_ctx(provider, model="gravion-chat"), "what is the refund window?")
    assert out == "recorded reply"
    (messages, model), = provider.calls
    assert model == "gravion-chat"
    assert messages[0]["role"] == "system"
    assert messages[0]["content"].startswith("You are Gravion")
    assert SUPPORT_ADDENDUM.strip() in messages[0]["content"]
    assert "never invent prices" in messages[0]["content"]
    assert messages[1] == {"role": "user", "content": "what is the refund window?"}


@pytest.mark.parametrize(
    "prompt",
    [
        "can I talk to a human please",
        "I want a real person, not a bot",
        "escalate this to your manager",
        "let me speak to an agent",
        "please call me about this",
    ],
)
def test_escalation_requests_skip_the_model(prompt):
    module = default_registry().instantiate("support")
    assert needs_human(prompt)
    assert module.run(_ctx(ExplodingProvider()), prompt) == ESCALATION_RESPONSE


@pytest.mark.parametrize(
    "prompt",
    [
        "where is my invoice?",
        "how do I reset my password?",
        "what are your opening hours?",
        "my order has not arrived yet",
    ],
)
def test_ordinary_questions_do_not_escalate(prompt):
    assert not needs_human(prompt)


@pytest.mark.parametrize(
    "prompt",
    [
        # Regression: escalation phrases used to match as substrings
        # inside longer words, escalating customers who never asked
        # for a person ("agent" in "reagent", "manager" in
        # "managerial", "human" in "humane").
        "do you sell reagent refills?",
        "how do I update my managerial title in settings?",
        "is this product humane certified?",
        # "Human resources" is a support topic, not a request for a
        # person.
        "what is your human resources policy?",
    ],
)
def test_escalation_phrases_inside_other_words_do_not_escalate(prompt):
    assert not needs_human(prompt)


@pytest.mark.parametrize(
    "prompt",
    [
        # Regression: escalation words inside topic phrases and
        # negated requests escalated customers who never asked
        # for a person.
        "my password manager is broken",
        "the user agent string looks wrong",
        "what is human readable format?",
        # Regression: job-title topic phrases escalated customers
        # asking about a role, not requesting a person.
        "who is the product manager for this plan?",
        "can a travel agent book this for me?",
        "do not escalate, just answer my question",
        "no human needed, what are your hours?",
        "you can call me Priyanshu",
    ],
)
def test_topic_phrases_and_negations_do_not_escalate(prompt):
    assert not needs_human(prompt)


@pytest.mark.parametrize(
    "prompt",
    [
        # Regression: a negator anywhere earlier in the sentence
        # used to cancel a real request, so an unhappy customer
        # asking for a person was never handed over. Only a negator
        # governing the escalation phrase itself counts now.
        "I am not happy, get me a human",
        "I said no yesterday, please get a human",
        "no, I need a human",
        "my order has not arrived, can I talk to a human?",
    ],
)
def test_distant_negators_do_not_suppress_escalation(prompt):
    assert needs_human(prompt)


@pytest.mark.parametrize(
    "prompt",
    [
        "please do not get me a human, just answer here",
        "I don't want a human, the bot is fine",
        "never escalate this ticket",
    ],
)
def test_governing_negators_still_suppress_escalation(prompt):
    assert not needs_human(prompt)


def test_support_runtime_dev_end_to_end(make_config):
    runtime = GravionRuntime(make_config(mode="development"))
    out = runtime.support("how do I change my plan?")
    assert "how do I change my plan?" in out


def test_support_manifest_signs_and_verifies(issuer, support_module_copy):
    payload = build_manifest(
        support_module_copy,
        name="support",
        module_type="support",
        version="0.1.0",
        entrypoint="gravion_ai.modules.support.module:SupportModule",
    )
    write_signed_manifest(support_module_copy, sign_manifest(payload, issuer))
    verified = verify_module(support_module_copy, issuer)
    assert verified["name"] == "support"
    assert verified["type"] == "support"
    assert "module.py" in verified["files"]


def test_bundled_support_manifest_is_unsigned(issuer):
    import gravion_ai.modules.support as support_pkg
    from pathlib import Path

    with pytest.raises(ModuleTamperError):
        verify_module(Path(support_pkg.__file__).parent, issuer)


def _support_registry(path):
    registry = ModuleRegistry()
    registry.register(
        ModuleManifest(
            name="support",
            type=ModuleType.SUPPORT,
            version="0.1.0",
            entrypoint="gravion_ai.modules.support.module:SupportModule",
            status=ModuleStatus.IMPLEMENTED,
            path=path,
        )
    )
    return registry


def test_support_production_runs_with_license_and_signed_module(
    issuer, licensed_home, support_module_copy
):
    config, authority, _ = licensed_home(modules=("support",))
    payload = build_manifest(
        support_module_copy,
        name="support",
        module_type="support",
        version="0.1.0",
        entrypoint="gravion_ai.modules.support.module:SupportModule",
    )
    write_signed_manifest(support_module_copy, sign_manifest(payload, issuer))
    runtime = GravionRuntime(
        config,
        issuer_public=issuer,
        client=MockAuthorityClient(authority),
        registry=_support_registry(support_module_copy),
    )
    out = runtime.support("how do I reset my password?")
    assert "how do I reset my password?" in out


def test_support_production_refuses_tampered_module(
    issuer, licensed_home, support_module_copy
):
    config, authority, _ = licensed_home(modules=("support",))
    payload = build_manifest(
        support_module_copy,
        name="support",
        module_type="support",
        version="0.1.0",
        entrypoint="gravion_ai.modules.support.module:SupportModule",
    )
    write_signed_manifest(support_module_copy, sign_manifest(payload, issuer))
    target = support_module_copy / "module.py"
    target.write_text(target.read_text(encoding="utf-8") + "\n# patched\n", encoding="utf-8")
    runtime = GravionRuntime(
        config,
        issuer_public=issuer,
        client=MockAuthorityClient(authority),
        registry=_support_registry(support_module_copy),
    )
    with pytest.raises(ModuleTamperError):
        runtime.support("hello")
    events = [e["event"] for e in runtime.audit.entries()]
    assert "module.tamper" in events
