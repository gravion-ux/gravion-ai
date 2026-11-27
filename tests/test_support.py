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
        # Regression: a negator governing a transfer phrase
        # sits past the negation window ("do not connect me
        # to a human"), so customers who refused a handoff
        # were handed over anyway.
        "do not connect me to a human",
        "don't transfer me to an agent",
        "please do not put me through to a human",
        "no need to connect me to a human",
    ],
)
def test_governing_negators_still_suppress_escalation(prompt):
    assert not needs_human(prompt)


def test_negator_governing_another_verb_does_not_suppress_transfer():
    # Guard for the transfer bridge above: a negator that
    # governs a different verb ("know") must not cancel the
    # real request that follows it in the same clause.
    assert needs_human("I do not know why but get me a human")


@pytest.mark.parametrize(
    "prompt",
    [
        # Regression: a lowercase name introduction escalated
        # ("call me priyanshu"), a negator after the phrase
        # ("a human is not needed") never cancelled it, and
        # topic phrases ("human-like", "ai agent", "account
        # manager") read as requests for a person.
        "you can call me priyanshu",
        "a human is not needed, just answer here",
        "human not needed thanks",
        "does the voice sound human-like?",
        "how does the ai agent work?",
        "my account manager sent an invoice",
    ],
)
def test_introductions_postposed_negations_and_topics_do_not_escalate(prompt):
    assert not needs_human(prompt)


@pytest.mark.parametrize(
    "prompt",
    [
        # Guards for the fixes above: phone-request words after
        # "call me" and "not a bot" after a real request still
        # escalate, and explicit person requests the phrase
        # list used to miss now hand over.
        "please call me tomorrow",
        "call me at 5pm",
        "I want a human, not a bot",
        "speak to a live person",
        "I want to speak to a representative",
        "can I talk to someone about my order",
        # Regression: the "with" forms of the someone-request
        # were missing, so "speak with someone" was answered
        # by the bot instead of handed over.
        "I want to speak with someone",
        "can I talk with someone about this?",
        # Guards for the topic phrases below: a bare agent
        # request still hands over.
        "get me an agent please",
        # Regression: a callback asked for as "phone me" or
        # "ring me" requests the same phone call as "call
        # me", yet never handed over.
        "phone me please",
        "ring me back when you can",
        # Regression: a Hinglish need-a-human request
        # without the talk verb, a named department, and
        # the operator asked for plainly all stayed with
        # the bot instead of handing over.
        "insaan chahiye",
        "aadmi chahiye",
        "kisi insaan ko bulao",
        "talk to sales",
        "connect me to billing",
        "sales team se baat karni hai",
        "billing se baat karni hai",
        "talk to technical support",
        "operator please",
        "i need an operator",
    ],
)
def test_phone_requests_and_person_requests_still_escalate(prompt):
    assert needs_human(prompt)


@pytest.mark.parametrize(
    "prompt",
    [
        # Regression: "human rights" and "agent fee" name
        # topics, not a request for a person, yet both
        # escalated a policy/price question.
        "this is about human rights",
        "what is your human rights policy?",
        "the agent fee is too high",
        "how are agent fees calculated?",
    ],
)
def test_rights_and_fee_topics_do_not_escalate(prompt):
    assert not needs_human(prompt)


@pytest.mark.parametrize(
    "prompt,expected",
    [
        # Regression: an identity question escalated on the
        # bare word "human" while "are you a robot?" stayed
        # with the bot; a declined Hinglish human and a
        # topic naming of a department state no request,
        # and a real request after an identity question
        # still hands over.
        ("are you human?", False),
        ("are you a human?", False),
        ("am i talking to a human?", False),
        ("insaan nahi chahiye", False),
        ("what does the sales team do?", False),
        ("are you human? get me a human", True),
    ],
)
def test_identity_questions_and_declined_humans(prompt, expected):
    assert needs_human(prompt) is expected


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


def test_team_requests_escalate_and_human_topics_do_not():
    # Regression: "connect me with customer care" and "talk to
    # customer service" asked for the team by its service name
    # yet never escalated, while "how does the human body
    # work?" escalated on the bare word "human".
    assert needs_human("connect me with customer care")
    assert needs_human("talk to customer service please")
    assert needs_human("speak to your support team")
    assert needs_human("get me an operator")
    # Regression: "customer support" is the same service
    # name as care/service/team, yet "talk to customer
    # support" never escalated; a topic question naming it
    # still states no request.
    assert needs_human("talk to customer support")
    assert needs_human("connect me with customer support")
    assert needs_human("put me through to customer support")
    assert not needs_human("what are your customer support hours?")
    assert not needs_human("do not connect me to customer support")
    # Regression: the team named by its bare service word
    # with a transfer preposition ("connect me to support")
    # asked for the same handoff yet never escalated.
    assert needs_human("connect me to support")
    assert needs_human("talk to support please")
    assert not needs_human("do not connect me to support")
    # A topic question naming the service states no request.
    assert not needs_human("what are your customer service hours?")
    assert not needs_human("how does chat support work?")
    assert not needs_human("do not connect me with customer care")
    # Regression: "agent orange" names a topic, not a
    # request for a person, yet escalated on "agent".
    assert not needs_human("agent orange history question")
    # Human topics are answered, not handed over.
    assert not needs_human("how does the human body work?")
    assert not needs_human("human anatomy question: what is a femur?")
    assert needs_human("talk to a real human please")


def test_hinglish_and_transfer_requests_escalate():
    # Regression: a Hinglish customer asking for a person
    # ("mujhe insaan se baat karni hai"), a callback asked
    # for without the pronoun ("please call back") and a
    # handoff asked for by its transfer verbs ("transfer
    # me to billing") never escalated; a Hindi negator
    # after the phrase ("human nahi chahiye") escalated a
    # customer declining a human.
    assert needs_human("mujhe insaan se baat karni hai")
    assert needs_human("aadmi se baat karwao")
    assert needs_human("transfer me to billing")
    assert needs_human("put me through")
    assert needs_human("please call back")
    assert needs_human("ring back please")
    assert needs_human("request a callback")
    assert not needs_human("mujhe insaan se baat nahi karni")
    assert not needs_human("human nahi chahiye")
    # Regression: a Hindi negator before the phrase ("nahi
    # chahiye human", "mat karo escalate") and the team named
    # in Roman Hindi ("customer care se baat karni hai")
    # were read the wrong way: the declining customer was
    # handed over and the requesting customer was not. A
    # verb-gated live-chat request asks for the human chat
    # channel the same way.
    assert not needs_human("nahi chahiye human")
    assert not needs_human("mat karo escalate")
    assert not needs_human("escalate mat karo")
    assert needs_human("nahi, human chahiye")
    assert needs_human("customer care se baat karni hai")
    assert needs_human("support se baat karao")
    assert not needs_human("customer care se baat nahi karni")
    assert needs_human("connect to live chat")
    assert needs_human("start live chat please")
    assert not needs_human("how does live chat work?")
    assert not needs_human("do not connect to live chat")
    assert not needs_human("what is the call back number")
    assert not needs_human("change my callback number")
    assert not needs_human("don't transfer me to an agent")
    # Regression: the callback asked for by its bare noun
    # ("callback please", "callback chahiye") and the call
    # asked for in Roman Hindi ("mujhe call karo") never
    # escalated, while the declined forms stay declined.
    assert needs_human("callback please")
    assert needs_human("callback chahiye")
    assert needs_human("phone call please")
    assert needs_human("mujhe call karo")
    assert needs_human("call kar do please")
    assert needs_human("manushya chahiye")
    assert needs_human("manushya se baat karni hai")
    assert not needs_human("callback nahi chahiye")
    assert not needs_human("call mat karo")


def test_bot_rejection_and_call_centre_escalate():
    # Regression: rejecting the bot asks for a person by
    # elimination, yet none of the escalation phrases
    # matched, so the bot kept answering.
    assert needs_human("i dont want a bot")
    assert needs_human("no bots please")
    assert needs_human("bot nahi chahiye")
    assert needs_human("bot se baat nahi karni")
    # The identity question and the bot-accepting form
    # state no rejection.
    assert not needs_human("are you a bot?")
    assert not needs_human("bot se hi baat karni hai")
    # The call centre is the same human team in both
    # languages.
    assert needs_human("call centre se baat karni hai")
    assert needs_human("talk to the call centre")
    assert needs_human("senior se baat karni hai")


def test_hinglish_send_and_wala_requests_escalate():
    # Regression: the person asked for by the send verb
    # ("insaan ko bhej do"), the senior by the call verb
    # ("senior ko bulao"), the team as its person
    # ("customer care wala chahiye"), and a phone
    # conversation asked for directly ("phone par baat
    # karni hai") never escalated, while declined and
    # topic forms stay with the bot.
    assert needs_human("insaan ko bhej do")
    assert needs_human("aadmi ko bhej do")
    assert needs_human("manushya ko bulao")
    assert needs_human("senior ko bulao")
    assert needs_human("senior ko bhej do")
    assert needs_human("customer care wala chahiye")
    assert needs_human("support wala chahiye")
    assert needs_human("help desk se baat karni hai")
    assert needs_human("complaint department se baat")
    assert needs_human("phone par baat karni hai")
    assert needs_human("kisi se baat karni hai")
    assert not needs_human("kisi se baat nahi karni")
    assert not needs_human("phone par baat nahi karni")
    assert not needs_human("customer care wala number kya hai")
    assert not needs_human("help desk ke hours kya hain")


def test_executive_requests_and_declined_callbacks():
    # Regression: the executive asked for by role
    # ("executive se connect karo") never escalated, the
    # singular topic "human resource policy" escalated on
    # the bare word "human", and a callback declined one
    # verb later ("call back karna mat") was handed over
    # anyway.
    assert needs_human("executive se connect karo")
    assert needs_human("executive se baat karni hai")
    assert needs_human("executive ko bulao")
    assert needs_human("call back karna hai")
    # Regression: the team lead and officer asked for by
    # role ("team lead se connect karo", "officer se baat
    # karni hai") never escalated, while naming either role
    # as a topic states no request.
    assert needs_human("team lead se connect karo")
    assert needs_human("team lead se baat karni hai")
    assert needs_human("officer se baat karni hai")
    assert needs_human("officer ko bulao")
    assert not needs_human("team lead ka naam kya hai")
    assert not needs_human("officer ki report bhej do")
    assert not needs_human("human resource policy kya hai")
    assert not needs_human("executive summary bhej do")
    assert not needs_human("call back karna mat")
    assert not needs_human("insaan se baat karna mat")
