from pathlib import Path

import pytest

from gravion_ai.errors import ModuleError, ModuleTamperError
from gravion_ai.licensing.modules_sig import (
    build_manifest,
    sign_manifest,
    verify_module,
    write_signed_manifest,
)
from gravion_ai.modules.all_in_one.module import infer_route, parse_route
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
    "prompt,expected",
    [
        # Regression: the underscore spelling the module
        # packages use was not a directive, so the directive
        # text stayed inside the routed prompt.
        ("website_builder: a site for our cafe",
         ("website-builder", "a site for our cafe")),
        ("data_analysis: mean of 2 4 6", ("data-analysis", "mean of 2 4 6")),
        ("document_qa: document: x question: y",
         ("document-qa", "document: x question: y")),
    ],
)
def test_parse_route_underscore_directives(prompt, expected):
    assert parse_route(prompt) == expected


def test_parse_route_refuses_underscore_self_routing():
    with pytest.raises(ModuleError, match="cannot route to itself"):
        parse_route("all_in_one: coder: fix this")


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
        # Regression: "optimise this algorithm" scored the
        # coder noun alone and fell through to chat, and
        # "search for cats in this document" scored the
        # retrieval verb alone because only the plural
        # "documents" was listed as its noun.
        ("optimise this algorithm", "coder"),
        ("optimize this python function", "coder"),
        ("search for cats in this document", "retrieval"),
        ("tldr this passage about cats", "summarization"),
        # Regression: edit verbs on an image or video, the
        # cyber response verbs and artefact nouns, the
        # translated text itself, searched files and a
        # spreadsheet each scored one point and fell
        # through to chat.
        ("edit this photo", "image"),
        ("upscale this image", "image"),
        ("remove background from this image", "image"),
        ("trim this video", "video"),
        ("subtitle this video", "video"),
        ("dub this clip", "video"),
        ("scan this server", "cyber"),
        ("block this ip address", "cyber"),
        ("is this ip malicious?", "cyber"),
        ("translate this document", "translation"),
        ("search my files for budget", "retrieval"),
        ("analyze this spreadsheet", "data-analysis"),
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
        # Regression: the noun forms ("summarize the following
        # text:") scored one intent point ("text" is not an
        # intent noun) and fell through to chat.
        ("summarize the following text: Cats sleep for long hours each day.", "summarization"),
        ("summarize this passage: Cats sleep for long hours each day.", "summarization"),
        # Regression: the "tldr:" prefix asks for the same
        # summary in shorthand but scored no intent point.
        ("tldr: Cats sleep for long hours each day.", "summarization"),
        ("tl;dr: Cats sleep for long hours each day.", "summarization"),
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
        # Regression: a statistic named with its numbers scored
        # one intent point and fell through to chat, so the
        # data-analysis module's local computation never ran.
        ("what is the mean of 2, 4, 6?", "data-analysis"),
        ("find the median of 3 1 2", "data-analysis"),
        ("average of 10 20 30", "data-analysis"),
        # Guards: a statistic without supplied numbers stays a
        # general question, a coding request keeps its coder
        # signal, and a negated statistic is not computed.
        ("explain what median means", "chat"),
        ("write code to compute the mean of 2 4 6", "coder"),
        ("do not compute the mean of 2 4 6, just explain averages", "chat"),
    ],
)
def test_parse_route_structural_markers_are_decisive(prompt, expected_module):
    name, routed = parse_route(prompt)
    assert name == expected_module
    assert routed == prompt


def test_all_in_one_defaults_to_chat():
    module = default_registry().instantiate("all-in-one")
    out = module.run(_ctx(GravionCoreProvider()), "what can you do?")
    # Routed to chat, whose deterministic capabilities answer
    # replaces the old echo fallback (28 Nov 2026 cycle).
    assert "write and review code" in out
    assert "You said" not in out


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


def test_plural_and_security_intent_routes():
    # Regression: intent words matched raw, so plural requests
    # ("draw logos", "triage alerts") scored one point and fell
    # through to chat, and a plain security ask ("secure my
    # website against XSS") carried no cyber signal at all.
    assert infer_route("draw logos for my shop") == "image"
    assert infer_route("make songs about rain") == "music"
    assert infer_route("edit videos of my trip") == "video"
    assert infer_route("build websites for bakeries") == "website-builder"
    assert infer_route("triage alerts from the server") == "cyber"
    assert infer_route("summarize articles about cats") == "summarization"
    assert infer_route("calculate averages of datasets") == "data-analysis"
    assert infer_route("how do I secure my website against XSS?") == "cyber"
    assert infer_route("protect my app from SQL injection") == "cyber"
    assert infer_route("scan my server for vulnerabilities") == "cyber"
    # Negated and tied requests keep their existing handling.
    assert infer_route("do not draw logos") == "chat"
    assert infer_route("do not compute the averages of 2 4 6") == "chat"
    assert infer_route("build a secure website for a bakery") == "website-builder"
    assert infer_route("make videos and songs") == "chat"
    # The verb "means" is not the statistic "mean".
    assert infer_route("explain what median means") == "chat"


def test_review_and_explain_code_requests_route_to_coder():
    # Regression: review/explain/check verbs were not coder
    # intent signals, so "review this function" scored the
    # noun alone and fell through to chat, losing the coder
    # module's review/explain guidance.
    assert infer_route("review this function please") == "coder"
    assert infer_route("explain this function") == "coder"
    assert infer_route("describe this algorithm") == "coder"
    assert infer_route("check this function for errors") == "coder"
    assert infer_route("find the bug in this function") == "coder"
    # A lone verb or noun still cannot hijack a general
    # question, and a negated review verb routes nowhere.
    assert infer_route("review the report") == "chat"
    assert infer_route("explain recursion") == "chat"
    assert infer_route("do not review this code") == "chat"


def test_summary_text_and_media_signals_route():
    # Regression: "give me a summary of this text" scored
    # one summarization point, the translation module's
    # what-is/say shapes scored one translation point, and
    # sketch/animate/sing/site/sum requests each missed a
    # verb or noun, so all fell through to chat.
    assert infer_route("give me a summary of this text") == "summarization"
    assert infer_route("summarize this text") == "summarization"
    assert infer_route("what is hello in hindi?") == "translation"
    assert infer_route("say hello in hindi") == "translation"
    assert infer_route("sketch a portrait") == "image"
    assert infer_route("illustrate a story") == "image"
    assert infer_route("make a gif of a cat") == "video"
    assert infer_route("sing a song about rain") == "music"
    assert infer_route("build a site for a cafe") == "website-builder"
    assert infer_route("calculate the sum of 2, 4, 6") == "data-analysis"
    assert infer_route("what is the sum of 2, 4, 6?") == "data-analysis"
    assert infer_route("calculate the total of 2, 4") == "data-analysis"
    # "Sum up" asks for a summary, not a total, and a lone
    # verb still cannot hijack a general request.
    assert infer_route("sum up the meeting notes") == "chat"
    assert infer_route("draw a cat") == "chat"
    assert infer_route("animate a bouncing ball") == "chat"


def test_human_requests_and_order_status_route_to_support():
    # Regression: an explicit request for a human scored
    # no intent point and fell through to chat, losing the
    # support module's deterministic handoff.
    assert infer_route("i need a human agent") == "support"
    assert infer_route("get me your manager") == "support"
    assert infer_route("bot nahi chahiye") == "support"
    # An order-status question names no support verb.
    assert infer_route("where is my order") == "support"
    assert infer_route("mera order kahan hai") == "support"
    assert infer_route("order status please") == "support"
    # Topic questions and general requests stay with chat.
    assert infer_route("what are your customer service hours?") == "chat"
    assert infer_route("where is the capital of france?") == "chat"
