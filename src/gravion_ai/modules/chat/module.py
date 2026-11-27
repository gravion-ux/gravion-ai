"""ChatModule — the one module implemented end-to-end in Piece 1.

It is deliberately small: it exists to prove the whole path — config →
system context (name + company policy) → model catalog resolution →
provider → entitlement gate → signed module verification — works, so
every later module piece follows the same shape.

One deterministic addition sits in front of that path, mirroring the
support module's escalation short-circuit: greetings, identity
questions ("who are you?"), capability questions ("what can you
do?") and the other conversational lines a demo opens and closes
with ("how are you?", "who built you?", "thank you", "goodbye") are
answered by the module itself instead of the model. The bundled
local model is trained on Gravion documentation and has no confident
match for these lines, so they used to fall through to the echo
fallback — the first thing a new user (or a demo audience) says came
back as "[gravion-core] You said: ...". The short-circuit is narrow:
only a short greeting, an explicit identity, capability or builder
question, or a whole-message thanks or farewell matches, negated
forms ("what can you not do?") stay with the model, and everything
else follows the reference path unchanged.
"""

from __future__ import annotations

import re

from ...registry import ModuleContext, ModuleManifest

GREETING_RESPONSE = (
    "Hello, I am Gravion. Tell me what you would like to work on — "
    "a question, code, a website, a security review or an agent plan — "
    "and I will help step by step."
)

IDENTITY_RESPONSE = (
    "I am Gravion, a modular AI platform built by Priyanshu Kumar "
    "Thakur. One module system runs chat, support, coding, website "
    "building, image, music, video, cyber security and agentic "
    "tools, and the built-in Gravion models run fully offline."
)

CAPABILITIES_RESPONSE = (
    "I can chat and explain ideas, write and review code, build "
    "websites, create images, music and video briefs, analyse and "
    "triage security alerts, draft agent plans, translate and "
    "summarize text, answer questions from documents, and help with "
    "customer support. Ask for any of these and I will route it to "
    "the right Gravion module."
)

WELLBEING_RESPONSE = (
    "I am doing well and ready to help. How are you? Tell me what "
    "you would like to work on — a question, code, a website, a "
    "security review or an agent plan."
)

BUILDER_RESPONSE = (
    "Gravion was built by Priyanshu Kumar Thakur. It is a modular "
    "AI platform: one module system runs chat, support, coding, "
    "website building, image, music, video, cyber security and "
    "agentic tools, with built-in models that run fully offline."
)

ABOUT_RESPONSE = (
    "Gravion is a modular AI platform built by Priyanshu Kumar "
    "Thakur. Startups can build on its module system — chat, "
    "support, coding, website building, image, music, video, cyber "
    "security and agentic tools — instead of building each AI type "
    "from scratch, and the built-in Gravion models run fully "
    "offline with no API key."
)

AI_IDENTITY_RESPONSE = (
    "Yes — I am Gravion, an AI platform built by Priyanshu Kumar "
    "Thakur, not a human. I answer through Gravion modules for "
    "chat, code, websites, security and more, running fully "
    "offline."
)

LANGUAGE_RESPONSE = (
    "I can work in English and Roman Hindi here, and the "
    "translation module handles Hindi phrases. Ask in either and "
    "I will follow your language."
)

THANKS_RESPONSE = (
    "You are welcome. Ask for anything else — a question, code, a "
    "website, a security review or an agent plan — and I will help."
)

FAREWELL_RESPONSE = (
    "Goodbye, and thank you for trying Gravion. Come back any "
    "time — I will be ready to help."
)

_GREETING = re.compile(
    r"^(?:hello|hi|hey|namaste|greetings|good (?:morning|afternoon|evening))\b"
)
_IDENTITY = re.compile(
    r"\b(?:who are you|what is your name|what s your name|"
    r"introduce yourself|tell me about yourself|"
    r"your name|"
    # Roman Hindi identity asks (28 Nov 2026 cycle): a demo
    # audience in India opens with "tum kaun ho" as plainly as
    # "who are you", and it used to echo back unanswered.
    r"tum kaun ho|aap kaun hain|tumhara naam kya hai|"
    r"apna naam batao|tum kaun|aap kaun)\b"
)
_CAPABILITIES = re.compile(
    r"\b(?:what can you do|what do you do|your capabilities|"
    r"what are your capabilities|what are you able to do|"
    r"how can you help|what modules do you have|"
    # Roman Hindi capability asks, same reasoning as above:
    # "tum kya kar sakte ho" is "what can you do".
    r"tum kya kar sakte ho|aap kya kar sakte hain|"
    r"kya kar sakte ho|tum kya kya kar sakte ho)\b"
)
_NEGATION = re.compile(
    r"\b(?:not|never|no|cannot|can t|don t|nahi|nahin)\b"
)
# A bare help request ("can you help me today?") is the same
# opening as a capability question, but it carries no task, so
# it used to echo while "how can you help" answered. The match
# is anchored to the whole message: "can you help me write
# code" names a task and stays with the model.
_BARE_HELP = re.compile(r"^(?:can you help me|help me|please help me)(?: today| please)?$")
# Conversational lines a demo opens and closes with (28 Nov
# 2026 cycle, 03:16 run): "how are you?", "who built you?",
# "tell me about Gravion", "are you an AI?", "thank you" and
# "goodbye" all echoed under the default model while the
# greeting and identity lines answered. Each pattern stays
# narrow: builder and about asks name Gravion or its maker,
# wellbeing asks are short, and thanks and farewell match the
# whole message so "write a thank you note" stays a task.
_WELLBEING = re.compile(
    r"\b(?:how are you|how are you doing|how is it going|"
    r"kaise ho|aap kaise hain|kya haal hai|haal kaisa hai)\b"
)
_BUILDER = re.compile(
    r"\b(?:who built you|who made you|who created you|"
    r"who built gravion|who made gravion|who created gravion|"
    r"who is your creator|who is your maker|"
    r"tumhe kisne banaya|gravion kisne banaya)\b"
)
_ABOUT = re.compile(
    r"\b(?:what is gravion|tell me about gravion|"
    r"about gravion|gravion kya hai|gravion ke baare me batao)\b"
)
_AI_IDENTITY = re.compile(
    r"\b(?:are you an ai|are you a robot|are you human|"
    r"are you a human|are you real|kya tum ai ho|"
    r"tum ai ho kya)\b"
)
_LANGUAGE = re.compile(
    r"\b(?:can you speak hindi|do you speak hindi|"
    r"can you speak hinglish|hindi aati hai|"
    r"hindi bol sakte ho)\b"
)
_THANKS = re.compile(
    r"^(?:thank you|thanks|thanks a lot|thank you so much|"
    r"many thanks|dhanyavaad|shukriya|bahut dhanyavaad)"
    r"(?: very much| so much)?$"
)
_FAREWELL = re.compile(
    r"^(?:goodbye|bye|bye bye|see you|see you later|"
    r"good night|phir milenge|alvida)(?: soon| later)?$"
)


def _normalized(prompt: str) -> str:
    text = prompt.lower().replace("'", " ")
    return re.sub(r"[^a-z0-9 ]+", " ", text).strip()


def opening_response(prompt: str) -> str | None:
    """Deterministic reply for greeting/identity/capability openings.

    Returns None for everything else, including negated capability
    questions, so the model keeps answering real requests.
    """

    text = _normalized(prompt)
    if not text or _NEGATION.search(text):
        return None
    # A translation request that quotes an opening line ("translate
    # how are you into hindi") asks for the translation, not the
    # opening reply, so it stays with the model.
    if "translate" in text.split():
        return None
    words = len(text.split())
    if _BUILDER.search(text):
        return BUILDER_RESPONSE
    if _AI_IDENTITY.search(text):
        return AI_IDENTITY_RESPONSE
    if _ABOUT.search(text) and words <= 8:
        return ABOUT_RESPONSE
    if _IDENTITY.search(text):
        return IDENTITY_RESPONSE
    if _CAPABILITIES.search(text) or _BARE_HELP.match(text):
        return CAPABILITIES_RESPONSE
    if _LANGUAGE.search(text):
        return LANGUAGE_RESPONSE
    if _WELLBEING.search(text) and words <= 8:
        return WELLBEING_RESPONSE
    if _THANKS.match(text):
        return THANKS_RESPONSE
    if _FAREWELL.match(text):
        return FAREWELL_RESPONSE
    if _GREETING.match(text) and words <= 6:
        return GREETING_RESPONSE
    return None


class ChatModule:
    def __init__(self, manifest: ModuleManifest):
        self.manifest = manifest

    def run(self, ctx: ModuleContext, prompt: str) -> str:
        opening = opening_response(prompt)
        if opening is not None:
            return opening
        return ctx.chat(prompt)
