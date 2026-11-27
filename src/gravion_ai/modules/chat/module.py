"""ChatModule — the one module implemented end-to-end in Piece 1.

It is deliberately small: it exists to prove the whole path — config →
system context (name + company policy) → model catalog resolution →
provider → entitlement gate → signed module verification — works, so
every later module piece follows the same shape.

One deterministic addition sits in front of that path, mirroring the
support module's escalation short-circuit: greetings, identity
questions ("who are you?") and capability questions ("what can you
do?") are answered by the module itself instead of the model. The
bundled local model is trained on Gravion documentation and has no
confident match for these opening lines, so they used to fall through
to the echo fallback — the first thing a new user (or a demo audience)
says came back as "[gravion-core] You said: ...". The short-circuit is
narrow: only a short greeting, an explicit identity question or an
explicit capability question matches, negated forms ("what can you
not do?") stay with the model, and everything else follows the
reference path unchanged.
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
    if _IDENTITY.search(text):
        return IDENTITY_RESPONSE
    if _CAPABILITIES.search(text):
        return CAPABILITIES_RESPONSE
    if _GREETING.match(text) and len(text.split()) <= 6:
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
