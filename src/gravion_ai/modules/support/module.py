"""SupportModule — the customer support bot (Piece 2).

The support module answers customer questions from the deployment's
own material: the company policy and facts that travel in the system
context (name + policy, built by config.system_context). It adds two
things of its own on top of the shared module path:

1. **Grounding rules.** A support addendum is appended to the system
   context: answer only from the material in context, never invent
   prices, timelines or policies, and say plainly when the answer is
   not in the material.

2. **Human escalation.** Some messages should never be answered by a
   bot. When the customer explicitly asks for a human (or a manager /
   supervisor / agent), the module does not call the model at all —
   it returns the escalation response immediately, so the handoff is
   deterministic and cannot be talked out of by the model.

Everything else follows the reference module's shape: manifest in,
`run(ctx, prompt)` out, model resolved by the Gravion registry
(Gravion Core serves support out of the box; Gravion Chat is the
trained support model from Piece 2's Tier 2 bundle).
"""

from __future__ import annotations

import re

from ...registry import ModuleContext, ModuleManifest

SUPPORT_ADDENDUM = """

# Support mode
You are handling customer support. Answer using only the company
policy and material in the context above. If the answer is not in
that material, say so plainly and offer to escalate to a human —
never invent prices, timelines, guarantees or policies. Be brief,
concrete and kind."""

# Phrases that mean "get me a person". Matched case-insensitively on
# word boundaries, so a phrase embedded in a longer word ("reagent",
# "managerial", "humane") never escalates a customer who never asked
# for a person. The list is deliberately narrow — it catches explicit
# requests for a human, not merely unhappy customers (the model, under
# the grounding rules, offers escalation for those).
ESCALATION_PHRASES = (
    "human",
    "real person",
    "actual person",
    "live person",
    "a person",
    "agent",
    "manager",
    "supervisor",
    "representative",
    "escalate",
    "escalation",
    "someone real",
    "talk to someone",
    "speak to someone",
    # "Speak with someone" asks for a person just as plainly
    # as "speak to someone"; with only the "to" forms listed,
    # the "with" request was answered by the bot instead of
    # handed over.
    "talk with someone",
    "speak with someone",
    "call me",
    # A callback asked for by its other verbs ("phone me
    # please", "ring me back") requests the same phone call
    # as "call me"; with only "call me" listed, those
    # customers were answered by the bot instead.
    "phone me",
    "ring me",
    # The callback asked for without the pronoun ("please
    # call back", "request a callback") and the handoff
    # asked for by its transfer verbs ("transfer me to
    # billing", "put me through") request the same person:
    # none of the phrases above matched, so the bot answered
    # customers who had asked for the team. The callback
    # number itself is a topic phrase below, not a request.
    "call back",
    "ring back",
    "a callback",
    "transfer me",
    "put me through",
    # The same request in Roman Hindi ("mujhe insaan se
    # baat karni hai" — I want to talk to a human): the
    # translation module covers Hinglish, so a Hinglish
    # customer asking for a person was answered by the bot
    # instead of handed over. A Hindi negator right after
    # the phrase ("insaan se baat nahi karni") cancels it
    # in _POSTPOSED_NEGATION_RE below.
    "insaan se baat",
    "aadmi se baat",
    # The same request without the talk verb ("insaan
    # chahiye" — I need a human; "kisi insaan ko bulao" —
    # call a human): with only the "se baat" forms listed,
    # those customers were answered by the bot instead.
    # The declined forms state no match ("insaan nahi
    # chahiye" breaks the phrase, and a trailing "nahi"
    # cancels it below).
    "insaan chahiye",
    "aadmi chahiye",
    "insaan ko bulao",
    "aadmi ko bulao",
)

ESCALATION_RESPONSE = (
    "Of course — I'm handing this over to a human on our support "
    "team. Your message has been passed on with its full context, so "
    "you won't need to explain it again. A person will follow up "
    "from here."
)


_ESCALATION_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(phrase) for phrase in ESCALATION_PHRASES) + r")\b"
)

# "Human resources" is a support topic, not a request for a person:
# the phrase is removed before matching so an HR-policy question is
# answered from the grounding material instead of escalated. The
# same holds for compound phrases where an escalation word is part
# of a product or topic name ("password manager", "user agent",
# "human readable") rather than a request for a person.
_TOPIC_PHRASES_RE = re.compile(
    r"\b(?:human resources|human readable|human error|human-like|"
    # "Human rights" and an "agent fee" name topics, not a
    # request for a person: both escalated a policy/price
    # question that the grounding material should answer. The
    # same holds for "human anatomy" and "human body"
    # questions, which escalated on the bare word "human".
    r"human rights|human anatomy|human body|"
    r"password manager|task manager|file manager|download manager|"
    r"user agent|ai agent|agent fee|agent fees|agent orange|"
    r"product manager|project manager|"
    r"account manager|travel agent|insurance agent|estate agent|"
    # An identity question ("are you human?", "am i talking
    # to a human?") asks what the bot is, not for a person:
    # it escalated on the bare word "human" while "are you a
    # robot?" stayed with the bot. A real request after it
    # ("are you human? get me a human") still escalates on
    # its own phrase.
    r"are you a human|are you human|is this a human|"
    r"am i talking to a human|"
    # A callback number is account data the grounding
    # material holds ("change my callback number"), not a
    # request for the call itself.
    r"sales representative|call ?back number)\b"
)

# "Call me <Name>" introduces the customer; it is not a request for
# a phone call. Matched on the original casing, before lowering.
# A lowercase name introduces the customer just the same ("call me
# priyanshu"), so the lowercase form is stripped too — unless the
# word after "call me" is a phone-request word ("call me about
# this", "call me tomorrow"), which keeps the escalation.
_CALL_ME_NAME_RE = re.compile(r"(?i:\bcall me) [A-Z][a-z]+\b")
_CALL_ME_LOWER_NAME_RE = re.compile(r"(?i:\bcall me) ([a-z]+)\b")
_CALL_ME_NON_NAMES = frozenset(
    {
        "about",
        "after",
        "again",
        "anytime",
        "asap",
        "at",
        "back",
        "before",
        "directly",
        "first",
        "if",
        "in",
        "later",
        "maybe",
        "next",
        "now",
        "on",
        "please",
        "sometime",
        "soon",
        "today",
        "tomorrow",
        "when",
    }
)


def _strip_call_me_name(prompt: str) -> str:
    text = _CALL_ME_NAME_RE.sub(" ", prompt)

    def _lower_repl(match: re.Match[str]) -> str:
        return " " if match.group(1) not in _CALL_ME_NON_NAMES else match.group(0)

    return _CALL_ME_LOWER_NAME_RE.sub(_lower_repl, text)

# A negated request ("do not escalate", "no human needed") asks for
# the opposite of escalation. A negator only cancels a match when it
# governs that match: it must sit in the same clause (commas break
# clauses too, so "I am not happy, get me a human" still escalates)
# and within the few words immediately before the escalation phrase
# ("please do not get me a human"). A distant negator earlier in the
# sentence ("I said no yesterday, please get a human") negates
# something else and must not suppress a real request.
# The Hindi negators govern the same way before the phrase
# ("nahi chahiye human", "mat karo escalate"): with only the
# English negators listed, a Hinglish customer declining a
# human in that order was handed over anyway. A comma still
# breaks the clause, so "nahi, human chahiye" escalates.
_NEGATOR_RE = re.compile(
    r"(?:\bno\b|\bnot\b|\bnever\b|\bwithout\b|\bnahi\b|\bnhi\b"
    r"|\bnahin\b|\bmat\b|n't\b)"
)
_CLAUSE_BREAKS = ".!?;,"
_NEGATION_WINDOW_WORDS = 4
# A negator can also follow the phrase it cancels ("a human is not
# needed", "human not needed"). It only cancels when it sits in
# the same clause right after the match and governs a
# needed/required word, so "I want a human, not a bot" and "a
# human not a bot" still escalate.
_POSTPOSED_NEGATION_RE = re.compile(
    r"\s*(?:(?:is|are|was|were)\s+)?(?:not|never)\s+"
    r"(?:needed|required|necessary|wanted)\b"
    r"|\s*(?:isn't|aren't|wasn't|weren't)\s+"
    r"(?:needed|required|necessary|wanted)\b"
    # Hindi negation follows the phrase it cancels
    # ("insaan se baat nahi karni", "human nahi chahiye",
    # "escalate mat karo"): without it a Hinglish customer
    # declining a human was handed over anyway.
    r"|\s+(?:nahi|nhi|nahin|mat)\b"
)


# A negator can govern the match through a transfer phrase
# ("do not connect me to a human", "don't put me through to
# an agent"): the negator sits five or more words before the
# person word, past the window above, yet it plainly cancels
# the request — those customers were handed over anyway. It
# only governs through the bridge when every word between
# the negator and the match is part of the transfer phrase
# itself, so "I do not know why but get me a human" (the
# negator governs "know") still escalates.
_TRANSFER_BRIDGE_WORDS = frozenset(
    {
        "a",
        "an",
        "connect",
        "directly",
        "ever",
        "get",
        "hand",
        "just",
        "me",
        "move",
        "need",
        "over",
        "pass",
        "please",
        "put",
        "really",
        "route",
        "send",
        "switch",
        "the",
        "through",
        "to",
        "transfer",
        "us",
        "with",
    }
)


def _is_negated(text: str, match_start: int, match_end: int) -> bool:
    clause_start = (
        max(text.rfind(break_char, 0, match_start) for break_char in _CLAUSE_BREAKS) + 1
    )
    words = text[clause_start:match_start].split()
    tail = " ".join(words[-_NEGATION_WINDOW_WORDS:])
    if bool(tail) and _NEGATOR_RE.search(tail) is not None:
        return True
    cleaned = [word.strip(".,!?;:\"'()") for word in words]
    for index in range(len(cleaned) - 1, -1, -1):
        word = cleaned[index]
        if word in (
            "no",
            "not",
            "never",
            "without",
            "nahi",
            "nhi",
            "nahin",
            "mat",
        ) or word.endswith("n't"):
            bridge = cleaned[index + 1 :]
            if bridge and all(part in _TRANSFER_BRIDGE_WORDS for part in bridge):
                return True
            break
    return _POSTPOSED_NEGATION_RE.match(text, match_end) is not None


# Asking to be put through to the team by its service name
# ("connect me with customer care", "talk to customer
# service", "get me an operator") requests a person just as
# plainly as "get me a human", yet none of the phrases above
# matched, so the bot answered a customer who had asked for
# the team. The transfer verb is required: a topic question
# naming the service ("what are your customer service
# hours?") states no request and stays with the bot.
# "Customer support" is the same service name ("talk to
# customer support"): with only care/service/team listed,
# that request was answered by the bot instead of handed
# over, while a topic question about it still states no
# request and stays with the bot.
_TEAM_REQUEST_RE = re.compile(
    r"\b(?:talk|speak|connect|transfer|chat|put me through)\b"
    r"[^.?!]*\b(?:customer care|customer service|customer support|support team"
    r"|technical support|tech support|sales team|billing team)\b"
    # A named department is the same human team ("talk to
    # sales", "connect me to billing"): with only the
    # support names listed, those customers were answered
    # by the bot instead of handed over. The transfer
    # preposition keeps a topic naming ("what does the
    # sales team do?") with the bot.
    r"|\b(?:talk|speak|connect|transfer|chat)\b"
    r"[^.?!]*\b(?:to|with) (?:sales|billing)\b"
    r"|\bget me an? operator\b"
    # The operator asked for without the "get me" frame
    # ("operator please", "i need an operator") requests
    # the same person.
    r"|\boperator please\b"
    r"|\b(?:need|want) an? operator\b"
    # The team named by its bare service word with a
    # transfer preposition ("connect me to support", "talk
    # to support") asks for the same handoff; the
    # preposition keeps the "chat support" topic phrase,
    # which states no request, with the bot.
    r"|\b(?:talk|speak|connect|transfer|chat)\b"
    r"[^.?!]*\b(?:to|with) support\b"
    # The same team request in Roman Hindi ("customer care se
    # baat karni hai", "support se baat karao") asks for the
    # team just as plainly; the "se baat" phrasing is the
    # request, so a topic naming of the service stays with
    # the bot, and a following "nahi" cancels it above.
    r"|\b(?:customer care|customer service|support team|support"
    r"|sales team|billing team|sales|billing) se baat\b"
    # A live-chat request with its transfer verb ("connect to
    # live chat", "start live chat") asks for the human chat
    # channel; the verb is required, so "how does live chat
    # work?" stays with the bot.
    r"|\b(?:connect|start|open|begin|switch|transfer)\b[^.?!]*\blive chat\b"
    r"|\blive chat please\b"
)


def needs_human(prompt: str) -> bool:
    """True when the customer explicitly asks for a human."""
    text = _strip_call_me_name(prompt)
    text = _TOPIC_PHRASES_RE.sub(" ", text.lower())
    for match in _ESCALATION_RE.finditer(text):
        if _is_negated(text, match.start(), match.end()):
            continue
        return True
    for match in _TEAM_REQUEST_RE.finditer(text):
        if _is_negated(text, match.start(), match.end()):
            continue
        return True
    return False


class SupportModule:
    def __init__(self, manifest: ModuleManifest):
        self.manifest = manifest

    def run(self, ctx: ModuleContext, prompt: str) -> str:
        if needs_human(prompt):
            return ESCALATION_RESPONSE
        messages = [
            {"role": "system", "content": ctx.system_context + SUPPORT_ADDENDUM},
            {"role": "user", "content": prompt},
        ]
        return ctx.provider.complete(messages, model=ctx.model)
