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
    # question that the grounding material should answer.
    r"human rights|"
    r"password manager|task manager|file manager|download manager|"
    r"user agent|ai agent|agent fee|agent fees|"
    r"product manager|project manager|"
    r"account manager|travel agent|insurance agent|estate agent|"
    r"sales representative)\b"
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
_NEGATOR_RE = re.compile(r"(?:\bno\b|\bnot\b|\bnever\b|\bwithout\b|n't\b)")
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
)


def _is_negated(text: str, match_start: int, match_end: int) -> bool:
    clause_start = (
        max(text.rfind(break_char, 0, match_start) for break_char in _CLAUSE_BREAKS) + 1
    )
    words = text[clause_start:match_start].split()
    tail = " ".join(words[-_NEGATION_WINDOW_WORDS:])
    if bool(tail) and _NEGATOR_RE.search(tail) is not None:
        return True
    return _POSTPOSED_NEGATION_RE.match(text, match_end) is not None


def needs_human(prompt: str) -> bool:
    """True when the customer explicitly asks for a human."""
    text = _strip_call_me_name(prompt)
    text = _TOPIC_PHRASES_RE.sub(" ", text.lower())
    for match in _ESCALATION_RE.finditer(text):
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
