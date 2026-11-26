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
    "a person",
    "agent",
    "manager",
    "supervisor",
    "escalate",
    "escalation",
    "someone real",
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
# answered from the grounding material instead of escalated.
_HUMAN_RESOURCES_RE = re.compile(r"\bhuman resources\b")


def needs_human(prompt: str) -> bool:
    """True when the customer explicitly asks for a human."""
    text = _HUMAN_RESOURCES_RE.sub(" ", prompt.lower())
    return _ESCALATION_RE.search(text) is not None


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
