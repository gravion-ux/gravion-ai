"""CoderModule — the coding assistant (Piece 2).

The coder module writes, reviews and explains code. It follows the
reference module's shape (manifest in, `run(ctx, prompt)` out) and
adds one piece of its own logic: **task detection**. A coding request
is one of three tasks — generate, review, explain — and each task
gets its own guidance appended to the deployment's system context,
so the same model behaves as a generator, a reviewer or a teacher
depending on what was actually asked.

Detection is deterministic and deliberately simple, in priority
order: an explicit "review:" / "explain:" style prefix; a request
that opens with a generation verb ("write code to explain…") is
generation; then a review or explain verb aimed at code ("can you
review my code", "what does this code do") anywhere in the
request. Anything else is generation. The rules are small enough
to state in a docstring and pin in tests — no hidden heuristics.

The coder module is served by **Gravion Code**, whose first trained
bundle ships at Tier 2 in this piece (see data/models.json). A
deployment therefore configures its model to `gravion-code` (or uses
its own endpoint model id) before running coder; the runtime's
registry check refuses a model that does not serve the coder
capability instead of silently answering with the wrong model.
"""

from __future__ import annotations

import re

from ...registry import ModuleContext, ModuleManifest

TASK_GENERATE = "generate"
TASK_REVIEW = "review"
TASK_EXPLAIN = "explain"

TASK_GUIDANCE = {
    TASK_GENERATE: """

# Coder mode: generate
Write the code the user asked for, in the stack and style of the
codebase in context. Return complete, runnable code — no placeholders,
no TODO stubs — with a one-line note on how to run or test it.""",
    TASK_REVIEW: """

# Coder mode: review
Review the code the user supplied. Lead with correctness and security
problems, then clarity and performance. Cite the exact lines you mean,
explain why each point matters, and show the corrected snippet for
anything you flag. Do not rewrite the whole file unless asked.""",
    TASK_EXPLAIN: """

# Coder mode: explain
Explain the code the user supplied so a competent developer new to
this codebase can follow it: what it does, the flow of control and
data, and any non-obvious choices or pitfalls. Use the code's own
names; do not change the code.""",
}

_REVIEW_PREFIXES = ("review:", "review this", "review the", "please review")
_EXPLAIN_PREFIXES = ("explain:", "explain this", "explain the", "please explain")

# A request that opens with a generation verb is generation even
# when it later mentions explaining or reviewing ("write code to
# explain how sorting works" asks for code, not an explanation).
_GENERATE_OPENERS = (
    "write",
    "create",
    "build",
    "implement",
    "generate",
    "make",
    "develop",
    "code",
    "add",
    "fix",
    "refactor",
)

_CODE_OBJECT = r"(?:code|function|script|snippet|program|module|class)"
_REVIEW_RE = re.compile(
    r"\b(?:review|audit|check)\b[^.?!]*\b" + _CODE_OBJECT + r"\b"
)
_EXPLAIN_RES = (
    re.compile(r"\b(?:explain|describe)\b[^.?!]*\b" + _CODE_OBJECT + r"\b"),
    re.compile(r"\bwhat does (?:this|the|it)\b[^.?!]*\b(?:do|mean)\b"),
    re.compile(r"\bhow does (?:this|the|it)\b[^.?!]*\bwork\b"),
)


def detect_task(prompt: str) -> str:
    """Classify a coder prompt as generate / review / explain."""
    text = prompt.strip().lower()
    if text.startswith(_REVIEW_PREFIXES) or "code review" in text:
        return TASK_REVIEW
    if text.startswith(_EXPLAIN_PREFIXES):
        return TASK_EXPLAIN
    first_word = text.split(None, 1)[0].strip(",:;") if text else ""
    if first_word in _GENERATE_OPENERS:
        return TASK_GENERATE
    if _REVIEW_RE.search(text):
        return TASK_REVIEW
    if any(pattern.search(text) for pattern in _EXPLAIN_RES):
        return TASK_EXPLAIN
    return TASK_GENERATE


class CoderModule:
    def __init__(self, manifest: ModuleManifest):
        self.manifest = manifest

    def run(self, ctx: ModuleContext, prompt: str) -> str:
        guidance = TASK_GUIDANCE[detect_task(prompt)]
        messages = [
            {"role": "system", "content": ctx.system_context + guidance},
            {"role": "user", "content": prompt},
        ]
        return ctx.provider.complete(messages, model=ctx.model)
