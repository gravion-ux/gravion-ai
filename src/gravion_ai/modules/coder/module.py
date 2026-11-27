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
generation and one that opens with an explanation verb ("explain
code review…") is explanation, even when a later noun phrase
mentions the other task; then a "code review" request or a review
or explain verb aimed at code ("can you review my code", "what
does this code do") anywhere in the request. A negated signal
does not count ("do not review, just explain this code" is an
explanation request). Anything else is generation. The rules
are small enough to state in a docstring
and pin in tests — no hidden heuristics.

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
# "Look over" / "go over" ask for a review just as plainly as
# "review" does ("look over this function please" fell through
# to generation, so a review request was answered with new code).
_REVIEW_RE = re.compile(
    r"\b(?:review|reviewing|audit|auditing|check|checking|"
    r"look over|go over)\b[^.?!]*\b"
    + _CODE_OBJECT
    + r"\b"
)

# A negated task verb asks for the opposite task ("do not
# review, just explain this code" is an explanation request).
# A negator only cancels a signal when it governs that signal:
# it must sit in the same clause (commas break clauses, so
# "do not write code, review this function" still reviews) and
# within the few words immediately before the signal, mirroring
# the support module's escalation negation.
_NEGATOR_RE = re.compile(r"(?:\bno\b|\bnot\b|\bnever\b|\bwithout\b|n't\b)")
_CLAUSE_BREAKS = ".!?;,"
_NEGATION_WINDOW_WORDS = 4


def _is_negated(text: str, match_start: int) -> bool:
    clause_start = (
        max(text.rfind(break_char, 0, match_start) for break_char in _CLAUSE_BREAKS) + 1
    )
    words = text[clause_start:match_start].split()
    tail = " ".join(words[-_NEGATION_WINDOW_WORDS:])
    return bool(tail) and _NEGATOR_RE.search(tail) is not None
_EXPLAIN_RES = (
    re.compile(
        r"\b(?:explain|explaining|describe|describing)\b[^.?!]*\b"
        + _CODE_OBJECT
        + r"\b"
    ),
    # "What does this function return?" asks what the code
    # does just as plainly as "...do?": with only do/mean the
    # return/output forms fell through to generation, so a
    # question about existing code was answered with new
    # code. The same holds for the "what is this code
    # doing?", "what this code does" and "walk me through
    # this code" shapes.
    re.compile(
        r"\bwhat does (?:this|the|it)\b[^.?!]*\b(?:do|mean|return|output|produce)\b"
    ),
    re.compile(
        r"\bwhat (?:will|would) (?:this|the|it)\b[^.?!]*\b(?:output|return|produce|print)\b"
    ),
    re.compile(r"\bwhat is (?:this|the|it)\b[^.?!]*\bdoing\b"),
    re.compile(r"\bwhat (?:this|the|it)\b[^.?!]*\b(?:does|do)\b"),
    re.compile(r"\bhow does (?:this|the|it)\b[^.?!]*\bwork\b"),
    re.compile(r"\bwalk\b[^.?!]*\bthrough\b[^.?!]*\b" + _CODE_OBJECT + r"\b"),
)


def detect_task(prompt: str) -> str:
    """Classify a coder prompt as generate / review / explain."""
    text = prompt.strip().lower()
    if text.startswith(_REVIEW_PREFIXES):
        return TASK_REVIEW
    if text.startswith(_EXPLAIN_PREFIXES):
        return TASK_EXPLAIN
    # A request opening with the noun phrase itself ("code review:
    # …") asks for a review; the verb "code" ("code a dashboard")
    # is a generation opener and is handled just below.
    if text.startswith("code review"):
        return TASK_REVIEW
    first_word = text.split(None, 1)[0].strip(",:;") if text else ""
    if first_word in _GENERATE_OPENERS:
        return TASK_GENERATE
    # An explanation verb opening the request ("explain code
    # review process") asks for an explanation, not a review of
    # the code named later.
    if first_word in ("explain", "describe"):
        return TASK_EXPLAIN
    # When both a review and an explanation signal are present,
    # the one the request states first governs: "can you explain
    # the code review process" asks for an explanation of the
    # process, while "do a code review and explain the issues"
    # asks for a review. The bare "code review" noun phrase is a
    # review signal at its own position. A negated signal does
    # not count at all: "do not review, just explain this code"
    # states the review only to cancel it, so the first
    # non-negated signal of each kind governs.
    review_at: int | None = None
    phrase_start = 0
    while True:
        phrase_at = text.find("code review", phrase_start)
        if phrase_at < 0:
            break
        if not _is_negated(text, phrase_at):
            review_at = phrase_at
            break
        phrase_start = phrase_at + 1
    for review_match in _REVIEW_RE.finditer(text):
        if _is_negated(text, review_match.start()):
            continue
        if review_at is None or review_match.start() < review_at:
            review_at = review_match.start()
        break
    explain_at: int | None = None
    for pattern in _EXPLAIN_RES:
        for explain_match in pattern.finditer(text):
            if _is_negated(text, explain_match.start()):
                continue
            if explain_at is None or explain_match.start() < explain_at:
                explain_at = explain_match.start()
            break
    if review_at is not None and explain_at is not None:
        return TASK_EXPLAIN if explain_at < review_at else TASK_REVIEW
    if review_at is not None:
        return TASK_REVIEW
    if explain_at is not None:
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
