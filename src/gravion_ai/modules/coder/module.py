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

# An algorithm, method, implementation or (in this module's
# context) file is supplied code just as a function is: "can
# you review this algorithm?", "check this method for bugs"
# and "is this algorithm correct?" fell through to generation,
# so a request to examine existing code was answered with new
# code instead of the problems in it. A loop or a line is
# supplied code the same way: "explain this loop" and "trace
# through this loop" fell through the same way.
_CODE_OBJECT = (
    r"(?:code|function|script|snippet|program|module|class"
    r"|algorithm|method|implementation|file|loops?|lines?)"
)
# "Look over" / "go over" ask for a review just as plainly as
# "review" does ("look over this function please" fell through
# to generation, so a review request was answered with new code).
# Debugging and correctness questions about supplied code ask
# for the same review ("debug this code", "find the bug in
# this function", "is this code correct?", "what's wrong with
# this code?"): without them they fell through to generation,
# so a request to examine existing code was answered with new
# code instead of the problems in it. Bug-hunt verbs beyond
# "find" ("look for bugs", "spot the error") and bug-presence
# questions ("is there a bug in this code?", "where is the
# bug in this function?", "does this code have any bugs?")
# ask for the same examination and fell through the same way.
# A pronoun object aims "debug" at the supplied code just as
# plainly ("can you debug this?"), a vulnerability hunt is a
# bug hunt ("find vulnerabilities in this code"), and a
# broken-code statement ("my code is broken", "this function
# throws an error") asks for the same examination with the
# code named before the problem instead of after it.
_REVIEW_RES = (
    re.compile(
        r"\b(?:review|reviewing|audit|auditing|check|checking|"
        r"look over|go over)\b[^.?!]*\b"
        + _CODE_OBJECT
        + r"\b"
    ),
    re.compile(r"\bdebug(?:ging)?\b[^.?!]*\b" + _CODE_OBJECT + r"\b"),
    re.compile(r"\bdebug(?:ging)?\s+(?:this|it)\b"),
    re.compile(
        r"\b(?:find|look for|search for|spot|hunt for)\b[^.?!]*"
        r"\b(?:bugs?|errors?|problems?|issues?|vulnerabilit(?:y|ies)|flaws?)\b"
    ),
    re.compile(
        r"\b" + _CODE_OBJECT + r"\b[^.?!]*"
        r"\b(?:is|are|was|were|keeps?|keep)?\s*"
        r"(?:broken|failing|fails?|not working|throws?|throwing|crashes?|crashing)\b"
    ),
    re.compile(
        r"\b" + _CODE_OBJECT + r"\b[^.?!]*"
        r"\bwhat(?:'s| is| went) wrong\b"
    ),
    re.compile(
        r"\b(?:is|are) there\b[^.?!]*\b(?:bugs?|errors?|problems?|issues?)\b"
        r"[^.?!]*\b" + _CODE_OBJECT + r"\b"
    ),
    re.compile(
        r"\bany\b[^.?!]*\b(?:bugs?|errors?|problems?|issues?)\b"
        r"[^.?!]*\b" + _CODE_OBJECT + r"\b"
    ),
    re.compile(
        r"\bwhere\b[^.?!]*\b(?:bugs?|errors?|problems?|issues?)\b"
        r"[^.?!]*\b" + _CODE_OBJECT + r"\b"
    ),
    re.compile(
        r"\bdoes (?:this|the|it)\b[^.?!]*\b" + _CODE_OBJECT
        + r"\b[^.?!]*\bhave\b[^.?!]*\b(?:bugs?|errors?|problems?|issues?)\b"
    ),
    re.compile(
        r"\bis (?:this|the|it)\b[^.?!]*\b"
        + _CODE_OBJECT
        + r"\b[^.?!]*\b(?:correct|secure|safe|valid|working|efficient|right)\b"
    ),
    # A compile/run question asks the same correctness verdict
    # as "does this code work?", and the future forms ("will
    # this code work?") ask it too: with only does/work listed,
    # "does this code compile?" fell through to generation.
    re.compile(
        r"\b(?:does|will|would|can) (?:this|the|it)\b[^.?!]*\b"
        + _CODE_OBJECT
        + r"\b[^.?!]*\b(?:work|compile|run)\b"
    ),
    # A speed question about supplied code asks for the
    # performance assessment review gives ("how fast is this
    # algorithm?", "how efficient is this code?"): without it
    # they fell through to generation.
    re.compile(
        r"\bhow (?:fast|efficient|slow)\b[^.?!]*\b" + _CODE_OBJECT + r"\b"
    ),
    re.compile(
        r"\bwhat(?:'s| is| went) wrong\b[^.?!]*\b" + _CODE_OBJECT + r"\b"
    ),
    # A wrong-behaviour statement names the code before its
    # problem the same way a broken-code statement does
    # ("this code gives the wrong output", "this script
    # returns None instead of a list", "the output is
    # wrong"): without them they fell through to generation,
    # so a request to examine existing code was answered
    # with new code instead of the problem in it.
    re.compile(
        r"\b" + _CODE_OBJECT + r"\b[^.?!]*"
        r"\b(?:gives?|returns?|outputs?|produces?)\b[^.?!]*"
        r"\b(?:wrong|incorrect|instead)\b"
    ),
    re.compile(
        r"\b(?:output|result)\b[^.?!]*\b(?:is|was)?\s*"
        r"(?:wrong|incorrect)\b"
    ),
    re.compile(
        r"\bwhat(?:'s| is| went) wrong\b[^.?!]*"
        r"\b(?:output|result)\b"
    ),
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
    # The print/log/display/show forms ask what the code does
    # just as plainly as the output form ("what does this code
    # print?"): with only do/mean/return/output/produce listed,
    # they fell through to generation.
    re.compile(
        r"\bwhat does (?:this|the|it)\b[^.?!]*"
        r"\b(?:do|mean|return|output|produce|print|log|display|show)\b"
    ),
    re.compile(
        r"\bwhat (?:will|would) (?:this|the|it)\b[^.?!]*"
        r"\b(?:output|return|produce|print|log|display|show)\b"
    ),
    re.compile(r"\bwhat is (?:this|the|it)\b[^.?!]*\bdoing\b"),
    re.compile(r"\bwhat (?:this|the|it)\b[^.?!]*\b(?:does|do)\b"),
    re.compile(
        r"\bhow does (?:this|the|it)\b[^.?!]*\b(?:work|handle|behave|run|process|fail)\b"
    ),
    # Why/when questions aimed at existing code ("why does
    # this code fail?", "when does this function run?") ask
    # for an explanation of that code just as plainly as
    # "what does it do?": without them they fell through to
    # generation, so a question about supplied code was
    # answered with new code. The "what is this function
    # for?" shape is the same ask.
    re.compile(
        r"\bwhy (?:does|is|did|would|do)\b[^.?!]*\b" + _CODE_OBJECT + r"\b"
    ),
    re.compile(
        r"\bwhen (?:does|is|will|would)\b[^.?!]*\b" + _CODE_OBJECT + r"\b"
    ),
    re.compile(
        r"\bwhat is (?:this|the|it)\b[^.?!]*\b"
        + _CODE_OBJECT
        + r"\b[^.?!]*\bfor\b"
    ),
    re.compile(
        r"\b(?:walk|trace|step)\b[^.?!]*\bthrough\b[^.?!]*\b" + _CODE_OBJECT + r"\b"
    ),
    # "What is the output of this code?" asks what the code
    # produces just as plainly as "what does this code
    # output?", and "what does return do in python?" asks
    # what a language concept does: both fell through to
    # generation, so a question about existing code or a
    # concept was answered with new code. A why-question
    # aimed at a failing pronoun subject ("why is this
    # failing?") is the same explanation ask as "why does
    # this code fail?" above.
    re.compile(
        r"\bwhat is the (?:output|result|return value)\b[^.?!]*\b"
        + _CODE_OBJECT
        + r"\b"
    ),
    re.compile(r"\bwhat does \w+ (?:do|mean)\b"),
    re.compile(
        r"\bwhy (?:is|does|did|would)\b[^.?!]*"
        r"\b(?:failing|broken|crashing|not working)\b"
    ),
    # An error is supplied code's artefact just as a
    # function is: "can you explain the error?", "why am I
    # getting this error?" and "walk through the error" ask
    # what the error means, and "help me understand this
    # code" / "break down this function" ask for the same
    # explanation as "explain this code": without them they
    # fell through to generation, so a question about
    # existing code was answered with new code.
    re.compile(
        r"\b(?:explain|explaining|describe|describing|understand|"
        r"understanding)\b[^.?!]*"
        r"\b(?:error|exception|traceback|message)\b"
    ),
    re.compile(
        r"\bunderstand(?:ing)?\b[^.?!]*\b" + _CODE_OBJECT + r"\b"
    ),
    re.compile(
        r"\bbreak\b[^.?!]*\bdown\b[^.?!]*\b" + _CODE_OBJECT + r"\b"
    ),
    re.compile(
        r"\bbreak\b[^.?!]*\b" + _CODE_OBJECT + r"\b[^.?!]*\bdown\b"
    ),
    re.compile(
        r"\bwhy\b[^.?!]*\b(?:error|exception)\b"
    ),
    re.compile(
        r"\b(?:walk|trace|step)\b[^.?!]*\bthrough\b[^.?!]*"
        r"\b(?:error|exception|traceback)\b"
    ),
    # "Why does this return None?" asks why the code
    # behaves as it does, the same explanation ask as
    # "why does this code fail?" above.
    re.compile(
        r"\bwhy (?:does|did|would)\b[^.?!]*"
        r"\b(?:return|give|output|produce)\b"
    ),
    # A complexity question asks what the supplied code costs
    # ("what is the time complexity of this code?"), a run
    # outcome question asks what executing it does ("what
    # happens when this code runs?", "how many times does this
    # loop run?"), a bare trace asks for the walk-through
    # without the word "through" ("trace this code"), and a
    # summary of supplied code is a short explanation of it:
    # without them they fell through to generation, so a
    # question about existing code was answered with new code.
    re.compile(
        r"\bwhat(?:'s| is) the (?:time |space )?complexity\b[^.?!]*\b"
        + _CODE_OBJECT
        + r"\b"
    ),
    re.compile(
        r"\bwhat happens\b[^.?!]*\b(?:run|runs|running|execute|executes)\b"
    ),
    re.compile(r"\bwhat happens\b[^.?!]*\b" + _CODE_OBJECT + r"\b"),
    re.compile(
        r"\bhow (?:many times|often)\b[^.?!]*\b" + _CODE_OBJECT + r"\b"
    ),
    re.compile(
        r"\bhow (?:many times|often) does (?:this|it)\b[^.?!]*"
        r"\b(?:run|execute|loop)\b"
    ),
    re.compile(r"\btrace\b[^.?!]*\b" + _CODE_OBJECT + r"\b"),
    re.compile(
        r"\b(?:summarize|summarise|summarizing|summarising)\b[^.?!]*\b"
        + _CODE_OBJECT
        + r"\b"
    ),
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
    for pattern in _REVIEW_RES:
        for review_match in pattern.finditer(text):
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
