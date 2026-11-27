"""AllInOneModule — every module under one deployment (Piece 8).

The all-in-one module is the packaging of the whole platform: one
deployment, one model (Gravion Omni), every module. It does not
reimplement the other modules — it **routes** to them:

1. **Routing.** A request may name its module with a leading
   directive — `coder: fix this function`, `image: a logo for the
   bakery`, `cyber: triage: alert from 10.0.0.8`. The directive is
   stripped and the rest of the request is handed to that module,
   unchanged, with the same context (so the target module's own
   brief logic — the coder's task detection, the cyber module's
   indicator extraction — applies exactly as if it had been
   invoked directly). Directive names are case-insensitive and the
   space after the colon is optional (`coder:fix this`). A
   directive naming the all-in-one module itself is refused
   rather than routed in a circle.

   With no directive, a conservative intent check runs: a request
   with a clear verb-plus-object signal for exactly one module
   (``write a python function``, ``draw a logo``, ``build a
   website``, a CVE id to triage) routes to that module with the
   prompt unchanged. Anything ambiguous, tied, or without a clear
   signal goes to **chat**, the general module.

2. **Serving model.** The runtime only lets this module run on a
   model that serves *every* capability (Gravion Omni, or a custom
   model id the deployment's own endpoint understands) — a
   specialist model behind the all-in-one package would silently
   downgrade most modules, so `resolve_model` refuses it.

The submodules are instantiated from the default registry. Their
manifests are verified by the runtime's gate before this module
runs in production, exactly as when they are invoked directly.
"""

from __future__ import annotations

import re

from ...errors import ModuleError
from ...registry import ModuleContext, ModuleManifest

# Modules a directive may name: every module except this one.
ROUTEABLE = (
    "chat",
    "support",
    "coder",
    "image",
    "music",
    "video",
    "website-builder",
    "cyber",
    "agentic",
    "translation",
    "summarization",
    "retrieval",
    "data-analysis",
    "document-qa",
)

DEFAULT_ROUTE = "chat"

# Words that cancel a following intent verb ("do not draw",
# "never make a video", "don't write code").
_NEGATORS = frozenset({"not", "never", "no", "nor", "cannot"})

_DIRECTIVE_RE = re.compile(
    r"^\s*([a-z][a-z0-9-]*)\s*:\s*(.+)$", re.DOTALL | re.IGNORECASE
)
_TOKEN_RE = re.compile(r"[a-z0-9]+")
# Match the cyber module's CVE shape: sequence numbers run to
# four or more digits with no fixed cap, so an eight-digit id is
# just as decisive here as it is when cyber extracts it.
_CVE_RE = re.compile(r"(?<!\d)CVE-\d{4}-\d{4,}(?!\d)", re.IGNORECASE)

# Intent signals: (verbs, nouns) per module. A module scores one
# point per distinct verb hit and one per distinct noun hit; it
# needs at least two points and a strict lead over every other
# module, so a lone keyword never hijacks a general question.
_INTENT_SIGNALS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "coder": (
        frozenset({"write", "fix", "debug", "refactor", "implement", "code"}),
        frozenset(
            {
                "function",
                "script",
                "code",
                "program",
                "algorithm",
                "python",
                "javascript",
                "typescript",
                "rust",
                "golang",
                "regex",
            }
        ),
    ),
    "image": (
        frozenset({"draw", "generate", "create", "make", "design", "render", "paint"}),
        frozenset(
            {
                "image",
                "picture",
                "logo",
                "poster",
                "illustration",
                "photo",
                "portrait",
                "wallpaper",
                "icon",
                "banner",
            }
        ),
    ),
    "music": (
        frozenset({"compose", "make", "create", "write", "produce"}),
        frozenset(
            {"song", "music", "melody", "tune", "beat", "lyrics", "soundtrack", "jingle"}
        ),
    ),
    "video": (
        frozenset({"make", "create", "generate", "edit", "render", "film"}),
        frozenset({"video", "clip", "animation", "film", "reel"}),
    ),
    "website-builder": (
        frozenset({"build", "make", "create", "design"}),
        frozenset({"website", "webpage", "landing", "homepage"}),
    ),
    "cyber": (
        frozenset({"triage", "analyse", "analyze", "investigate"}),
        frozenset(
            {
                "alert",
                "incident",
                "malware",
                "phishing",
                "ransomware",
                "vulnerability",
                "ioc",
                "cve",
            }
        ),
    ),
    "support": (
        frozenset({"refund", "cancel", "track", "resolve"}),
        frozenset({"refund", "invoice", "billing", "subscription", "ticket", "warranty", "order"}),
    ),
    "agentic": (
        frozenset({"automate", "plan", "orchestrate"}),
        frozenset({"agent", "workflow", "automation", "steps"}),
    ),
    # The five text modules below were directive-only: without a
    # leading "<module>:" they could never be reached by intent,
    # so plain requests ("translate this into hindi", "summarize
    # this article") silently fell through to chat. Their nouns
    # stay narrow (a language, a summary object, a passage, a
    # statistic, a document) so a lone verb still cannot hijack
    # a general question.
    "translation": (
        frozenset({"translate", "convert"}),
        frozenset({"hindi", "english", "hinglish", "urdu", "translation"}),
    ),
    "summarization": (
        frozenset({"summarize", "summarise", "shorten", "condense"}),
        frozenset({"summary", "article", "paragraph", "passage"}),
    ),
    "retrieval": (
        frozenset({"search", "rank", "retrieve", "find"}),
        frozenset({"passage", "passages", "query", "documents"}),
    ),
    "data-analysis": (
        frozenset({"analyze", "analyse", "compute", "calculate"}),
        frozenset(
            {"mean", "median", "mode", "numbers", "statistics", "average", "dataset", "data"}
        ),
    ),
    "document-qa": (
        frozenset({"answer"}),
        frozenset({"document", "question"}),
    ),
}

# Structural input markers the text modules document as their
# input format. A prompt carrying both markers of a pair is a
# decisive signal on its own, like a code fence or a CVE id:
# "document: ... question: ..." can only be document Q&A, and
# "query: ... passage: ..." can only be retrieval.
_DOCUMENT_MARKER_RE = re.compile(r"^\s*document\s*:", re.IGNORECASE | re.MULTILINE)
_QUESTION_MARKER_RE = re.compile(r"^\s*question\s*:", re.IGNORECASE | re.MULTILINE)
_QUERY_MARKER_RE = re.compile(r"^\s*query\s*:", re.IGNORECASE | re.MULTILINE)
_PASSAGE_MARKER_RE = re.compile(r"^\s*passage\s*:", re.IGNORECASE | re.MULTILINE)
# The "summarize this:" / "summarize the following:" forms are
# the same decisive prefix with the object named: without them
# the request scored one intent point and fell through to chat,
# so the module's own input form could never reach it.
_SUMMARIZE_PREFIX_RE = re.compile(
    r"^\s*summar(?:ize|ise)(?:\s+(?:this|the following))?\s*:", re.IGNORECASE
)
# The translation module documents leading "translate:" and
# "translate to/into/in hindi|english:" forms; without a
# decisive check here they scored one intent point and fell
# through to chat, so the module's own input format could
# never reach it through the all-in-one package. The prompt
# is routed unchanged, exactly as the module expects it.
_TRANSLATE_PREFIX_RE = re.compile(
    r"^\s*translate\s*(?::|\s+(?:to|into|in)\s+(?:hindi|english)\s*:)",
    re.IGNORECASE,
)


def infer_route(prompt: str) -> str:
    """Infer a module from a clear verb-plus-object intent signal.

    Returns the single best-scoring module, or ``chat`` when no
    module scores at least two points with a strict lead. A code
    fence is a decisive coder signal and a CVE id is a decisive
    cyber signal on its own; the documented ``document:`` /
    ``question:`` and ``query:`` / ``passage:`` marker pairs and
    a leading ``summarize:`` or ``translate:`` are likewise
    decisive for their text modules.
    """
    if "```" in prompt:
        return "coder"
    if _CVE_RE.search(prompt):
        return "cyber"
    if _DOCUMENT_MARKER_RE.search(prompt) and _QUESTION_MARKER_RE.search(prompt):
        return "document-qa"
    if _QUERY_MARKER_RE.search(prompt) and _PASSAGE_MARKER_RE.search(prompt):
        return "retrieval"
    if _SUMMARIZE_PREFIX_RE.match(prompt):
        return "summarization"
    if _TRANSLATE_PREFIX_RE.match(prompt):
        return "translation"
    # A negated verb cancels its module: "do not draw a logo"
    # asks for an explanation, not an image, yet the bare
    # verb+noun signal routed it to image. Contractions are
    # expanded first ("don't" -> "do not") so the negator is a
    # token of its own; a verb counts as negated when a negator
    # stands within the two tokens before it ("do not draw",
    # "never make") — a wider window would let one clause's
    # negation cancel the next clause's verb.
    ordered = _TOKEN_RE.findall(prompt.lower().replace("n't", " not"))
    tokens = frozenset(ordered)
    negated: set[str] = set()
    for index, token in enumerate(ordered):
        if any(t in _NEGATORS for t in ordered[max(0, index - 2) : index]):
            negated.add(token)
    scores: dict[str, int] = {}
    for name, (verbs, nouns) in _INTENT_SIGNALS.items():
        if verbs & negated:
            continue
        score = len(tokens & verbs) + len(tokens & nouns)
        if score:
            scores[name] = score
    if not scores:
        return DEFAULT_ROUTE
    best = max(scores.values())
    leaders = [name for name, score in scores.items() if score == best]
    if best >= 2 and len(leaders) == 1:
        return leaders[0]
    return DEFAULT_ROUTE


def parse_route(prompt: str) -> tuple[str, str]:
    """Split a request into (module name, routed prompt).

    A leading `<module>:` directive naming a routeable module
    selects it; the all-in-one module itself is refused (routing
    in a circle helps no one). With no usable directive, a clear
    intent signal selects its module with the prompt unchanged;
    anything else — an unknown word before the colon, an empty
    remainder, an ambiguous request — goes to chat unchanged.
    """
    stripped = prompt.strip()
    match = _DIRECTIVE_RE.match(stripped)
    if match:
        name, rest = match.group(1).lower(), match.group(2).strip()
        if name == "all-in-one":
            raise ModuleError(
                "the all-in-one module cannot route to itself; name "
                "one of: " + ", ".join(ROUTEABLE)
            )
        if name in ROUTEABLE and rest:
            return name, rest
    return infer_route(stripped), stripped


class AllInOneModule:
    def __init__(self, manifest: ModuleManifest):
        self.manifest = manifest

    def run(self, ctx: ModuleContext, prompt: str) -> str:
        from ...registry import default_registry

        name, routed = parse_route(prompt)
        module = default_registry().instantiate(name)
        return module.run(ctx, routed)
