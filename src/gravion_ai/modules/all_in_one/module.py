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

_DIRECTIVE_RE = re.compile(
    r"^\s*([a-z][a-z0-9-]*)\s*:\s*(.+)$", re.DOTALL | re.IGNORECASE
)
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_CVE_RE = re.compile(r"\bcve-\d{4}-\d{4,7}\b", re.IGNORECASE)

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
}


def infer_route(prompt: str) -> str:
    """Infer a module from a clear verb-plus-object intent signal.

    Returns the single best-scoring module, or ``chat`` when no
    module scores at least two points with a strict lead. A code
    fence is a decisive coder signal and a CVE id is a decisive
    cyber signal on its own.
    """
    if "```" in prompt:
        return "coder"
    if _CVE_RE.search(prompt):
        return "cyber"
    tokens = frozenset(_TOKEN_RE.findall(prompt.lower()))
    scores: dict[str, int] = {}
    for name, (verbs, nouns) in _INTENT_SIGNALS.items():
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
