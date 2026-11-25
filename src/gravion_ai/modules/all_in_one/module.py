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
   invoked directly). With no directive, the request goes to
   **chat**, the general module. A directive naming the all-in-one
   module itself is refused rather than routed in a circle.

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
)

DEFAULT_ROUTE = "chat"

_DIRECTIVE_RE = re.compile(r"^\s*([a-z][a-z0-9-]*)\s*:\s+(.+)$", re.DOTALL)


def parse_route(prompt: str) -> tuple[str, str]:
    """Split a request into (module name, routed prompt).

    A leading `<module>:` directive naming a routeable module
    selects it; the all-in-one module itself is refused (routing
    in a circle helps no one). Anything else — no directive, an
    unknown word before the colon, an empty remainder — goes to
    chat with the prompt unchanged.
    """
    match = _DIRECTIVE_RE.match(prompt.strip())
    if match:
        name, rest = match.group(1), match.group(2).strip()
        if name == "all-in-one":
            raise ModuleError(
                "the all-in-one module cannot route to itself; name "
                "one of: " + ", ".join(ROUTEABLE)
            )
        if name in ROUTEABLE and rest:
            return name, rest
    return DEFAULT_ROUTE, prompt.strip()


class AllInOneModule:
    def __init__(self, manifest: ModuleManifest):
        self.manifest = manifest

    def run(self, ctx: ModuleContext, prompt: str) -> str:
        from ...registry import default_registry

        name, routed = parse_route(prompt)
        module = default_registry().instantiate(name)
        return module.run(ctx, routed)
