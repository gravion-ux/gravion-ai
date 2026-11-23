"""ImageModule — image generation and editing (Piece 3).

The image module turns a plain-language request into a structured
**image brief** for Gravion Vision, following the same shape as the
other modules (manifest in, `run(ctx, prompt)` out) and adding two
pieces of deterministic logic of its own:

1. **Mode detection.** A request is either *generate* (a new image
   from a description) or *edit* (change an existing image). Edit is
   detected from explicit phrasing — an "edit:" / "modify:" style
   prefix, or a request to edit/modify *this* or *the* image —
   mirroring the coder module's task detection. Anything else is
   generation.

2. **Brief parsing.** Size intent is extracted from the request:
   an explicit pixel size ("1920x1080") wins; otherwise a named
   shape — square, portrait, landscape or banner — maps to the
   aspect ratio the brief records. No size intent means the brief
   records the model default rather than inventing one.

The brief is appended to the deployment's system context together
with the mode guidance, so the model receives the request already
structured: mode, aspect, and the brand context from the deployment
policy. The module is served by **Gravion Vision**, whose first
trained bundle ships at Tier 2 in this piece (see data/models.json);
the runtime's registry check refuses a model that does not serve
the image capability instead of silently answering with the wrong
model.
"""

from __future__ import annotations

import re

from ...registry import ModuleContext, ModuleManifest

MODE_GENERATE = "generate"
MODE_EDIT = "edit"

MODE_GUIDANCE = {
    MODE_GENERATE: """

# Image mode: generate
Create the image described in the brief below. Follow the
deployment's brand and policy context above — colours, tone and
content rules there govern every image this deployment produces.
Honour the brief's aspect ratio; if the brief records the model
default, choose the aspect that fits the subject.""",
    MODE_EDIT: """

# Image mode: edit
Edit the existing image as described in the brief below. Change
only what the request asks to change; preserve everything else —
subject, composition and the deployment's brand elements — exactly
as it is. Follow the deployment's brand and policy context above.""",
}

_EDIT_PREFIXES = (
    "edit:",
    "edit this",
    "edit the",
    "please edit",
    "modify:",
    "modify this",
    "modify the",
    "please modify",
)

# Named shapes, in the order they are checked. An explicit pixel
# size in the request beats every named shape.
_SHAPES = (
    ("square", "1:1"),
    ("portrait", "3:4"),
    ("landscape", "16:9"),
    ("banner", "21:9"),
)

_SIZE_RE = re.compile(r"(\d{3,5})\s*[x×]\s*(\d{3,5})")


def detect_mode(prompt: str) -> str:
    """Classify an image prompt as generate / edit."""
    text = prompt.strip().lower()
    if text.startswith(_EDIT_PREFIXES):
        return MODE_EDIT
    return MODE_GENERATE


def parse_aspect(prompt: str) -> str | None:
    """Extract the requested aspect/size, or None for the default.

    An explicit pixel size ("1920x1080") is returned verbatim; a
    named shape returns its aspect ratio ("16:9"); anything else
    returns None, meaning the brief records the model default.
    """
    match = _SIZE_RE.search(prompt)
    if match:
        return f"{match.group(1)}x{match.group(2)}"
    text = prompt.lower()
    for shape, ratio in _SHAPES:
        if shape in text:
            return ratio
    return None


def build_brief(prompt: str) -> str:
    """Render the structured image brief for a request."""
    aspect = parse_aspect(prompt)
    return (
        "\n\n# Image brief\n"
        f"Mode: {detect_mode(prompt)}\n"
        f"Aspect: {aspect if aspect else 'model default'}\n"
        f"Request: {prompt.strip()}"
    )


class ImageModule:
    def __init__(self, manifest: ModuleManifest):
        self.manifest = manifest

    def run(self, ctx: ModuleContext, prompt: str) -> str:
        guidance = MODE_GUIDANCE[detect_mode(prompt)]
        messages = [
            {
                "role": "system",
                "content": ctx.system_context + guidance + build_brief(prompt),
            },
            {"role": "user", "content": prompt},
        ]
        return ctx.provider.complete(messages, model=ctx.model)
