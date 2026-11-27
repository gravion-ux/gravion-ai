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
   an explicit pixel size ("1920x1080") wins, then an explicit
   ratio ("4:5"); otherwise a named shape — square, portrait,
   landscape or banner — maps to the aspect ratio the brief
   records. No size intent means the brief
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

# An edit verb aimed at an image anywhere in the request asks
# for an edit just as plainly as the prefixes above: "can you
# edit this image", "edit my image" and "could you modify this
# picture" all fell through to generation, so an edit request
# was answered with a new image. "Change"/"retouch"/"adjust"
# aimed at an image noun ("change this photo background",
# "retouch this portrait") name the same task, as do
# "fix"/"crop"/"remove" ("fix this image", "crop this photo",
# "remove the background from this photo"). The pronoun form
# ("can you edit this please") counts for edit/modify/fix/crop.
# The transform verbs name the same task ("resize this
# image", "upscale this photo", "rotate this picture",
# "enhance this shot"), as do removing the background with no
# image noun named ("remove background" edits the image in
# hand) and a comparative aimed at the image ("make it
# brighter"): all fell through to generation.
_IMAGE_OBJECT = (
    r"(?:image|photo|picture|portrait|logo|banner|poster|icon|shot)"
)
_EDIT_RES = (
    re.compile(
        r"\b(?:edit|modify|change|retouch|adjust|fix|crop|remove"
        r"|resize|upscale|enhance|rotate|flip|restore|sharpen|blur"
        r"|brighten|darken|colorize|colourize)\b[^.?!]*\b"
        + _IMAGE_OBJECT
        + r"\b"
    ),
    re.compile(
        r"\b(?:edit|modify|fix|crop|resize|upscale|enhance|rotate"
        r"|flip|restore|sharpen|blur)\s+(?:this|that|it)\b"
    ),
    re.compile(r"\bremove\b[^.?!]*\bbackground\b"),
    # Changing the background works on the image in hand the
    # same way removing it does ("change the background to a
    # beach" fell through to generation).
    re.compile(r"\b(?:change|replace|swap)\b[^.?!]*\bbackground\b"),
    re.compile(
        r"\bmake (?:this|that|it) "
        r"(?:brighter|darker|sharper|bigger|smaller|clearer|blurrier)\b"
    ),
)

# A negated edit verb asks for a new image ("do not edit this
# image, draw a new one"), so a negator governing the verb in
# its clause cancels it — the same guard the coder module
# gives its task verbs. Commas break clauses, so "do not
# draw, edit this image" still edits.
_NEGATOR_RE = re.compile(r"(?:\bno\b|\bnot\b|\bnever\b|\bwithout\b|n't\b)")
_CLAUSE_BREAKS = ".!?;,"
_NEGATION_WINDOW_WORDS = 4


def _is_negated(text: str, match_start: int) -> bool:
    clause_start = (
        max(text.rfind(break_char, 0, match_start) for break_char in _CLAUSE_BREAKS)
        + 1
    )
    words = text[clause_start:match_start].split()
    tail = " ".join(words[-_NEGATION_WINDOW_WORDS:])
    return bool(tail) and _NEGATOR_RE.search(tail) is not None

# Named shapes, in the order they are checked. An explicit pixel
# size or ratio in the request beats every named shape. Shapes
# match on word boundaries (a shape buried inside a longer word
# is not a shape request), and a negated shape ("not square,
# make it landscape") is the shape the request rules out, so a
# negator governing the shape in its clause cancels it — the
# same guard the music module gives its kinds.
_SHAPES = (
    ("square", "1:1"),
    ("portrait", "3:4"),
    # "Vertical"/"horizontal" name the portrait/landscape
    # orientations in their other words ("vertical image"
    # recorded the model default while the video module
    # already read the same words).
    ("vertical", "3:4"),
    ("landscape", "16:9"),
    ("horizontal", "16:9"),
    ("banner", "21:9"),
)
_SHAPE_RES = tuple(
    (re.compile(rf"\b{shape}s?\b"), ratio) for shape, ratio in _SHAPES
)

# "800 by 600" states a pixel size just as "800x600" does.
_SIZE_RE = re.compile(r"(\d{3,5})\s*(?:[x×]|\bby\b)\s*(\d{3,5})", re.IGNORECASE)

# An explicit ratio ("4:5", "3:2") is the canonical way to state
# an image aspect, and resolve_size already turns ratio strings
# into dimensions — but the parser never produced one, so "a
# 4:5 poster" recorded the model default. Parts are capped at
# 21 and the value must sit between 1:3 and 3:1, so clock
# readings ("10:30") and durations ("1:30") are not read as
# aspects; an am/pm suffix also marks a clock reading.
_RATIO_RE = re.compile(
    r"(?<![0-9:])(\d{1,2})\s*:\s*(\d{1,2})(?![0-9:])(?!\s*(?:am|pm)\b)",
    re.IGNORECASE,
)


def _explicit_ratio(prompt: str) -> str | None:
    for match in _RATIO_RE.finditer(prompt):
        width, height = int(match.group(1)), int(match.group(2))
        if 1 <= width <= 21 and 1 <= height <= 21 and 1 / 3 <= width / height <= 3:
            return f"{width}:{height}"
    return None


def detect_mode(prompt: str) -> str:
    """Classify an image prompt as generate / edit."""
    text = prompt.strip().lower()
    if text.startswith(_EDIT_PREFIXES):
        return MODE_EDIT
    for pattern in _EDIT_RES:
        for match in pattern.finditer(text):
            if not _is_negated(text, match.start()):
                return MODE_EDIT
    return MODE_GENERATE


def parse_aspect(prompt: str) -> str | None:
    """Extract the requested aspect/size, or None for the default.

    An explicit pixel size ("1920x1080") is returned verbatim and
    beats an explicit ratio ("4:5"), which beats a named shape
    ("16:9" for landscape); anything else returns None, meaning
    the brief records the model default.
    """
    match = _SIZE_RE.search(prompt)
    if match:
        return f"{match.group(1)}x{match.group(2)}"
    ratio = _explicit_ratio(prompt)
    if ratio:
        return ratio
    text = prompt.lower()
    for pattern, ratio in _SHAPE_RES:
        for match in pattern.finditer(text):
            if not _is_negated(text, match.start()):
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


def resolve_size(prompt: str, width: int | None = None,
                 height: int | None = None) -> tuple[int, int]:
    """Resolve output dimensions for a local generation.

    Explicit width/height win; otherwise an explicit pixel size in
    the prompt is used, then an explicit ratio or a named shape's
    aspect at a 1024px long edge, then the 1024x1024 default.
    """
    if width is not None and height is not None:
        return width, height
    aspect = parse_aspect(prompt)
    if aspect and "x" in aspect:
        w, h = aspect.split("x", 1)
        return int(w), int(h)
    if aspect and ":" in aspect:
        rw, rh = (int(part) for part in aspect.split(":", 1))
        long_edge = 1024
        if rw >= rh:
            return long_edge, max(int(long_edge * rh / rw), 64)
        return max(int(long_edge * rw / rh), 64), long_edge
    return (width or 1024), (height or 1024)


class ImageModule:
    def __init__(self, manifest: ModuleManifest):
        self.manifest = manifest

    def generate_local(self, prompt: str, width: int | None = None,
                       height: int | None = None, quality: str = "high",
                       seed: int | None = None, output_path=None):
        """Generate an HQ image locally (no provider call).

        Returns the GeneratedImage; when output_path is given the
        PNG is also written there. Sizes follow resolve_size.
        """
        from .generator import generate_image

        w, h = resolve_size(prompt, width, height)
        result = generate_image(prompt, width=w, height=h,
                                quality=quality, seed=seed)
        if output_path is not None:
            result.save(output_path)
        return result

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
