"""VideoModule — video generation and editing (Piece 4).

The video module turns a plain-language request into a structured
**video brief** for Gravion Video, following the same shape as the
other modules (manifest in, `run(ctx, prompt)` out) and adding
deterministic logic of its own:

1. **Mode detection.** A request is either *generate* (a new video
   from a description) or *edit* (change an existing video),
   detected from explicit phrasing exactly as the image module
   detects it — an "edit:" / "modify:" style prefix, or a request
   to edit/modify *this* or *the* video.

2. **Brief parsing.** Three pieces of intent are extracted:
   duration ("30 seconds", "2 minutes", as in the music module);
   aspect — an explicit pixel size wins, otherwise a named shape,
   where *vertical* and *portrait* mean 9:16, the shape of phone
   video; and reference frames — image files named in the request
   (".png" / ".jpg" / ".jpeg" / ".webp") that the video should be
   built from. Anything not stated is recorded in the brief as the
   model default rather than invented.

The brief is appended to the deployment's system context together
with the mode guidance. The module is served by **Gravion Video**,
whose first trained bundle ships at Tier 2 in this piece (see
data/models.json); the runtime's registry check refuses a model
that does not serve the video capability instead of silently
answering with the wrong model.

`run` is the model path. `render` is the development path: it hands
the request to the local render pipeline (see pipeline.py), which
produces a real, short MP4 offline — honestly labelled as the local
renderer, never as the trained model. Long-form output (up to ~1
hour) comes from the scene-planning and stitching pipeline that
ships with the trained bundle and remains planned in this piece.
"""

from __future__ import annotations

import re
from pathlib import Path

from ...registry import ModuleContext, ModuleManifest

MODE_GENERATE = "generate"
MODE_EDIT = "edit"

MODE_GUIDANCE = {
    MODE_GENERATE: """

# Video mode: generate
Create the video described in the brief below. Follow the
deployment's brand and policy context above — colours, tone and
content rules there govern every video this deployment produces.
Honour the brief's duration and aspect; if the brief records the
model default, choose what fits the subject. Where the brief names
reference frames, build the video from them.""",
    MODE_EDIT: """

# Video mode: edit
Edit the existing video as described in the brief below. Change
only what the request asks to change; preserve everything else —
footage, pacing and the deployment's brand elements — exactly as
it is. Follow the deployment's brand and policy context above.""",
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
# size in the request beats every named shape. Portrait video is
# vertical: 9:16, not the still-image 3:4.
_SHAPES = (
    ("square", "1:1"),
    ("vertical", "9:16"),
    ("portrait", "9:16"),
    ("landscape", "16:9"),
    ("banner", "21:9"),
)

_SIZE_RE = re.compile(r"(\d{3,5})\s*[x×]\s*(\d{3,5})")

_DURATION_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*-?\s*(hours?|hrs?|minutes?|mins?|seconds?|secs?)\b",
    re.IGNORECASE,
)
_UNIT_SECONDS = {"hour": 3600, "hr": 3600, "minute": 60, "min": 60, "second": 1, "sec": 1}

_FRAME_RE = re.compile(r"[\w./-]*[\w]\.(?:png|jpe?g|webp)\b", re.IGNORECASE)


def detect_mode(prompt: str) -> str:
    """Classify a video prompt as generate / edit."""
    text = prompt.strip().lower()
    if text.startswith(_EDIT_PREFIXES):
        return MODE_EDIT
    return MODE_GENERATE


def parse_duration(prompt: str) -> int | None:
    """Extract a requested duration in seconds, or None.

    Compound durations add up ("1 minute 30 seconds" is 90) and
    fractional values count ("1.5 minutes" is 90).
    """
    matches = _DURATION_RE.findall(prompt)
    if not matches:
        return None
    total = 0.0
    for value, unit in matches:
        total += float(value) * _UNIT_SECONDS[unit.lower().rstrip("s")]
    return round(total)


def parse_aspect(prompt: str) -> str | None:
    """Extract the requested aspect/size, or None for the default.

    An explicit pixel size ("1920x1080") is returned verbatim; a
    named shape returns its aspect ratio ("9:16" for vertical or
    portrait); anything else returns None, meaning the brief
    records the model default.
    """
    match = _SIZE_RE.search(prompt)
    if match:
        return f"{match.group(1)}x{match.group(2)}"
    text = prompt.lower()
    for shape, ratio in _SHAPES:
        if shape in text:
            return ratio
    return None


def parse_reference_frames(prompt: str) -> list[str]:
    """Collect the image files named in the request, in request
    order — the frames the video should be built from."""
    frames: list[str] = []
    for match in _FRAME_RE.finditer(prompt):
        frame = match.group(0)
        if frame not in frames:
            frames.append(frame)
    return frames


def build_brief(prompt: str) -> str:
    """Render the structured video brief for a request."""
    duration = parse_duration(prompt)
    aspect = parse_aspect(prompt)
    frames = parse_reference_frames(prompt)
    return (
        "\n\n# Video brief\n"
        f"Mode: {detect_mode(prompt)}\n"
        f"Duration: {f'{duration} seconds' if duration else 'model default'}\n"
        f"Aspect: {aspect if aspect else 'model default'}\n"
        f"Reference frames: {', '.join(frames) if frames else 'none'}\n"
        f"Request: {prompt.strip()}"
    )


class VideoModule:
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

    def render(self, prompt: str, out_path, video_provider=None):
        """Render a real MP4 for the request via the local pipeline.

        This is the development path (see pipeline.py): the output
        is a short clip from the local renderer, honestly labelled
        with its provider name — not output of the trained Gravion
        Video model, which serves requests through `run`.
        """
        from .pipeline import LocalVideoProvider, job_from_prompt

        job = job_from_prompt(prompt)
        provider = video_provider or LocalVideoProvider()
        return provider.render(job, Path(out_path))
