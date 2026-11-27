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
   aspect — an explicit pixel size wins, then an explicit ratio
   ("9:16"), otherwise a named shape, where *vertical* and
   *portrait* mean 9:16, the shape of phone video; and reference
   frames — image files named in the request
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
from ..music.module import parse_duration

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

# An edit verb aimed at a video anywhere in the request asks
# for an edit just as plainly as the prefixes above: "can you
# edit this video" and "edit my video please" fell through to
# generation, so an edit request was answered with a new
# video. "Trim"/"cut"/"shorten" name the same task when they
# govern a video noun through a determiner ("trim the intro",
# "cut this clip shorter"); without one, "cut" is as often a
# noun ("a director's cut video") as a verb. The pronoun form
# ("can you edit this please") counts for edit/modify.
# "Change"/"adjust"/"fix"/"crop"/"rotate" name the same task
# under the same determiner rule ("change my video's ending",
# "crop this video to vertical", "adjust the brightness of
# this clip") — without a determiner-governed video noun they
# describe new content ("a video about changing seasons")
# and stay generation. Adding something *to* an existing
# video ("add subtitles to this video") is likewise an edit;
# adding the video itself somewhere ("add this video to my
# playlist") names no video object after "to" and is not.
_VIDEO_OBJECT = r"(?:video|clip|reel|footage|intro|ending|demo)"
_EDIT_RES = (
    re.compile(
        r"\b(?:edit|modify)\b[^.?!]*\b" + _VIDEO_OBJECT + r"\b"
    ),
    re.compile(
        r"\b(?:trim|cut|shorten|change|adjust|fix|crop|rotate)\b[^.?!]*"
        r"\b(?:this|the|my|that)\s+(?:\w+\s+)?" + _VIDEO_OBJECT + r"\b"
    ),
    re.compile(
        r"\badd\b[^.?!]*\bto\s+(?:this|the|my|that)\s+"
        r"(?:\w+\s+)?" + _VIDEO_OBJECT + r"\b"
    ),
    re.compile(r"\b(?:edit|modify|trim|cut|fix|crop)\s+(?:this|that|it)\b"),
)

# A negated edit verb asks for a new video ("do not edit this
# video, make a new one"), so a negator governing the verb in
# its clause cancels it — the same guard the coder module
# gives its task verbs. Commas break clauses, so "do not
# generate, edit this video" still edits.
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
# size or ratio in the request beats every named shape. Portrait
# video is vertical: 9:16, not the still-image 3:4. Shapes match
# on word boundaries, and a negated shape ("not vertical,
# landscape please") is the shape the request rules out, so a
# negator governing the shape in its clause cancels it — the
# same guard the music module gives its kinds.
_SHAPES = (
    ("square", "1:1"),
    ("vertical", "9:16"),
    ("portrait", "9:16"),
    ("landscape", "16:9"),
    ("banner", "21:9"),
)
_SHAPE_RES = tuple(
    (re.compile(rf"\b{shape}s?\b"), ratio) for shape, ratio in _SHAPES
)

# "800 by 600" states a pixel size just as "800x600" does.
_SIZE_RE = re.compile(r"(\d{3,5})\s*(?:[x×]|\bby\b)\s*(\d{3,5})", re.IGNORECASE)

# An explicit ratio ("9:16", "4:5") is the canonical way to state
# a video aspect, and the render pipeline already resolves ratio
# strings to dimensions — but the parser never produced one, so
# "make a 9:16 video" recorded the model default and rendered
# 16:9. Parts are capped at 21 and the value must sit between
# 1:3 and 3:1, so clock readings ("10:30") and durations
# ("1:30") are not read as aspects; an am/pm suffix also marks
# a clock reading, never a ratio.
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

_FRAME_RE = re.compile(r"[\w./-]*[\w]\.(?:png|jpe?g|webp)\b", re.IGNORECASE)


def detect_mode(prompt: str) -> str:
    """Classify a video prompt as generate / edit."""
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
    beats an explicit ratio ("9:16"), which beats a named shape
    ("9:16" for vertical or portrait); anything else returns
    None, meaning the brief records the model default.
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
