"""MusicModule — tracks, jingles and background scores (Piece 3).

The music module turns a plain-language request into a structured
**music brief** for Gravion Audio, following the same shape as the
other modules (manifest in, `run(ctx, prompt)` out) and adding two
pieces of deterministic logic of its own:

1. **Kind detection.** A request is one of three kinds — a *jingle*
   (a short brand sting), a *score* (background music that sits
   under other content) or a *track* (a standalone piece, the
   default). Detection is keyword-based and deliberately simple,
   mirroring the coder module's task detection: "jingle" means
   jingle; background phrasing ("background music", "underscore",
   "soundtrack", "bgm") means score; anything else is a track.

2. **Brief parsing.** Duration intent ("30 seconds", "2 minutes")
   is extracted into seconds, and mood words from a small fixed
   vocabulary are collected in request order. Anything not stated
   is recorded in the brief as the model default rather than
   invented.

The brief is appended to the deployment's system context together
with the kind guidance. The module is served by **Gravion Audio**,
whose first trained bundle ships at Tier 2 in this piece (see
data/models.json); the runtime's registry check refuses a model
that does not serve the music capability instead of silently
answering with the wrong model.
"""

from __future__ import annotations

import re

from ...registry import ModuleContext, ModuleManifest

KIND_TRACK = "track"
KIND_JINGLE = "jingle"
KIND_SCORE = "score"

KIND_GUIDANCE = {
    KIND_TRACK: """

# Music mode: track
Compose the standalone track described in the brief below. Follow
the deployment's brand and policy context above. Honour the brief's
duration and mood where recorded; where the brief records the model
default, choose what fits the request.""",
    KIND_JINGLE: """

# Music mode: jingle
Compose a short, memorable brand jingle from the brief below:
a simple hook that carries the deployment's name or slogan from
the context above and can be recognised in the first seconds.
Keep it to the brief's duration; a jingle that overruns stops
being a jingle.""",
    KIND_SCORE: """

# Music mode: score
Compose background music from the brief below. It sits under
other content — speech or video — so keep it even and unobtrusive:
no sudden peaks, no hooks competing for attention, and a clean
loop point where the duration allows one.""",
}

_SCORE_PHRASES = (
    "background music",
    "background score",
    "underscore",
    "soundtrack",
    "bgm",
)

# The mood vocabulary the brief understands. Collected in the order
# they appear in the request; anything outside this list is left
# for the model to read from the request itself.
MOODS = (
    "upbeat",
    "calm",
    "corporate",
    "epic",
    "lofi",
    "lo-fi",
    "cinematic",
    "playful",
    "dark",
    "warm",
)

_DURATION_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*-?\s*(hours?|hrs?|minutes?|mins?|seconds?|secs?)\b",
    re.IGNORECASE,
)
_UNIT_SECONDS = {"hour": 3600, "hr": 3600, "minute": 60, "min": 60, "second": 1, "sec": 1}


def detect_kind(prompt: str) -> str:
    """Classify a music prompt as jingle / score / track."""
    text = prompt.lower()
    if "jingle" in text:
        return KIND_JINGLE
    if any(phrase in text for phrase in _SCORE_PHRASES):
        return KIND_SCORE
    return KIND_TRACK


def parse_duration(prompt: str) -> int | None:
    """Extract a requested duration in seconds, or None.

    Compound durations add up ("2 minutes 30 seconds" is 150),
    fractional values count ("1.5 minutes" is 90, not the 300 a
    whole-number scan reads from the "5"), and hours are a unit.
    """
    matches = _DURATION_RE.findall(prompt)
    if not matches:
        return None
    total = 0.0
    for value, unit in matches:
        total += float(value) * _UNIT_SECONDS[unit.lower().rstrip("s")]
    return round(total)


def parse_moods(prompt: str) -> list[str]:
    """Collect the brief's mood words, in request order."""
    text = prompt.lower()
    found = [mood for mood in MOODS if mood in text]
    return sorted(found, key=lambda mood: text.index(mood))


def build_brief(prompt: str) -> str:
    """Render the structured music brief for a request."""
    duration = parse_duration(prompt)
    moods = parse_moods(prompt)
    return (
        "\n\n# Music brief\n"
        f"Kind: {detect_kind(prompt)}\n"
        f"Duration: {f'{duration} seconds' if duration else 'model default'}\n"
        f"Mood: {', '.join(moods) if moods else 'model default'}\n"
        f"Request: {prompt.strip()}"
    )


class MusicModule:
    def __init__(self, manifest: ModuleManifest):
        self.manifest = manifest

    def run(self, ctx: ModuleContext, prompt: str) -> str:
        guidance = KIND_GUIDANCE[detect_kind(prompt)]
        messages = [
            {
                "role": "system",
                "content": ctx.system_context + guidance + build_brief(prompt),
            },
            {"role": "user", "content": prompt},
        ]
        return ctx.provider.complete(messages, model=ctx.model)
