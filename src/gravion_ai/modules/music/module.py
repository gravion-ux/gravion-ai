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

# Kind phrases match on word boundaries, like the moods: "bgm"
# buried inside a longer word is not a score request. A negated
# kind ("not a jingle, a full track") is the kind the request
# rules out, so a negator governing the phrase in its clause
# cancels it — the same guard the agentic module gives tools.
_SCORE_RES = tuple(
    re.compile(rf"\b{re.escape(phrase)}\b") for phrase in _SCORE_PHRASES
)
_JINGLE_RE = re.compile(r"\bjingle\b")
_KIND_NEGATOR_RE = re.compile(r"(?:\bno\b|\bnot\b|\bnever\b|n't\b)")
_CLAUSE_BREAKS = ".!?;,"


def _kind_negated(text: str, match_start: int) -> bool:
    clause_start = (
        max(text.rfind(break_char, 0, match_start) for break_char in _CLAUSE_BREAKS)
        + 1
    )
    words = text[clause_start:match_start].split()
    tail = " ".join(words[-3:])
    return bool(tail) and _KIND_NEGATOR_RE.search(tail) is not None


def _kind_named(pattern: re.Pattern[str], text: str) -> bool:
    return any(
        not _kind_negated(text, match.start())
        for match in pattern.finditer(text)
    )

# The mood vocabulary the brief understands. Collected in the order
# they appear in the request; anything outside this list is left
# for the model to read from the request itself. Matching is on
# word boundaries: a mood word embedded in a longer word is not a
# mood ("swarm" is not "warm", "epicentre" is not "epic").
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
    r"(\d+(?:\.\d+)?)\s*-?\s*"
    r"(hours?|hrs?|h|minutes?|mins?|m|seconds?|secs?|s)(?![a-z])",
    re.IGNORECASE,
)
_UNIT_SECONDS = {
    "hour": 3600, "hr": 3600, "h": 3600,
    "minute": 60, "min": 60, "m": 60,
    "second": 1, "sec": 1, "s": 1,
}

# Clock-form durations ("1:30", "1:02:30") state the same length
# as the unit forms. A two-part clock is read as a duration only
# when it cannot be an aspect ratio: the image/video ratio
# parsers accept parts up to 21 with a value between 1:3 and
# 3:1, so "9:16" stays an aspect while "1:30" (30 > 21) and
# "0:45" are durations. A three-part clock is never a ratio.
_CLOCK3_RE = re.compile(r"(?<![0-9:])(\d+):([0-5]\d):([0-5]\d)(?![0-9:])")
_CLOCK2_RE = re.compile(r"(?<![0-9:])(\d+):([0-5]\d)(?![0-9:])")


def _clock_seconds(prompt: str) -> int | None:
    total = 0
    found = False
    for match in _CLOCK3_RE.finditer(prompt):
        hours, minutes, seconds = (int(part) for part in match.groups())
        total += hours * 3600 + minutes * 60 + seconds
        found = True
    for match in _CLOCK2_RE.finditer(prompt):
        minutes, seconds = int(match.group(1)), int(match.group(2))
        if 1 <= minutes <= 21 and 1 <= seconds <= 21 and 1 / 3 <= minutes / seconds <= 3:
            continue  # an aspect ratio, not a duration
        total += minutes * 60 + seconds
        found = True
    return total if found else None


def detect_kind(prompt: str) -> str:
    """Classify a music prompt as jingle / score / track."""
    text = prompt.lower()
    if _kind_named(_JINGLE_RE, text):
        return KIND_JINGLE
    if any(_kind_named(pattern, text) for pattern in _SCORE_RES):
        return KIND_SCORE
    return KIND_TRACK


def parse_duration(prompt: str) -> int | None:
    """Extract a requested duration in seconds, or None.

    Compound durations add up ("2 minutes 30 seconds" is 150),
    fractional values count ("1.5 minutes" is 90, not the 300 a
    whole-number scan reads from the "5"), hours are a unit, and
    the compact forms count too ("30s", "1m 30s", "1m30s").
    Clock forms ("1:30", "1:02:30") state the length precisely,
    so a clock reading wins over unit forms in the same request
    rather than adding to them. The video module reuses this
    parser instead of carrying a second copy that could drift
    from it.
    """
    clock = _clock_seconds(prompt)
    if clock is not None:
        return clock
    matches = _DURATION_RE.findall(prompt)
    if not matches:
        return None
    total = 0.0
    for value, unit in matches:
        key = unit.lower()
        if key not in _UNIT_SECONDS:
            key = key.rstrip("s")
        total += float(value) * _UNIT_SECONDS[key]
    return round(total)


_MOOD_RES = tuple(
    (mood, re.compile(r"\b" + re.escape(mood) + r"\b")) for mood in MOODS
)


def parse_moods(prompt: str) -> list[str]:
    """Collect the brief's mood words, in request order."""
    text = prompt.lower()
    found = [
        (match.start(), mood)
        for mood, pattern in _MOOD_RES
        if (match := pattern.search(text)) is not None
    ]
    return [mood for _start, mood in sorted(found)]


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
