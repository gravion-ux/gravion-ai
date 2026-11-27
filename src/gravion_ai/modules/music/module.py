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
# The guard is shared with the moods below: a negated mood
# ("not calm, make it upbeat") is likewise the mood the request
# rules out, not one it asks for.
_SCORE_RES = tuple(
    re.compile(rf"\b{re.escape(phrase)}\b") for phrase in _SCORE_PHRASES
)
_JINGLE_RE = re.compile(r"\bjingle\b")
# "Without" rules a kind or mood out just as "not" does ("a
# track without dark tones" asks for no dark), the same negator
# the image, video and coder modules already carry; music was
# the one module missing it, so a ruled-out mood was recorded
# in the brief as a requested one.
_NEGATOR_RE = re.compile(r"(?:\bno\b|\bnot\b|\bnever\b|\bwithout\b|n't\b)")
_CLAUSE_BREAKS = ".!?;,"


def _negated_before(text: str, match_start: int) -> bool:
    clause_start = (
        max(text.rfind(break_char, 0, match_start) for break_char in _CLAUSE_BREAKS)
        + 1
    )
    words = text[clause_start:match_start].split()
    tail = " ".join(words[-3:])
    return bool(tail) and _NEGATOR_RE.search(tail) is not None


def _kind_named(pattern: re.Pattern[str], text: str) -> bool:
    # A "-free" suffix rules the kind out just as a negator does:
    # "a jingle-free track" names the jingle only to exclude it.
    return any(
        not _negated_before(text, match.start())
        and not text[match.end():].startswith("-free")
        for match in pattern.finditer(text)
    )

# The mood vocabulary the brief understands. Collected in the order
# they appear in the request; anything outside this list is left
# for the model to read from the request itself. Matching is on
# word boundaries: a mood word embedded in a longer word is not a
# mood ("swarm" is not "warm", "epicentre" is not "epic"). A
# negated mood is skipped under the same clause guard as the
# kinds, so "no dark tones, warm and playful" records warm and
# playful only; a mood named again without a negator still counts,
# positioned at that later mention.
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
    # The plain feeling words a request states first ("a
    # sad piano melody", "a cheerful jingle") recorded no
    # mood at all, so the brief handed the model a default
    # where the request had stated one.
    "sad",
    "happy",
    "cheerful",
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

# Word-form durations ("thirty seconds", "two minutes") state
# the same lengths as the digit forms and are how a request is
# usually spoken; the digit scan alone left them at the model
# default. The article forms count for minutes and hours only:
# "a minute long" is a duration, but "a second version" is an
# ordinal, not one second. "Half a minute" is thirty seconds,
# counted once — the article inside it is not a second minute.
_WORD_ONES = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19,
}
_WORD_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_WORD_NUM_RE = re.compile(
    r"\b((?:"
    + "|".join(_WORD_TENS)
    + r")(?:[-\s](?:one|two|three|four|five|six|seven|eight|nine))?|"
    + "|".join(_WORD_ONES)
    + r")\s*-?\s*(hours?|minutes?|seconds?)\b",
    re.IGNORECASE,
)
_ARTICLE_DURATION_RE = re.compile(
    r"\b(?:a|an)\s+(hours?|minutes?)\b", re.IGNORECASE
)
_HALF_DURATION_RE = re.compile(
    r"\bhalf\s+(?:an?\s+)?(hours?|minutes?)\b", re.IGNORECASE
)


def _word_number_value(text: str) -> int:
    parts = text.lower().replace("-", " ").split()
    return sum(
        _WORD_TENS.get(part, 0) + _WORD_ONES.get(part, 0) for part in parts
    )


def _word_duration_seconds(prompt: str) -> float:
    total = 0.0
    half_spans = []
    for match in _HALF_DURATION_RE.finditer(prompt):
        half_spans.append(match.span())
        total += 0.5 * _UNIT_SECONDS[match.group(1).lower().rstrip("s")]
    for match in _WORD_NUM_RE.finditer(prompt):
        total += _word_number_value(match.group(1)) * _UNIT_SECONDS[
            match.group(2).lower().rstrip("s")
        ]
    for match in _ARTICLE_DURATION_RE.finditer(prompt):
        if any(start <= match.start() < end for start, end in half_spans):
            continue  # the "a minute" inside "half a minute"
        total += _UNIT_SECONDS[match.group(1).lower().rstrip("s")]
    return total


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
    rather than adding to them. Word forms ("thirty seconds",
    "a minute long", "half a minute") count alongside the digit
    forms and add up the same way. The video module reuses this
    parser instead of carrying a second copy that could drift
    from it.
    """
    clock = _clock_seconds(prompt)
    if clock is not None:
        return clock
    total = _word_duration_seconds(prompt)
    matches = _DURATION_RE.findall(prompt)
    if not matches and not total:
        return None
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
    """Collect the brief's mood words, in request order.

    A negated mention ("not calm", "no dark tones") is the mood
    the request rules out, so it is skipped; the mood still counts
    when a later mention carries no negator.
    """
    text = prompt.lower()
    found = []
    for mood, pattern in _MOOD_RES:
        for match in pattern.finditer(text):
            if not _negated_before(text, match.start()):
                found.append((match.start(), mood))
                break
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
