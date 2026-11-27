"""CyberModule — defensive security analysis, triage and reporting
(Piece 6).

The cyber module turns a plain-language security request into a
structured **cyber brief** for Gravion Shield, following the same
shape as the other modules (manifest in, `run(ctx, prompt)` out)
and adding deterministic logic of its own:

1. **Task detection.** A request is one of three tasks: *analyse*
   (understand an artefact — a log excerpt, an alert, a config, a
   code snippet), *triage* (decide priority and next steps for an
   alert or finding), or *report* (write the finding up). Detection
   is deterministic, in priority order: an explicit "triage:" /
   "report:" style prefix; a request that opens with an analysis
   verb ("analyse this triage report") is analysis even when it
   later names a report or triage as a noun; then a report or
   triage verb aimed at an item or at a named indicator anywhere
   in the request ("can you triage this alert", "triage
   CVE-2026-12345", "could you write up this finding") —
   mirroring the coder module's task detection. A determiner
   directly before "report" or "triage" marks the noun instead
   ("the report mentions CVE-2026-12345" is discussion, not a
   request), while a creation verb ("file a report") still
   counts as a report request. A negated verb rules its task
   out ("do not triage this alert, just analyse it" is
   analysis). Anything else is analysis.

2. **Indicator extraction.** Indicators of compromise named in the
   request are extracted deterministically and recorded in the
   brief, so the model works from the exact artefacts instead of
   paraphrase: CVE ids, IPv4 addresses (octet-validated), IPv6
   addresses (validated and recorded in compressed form),
   SHA-256, SHA-1 and MD5 hashes and URLs, each deduplicated in
   request order. Defanged indicators as shared in defensive
   reporting ("10[.]0[.]0[.]8", "hxxp://…") are refanged and
   recorded in their canonical form, like the IPv6 compression
   above. A stated severity ("severity: high") is recorded
   verbatim; an unstated one is recorded as not stated rather
   than guessed.

The module is **defensive only**, by design (see ARCHITECTURE.md):
analysis, triage and reporting assistance for the deployment's own
security work. The brief is appended to the deployment's system
context together with the task guidance. The module is served by
**Gravion Shield**, whose first trained bundle ships at Tier 2 in
this piece (see data/models.json); the runtime's registry check
refuses a model that does not serve the cyber capability instead
of silently answering with the wrong model.
"""

from __future__ import annotations

import ipaddress
import re

from ...registry import ModuleContext, ModuleManifest

TASK_ANALYSE = "analyse"
TASK_TRIAGE = "triage"
TASK_REPORT = "report"

TASK_GUIDANCE = {
    TASK_ANALYSE: """

# Cyber task: analyse
Analyse the artefact in the brief below for the deployment's own
defensive security work. Say what the artefact shows, which parts
are suspicious and why, and what would confirm or clear each
suspicion. Work from the extracted indicators exactly as recorded.
Do not invent indicators, and do not provide instructions for
attacking third-party systems — this module is defensive only.""",
    TASK_TRIAGE: """

# Cyber task: triage
Triage the item in the brief below for the deployment's own
defensive security work. Give a verdict (action now / investigate /
monitor / dismiss), the reasoning behind it, the severity you
assess and why, and the concrete next steps in priority order.
Work from the extracted indicators exactly as recorded; where the
evidence is thin, say what is missing instead of guessing.""",
    TASK_REPORT: """

# Cyber task: report
Write the finding in the brief below up as a professional security
report for the deployment's own records: title, affected asset,
severity with reasoning, evidence (the extracted indicators,
exactly as recorded), impact, and remediation. Do not invent
indicators, assets or evidence that the brief does not contain.""",
}

_TRIAGE_PREFIXES = (
    "triage:",
    "triage this",
    "triage the",
    "please triage",
)

_REPORT_PREFIXES = (
    "report:",
    "report this",
    "report the",
    "write up",
    "write a report",
    "please report",
)

# CVE sequence numbers run to four or more digits with no fixed
# cap, so the pattern takes the whole sequence instead of
# truncating an eight-digit id to seven. The digit guards keep a
# candidate embedded in a longer digit run from matching in part.
_CVE_RE = re.compile(r"(?<!\d)CVE-\d{4}-\d{4,}(?!\d)", re.IGNORECASE)
# An IPv4 candidate must stand alone: the guards reject a
# four-group window inside a longer dotted sequence (a version
# string such as "1.2.3.4.5") while still allowing sentence
# punctuation after a real address ("from 10.0.0.8."). A quad
# glued to a letter ("v1.2.3.4") is a version label, not an
# address, so a preceding letter also rejects the candidate.
_IPV4_RE = re.compile(
    r"(?<!\d)(?<!\d\.)(?<![A-Za-z])\d{1,3}(?:\.\d{1,3}){3}(?!\d)(?!\.\d)"
)
# A defanged IPv4 ("10[.]0[.]0[.]8", "10(.)0(.)0(.)8") is how
# indicators arrive in shared defensive reporting: extracted
# nowhere, a triage request naming one recorded no indicator
# at all. Candidates are refanged and octet-validated exactly
# like plain addresses, and recorded in canonical form. The
# defanging is often partial — only the first dot neutralised
# to break linkification ("10[.]0.0.8") — so each dot may be
# plain or defanged; the guards mirror the plain pattern's, in
# every dot spelling, so a five-group chain ("1[.]2[.]3[.]4[.]5")
# still yields no four-group window.
_DEFANGED_DOT = r"(?:\.|\[\.\]|\(\.\))"
_DEFANGED_IPV4_RE = re.compile(
    r"(?<!\d)(?<!\d\.)(?<!\[\.\])(?<!\(\.\))(?<![A-Za-z])"
    r"\d{1,3}(?:" + _DEFANGED_DOT + r"\d{1,3}){3}"
    r"(?!\d)(?!\.\d)(?!\[\.\]\d)(?!\(\.\)\d)"
)
# An IPv6 candidate is a colon-hex sequence (full, "::"-compressed
# or IPv4-embedded); the guards keep a window inside a longer
# hex/colon run from matching in part while still allowing
# sentence punctuation after a real address. Candidates are
# validated with ipaddress before recording, so lookalikes that
# share the shape — clock readings ("1:05:30"), MAC addresses
# ("aa:bb:cc:dd:ee:ff") — record nothing.
_IPV6_RE = re.compile(
    r"(?<![0-9A-Fa-f:.])"
    r"(?:[0-9A-Fa-f]{0,4}:){2,7}"
    r"(?:[0-9A-Fa-f]{0,4}|(?:\d{1,3}\.){3}\d{1,3})"
    r"(?![0-9A-Fa-f:])(?!\.\d)"
)
_SHA256_RE = re.compile(r"\b[0-9a-fA-F]{64}\b")
# The word-boundary guards keep a shorter hash pattern from
# matching inside a longer hash: a 64-hex SHA-256 yields no
# 40- or 32-hex fragment, and a 40-hex SHA-1 yields no MD5.
_SHA1_RE = re.compile(r"\b[0-9a-fA-F]{40}\b")
_MD5_RE = re.compile(r"\b[0-9a-fA-F]{32}\b")
# Brackets and closing parens are allowed inside the match so a
# URL that legitimately contains them (an IPv6 literal, a
# parenthesised path segment) is recorded whole; _clean_url
# trims the unbalanced trailing ones that belong to the
# surrounding sentence instead. The scheme match is
# case-insensitive ("HTTPS://…"), like the defanged pattern
# below: schemes are case-insensitive, and an uppercase one
# recorded no URL at all.
_URL_RE = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)
# A defanged URL ("hxxp://evil[.]example.com/path") is the
# same artefact with the scheme and dots neutralised; it is
# refanged before recording. A plain URL carrying defanged
# dots ("https://evil[.]example.com") is refanged the same
# way, so one artefact is never recorded in two spellings.
_DEFANGED_URL_RE = re.compile(r"hxxps?://[^\s\"'<>]+", re.IGNORECASE)
# A stated severity arrives as "severity: high", "severity is
# high" or "(high|high-)severity"; all three are recorded. The
# same statement also arrives with a "level" or "rating" noun
# ("severity level is high", "severity rating: high", the
# hyphenated "severity-rating: high"), an "of" ("severity of
# high") or an "=" or "-" separator ("severity=high"), and
# "moderate" is recorded verbatim as the level it states. The
# copula form requires the level directly after "is"/"was"/"of"
# (allowing a rating verb, "severity is rated high",
# "severity is rated as high" / "severity is assessed as
# high"), so "severity is not high"
# and "severity is not rated high" state no level and record
# none. A parenthesised level ("severity (high)") and a copula
# with a short noun phrase in between ("severity of the
# finding is high") state a level just as plainly, and
# "informational"/"info" are recorded verbatim as the levels
# they state. The level-first form carries the same negation
# guard as the copula forms: "not high severity" and "isn't
# high severity" state no level, so a negation directly
# before the level refuses the match — "never" negates just
# as directly ("never high severity" recorded high). An
# intensifier between the copula and the level ("severity is
# quite high", "severity is extremely high", "severity is
# somewhat low") still states that level plainly.
_LEVEL = r"(critical|high|medium|moderate|low|informational|info)"
_SEV_NOUNS = r"(?:[\s-]+(?:level|rating))*"
_RATED = r"(?:rated\s+(?:as\s+)?|assessed\s+as\s+)?"
_INTENSIFIER = r"(?:quite\s+|very\s+|really\s+|extremely\s+|fairly\s+|somewhat\s+|rather\s+)?"
_SEVERITY_RE = re.compile(
    r"severity" + _SEV_NOUNS + r"[\s:=\-]+" + _RATED + _LEVEL + r"\b"
    r"|severity" + _SEV_NOUNS + r"\s+(?:is|was)\s+" + _RATED + _INTENSIFIER + _LEVEL + r"\b"
    r"|severity" + _SEV_NOUNS + r"\s+of\s+" + _LEVEL + r"\b"
    r"|severity\s+of\s+(?:the\s+)?\w+\s+(?:is|was)\s+" + _RATED + _INTENSIFIER + _LEVEL + r"\b"
    r"|severity" + _SEV_NOUNS + r"\s*\(\s*" + _LEVEL + r"\s*\)"
    r"|(?<!not )(?<!no )(?<!n't )(?<!never )\b" + _LEVEL + r"[\s-]severity\b",
    re.IGNORECASE,
)


# A request that opens with an analysis verb is analysis even when
# it later mentions a report or triage as a noun ("analyse this
# triage report", "what does this alert mean").
_ANALYSE_OPENERS = (
    "analyse",
    "analyze",
    "explain",
    "describe",
    "what",
    "how",
    "why",
)

_ITEM = r"(?:this|the|that|incident|finding|alert|issue|vulnerability|event)"
# A triage/report verb aimed straight at a named indicator
# ("triage CVE-2026-12345", "report 10.0.0.8") targets an item
# just as surely as one aimed at "this alert"; without this the
# request fell through to analysis. The text is already lowered,
# so the hash form needs no case flag.
_INDICATOR_HINT = (
    r"(?:cve-\d{4}-\d{4,}"
    r"|(?<![a-z])\d{1,3}(?:(?:\.|\[\.\]|\(\.\))\d{1,3}){3}"
    r"|[0-9a-f]{0,4}::[0-9a-f:]*|(?:[0-9a-f]{1,4}:){4,}[0-9a-f]{1,4}"
    r"|[0-9a-f]{64}|[0-9a-f]{40}|[0-9a-f]{32}|https?://\S+|hxxps?://\S+)"
)
# "Report" and "triage" are nouns as often as verbs: "the report
# mentions CVE-2026-12345" discusses a report, it does not ask
# for one. A determiner or possessive directly before the word
# marks the noun ("the report", "this triage report", "my
# report"), so the verb patterns refuse those positions; a
# creation verb ("write a report", "file a report") is checked
# separately and still counts as a report request.
_NOUN_PREFIX = (
    r"(?<!the )(?<!this )(?<!that )(?<!a )(?<!an )(?<!its )"
    r"(?<!their )(?<!his )(?<!her )(?<!my )(?<!your )(?<!our )"
)
_REPORT_RES = (
    re.compile(r"\bwrite up\b"),
    re.compile(
        r"\b(?:write|file|create|draft|prepare) (?:a|the|up)\b[^.?!]*\breport\b"
    ),
    re.compile(_NOUN_PREFIX + r"(?<!triage )\breport\b[^.?!]*\b" + _ITEM + r"\b"),
    re.compile(_NOUN_PREFIX + r"(?<!triage )\breport\b[^.?!]*" + _INDICATOR_HINT),
)
_TRIAGE_RE = re.compile(_NOUN_PREFIX + r"\btriage\b[^.?!]*\b" + _ITEM + r"\b")
_TRIAGE_INDICATOR_RE = re.compile(
    _NOUN_PREFIX + r"\btriage\b[^.?!]*" + _INDICATOR_HINT
)

# A negated task verb rules its task out, mirroring the agentic
# module's negated build/run verbs: "do not triage this alert,
# just analyse it" asks for analysis, and recording it as a
# triage brief handed the model the opposite task. A negator
# cancels a signal only when it governs it: same clause (commas
# break clauses) and within the few words immediately before
# the verb.
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


def _has_live_match(patterns, text: str) -> bool:
    return any(
        not _negated_before(text, match.start())
        for pattern in patterns
        for match in pattern.finditer(text)
    )


def detect_task(prompt: str) -> str:
    """Classify a cyber prompt as analyse / triage / report."""
    text = prompt.strip().lower()
    if text.startswith(_REPORT_PREFIXES):
        return TASK_REPORT
    if text.startswith(_TRIAGE_PREFIXES):
        return TASK_TRIAGE
    first_word = text.split(None, 1)[0].strip(",:;") if text else ""
    if first_word in _ANALYSE_OPENERS:
        return TASK_ANALYSE
    if _has_live_match(_REPORT_RES, text):
        return TASK_REPORT
    if _has_live_match((_TRIAGE_RE, _TRIAGE_INDICATOR_RE), text):
        return TASK_TRIAGE
    return TASK_ANALYSE


def _valid_ipv4(candidate: str) -> bool:
    return all(0 <= int(octet) <= 255 for octet in candidate.split("."))


def _normalised_ipv6(candidate: str) -> str | None:
    """Compressed form of a valid IPv6 candidate, else None."""
    try:
        return ipaddress.IPv6Address(candidate).compressed
    except ValueError:
        return None


def _refang(text: str) -> str:
    """Canonical form of a defanged indicator ("10[.]0[.]0[.]8"
    -> "10.0.0.8", "hxxps://…" -> "https://…").

    A URL's scheme and host are case-insensitive, so they are
    lowercased: "HTTPS://Evil.Example.COM/Path" and its plain
    spelling are one artefact and must dedupe to one record.
    The path and query keep their case — they are
    case-sensitive."""
    refanged = text.replace("[.]", ".").replace("(.)", ".")
    if refanged[:5].lower() == "hxxps":
        refanged = "https" + refanged[5:]
    elif refanged[:4].lower() == "hxxp":
        refanged = "http" + refanged[4:]
    scheme, sep, rest = refanged.partition("://")
    if not sep:
        return refanged
    authority_match = re.match(r"([^/?#]*)(.*)", rest, re.DOTALL)
    authority, tail = authority_match.group(1), authority_match.group(2)
    userinfo, at, hostport = authority.rpartition("@")
    if hostport.startswith("["):
        end = hostport.find("]")
        host, port = hostport[: end + 1], hostport[end + 1 :]
    else:
        host, colon, port_number = hostport.partition(":")
        port = colon + port_number
    authority = (userinfo + at if at else "") + host.lower() + port
    return scheme.lower() + sep + authority + tail


def _clean_url(url: str) -> str:
    """Trim sentence punctuation glued to a URL match.

    Trailing sentence punctuation ("…/path. Next …") is not part
    of the address. A closing paren or bracket is trimmed only
    while it is unbalanced — "(see https://example.com/path)"
    loses the sentence's paren, while a URL whose own path
    carries a balanced pair keeps it whole.
    """
    url = url.rstrip(".,;:!?")
    while url.endswith(")") and url.count(")") > url.count("("):
        url = url[:-1].rstrip(".,;:!?")
    while url.endswith("]") and url.count("]") > url.count("["):
        url = url[:-1].rstrip(".,;:!?")
    return url


def extract_indicators(prompt: str) -> dict[str, list[str]]:
    """Extract the indicators named in the request, by kind.

    Deterministic and conservative: CVE ids (normalised to upper
    case), IPv4 addresses whose octets are all valid (defanged
    forms refanged), IPv6 addresses (validated, normalised to
    compressed form), SHA-256, SHA-1 and MD5 hashes (normalised
    to lower case) and URLs (defanged forms refanged) — each
    deduplicated in request order. Kinds with no matches are
    omitted.
    """
    indicators: dict[str, list[str]] = {}

    def collect(kind: str, values: list[str]) -> None:
        seen: list[str] = []
        for value in values:
            if value not in seen:
                seen.append(value)
        if seen:
            indicators[kind] = seen

    collect("cve", [m.group(0).upper() for m in _CVE_RE.finditer(prompt)])
    ipv4_found = [
        (m.start(), m.group(0))
        for m in _IPV4_RE.finditer(prompt)
        if _valid_ipv4(m.group(0))
    ] + [
        (m.start(), _refang(m.group(0)))
        for m in _DEFANGED_IPV4_RE.finditer(prompt)
        if _valid_ipv4(_refang(m.group(0)))
    ]
    collect("ipv4", [value for _pos, value in sorted(ipv4_found)])
    collect(
        "ipv6",
        [
            normalised
            for m in _IPV6_RE.finditer(prompt)
            if (normalised := _normalised_ipv6(m.group(0))) is not None
        ],
    )
    collect("sha256", [m.group(0).lower() for m in _SHA256_RE.finditer(prompt)])
    collect("sha1", [m.group(0).lower() for m in _SHA1_RE.finditer(prompt)])
    collect("md5", [m.group(0).lower() for m in _MD5_RE.finditer(prompt)])
    url_found = [
        (m.start(), _refang(_clean_url(m.group(0))))
        for m in _URL_RE.finditer(prompt)
    ] + [
        (m.start(), _refang(_clean_url(m.group(0))))
        for m in _DEFANGED_URL_RE.finditer(prompt)
    ]
    collect("url", [value for _pos, value in sorted(url_found)])
    return indicators


def parse_severity(prompt: str) -> str | None:
    """Extract a stated severity ("severity: high", "severity is
    high", "severity level is high", "high severity" or
    "high-severity"), or None."""
    match = _SEVERITY_RE.search(prompt)
    if match:
        return next(group for group in match.groups() if group).lower()
    return None


def build_brief(prompt: str) -> str:
    """Render the structured cyber brief for a request."""
    indicators = extract_indicators(prompt)
    if indicators:
        rendered = "; ".join(
            f"{kind}: {', '.join(values)}"
            for kind, values in indicators.items()
        )
    else:
        rendered = "none extracted"
    severity = parse_severity(prompt)
    return (
        "\n\n# Cyber brief\n"
        f"Task: {detect_task(prompt)}\n"
        f"Indicators: {rendered}\n"
        f"Stated severity: {severity if severity else 'not stated'}\n"
        f"Request: {prompt.strip()}"
    )


class CyberModule:
    def __init__(self, manifest: ModuleManifest):
        self.manifest = manifest

    def run(self, ctx: ModuleContext, prompt: str) -> str:
        guidance = TASK_GUIDANCE[detect_task(prompt)]
        messages = [
            {
                "role": "system",
                "content": ctx.system_context + guidance + build_brief(prompt),
            },
            {"role": "user", "content": prompt},
        ]
        return ctx.provider.complete(messages, model=ctx.model)
