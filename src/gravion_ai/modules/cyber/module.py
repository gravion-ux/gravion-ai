"""CyberModule — defensive security analysis, triage and reporting
(Piece 6).

The cyber module turns a plain-language security request into a
structured **cyber brief** for Gravion Shield, following the same
shape as the other modules (manifest in, `run(ctx, prompt)` out)
and adding deterministic logic of its own:

1. **Task detection.** A request is one of three tasks: *analyse*
   (understand an artefact — a log excerpt, an alert, a config, a
   code snippet), *triage* (decide priority and next steps for an
   alert or finding), or *report* (write the finding up). Triage
   and report are detected from explicit phrasing — a "triage:" /
   "report:" style prefix, or a request to triage / write up *this*
   or *the* item — mirroring the coder module's task detection.
   Anything else is analysis.

2. **Indicator extraction.** Indicators of compromise named in the
   request are extracted deterministically and recorded in the
   brief, so the model works from the exact artefacts instead of
   paraphrase: CVE ids, IPv4 addresses (octet-validated), SHA-256
   hashes and URLs, each deduplicated in request order. A stated
   severity ("severity: high") is recorded verbatim; an unstated
   one is recorded as not stated rather than guessed.

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

_CVE_RE = re.compile(r"CVE-\d{4}-\d{4,7}", re.IGNORECASE)
_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_SHA256_RE = re.compile(r"\b[0-9a-fA-F]{64}\b")
_URL_RE = re.compile(r"https?://[^\s\"'<>)\]]+")
_SEVERITY_RE = re.compile(
    r"severity[:\s]+(critical|high|medium|low)\b"
    r"|\b(critical|high|medium|low)\s+severity\b",
    re.IGNORECASE,
)


def detect_task(prompt: str) -> str:
    """Classify a cyber prompt as analyse / triage / report."""
    text = prompt.strip().lower()
    if text.startswith(_REPORT_PREFIXES):
        return TASK_REPORT
    if text.startswith(_TRIAGE_PREFIXES):
        return TASK_TRIAGE
    return TASK_ANALYSE


def _valid_ipv4(candidate: str) -> bool:
    return all(0 <= int(octet) <= 255 for octet in candidate.split("."))


def extract_indicators(prompt: str) -> dict[str, list[str]]:
    """Extract the indicators named in the request, by kind.

    Deterministic and conservative: CVE ids (normalised to upper
    case), IPv4 addresses whose octets are all valid, SHA-256
    hashes (normalised to lower case) and URLs — each deduplicated
    in request order. Kinds with no matches are omitted.
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
    collect(
        "ipv4",
        [m.group(0) for m in _IPV4_RE.finditer(prompt) if _valid_ipv4(m.group(0))],
    )
    collect("sha256", [m.group(0).lower() for m in _SHA256_RE.finditer(prompt)])
    collect("url", [m.group(0) for m in _URL_RE.finditer(prompt)])
    return indicators


def parse_severity(prompt: str) -> str | None:
    """Extract a stated severity ("severity: high" or "high
    severity"), or None."""
    match = _SEVERITY_RE.search(prompt)
    if match:
        return (match.group(1) or match.group(2)).lower()
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
