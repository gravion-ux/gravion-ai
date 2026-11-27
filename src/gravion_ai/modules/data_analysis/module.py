"""DataAnalysisModule — descriptive statistics, computed locally.

Numbers supplied in the prompt (comma, space or line separated, or
CSV rows) are parsed and summarized with real arithmetic: count,
sum, mean, median, mode, range, standard deviation and a simple
first-half/second-half trend. Nothing is estimated or invented.
Requests without numbers are answered by the resolved model (Gravion
Analyst), whose trained weights cover analysis guidance.
"""

from __future__ import annotations

import re
import statistics
from collections import Counter

from ...registry import ModuleContext, ModuleManifest

# One numeric token: optional sign, then either a grouped integer
# (Western "1,000,000" or Indian "1,00,000" digit grouping, or
# underscore grouping "1_000_000" as written in code), a plain
# integer/decimal, or a leading-dot decimal (".5"), with an optional
# scientific-notation exponent ("2.5e3"). A comma only joins a
# number when it sits inside a valid digit group, so comma-space
# separated lists ("4, 8, 15") still parse as separate values. The
# leading lookbehind keeps a minus glued to a preceding digit or
# letter from becoming a sign: in a range ("5-10") or a date
# ("2026-11-27") the hyphen is a separator, so both sides parse as
# positive values, and digits inside a word ("version2") are not
# numbers at all. Without the underscore alternative "1_000" read
# as the phantom pair 1 and 0 and skewed every statistic.
_NUMBER_RE = re.compile(
    r"(?<![0-9A-Za-z])-?(?:\d{1,3}(?:,\d{3})+|\d{1,2}(?:,\d{2})+,\d{3}"
    r"|\d+(?:_\d+)+|\d+|\.\d+)"
    r"(?:\.\d+)?(?:[eE][+-]?\d+)?"
)


# A dotted sequence with two or more dots ("3.14.15", an IPv4
# address) is a version or address label, never a measurement:
# left in place, the number pattern read "3.14.15" as the pair
# 3.14 and 15 and inflated every statistic. Such sequences are
# blanked out before parsing; a plain decimal ("3.14") has one
# dot and is untouched.
_DOTTED_SEQUENCE_RE = re.compile(r"\d+(?:\.\d+){2,}")

# Machine literals ("0xFF", "0b101", "0o17") are code, not
# decimal measurements: left in place, "0x10" read as the
# phantom value 0. They are blanked out before parsing, like
# dotted labels above.
_MACHINE_LITERAL_RE = re.compile(r"\b0[xX][0-9a-fA-F]+\b|\b0[bB][01]+\b|\b0[oO][0-7]+\b")


def parse_numbers(prompt: str) -> list[float]:
    cleaned = _MACHINE_LITERAL_RE.sub(" ", prompt)
    cleaned = _DOTTED_SEQUENCE_RE.sub(" ", cleaned)
    return [
        float(m.replace(",", "").replace("_", ""))
        for m in _NUMBER_RE.findall(cleaned)
    ]


def _mode_line(numbers: list[float]) -> str:
    """Honest mode: tied modes are all named, and a list where
    every value appears once has no mode instead of reporting the
    first value as one."""
    counts = Counter(numbers)
    mode_count = max(counts.values())
    if mode_count == 1:
        return "Mode: none (every value appears once)"
    modes = sorted(value for value, seen in counts.items() if seen == mode_count)
    if len(modes) == 1:
        return f"Mode: {modes[0]:g} (appears {mode_count} times)"
    listed = ", ".join(f"{value:g}" for value in modes)
    return f"Mode: {listed} (each appears {mode_count} times)"


def analyze(numbers: list[float]) -> str:
    if not numbers:
        return "No numbers to analyze."
    count = len(numbers)
    total = sum(numbers)
    mean = total / count
    median = statistics.median(numbers)
    low, high = min(numbers), max(numbers)
    stdev = statistics.pstdev(numbers) if count > 1 else 0.0
    half = count // 2
    if half:
        first = sum(numbers[:half]) / half
        second = sum(numbers[half:]) / (count - half)
        if second > first:
            trend = "rising (second half averages higher than the first)"
        elif second < first:
            trend = "falling (second half averages lower than the first)"
        else:
            trend = "flat (both halves average the same)"
    else:
        trend = "not enough values for a trend"
    lines = [
        f"Count: {count}",
        f"Sum: {total:g}",
        f"Mean: {mean:g}",
        f"Median: {median:g}",
        _mode_line(numbers),
        f"Min: {low:g}",
        f"Max: {high:g}",
        f"Range: {high - low:g}",
        f"Standard deviation: {stdev:g}",
        f"Trend: {trend}",
    ]
    return "\n".join(lines)


class DataAnalysisModule:
    def __init__(self, manifest: ModuleManifest):
        self.manifest = manifest

    def run(self, ctx: ModuleContext, prompt: str) -> str:
        numbers = parse_numbers(prompt)
        if len(numbers) >= 3:
            return analyze(numbers)
        return ctx.chat(prompt)
