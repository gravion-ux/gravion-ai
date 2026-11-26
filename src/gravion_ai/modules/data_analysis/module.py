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
# (Western "1,000,000" or Indian "1,00,000" digit grouping), a plain
# integer/decimal, or a leading-dot decimal (".5"), with an optional
# scientific-notation exponent ("2.5e3"). A comma only joins a
# number when it sits inside a valid digit group, so comma-space
# separated lists ("4, 8, 15") still parse as separate values. The
# leading lookbehind keeps a minus glued to a preceding digit or
# letter from becoming a sign: in a range ("5-10") or a date
# ("2026-11-27") the hyphen is a separator, so both sides parse as
# positive values, and digits inside a word ("version2") are not
# numbers at all.
_NUMBER_RE = re.compile(
    r"(?<![0-9A-Za-z])-?(?:\d{1,3}(?:,\d{3})+|\d{1,2}(?:,\d{2})+,\d{3}|\d+|\.\d+)"
    r"(?:\.\d+)?(?:[eE][+-]?\d+)?"
)


def parse_numbers(prompt: str) -> list[float]:
    return [float(m.replace(",", "")) for m in _NUMBER_RE.findall(prompt)]


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
