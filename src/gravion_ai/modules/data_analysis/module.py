"""DataAnalysisModule — descriptive statistics, computed locally.

Numbers supplied in the prompt (comma, space or line separated, or
CSV rows) are parsed and summarized with real arithmetic: count,
sum, mean, median, mode, range, standard deviation and a simple
first-half/second-half trend. Nothing is estimated or invented.
Requests without numbers are answered by the resolved model (Gravion
Analyst), whose trained weights cover analysis guidance.
"""

from __future__ import annotations

import math
import re
import statistics
from collections import Counter

from ...registry import ModuleContext, ModuleManifest

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def parse_numbers(prompt: str) -> list[float]:
    return [float(m) for m in _NUMBER_RE.findall(prompt)]


def analyze(numbers: list[float]) -> str:
    count = len(numbers)
    total = sum(numbers)
    mean = total / count
    median = statistics.median(numbers)
    low, high = min(numbers), max(numbers)
    stdev = statistics.pstdev(numbers) if count > 1 else 0.0
    counts = Counter(numbers)
    mode_value, mode_count = counts.most_common(1)[0]
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
        f"Mode: {mode_value:g} (appears {mode_count} time{'s' if mode_count != 1 else ''})",
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
