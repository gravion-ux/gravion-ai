"""DataAnalysisModule — descriptive statistics, computed locally.

Numbers supplied in the prompt (comma, space or line separated, or
CSV rows) are parsed and summarized with real arithmetic: count,
sum, mean, median, mode, range, standard deviation and a simple
first-half/second-half trend. Nothing is estimated or invented.
Requests without numbers are answered by the resolved model (Gravion
Analyst), whose trained weights cover analysis guidance.
"""

from __future__ import annotations

import ipaddress
import math
import re
import statistics
from collections import Counter
from fractions import Fraction

from ...registry import ModuleContext, ModuleManifest

# One numeric token: optional sign, then either a grouped integer
# (Western "1,000,000" or Indian "1,00,000" digit grouping, or
# underscore grouping "1_000_000" as written in code), a plain
# integer/decimal, or a leading-dot decimal (".5"), with an optional
# scientific-notation exponent ("2.5e3"). A comma only joins a
# number when it sits inside a valid digit group, so comma-space
# separated lists ("4, 8, 15") still parse as separate values. The
# leading lookbehind keeps a minus glued to a preceding digit or
# letter from becoming a sign: in a range ("5-10") the hyphen is
# a separator, so both sides parse as positive values, and digits
# inside a word ("version2") are not numbers at all. Dates are
# blanked before parsing (see below), so no date part reaches
# this pattern. Without the underscore alternative "1_000" read
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

# An IPv6 address ("fe80::1", "2001:db8::1", "::ffff:192.0.2.1")
# is an address label like its dotted IPv4 form: left in place,
# the number pattern read the trailing group of "fe80::1" as the
# phantom value 1. Candidates use the cyber module's colon-hex
# shape and are blanked only when ipaddress validates them, so
# clock readings ("1:05:30"), ratios ("1:2") and other colon
# chains keep their existing handling.
_IPV6_RE = re.compile(
    r"(?<![0-9A-Fa-f:.])"
    r"(?:[0-9A-Fa-f]{0,4}:){2,7}"
    r"(?:[0-9A-Fa-f]{0,4}|(?:\d{1,3}\.){3}\d{1,3})"
    r"(?![0-9A-Fa-f:])(?!\.\d)"
)


def _blank_ipv6(match: re.Match[str]) -> str:
    try:
        ipaddress.IPv6Address(match.group(0))
    except ValueError:
        return match.group(0)
    return " "


# A calendar date ("2026-11-27", "27/11/2026", "11-27-2026")
# is a label, never three measurements: left in place, the
# number pattern read a date's year, month and day as values
# and skewed every statistic computed from the real numbers
# beside it. Plausible dates are blanked out before parsing,
# like dotted labels above. Only plausible dates go: the
# month must be 1-12 and the day 1-31, so a subtraction chain
# ("10-5-2") or an implausible chain ("13/40/2026") keeps its
# values, and a single-hyphen range ("5-10") is untouched.
# The year-first slash form ("2026/11/27") is a date too: the
# day-first slash pattern cannot see it (its first group takes
# at most two digits), so it was read as three values. The
# unpadded year-first hyphen form ("2026-1-5") is the same
# label: requiring two-digit month/day groups read it as the
# phantom values 2026, 1 and 5. The hyphen day-first form
# takes a two-digit year like its slash twin ("27-11-26");
# a one-digit tail ("10-5-2") still keeps its values.
_ISO_DATE_RE = re.compile(r"(?<!\d)(\d{4})-(\d{1,2})-(\d{1,2})(?!\d)")
_ISO_SLASH_DATE_RE = re.compile(r"(?<![0-9/])(\d{4})/(\d{1,2})/(\d{1,2})(?![0-9/])")
_SLASH_DATE_RE = re.compile(r"(?<![0-9/])(\d{1,2})/(\d{1,2})/(\d{4}|\d{2})(?![0-9/])")
_HYPHEN_DATE_RE = re.compile(r"(?<!\d)(\d{1,2})-(\d{1,2})-(\d{4}|\d{2})(?!\d)")


def _blank_iso_date(match: re.Match[str]) -> str:
    _year, month, day = (int(group) for group in match.groups())
    if 1 <= month <= 12 and 1 <= day <= 31:
        return " "
    return match.group(0)


def _blank_day_month_date(match: re.Match[str]) -> str:
    first, second, _year = (int(group) for group in match.groups())
    # Day-first and month-first orders are both in use, so
    # either reading makes the chain a date.
    if (1 <= first <= 31 and 1 <= second <= 12) or (
        1 <= first <= 12 and 1 <= second <= 31
    ):
        return " "
    return match.group(0)


# A month-name date ("27 Nov 2026", "November 27, 2026") is a
# label like its numeric forms: left in place, the number
# pattern read the day and year as values and skewed every
# statistic beside them. Day-first forms are unambiguous (a
# number glued to a month name is a date), so the year is
# optional there; month-first forms require a year or an
# ordinal day, so a verb use ("we march 5 miles") keeps its
# value. The day must be 1-31 either way.
_MONTH_NAME = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?"
    r"|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?"
    r"|nov(?:ember)?|dec(?:ember)?)"
)
_DAY_FIRST_DATE_RE = re.compile(
    r"(?<!\d)(\d{1,2})(?:st|nd|rd|th)?\s+" + _MONTH_NAME + r"\.?(?:\s*,?\s*\d{4})?(?!\d)",
    re.IGNORECASE,
)
_MONTH_FIRST_DATE_RE = re.compile(
    _MONTH_NAME + r"\.?\s+(\d{1,2})(?:st|nd|rd|th)?(?:\s*,\s*\d{4}|\s+\d{4})(?!\d)",
    re.IGNORECASE,
)
# The hyphenated month-name forms ("27-Nov-2026", "Nov-27-2026")
# are the same labels with hyphens instead of spaces: left in
# place, the day and year were read as values. The year is
# required, as in the month-first space form, so a bare
# "27-Nov" keeps its existing handling.
_DAY_FIRST_HYPHEN_DATE_RE = re.compile(
    r"(?<!\d)(\d{1,2})(?:st|nd|rd|th)?-" + _MONTH_NAME + r"-\d{4}(?!\d)",
    re.IGNORECASE,
)
_MONTH_FIRST_HYPHEN_DATE_RE = re.compile(
    r"(?<![A-Za-z])" + _MONTH_NAME + r"-(\d{1,2})(?:st|nd|rd|th)?-\d{4}(?!\d)",
    re.IGNORECASE,
)


def _blank_month_name_date(match: re.Match[str]) -> str:
    day = int(match.group(1))
    if not 1 <= day <= 31:
        return match.group(0)
    text = match.group(0)
    # "may" is also a modal verb: "chapter 5 may contain" is
    # not a date. A bare day-first "5 may" only reads as the
    # month when a year or an ordinal day pins it down.
    if re.search(r"\bmay\b", text, re.IGNORECASE) and not re.search(
        r"\d{4}|(?:st|nd|rd|th)\b", text, re.IGNORECASE
    ):
        return match.group(0)
    return " "


# A clock time ("10:30", "3:30 pm", "1:05:30") is a label,
# not two or three measurements: left in place, the number
# pattern read the hour and minutes as values. Only plausible
# clock readings go: minutes and seconds are two digits in
# 00-59 and the hour is 0-23, so a ratio ("3:1") or an
# implausible chain ("25:99") keeps its values.
_CLOCK_TIME_RE = re.compile(
    r"(?<![0-9:])(\d{1,2}):(\d{2})(?::(\d{2}))?(?:\s*[aApP]\.?[mM]\.?)?(?![0-9:])"
)


def _blank_clock_time(match: re.Match[str]) -> str:
    hour, minute, second = match.groups()
    if (
        0 <= int(hour) <= 23
        and int(minute) <= 59
        and (second is None or int(second) <= 59)
    ):
        return " "
    return match.group(0)

# An ordinal ("1st", "2nd", "21st") is a rank label, not a
# measurement: left in place, "ranked 1st, 2nd and 3rd with
# scores 10 20 30" counted the ranks as values and skewed every
# statistic. Ordinals are blanked out before parsing, like the
# date and clock labels above; the letter guard mirrors the
# number pattern's, so digits glued inside a word are untouched
# either way.
_ORDINAL_RE = re.compile(r"(?<![0-9A-Za-z])\d+(?:st|nd|rd|th)\b", re.IGNORECASE)

# Machine literals ("0xFF", "0b101", "0o17") are code, not
# decimal measurements: left in place, "0x10" read as the
# phantom value 0. They are blanked out before parsing, like
# dotted labels above.
_MACHINE_LITERAL_RE = re.compile(r"\b0[xX][0-9a-fA-F]+\b|\b0[bB][01]+\b|\b0[oO][0-7]+\b")

# A UUID ("123e4567-e89b-12d3-a456-426614174000") is an
# identifier label like an address: left in place, its first
# group read as the scientific-notation value 123e4567, which
# overflows to infinity and crashed the statistics, and its
# trailing group read as a huge phantom measurement. UUIDs
# are blanked out before parsing.
_UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)

# A fraction ("1/2") is one value, not the phantom pair 1 and
# 2 the number pattern read from it, and a mixed number
# ("1 1/2") is one value too. Fractions are folded to their
# decimal value before parsing. The slash guards keep a
# slash-date chain ("27/11/2026") out: neither half of the
# chain is a standalone fraction. A zero denominator states
# no value at all, so it is blanked instead of read as 1 and 0.
_MIXED_FRACTION_RE = re.compile(
    r"(?<![0-9/])(\d+)\s+(\d+)\s*/\s*(\d+)(?![0-9/])"
)
_FRACTION_RE = re.compile(r"(?<![0-9/])(\d+)\s*/\s*(\d+)(?![0-9/])")


def _fold_mixed_fraction(match: re.Match[str]) -> str:
    try:
        whole, numerator, denominator = (int(group) for group in match.groups())
        if denominator == 0 or numerator >= denominator:
            # Not a mixed number: leave the text for the plain
            # fraction fold (or blanking) below.
            return match.group(0)
        return repr(whole + numerator / denominator)
    except (ValueError, OverflowError):
        # A digit run past the integer-conversion limit (or a
        # value past float range) states no finite measurement.
        # Leave the text for the plain fraction fold below,
        # which blanks only the fraction itself instead of
        # crashing the parse or swallowing the whole part.
        return match.group(0)


def _fold_fraction(match: re.Match[str]) -> str:
    try:
        numerator, denominator = (int(group) for group in match.groups())
        if denominator == 0:
            return " "
        return repr(numerator / denominator)
    except (ValueError, OverflowError):
        return " "


def parse_numbers(prompt: str) -> list[float]:
    # Text pasted from documents writes negatives with the
    # Unicode minus (U+2212): left as-is, "−5" lost its sign
    # and read as +5. Normalise it to the ASCII hyphen the
    # number pattern understands; a Unicode range dash then
    # behaves like its ASCII twin too.
    cleaned = prompt.replace("−", "-")
    cleaned = _MACHINE_LITERAL_RE.sub(" ", cleaned)
    cleaned = _UUID_RE.sub(" ", cleaned)
    cleaned = _IPV6_RE.sub(_blank_ipv6, cleaned)
    cleaned = _ISO_DATE_RE.sub(_blank_iso_date, cleaned)
    cleaned = _ISO_SLASH_DATE_RE.sub(_blank_iso_date, cleaned)
    cleaned = _SLASH_DATE_RE.sub(_blank_day_month_date, cleaned)
    cleaned = _HYPHEN_DATE_RE.sub(_blank_day_month_date, cleaned)
    cleaned = _DAY_FIRST_DATE_RE.sub(_blank_month_name_date, cleaned)
    cleaned = _MONTH_FIRST_DATE_RE.sub(_blank_month_name_date, cleaned)
    cleaned = _DAY_FIRST_HYPHEN_DATE_RE.sub(_blank_month_name_date, cleaned)
    cleaned = _MONTH_FIRST_HYPHEN_DATE_RE.sub(_blank_month_name_date, cleaned)
    cleaned = _CLOCK_TIME_RE.sub(_blank_clock_time, cleaned)
    cleaned = _ORDINAL_RE.sub(" ", cleaned)
    cleaned = _DOTTED_SEQUENCE_RE.sub(" ", cleaned)
    cleaned = _MIXED_FRACTION_RE.sub(_fold_mixed_fraction, cleaned)
    cleaned = _FRACTION_RE.sub(_fold_fraction, cleaned)
    # A token whose value overflows float ("1e999") is not a
    # finite measurement: kept, it turned the standard
    # deviation into a crash instead of a statistic, so
    # non-finite results state no value and are dropped.
    values: list[float] = []
    for match in _NUMBER_RE.findall(cleaned):
        value = float(match.replace(",", "").replace("_", ""))
        if math.isfinite(value):
            values.append(value)
    return values


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


def _exact_total(numbers: list[float]) -> float:
    """Sum without intermediate overflow.

    ``math.fsum`` is exact until an intermediate partial sum
    leaves float range, where it raises even when the true
    total is representable ("1e308 + 1e308 - 1e308"). The
    fraction fallback keeps the exact total in that case; a
    total past float range itself ("1e308 + 1e308") is
    reported as signed infinity, the honest float answer.
    """
    try:
        return math.fsum(numbers)
    except OverflowError:
        exact = sum((Fraction(value) for value in numbers), Fraction(0))
        try:
            return float(exact)
        except OverflowError:
            return math.inf if exact > 0 else -math.inf


def analyze(numbers: list[float]) -> str:
    if not numbers:
        return "No numbers to analyze."
    count = len(numbers)
    total = _exact_total(numbers)
    # statistics.mean works in exact fractions: total / count
    # overflowed to infinity for values like 1e308 whose mean
    # is plainly 1e308, and the trend halves inherited the
    # same phantom infinity.
    mean = statistics.mean(numbers)
    median = statistics.median(numbers)
    low, high = min(numbers), max(numbers)
    stdev = statistics.pstdev(numbers) if count > 1 else 0.0
    half = count // 2
    if half:
        first = statistics.mean(numbers[:half])
        second = statistics.mean(numbers[half:])
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
