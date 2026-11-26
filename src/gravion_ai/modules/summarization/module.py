"""SummarizationModule — extractive summarization, computed locally.

The summary is built only from the source's own sentences: sentences
are scored by term frequency and position, the strongest are kept in
their original order, and nothing is rewritten or invented. Requests
without source text are answered by the resolved model (Gravion
Summary), whose trained weights cover summarization guidance.
"""

from __future__ import annotations

import re
from collections import Counter

from ...registry import ModuleContext, ModuleManifest

_STOP = frozenset(
    "a an the and or but if then else of at by for with about into "
    "through after over between out against during without before "
    "under around among is are was were be been being have has had "
    "do does did will would shall should can could may might must "
    "i you he she it we they this that these those as in on to from".split()
)

_SUMMARIZE_RE = re.compile(r"summarize\s*:\s*(.+)", re.IGNORECASE | re.DOTALL)

# Abbreviations whose final dot does not end a sentence, plus
# dotted forms (e.g. "e.g", "u.s"). Single uppercase initials
# ("A. Smith") are handled separately in split_sentences.
_ABBREVIATIONS = frozenset(
    "dr mr mrs ms prof sr jr st vs etc e.g i.e u.s u.k a.m p.m".split()
)
_CLOSERS = "\"'”’)]"


def stem_word(word: str) -> str:
    """Light plural stemmer shared by the text modules.

    Singular and plural forms must land on the same stem, or a
    question about a "class" never matches a document sentence
    about "classes". The rules only strip a plural ending: "ies"
    becomes "y" ("cities" -> "city"), "es" drops after s/x/z/ch/sh
    ("classes" -> "class", "boxes" -> "box"), and a final "s"
    drops unless the word ends in "ss", "us" or "is" ("class",
    "status" and "analysis" are not plurals). Anything else is
    returned unchanged; irregular plurals are out of scope.
    """
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith("es"):
        stem = word[:-2]
        if stem.endswith(("s", "x", "z", "ch", "sh")):
            return stem
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def _is_abbreviation_dot(text: str, index: int) -> bool:
    """True when the dot at ``index`` belongs to an abbreviation/initial."""
    start = index - 1
    while start >= 0 and (text[start].isalpha() or text[start] == "."):
        start -= 1
    word = text[start + 1 : index]
    if word.lower() in _ABBREVIATIONS:
        return True
    # A single uppercase letter before the dot is a name initial.
    return len(word) == 1 and word.isupper()


def split_sentences(text: str) -> list[str]:
    """Split ``text`` into sentences, keeping a trailing fragment.

    A dot between two digits is a decimal point, not a sentence
    boundary, and abbreviation dots ("Dr.", "e.g.") do not end a
    sentence either. Text after the last terminator is kept as its
    own sentence instead of being silently dropped, so callers that
    quote sentences (document Q&A) can still find an answer in a
    final line that has no closing punctuation.
    """
    stripped = text.strip()
    if not stripped:
        return []
    sentences: list[str] = []
    start = 0
    index = 0
    while index < len(stripped):
        char = stripped[index]
        if char in ".!?":
            if char == ".":
                # Decimal point: "3.14" stays inside its sentence.
                if (
                    0 < index < len(stripped) - 1
                    and stripped[index - 1].isdigit()
                    and stripped[index + 1].isdigit()
                ):
                    index += 1
                    continue
                # Ellipsis: only the last dot can end the sentence.
                if index + 1 < len(stripped) and stripped[index + 1] == ".":
                    index += 1
                    continue
                if _is_abbreviation_dot(stripped, index):
                    index += 1
                    continue
            end = index + 1
            while end < len(stripped) and stripped[end] in _CLOSERS:
                end += 1
            if end == len(stripped) or stripped[end].isspace():
                sentence = stripped[start:end].strip()
                if sentence:
                    sentences.append(sentence)
                start = end
                index = end
                continue
        index += 1
    tail = stripped[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def summarize(text: str, max_sentences: int = 3) -> str:
    """Extractive summary: top-scored sentences in original order."""
    # Drop exact duplicate sentences first (keeping the first
    # occurrence): a repeated sentence inflates its own term
    # frequencies, so without this a summary could be the same
    # sentence repeated max_sentences times.
    sentences = list(dict.fromkeys(split_sentences(text)))
    if len(sentences) <= max_sentences:
        return " ".join(sentences)
    # Stem before counting so singular and plural forms of a word
    # ("cat"/"cats") share one frequency instead of splitting it.
    words = [
        stem_word(w)
        for w in re.findall(r"[a-z0-9]+", " ".join(sentences).lower())
        if w not in _STOP
    ]
    freq = Counter(words)
    scored: list[tuple[float, int]] = []
    for index, sentence in enumerate(sentences):
        terms = [
            stem_word(w)
            for w in re.findall(r"[a-z0-9]+", sentence.lower())
            if w not in _STOP
        ]
        if not terms:
            continue
        score = sum(freq[w] for w in terms) / len(terms)
        # Mild position bonus: openings and closings carry the thesis.
        if index == 0:
            score *= 1.25
        elif index == len(sentences) - 1:
            score *= 1.1
        scored.append((score, index))
    keep = sorted(index for _, index in sorted(scored, reverse=True)[:max_sentences])
    return " ".join(sentences[i] for i in keep)


class SummarizationModule:
    def __init__(self, manifest: ModuleManifest):
        self.manifest = manifest

    def run(self, ctx: ModuleContext, prompt: str) -> str:
        match = _SUMMARIZE_RE.search(prompt)
        if match:
            return summarize(match.group(1))
        if len(split_sentences(prompt)) >= 4:
            return summarize(prompt)
        return ctx.chat(prompt)
