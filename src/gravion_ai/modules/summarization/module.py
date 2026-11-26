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
_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]")


def split_sentences(text: str) -> list[str]:
    sentences = [s.strip() for s in _SENTENCE_RE.findall(text)]
    if not sentences and text.strip():
        sentences = [text.strip()]
    return sentences


def summarize(text: str, max_sentences: int = 3) -> str:
    """Extractive summary: top-scored sentences in original order."""
    sentences = split_sentences(text)
    if len(sentences) <= max_sentences:
        return " ".join(sentences)
    words = [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP]
    freq = Counter(words)
    scored: list[tuple[float, int]] = []
    for index, sentence in enumerate(sentences):
        terms = [w for w in re.findall(r"[a-z0-9]+", sentence.lower()) if w not in _STOP]
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
