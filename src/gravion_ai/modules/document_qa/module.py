"""DocumentQAModule — extractive document Q&A, computed locally.

The answer is quoted from the supplied document: sentences are
scored by overlap with the question's content terms, and the best
matching sentences are returned as the answer with their evidence.
When no sentence overlaps the question, the module says so plainly
instead of inventing an answer. Requests without a document are
answered by the resolved model (Gravion DocQA), whose trained
weights cover document Q&A guidance.

Input format:
    document: <text>
    question: <question>
"""

from __future__ import annotations

import re

from ...registry import ModuleContext, ModuleManifest
from ..summarization.module import split_sentences, stem_word, word_tokens

_STOP = frozenset(
    "a an the and or but of at by for with is are was were be been "
    "do does did what when where who whom which why how can could "
    "should would i you he she it we they this that in on to from".split()
)


def _terms(text: str) -> set[str]:
    return {stem_word(w) for w in word_tokens(text) if w not in _STOP}


def answer_question(document: str, question: str, max_sentences: int = 2) -> str:
    question_terms = _terms(question)
    sentences = split_sentences(document)
    # One shared common word is not evidence: require at least two
    # matching content terms (or one for a single-term question).
    required = min(2, len(question_terms)) if question_terms else 1
    scored: list[tuple[int, int, str]] = []
    for index, sentence in enumerate(sentences):
        overlap = len(question_terms & _terms(sentence))
        if overlap >= required:
            scored.append((overlap, -index, sentence))
    if not scored:
        return (
            "The document does not contain an answer to that question: "
            "no sentence in it matches the question's terms."
        )
    scored.sort(reverse=True)
    chosen = sorted(scored[:max_sentences], key=lambda item: -item[1])
    evidence = " ".join(sentence for _, _, sentence in chosen)
    return f"From the document: {evidence}"


def parse_document_prompt(prompt: str) -> tuple[str, str] | None:
    doc_match = re.search(
        r"document\s*:\s*(.+?)\s*question\s*:\s*(.+)",
        prompt,
        re.IGNORECASE | re.DOTALL,
    )
    if doc_match:
        return doc_match.group(1).strip(), doc_match.group(2).strip()
    # The question may come first ("question: ... document:
    # ..."): the all-in-one router treats the marker pair as
    # decisive in either order, so refusing the reversed form
    # here silently dropped a routed request to chat.
    reversed_match = re.search(
        r"question\s*:\s*(.+?)\s*document\s*:\s*(.+)",
        prompt,
        re.IGNORECASE | re.DOTALL,
    )
    if not reversed_match:
        return None
    return reversed_match.group(2).strip(), reversed_match.group(1).strip()


class DocumentQAModule:
    def __init__(self, manifest: ModuleManifest):
        self.manifest = manifest

    def run(self, ctx: ModuleContext, prompt: str) -> str:
        parsed = parse_document_prompt(prompt)
        if parsed is None:
            return ctx.chat(prompt)
        document, question = parsed
        return answer_question(document, question)
