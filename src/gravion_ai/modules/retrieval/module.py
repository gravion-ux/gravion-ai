"""RetrievalModule — sparse embedding search, computed locally.

Passages supplied in the prompt are embedded as hashed n-gram bucket
vectors (the same sparse feature family the Gravion core model uses)
and ranked against the query by cosine similarity. Ranking weights
each bucket by inverse document frequency across the supplied
passages, so a rare query term that only one passage carries counts
for more than a word every passage shares, and stopwords carry no
weight at all. The module only ranks what it was given; it never
invents passages. Requests without passages are answered by the
resolved model (Gravion Search), whose trained weights cover
retrieval guidance.

Input format:
    query: <question>
    passage: <text>
    passage: <text>
"""

from __future__ import annotations

import hashlib
import math
import re

from ...registry import ModuleContext, ModuleManifest
from ..summarization.module import stem_word, word_tokens

_BUCKETS = 4096
_STOP = frozenset(
    "a an the and or but of at by for with is are was were be been "
    "do does did what when where who whom which why how can could "
    "should would i you he she it we they this that in on to from".split()
)


def _tokens(text: str) -> list[str]:
    # Stem so a query for "cats" matches a passage about a "cat":
    # without it the two forms hash to different buckets and the
    # relevant passage scores zero.
    return [stem_word(t) for t in word_tokens(text) if t not in _STOP]


def embed(text: str) -> dict[int, float]:
    """Sparse hashed n-gram embedding (unigrams + bigrams).

    Stopwords are dropped before terms are formed, so a passage
    padded with "the the of of" gains no similarity from them.
    """
    tokens = _tokens(text)
    vector: dict[int, float] = {}
    terms = list(tokens) + [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]
    for term in terms:
        digest = hashlib.sha256(term.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "little") % _BUCKETS
        vector[bucket] = vector.get(bucket, 0.0) + 1.0
    return vector


def cosine(a: dict[int, float], b: dict[int, float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(value * b.get(bucket, 0.0) for bucket, value in a.items())
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def rank_passages(query: str, passages: list[str]) -> list[tuple[float, str]]:
    """Rank passages by IDF-weighted cosine similarity to the query.

    Bucket weights are smoothed inverse document frequencies over
    the supplied passages: ``log((N + 1) / (df + 1)) + 1``. A term
    present in every passage keeps weight 1.0 while a term unique
    to one passage weighs more, so the passage carrying the query's
    distinctive terms wins over one that merely shares common words.
    """
    passage_vectors = [embed(p) for p in passages]
    total = len(passages)
    document_frequency: dict[int, int] = {}
    for vector in passage_vectors:
        for bucket in vector:
            document_frequency[bucket] = document_frequency.get(bucket, 0) + 1

    def weighted(vector: dict[int, float]) -> dict[int, float]:
        return {
            bucket: value
            * (math.log((total + 1) / (document_frequency.get(bucket, 0) + 1)) + 1.0)
            for bucket, value in vector.items()
        }

    query_vector = weighted(embed(query))
    scored = [
        (cosine(query_vector, weighted(vector)), passage)
        for vector, passage in zip(passage_vectors, passages)
    ]
    return sorted(scored, key=lambda item: item[0], reverse=True)


def _collect_retrieval_parts(parts: list[str]) -> tuple[str, list[str]] | None:
    # re.split with one capture group yields
    # [before, label, body, label, body, ...].
    if len(parts) < 5:
        return None
    query: str | None = None
    passages: list[str] = []
    for index in range(1, len(parts) - 1, 2):
        label = parts[index].lower()
        body = " ".join(parts[index + 1].split())
        if not body:
            continue
        if label == "query":
            if query is None:
                query = body
        else:
            passages.append(body)
    if query is None or not passages:
        return None
    return query, passages


def parse_retrieval_prompt(prompt: str) -> tuple[str, list[str]] | None:
    # A query or passage runs until the next "query:"/"passage:"
    # marker or the end of the prompt, so a passage wrapped over
    # several lines is kept whole instead of truncated to its
    # first line. Wrapped lines are joined with single spaces.
    parsed = _collect_retrieval_parts(
        re.split(r"(?im)^\s*(query|passage)\s*:\s*", prompt)
    )
    if parsed is not None:
        return parsed
    # The same markers written on one line ("query: ... passage:
    # ...") are the documented input in single-line form: the
    # all-in-one router already sends them here on the
    # query+passage signal, so refusing them dropped a routed
    # request to chat. The inline split only runs when the
    # line-anchored form is absent, so a multi-line passage
    # mentioning "query:" mid-line keeps its line-form reading.
    return _collect_retrieval_parts(
        re.split(r"(?i)\b(query|passage)\s*:\s*", prompt)
    )


class RetrievalModule:
    def __init__(self, manifest: ModuleManifest):
        self.manifest = manifest

    def run(self, ctx: ModuleContext, prompt: str) -> str:
        parsed = parse_retrieval_prompt(prompt)
        if parsed is None:
            return ctx.chat(prompt)
        query, passages = parsed
        ranked = rank_passages(query, passages)
        lines = [f"Top matches for: {query}"]
        for rank, (score, passage) in enumerate(ranked, start=1):
            lines.append(f"{rank}. [{score:.3f}] {passage}")
        return "\n".join(lines)
