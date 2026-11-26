"""RetrievalModule — sparse embedding search, computed locally.

Passages supplied in the prompt are embedded as hashed n-gram bucket
vectors (the same sparse feature family the Gravion core model uses)
and ranked against the query by cosine similarity. The module only
ranks what it was given; it never invents passages. Requests without
passages are answered by the resolved model (Gravion Search), whose
trained weights cover retrieval guidance.

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

_BUCKETS = 4096
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def embed(text: str) -> dict[int, float]:
    """Sparse hashed n-gram embedding (unigrams + bigrams)."""
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
    query_vector = embed(query)
    scored = [(cosine(query_vector, embed(p)), p) for p in passages]
    return sorted(scored, key=lambda item: item[0], reverse=True)


def parse_retrieval_prompt(prompt: str) -> tuple[str, list[str]] | None:
    query_match = re.search(r"query\s*:\s*(.+)", prompt, re.IGNORECASE)
    passages = re.findall(r"passage\s*:\s*(.+)", prompt, re.IGNORECASE)
    if not query_match or not passages:
        return None
    return query_match.group(1).strip(), [p.strip() for p in passages]


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
