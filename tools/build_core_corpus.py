"""Build the bundled Gravion Core training corpus from Gravion-owned text.

Sources (all inside this repository, all Gravion-owned):
- the bundled seed dataset (gravion-core-seed.jsonl)
- README.md, ARCHITECTURE.md, CHANGELOG.md, docs/*.md, examples/policy.md
- module docstrings under src/gravion_ai/

Markdown and docstring text is split into sentence-sized answer
classes. Training prompts combine section titles with salient terms
from each answer; held-out eval prompts use different title templates
and are written to tests/fixtures/core-eval.jsonl. No third-party
text is used.
"""

from __future__ import annotations

import ast
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEED = ROOT / "src" / "gravion_ai" / "data" / "gravion-core-seed.jsonl"
CORPUS_OUT = ROOT / "src" / "gravion_ai" / "data" / "gravion-core-corpus.jsonl"
EVAL_OUT = ROOT / "tests" / "fixtures" / "core-eval.jsonl"

_MD_FILES = ["README.md", "ARCHITECTURE.md", "CHANGELOG.md", "examples/policy.md"]
_SKIP_TITLES = {"license", "testing", "contributing", "changelog"}
_STOPWORDS = {
    "a",
    "all",
    "also",
    "an",
    "and",
    "any",
    "are",
    "as",
    "be",
    "by",
    "can",
    "down",
    "each",
    "every",
    "for",
    "from",
    "here",
    "how",
    "in",
    "into",
    "is",
    "it",
    "its",
    "may",
    "more",
    "most",
    "must",
    "no",
    "not",
    "of",
    "on",
    "only",
    "or",
    "other",
    "out",
    "over",
    "own",
    "same",
    "should",
    "so",
    "such",
    "than",
    "that",
    "the",
    "then",
    "there",
    "this",
    "to",
    "under",
    "up",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "would",
}


def _clean(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"[`*_>#|]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _chunks(text: str) -> list[str]:
    text = _clean(text)
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if len(sentence.strip()) >= 30
    ]
    chunks: list[str] = []
    for sentence in sentences:
        if len(sentence) > 320:
            sentence = sentence[:320].rsplit(" ", 1)[0].strip() + "."
        chunks.append(sentence)
    return chunks


def _sections(path: Path) -> list[tuple[str, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    sections: list[tuple[str, list[str]]] = []
    title = path.stem.replace("-", " ").replace("_", " ")
    body: list[str] = []
    for line in lines:
        match = re.match(r"#{1,6}\s+(.*)", line)
        if match:
            if body:
                sections.append((title, body))
            title = _clean(match.group(1)).lower()
            body = []
        else:
            body.append(line)
    if body:
        sections.append((title, body))

    out: list[tuple[str, str]] = []
    for title, body in sections:
        paragraphs = [p for p in re.split(r"\n\s*\n", "\n".join(body)) if _clean(p)]
        index = 0
        for paragraph in paragraphs:
            for answer in _chunks(paragraph):
                index += 1
                label = title if index == 1 else f"{title} detail {index}"
                out.append((label, answer))
    return out


def _docstring_classes() -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for path in sorted((ROOT / "src" / "gravion_ai").rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(
                node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
            ):
                continue
            doc = ast.get_docstring(node)
            if not doc:
                continue
            if isinstance(node, ast.Module):
                label = path.stem.replace("_", " ") + " module"
            elif isinstance(node, ast.ClassDef):
                label = node.name.replace("_", " ") + " class"
            else:
                label = node.name.replace("_", " ") + " function"
            for index, answer in enumerate(_chunks(doc)):
                out.append((label if index == 0 else f"{label} detail {index + 1}", answer))
    return out


def _keywords(answer: str) -> list[str]:
    tokens = [
        token
        for token in re.findall(r"[a-z0-9]+", answer.lower())
        if len(token) > 2 and token not in _STOPWORDS
    ]
    return [token for token, _ in Counter(tokens).most_common(8)]


def _train_prompts(title: str, answer: str) -> list[str]:
    prompts = [
        f"what is {title}",
        f"explain {title}",
        f"gravion {title}",
        f"what does gravion say about {title}",
    ]
    keywords = _keywords(answer)
    if keywords:
        prompts.extend(
            [
                f"gravion {' '.join(keywords[:3])}",
                f"explain {' '.join(keywords[:4])}",
                f"how does {' '.join(keywords[:3])} work in gravion",
                f"what should i know about {' '.join(keywords[:4])}",
            ]
        )
    return prompts


def _eval_prompts(title: str) -> list[str]:
    return [
        f"how does {title} work",
        f"tell me about {title}",
    ]


def _is_self_referential(answer: str) -> bool:
    text = answer.lower()
    if "parameters counted" in text or "held-out eval" in text:
        return True
    if "examples across" in text or "answer classes" in text:
        return True
    return bool(re.search(r"\b\d[\d,]*\s+parameters\b", text))


def main() -> None:
    classes: list[tuple[str, str]] = []
    seen_answers: set[str] = set()

    def add(title: str, answer: str) -> None:
        key = answer.lower()
        if key in seen_answers or title.lower() in _SKIP_TITLES:
            return
        if _is_self_referential(answer):
            return
        seen_answers.add(key)
        classes.append((title, answer))

    md_files = [ROOT / name for name in _MD_FILES]
    md_files += sorted((ROOT / "docs").glob("*.md"))
    for path in md_files:
        if path.exists():
            for title, answer in _sections(path):
                add(title, answer)
    for title, answer in _docstring_classes():
        add(title, answer)

    train: list[dict[str, str]] = []
    evaluation: list[dict[str, str]] = []
    train_prompt_set: set[str] = set()

    for line in SEED.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        train.append({"prompt": row["prompt"], "completion": row["completion"]})
        train_prompt_set.add(row["prompt"].lower())

    for title, answer in classes:
        for prompt in _train_prompts(title, answer):
            if prompt.lower() not in train_prompt_set:
                train.append({"prompt": prompt, "completion": answer})
                train_prompt_set.add(prompt.lower())
        for prompt in _eval_prompts(title):
            if prompt.lower() not in train_prompt_set:
                evaluation.append({"prompt": prompt, "completion": answer})

    CORPUS_OUT.write_text(
        "\n".join(json.dumps(row) for row in train) + "\n", encoding="utf-8"
    )
    EVAL_OUT.write_text(
        "\n".join(json.dumps(row) for row in evaluation) + "\n", encoding="utf-8"
    )
    tokens = sum(
        len(re.findall(r"[a-z0-9]+", (r["prompt"] + " " + r["completion"]).lower()))
        for r in train
    )
    distinct_classes = len({row["completion"] for row in train})
    print(
        f"classes={distinct_classes} train_examples={len(train)} "
        f"eval_examples={len(evaluation)} train_tokens={tokens}"
    )


if __name__ == "__main__":
    main()
