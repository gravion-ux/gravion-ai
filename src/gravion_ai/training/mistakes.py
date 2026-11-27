"""Mistake learning for Gravion local models.

A mistake case records one observed failure: a failing test, a wrong
route, a weak response, or a catalog failure. When the corrected
output is known, the case becomes a preference pair (rejected output
versus corrected output). Corrected cases also become regression
tests, and preference learning folds the corrected output back into
local training while keeping the rejected output out.

Verified catalog routing examples can join the same training set, so
model selection learns from catalog entries that passed verification
instead of from stale registry claims.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from importlib import resources
from pathlib import Path
from typing import Iterable, Sequence

from ..catalog import ModelCatalog
from ..catalog_verification import (
    CatalogVerificationReport,
    catalog_routing_examples,
    verify_catalog,
)
from ..core_model import GravionCoreLocalModel, load_bundled_model, train_local_core
from ..errors import TrainingError
from ..providers import GravionCoreProvider
from ..registry import ModuleContext, default_registry

MISTAKE_KINDS = frozenset(
    {"failing_test", "wrong_route", "weak_response", "catalog_failure"}
)
REGRESSION_TARGETS = frozenset({"none", "model_none", "provider", "module", "catalog"})


def _content_id(payload: dict[str, object]) -> str:
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return "mistake-" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _normalized_prompt(prompt: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", prompt.lower()))


@dataclass(frozen=True)
class MistakeCase:
    """One recorded failure and, when known, its correction."""

    kind: str
    prompt: str
    rejected_output: str
    corrected_output: str = ""
    id: str = ""
    model_id: str = ""
    module: str = ""
    expected_route: str = ""
    actual_route: str = ""
    source: str = ""
    notes: str = ""
    regression_target: str = "none"
    expected_contains: str = ""

    def __post_init__(self) -> None:
        if self.kind not in MISTAKE_KINDS:
            raise TrainingError(
                f"mistake kind must be one of {sorted(MISTAKE_KINDS)}, got {self.kind!r}"
            )
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise TrainingError("mistake prompt must be a non-empty string")
        if not isinstance(self.rejected_output, str) or not self.rejected_output.strip():
            raise TrainingError("mistake rejected_output must be a non-empty string")
        if self.regression_target not in REGRESSION_TARGETS:
            raise TrainingError(
                "mistake regression_target must be one of "
                f"{sorted(REGRESSION_TARGETS)}, got {self.regression_target!r}"
            )
        if not self.id:
            object.__setattr__(self, "id", _content_id(self._identity_payload()))

    def _identity_payload(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "prompt": self.prompt,
            "rejected_output": self.rejected_output,
            "corrected_output": self.corrected_output,
            "model_id": self.model_id,
            "module": self.module,
            "expected_route": self.expected_route,
            "actual_route": self.actual_route,
            "regression_target": self.regression_target,
            "expected_contains": self.expected_contains,
        }

    @property
    def corrected(self) -> bool:
        return bool(self.corrected_output.strip()) and (
            self.corrected_output.strip() != self.rejected_output.strip()
            or self.regression_target == "model_none"
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: object) -> "MistakeCase":
        if not isinstance(raw, dict):
            raise TrainingError("mistake case must be a JSON object")
        allowed = {
            "id",
            "kind",
            "prompt",
            "rejected_output",
            "corrected_output",
            "model_id",
            "module",
            "expected_route",
            "actual_route",
            "source",
            "notes",
            "regression_target",
            "expected_contains",
        }
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise TrainingError(f"mistake case has unknown fields: {unknown}")
        return cls(
            id=str(raw.get("id", "")),
            kind=str(raw.get("kind", "")),
            prompt=str(raw.get("prompt", "")),
            rejected_output=str(raw.get("rejected_output", "")),
            corrected_output=str(raw.get("corrected_output", "")),
            model_id=str(raw.get("model_id", "")),
            module=str(raw.get("module", "")),
            expected_route=str(raw.get("expected_route", "")),
            actual_route=str(raw.get("actual_route", "")),
            source=str(raw.get("source", "")),
            notes=str(raw.get("notes", "")),
            regression_target=str(raw.get("regression_target", "none")),
            expected_contains=str(raw.get("expected_contains", "")),
        )


@dataclass(frozen=True)
class PreferencePair:
    """A rejected output paired with the corrected output for one prompt."""

    prompt: str
    rejected_output: str
    chosen_output: str
    mistake_id: str
    kind: str
    model_id: str = ""
    module: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class MistakeLog:
    """Append-only JSONL mistake log with content-based deduplication."""

    def __init__(self, path: Path | str):
        self.path = Path(path)

    def load(self) -> list[MistakeCase]:
        if not self.path.exists():
            return []
        cases: list[MistakeCase] = []
        for lineno, line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                cases.append(MistakeCase.from_dict(json.loads(line)))
            except (json.JSONDecodeError, TrainingError) as exc:
                raise TrainingError(f"invalid mistake log line {lineno}: {exc}") from exc
        return cases

    def append(self, case: MistakeCase) -> MistakeCase:
        existing = {item.id: item for item in self.load()}
        if case.id in existing:
            return existing[case.id]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(case.to_dict(), sort_keys=True, ensure_ascii=False) + "\n")
        return case

    def record_failure(
        self,
        *,
        kind: str,
        prompt: str,
        rejected_output: str,
        corrected_output: str = "",
        model_id: str = "",
        module: str = "",
        expected_route: str = "",
        actual_route: str = "",
        source: str = "",
        notes: str = "",
        regression_target: str = "none",
        expected_contains: str = "",
    ) -> MistakeCase:
        return self.append(
            MistakeCase(
                kind=kind,
                prompt=prompt,
                rejected_output=rejected_output,
                corrected_output=corrected_output,
                model_id=model_id,
                module=module,
                expected_route=expected_route,
                actual_route=actual_route,
                source=source,
                notes=notes,
                regression_target=regression_target,
                expected_contains=expected_contains,
            )
        )


def load_bundled_mistakes() -> list[MistakeCase]:
    resource = resources.files("gravion_ai").joinpath("data", "mistakes.jsonl")
    return [
        MistakeCase.from_dict(json.loads(line))
        for line in resource.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_preference_pairs(cases: Iterable[MistakeCase]) -> list[PreferencePair]:
    """Build preference pairs from corrected mistake cases only."""

    pairs: list[PreferencePair] = []
    seen: set[str] = set()
    for case in cases:
        if not case.corrected or case.id in seen:
            continue
        seen.add(case.id)
        pairs.append(
            PreferencePair(
                prompt=case.prompt,
                rejected_output=case.rejected_output,
                chosen_output=case.corrected_output,
                mistake_id=case.id,
                kind=case.kind,
                model_id=case.model_id,
                module=case.module,
            )
        )
    return pairs


def apply_preference_learning(
    base_examples: Iterable[tuple[str, str]],
    cases: Iterable[MistakeCase],
    *,
    catalog_examples: Iterable[tuple[str, str]] = (),
) -> list[tuple[str, str]]:
    """Fold corrected outputs and verified catalog examples into training data.

    A corrected output wins over any base example for the same
    normalized prompt, so the rejected output cannot remain as the
    trained completion for that prompt.
    """

    by_prompt: dict[str, tuple[str, str]] = {}
    for prompt, completion in base_examples:
        by_prompt[_normalized_prompt(prompt)] = (prompt, completion)
    for prompt, completion in catalog_examples:
        by_prompt[_normalized_prompt(prompt)] = (prompt, completion)
    for pair in build_preference_pairs(cases):
        by_prompt[_normalized_prompt(pair.prompt)] = (pair.prompt, pair.chosen_output)
    return [by_prompt[key] for key in sorted(by_prompt)]


def evaluate_preference_pairs(
    model: GravionCoreLocalModel, pairs: Sequence[PreferencePair]
) -> dict[str, object]:
    wins = 0
    losses = 0
    other = 0
    for pair in pairs:
        prediction = model.predict(pair.prompt)
        if prediction == pair.chosen_output:
            wins += 1
        elif prediction == pair.rejected_output:
            losses += 1
        else:
            other += 1
    total = len(pairs)
    return {
        "pairs": total,
        "wins": wins,
        "losses": losses,
        "other": other,
        "accuracy": round(wins / total, 4) if total else 0.0,
    }


def train_with_mistake_learning(
    base_examples: Iterable[tuple[str, str]],
    cases: Iterable[MistakeCase],
    *,
    model_id: str,
    source: str = "<mistake-learning>",
    catalog: ModelCatalog | None = None,
    catalog_report: CatalogVerificationReport | None = None,
    include_catalog_examples: bool = True,
) -> tuple[GravionCoreLocalModel, dict[str, object]]:
    """Train a local model from base, preference, and verified catalog data."""

    case_list = list(cases)
    pairs = build_preference_pairs(case_list)
    routing_examples: list[tuple[str, str]] = []
    if include_catalog_examples:
        routing_examples = catalog_routing_examples(catalog, catalog_report)
    examples = apply_preference_learning(
        base_examples, case_list, catalog_examples=routing_examples
    )
    model = train_local_core(examples, model_id=model_id, source=source)
    report = {
        "model_id": model_id,
        "training_examples": len(examples),
        "preference_pairs": len(pairs),
        "catalog_examples": len(routing_examples),
        "preference_evaluation": evaluate_preference_pairs(model, pairs),
        "parameter_count": model.parameter_count,
    }
    return model, report


def run_regression_case(
    case: MistakeCase,
    *,
    provider: GravionCoreProvider | None = None,
    catalog_report: CatalogVerificationReport | None = None,
) -> bool:
    """Run the regression check attached to one corrected mistake."""

    if not case.corrected:
        return False
    if case.regression_target == "none":
        return True
    if case.regression_target == "model_none":
        if not case.model_id:
            raise TrainingError(f"mistake {case.id}: model_none regression needs model_id")
        return load_bundled_model(case.model_id).predict(case.prompt) is None
    if case.regression_target == "provider":
        provider = provider or GravionCoreProvider()
        output = provider.complete(
            [{"role": "user", "content": case.prompt}], model=case.model_id or None
        )
        if case.expected_contains:
            return case.expected_contains in output
        return output == case.corrected_output
    if case.regression_target == "module":
        if not case.module:
            raise TrainingError(f"mistake {case.id}: module regression needs module")
        provider = provider or GravionCoreProvider()
        module = default_registry().instantiate(case.module)
        context = ModuleContext("You are Gravion.", provider, model=case.model_id or None)
        output = module.run(context, case.prompt)
        if case.expected_contains:
            return case.expected_contains in output
        return output == case.corrected_output
    if case.regression_target == "catalog":
        report = catalog_report or verify_catalog()
        return report.ok
    raise TrainingError(f"mistake {case.id}: unsupported regression target")


def render_regression_tests(cases: Iterable[MistakeCase]) -> str:
    """Render one pytest regression test per corrected mistake case."""

    corrected = [case for case in cases if case.corrected]
    lines = [
        '"""Generated regression tests for corrected Gravion mistake cases."""',
        "",
        "import pytest",
        "",
        "from gravion_ai.training.mistakes import MistakeCase, run_regression_case",
        "",
        "CASES = [",
    ]
    for case in corrected:
        lines.append(f"    {case.to_dict()!r},")
    lines.extend(
        [
            "]",
            "",
            "",
            "@pytest.mark.parametrize('case', [MistakeCase.from_dict(item) for item in CASES], ids=[item['id'] for item in CASES])",
            "def test_corrected_mistake_stays_fixed(case):",
            "    assert run_regression_case(case)",
            "",
        ]
    )
    return "\n".join(lines)


def write_regression_tests(path: Path | str, cases: Iterable[MistakeCase]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_regression_tests(cases), encoding="utf-8")
    return destination
