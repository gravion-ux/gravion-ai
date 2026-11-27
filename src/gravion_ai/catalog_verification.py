"""Catalog verification for Gravion models and modules.

The model registry and module registry are product data, so this
module treats them as claims to verify rather than facts to trust.
Every bundled model must load from its stored weights, match its
catalog parameter count and weights hash, resolve through the local
provider, and answer its first seed prompt. Every module must have a
current manifest, load its entrypoint, and run a smoke prompt with a
verified serving model.

Only verified catalog entries feed routing/selection training and
evaluation. A stale entry is reported as an issue and excluded from
selection until it is fixed or removed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

from .catalog import CAPABILITIES, ModelCatalog, ModelEntry
from .core_model import LOCAL_MODEL_IDS, bundled_seed_path, count_parameters, load_bundled_model
from .providers import GravionCoreProvider
from .registry import ModuleContext, ModuleRegistry, ModuleStatus, default_registry
from .tiers import TIERS

MANIFEST_FILENAME = "manifest.json"
_EXCLUDED_PARTS = {MANIFEST_FILENAME, "__pycache__"}

MODULE_SMOKE_PROMPTS = {
    "chat": "Hello, what can you do?",
    "support": "What are your opening hours?",
    "coder": "Write a python function that adds two numbers",
    "image": "Draw a square logo for a bakery",
    "music": "Compose a calm 30 seconds track",
    "video": "Create a vertical 15 seconds video of a sunrise",
    "website-builder": "Build a landing page for a bakery, pages: home, about",
    "cyber": "Triage: alert from 10.0.0.8 severity: high",
    "agentic": "Plan an agent that checks email and drafts replies",
    "translation": "translate to hindi: thank you",
    "summarization": (
        "summarize: Cats sleep for long hours each day. Cats also groom "
        "themselves carefully. Dogs need daily walks and training. "
        "Birds sing at dawn in spring."
    ),
    "retrieval": (
        "query: how do cats groom\n"
        "passage: Cats groom themselves with their tongues.\n"
        "passage: Dogs enjoy long walks in parks."
    ),
    "data-analysis": "analyze these numbers: 4, 8, 15, 16, 23, 42",
    "document-qa": (
        "document: The library opens at nine in the morning. It closes "
        "at eight in the evening.\nquestion: when does the library open"
    ),
    "all-in-one": "coder: write a python function that adds two numbers",
}


@dataclass(frozen=True)
class CatalogIssue:
    subject: str
    check: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {"subject": self.subject, "check": self.check, "detail": self.detail}


@dataclass(frozen=True)
class ModelVerification:
    model_id: str
    ok: bool
    parameter_count: int | None = None
    output_length: int = 0
    issues: tuple[CatalogIssue, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "ok": self.ok,
            "parameter_count": self.parameter_count,
            "output_length": self.output_length,
            "issues": [issue.to_dict() for issue in self.issues],
        }


@dataclass(frozen=True)
class ModuleVerification:
    name: str
    ok: bool
    model_id: str | None = None
    output_length: int = 0
    issues: tuple[CatalogIssue, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "ok": self.ok,
            "model_id": self.model_id,
            "output_length": self.output_length,
            "issues": [issue.to_dict() for issue in self.issues],
        }


@dataclass(frozen=True)
class CatalogVerificationReport:
    ok: bool
    models: tuple[ModelVerification, ...]
    modules: tuple[ModuleVerification, ...]
    issues: tuple[CatalogIssue, ...] = field(default_factory=tuple)

    @property
    def verified_model_ids(self) -> frozenset[str]:
        return frozenset(model.model_id for model in self.models if model.ok)

    @property
    def verified_module_names(self) -> frozenset[str]:
        return frozenset(module.name for module in self.modules if module.ok)

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "verified_model_ids": sorted(self.verified_model_ids),
            "verified_module_names": sorted(self.verified_module_names),
            "models": [model.to_dict() for model in self.models],
            "modules": [module.to_dict() for module in self.modules],
            "issues": [issue.to_dict() for issue in self.issues],
        }


def _issue(subject: str, check: str, detail: str) -> CatalogIssue:
    return CatalogIssue(subject=subject, check=check, detail=detail)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _module_file_hashes(module_dir: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(module_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(module_dir)
        if any(part in _EXCLUDED_PARTS for part in relative.parts):
            continue
        if path.suffix in (".pyc", ".pyo"):
            continue
        hashes[relative.as_posix()] = _hash_file(path)
    return hashes


def _first_seed_example(model_id: str) -> tuple[str, str]:
    seed_path = bundled_seed_path(model_id)
    first_line = next(
        line for line in seed_path.read_text(encoding="utf-8").splitlines() if line.strip()
    )
    record = json.loads(first_line)
    return str(record["prompt"]), str(record["completion"])


def _verify_model(
    entry: ModelEntry,
    *,
    implemented_capabilities: set[str],
    provider: GravionCoreProvider,
    run_smoke: bool,
) -> ModelVerification:
    issues: list[CatalogIssue] = []
    output_length = 0

    if entry.status != "available":
        issues.append(_issue(entry.id, "status", f"model status is {entry.status!r}"))
    if not entry.weights_included:
        issues.append(
            _issue(entry.id, "weights_status", f"weights_status is {entry.weights_status!r}")
        )
    for capability in entry.capabilities:
        if capability not in implemented_capabilities:
            issues.append(
                _issue(entry.id, "capability", f"capability {capability!r} has no implemented module")
            )
    if entry.id not in LOCAL_MODEL_IDS:
        issues.append(_issue(entry.id, "local_model", "catalog entry has no bundled local model"))
        return ModelVerification(entry.id, False, entry.parameter_count, 0, tuple(issues))

    try:
        artifact_text = (
            resources.files("gravion_ai").joinpath(entry.weights_artifact).read_text(encoding="utf-8")
        )
        artifact = json.loads(artifact_text)
    except Exception as exc:  # noqa: BLE001 - verification reports, never crashes
        issues.append(_issue(entry.id, "weights_artifact", f"cannot load weights artifact: {exc}"))
        return ModelVerification(entry.id, False, entry.parameter_count, 0, tuple(issues))

    if artifact.get("model_id") != entry.id:
        issues.append(
            _issue(entry.id, "weights_model_id", f"artifact model_id is {artifact.get('model_id')!r}")
        )
    try:
        stored_parameter_count = count_parameters(artifact)
    except Exception as exc:  # noqa: BLE001
        issues.append(_issue(entry.id, "parameter_count", f"cannot count stored parameters: {exc}"))
        stored_parameter_count = None
    if stored_parameter_count != entry.parameter_count:
        issues.append(
            _issue(
                entry.id,
                "parameter_count",
                f"catalog={entry.parameter_count} stored={stored_parameter_count}",
            )
        )
    if artifact.get("weights_sha256") != entry.weights_sha256:
        issues.append(_issue(entry.id, "weights_sha256", "catalog weights hash does not match artifact"))

    try:
        model = load_bundled_model(entry.id)
        if model.model_id != entry.id:
            issues.append(_issue(entry.id, "load", f"loaded model_id is {model.model_id!r}"))
        if model.parameter_count != entry.parameter_count:
            issues.append(
                _issue(
                    entry.id,
                    "load_parameter_count",
                    f"loaded={model.parameter_count} catalog={entry.parameter_count}",
                )
            )
        if model.weights_sha256 != entry.weights_sha256:
            issues.append(_issue(entry.id, "load_weights_sha256", "loaded weights hash mismatch"))
    except Exception as exc:  # noqa: BLE001
        issues.append(_issue(entry.id, "load", f"cannot load bundled model: {exc}"))

    try:
        prompt, completion = _first_seed_example(entry.id)
    except Exception as exc:  # noqa: BLE001
        issues.append(_issue(entry.id, "seed", f"cannot read bundled seed: {exc}"))
        prompt, completion = "", ""

    if run_smoke and prompt:
        try:
            output = provider.complete([{"role": "user", "content": prompt}], model=entry.id)
            output_length = len(output)
            if output != completion:
                issues.append(
                    _issue(entry.id, "smoke", "provider output does not match first seed completion")
                )
        except Exception as exc:  # noqa: BLE001
            issues.append(_issue(entry.id, "smoke", f"provider smoke failed: {exc}"))

    return ModelVerification(
        model_id=entry.id,
        ok=not issues,
        parameter_count=entry.parameter_count,
        output_length=output_length,
        issues=tuple(issues),
    )


def _verify_module_manifest(manifest) -> list[CatalogIssue]:
    issues: list[CatalogIssue] = []
    subject = manifest.name
    if manifest.path is None:
        return [_issue(subject, "manifest", "module has no filesystem path")]
    module_dir = Path(manifest.path)
    manifest_path = module_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        return [_issue(subject, "manifest", "module manifest.json is missing")]
    try:
        signed = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload = signed["payload"]
    except Exception as exc:  # noqa: BLE001
        return [_issue(subject, "manifest", f"module manifest is invalid: {exc}")]

    expected = {
        "name": manifest.name,
        "type": manifest.type.value,
        "version": manifest.version,
        "entrypoint": manifest.entrypoint,
    }
    for field_name, expected_value in expected.items():
        if payload.get(field_name) != expected_value:
            issues.append(
                _issue(
                    subject,
                    "manifest_identity",
                    f"{field_name}={payload.get(field_name)!r} registry={expected_value!r}",
                )
            )
    recorded_files = payload.get("files")
    if not isinstance(recorded_files, dict):
        issues.append(_issue(subject, "manifest_files", "manifest files must be an object"))
        return issues
    actual_files = _module_file_hashes(module_dir)
    if set(recorded_files) != set(actual_files):
        missing = sorted(set(recorded_files) - set(actual_files))
        extra = sorted(set(actual_files) - set(recorded_files))
        issues.append(
            _issue(subject, "manifest_files", f"missing={missing} extra={extra}")
        )
    changed = sorted(
        name for name in recorded_files if recorded_files.get(name) != actual_files.get(name)
    )
    if changed:
        issues.append(_issue(subject, "manifest_hashes", f"stale file hashes: {changed}"))
    return issues


def _verify_module(
    manifest,
    *,
    catalog: ModelCatalog,
    registry: ModuleRegistry,
    verified_model_ids: frozenset[str],
    provider: GravionCoreProvider,
    run_smoke: bool,
) -> ModuleVerification:
    issues: list[CatalogIssue] = []
    model_id: str | None = None
    output_length = 0

    if manifest.status is not ModuleStatus.IMPLEMENTED:
        issues.append(_issue(manifest.name, "status", f"module status is {manifest.status.value!r}"))
        return ModuleVerification(manifest.name, False, None, 0, tuple(issues))

    issues.extend(_verify_module_manifest(manifest))

    try:
        module = registry.instantiate(manifest.name)
    except Exception as exc:  # noqa: BLE001
        issues.append(_issue(manifest.name, "entrypoint", f"cannot instantiate module: {exc}"))
        module = None

    if manifest.name == "all-in-one":
        model_id = "gravion-omni" if "gravion-omni" in verified_model_ids else None
    else:
        serving = [
            entry
            for entry in catalog.models_for(manifest.type.value)
            if entry.id in verified_model_ids
        ]
        model_id = serving[0].id if serving else None
    if model_id is None:
        issues.append(_issue(manifest.name, "serving_model", "no verified model serves this module"))

    if run_smoke and module is not None and model_id is not None:
        try:
            context = ModuleContext("You are Gravion. Be helpful.", provider, model=model_id)
            output = module.run(context, MODULE_SMOKE_PROMPTS.get(manifest.name, "Hello"))
            output_length = len(output)
            if not isinstance(output, str) or not output.strip():
                issues.append(_issue(manifest.name, "smoke", "module smoke output is empty"))
        except Exception as exc:  # noqa: BLE001
            issues.append(_issue(manifest.name, "smoke", f"module smoke failed: {exc}"))

    return ModuleVerification(
        name=manifest.name,
        ok=not issues,
        model_id=model_id,
        output_length=output_length,
        issues=tuple(issues),
    )


def verify_catalog(
    catalog: ModelCatalog | None = None,
    registry: ModuleRegistry | None = None,
    provider: GravionCoreProvider | None = None,
    *,
    run_smoke: bool = True,
) -> CatalogVerificationReport:
    """Verify every catalog model and module entry.

    The report is the gate for routing/selection data: only entries
    with ``ok=True`` are returned by :func:`select_verified_model` and
    included in routing training examples.
    """

    catalog = catalog or ModelCatalog.load()
    registry = registry or default_registry()
    provider = provider or GravionCoreProvider()

    global_issues: list[CatalogIssue] = []
    catalog_ids = {entry.id for entry in catalog.list()}
    local_ids = set(LOCAL_MODEL_IDS)
    for model_id in sorted(catalog_ids - local_ids):
        global_issues.append(_issue(model_id, "stale_entry", "catalog entry has no bundled local model"))
    for model_id in sorted(local_ids - catalog_ids):
        global_issues.append(_issue(model_id, "missing_entry", "bundled local model is missing from catalog"))

    implemented_capabilities = {
        manifest.type.value
        for manifest in registry.list()
        if manifest.status is ModuleStatus.IMPLEMENTED
    }
    for capability in sorted(CAPABILITIES):
        if capability not in implemented_capabilities:
            global_issues.append(
                _issue(capability, "capability_module", "capability has no implemented module")
            )
        for tier in sorted(TIERS):
            serving = [
                entry
                for entry in catalog.models_for(capability, tier=tier)
                if entry.status == "available"
            ]
            if not serving:
                global_issues.append(
                    _issue(capability, "tier_coverage", f"no available model for tier {tier}")
                )

    model_results = tuple(
        _verify_model(
            entry,
            implemented_capabilities=implemented_capabilities,
            provider=provider,
            run_smoke=run_smoke,
        )
        for entry in catalog.list()
    )
    verified_model_ids = frozenset(model.model_id for model in model_results if model.ok)

    module_results = tuple(
        _verify_module(
            manifest,
            catalog=catalog,
            registry=registry,
            verified_model_ids=verified_model_ids,
            provider=provider,
            run_smoke=run_smoke,
        )
        for manifest in registry.list()
    )

    all_issues = tuple(global_issues) + tuple(
        issue for result in (*model_results, *module_results) for issue in result.issues
    )
    return CatalogVerificationReport(
        ok=not all_issues,
        models=model_results,
        modules=module_results,
        issues=tuple(global_issues),
    )


def select_verified_model(
    catalog: ModelCatalog,
    report: CatalogVerificationReport,
    capability: str,
    *,
    tier: str | None = None,
) -> ModelEntry | None:
    """Select the first verified model serving a capability."""

    for entry in catalog.models_for(capability, tier=tier):
        if entry.id in report.verified_model_ids:
            return entry
    return None


def catalog_routing_examples(
    catalog: ModelCatalog | None = None,
    report: CatalogVerificationReport | None = None,
) -> list[tuple[str, str]]:
    """Build routing/selection training examples from verified entries."""

    catalog = catalog or ModelCatalog.load()
    report = report or verify_catalog(catalog)
    examples: list[tuple[str, str]] = []
    for capability in sorted(CAPABILITIES):
        for tier in sorted(TIERS):
            selected = select_verified_model(catalog, report, capability, tier=tier)
            if selected is not None:
                examples.append(
                    (f"route capability {capability} tier {tier}", selected.id)
                )
    return examples


def evaluate_catalog_routing(
    catalog: ModelCatalog | None = None,
    report: CatalogVerificationReport | None = None,
) -> dict[str, object]:
    """Evaluate verified routing selection over every capability/tier."""

    catalog = catalog or ModelCatalog.load()
    report = report or verify_catalog(catalog)
    cases = 0
    correct = 0
    failures: list[dict[str, str]] = []
    for capability in sorted(CAPABILITIES):
        for tier in sorted(TIERS):
            cases += 1
            selected = select_verified_model(catalog, report, capability, tier=tier)
            if (
                selected is not None
                and selected.serves(capability)
                and selected.offered_in(tier)
                and selected.id in report.verified_model_ids
            ):
                correct += 1
            else:
                failures.append({"capability": capability, "tier": tier})
    return {
        "cases": cases,
        "correct": correct,
        "accuracy": round(correct / cases, 4) if cases else 0.0,
        "failures": failures,
        "training_examples": len(catalog_routing_examples(catalog, report)),
    }
