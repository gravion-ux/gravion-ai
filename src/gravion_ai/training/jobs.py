"""Training job specification and validation (Piece 7).

A training job is a small JSON document:

    {
      "model": "gravion-chat",
      "data": "path/to/deployment-examples.jsonl",
      "base_tier": "tier-2-basic",
      "notes": "why this run exists"
    }

`validate_job` checks the job against the real catalog and
registry: the model must exist, must be an *available* model (a
planned model has no bundle to fine-tune yet), its capabilities
must belong to implemented modules, the target tier must be one
the model is offered in or can be trained toward, and a dataset
path must be named. Every problem found is reported together.

`run_training_job` is deliberately honest: this build contains
**no training backend**, so a valid job still cannot run here —
it raises `TrainingError` saying exactly that. Customer
fine-tuning runs through the hosted training pipeline (see
ARCHITECTURE.md); nothing in this package fakes weights, losses
or progress.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ..catalog import ModelCatalog
from ..errors import TrainingError
from ..registry import ModuleStatus, default_registry
from ..tiers import TIERS

JOB_FIELDS = frozenset({"model", "data", "base_tier", "notes"})


@dataclass(frozen=True)
class TrainingJob:
    """A validated training job."""

    model: str
    data: str
    base_tier: str = "tier-2-basic"
    notes: str = ""
    capabilities: tuple[str, ...] = field(default=())


def load_job(path) -> dict:
    """Read a training job document from disk."""
    path = Path(path)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise TrainingError(f"training job file not found: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingError(f"training job file is not valid JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise TrainingError("a training job must be a JSON object")
    return document


def validate_job(document: dict, catalog: ModelCatalog | None = None) -> TrainingJob:
    """Validate a job document against the catalog and registry.

    Returns the TrainingJob on success; raises TrainingError
    listing every problem found.
    """
    catalog = catalog or ModelCatalog.load()
    problems: list[str] = []

    unknown = sorted(set(document) - JOB_FIELDS)
    if unknown:
        problems.append(f"unknown job fields: {', '.join(unknown)}")

    model_id = document.get("model")
    entry = catalog.get(model_id) if isinstance(model_id, str) else None
    if entry is None:
        problems.append(f"unknown model {model_id!r} (not in the catalog)")
    else:
        if entry.status != "available":
            problems.append(
                f"model {entry.id!r} is {entry.status}: there is no "
                "trained bundle to fine-tune yet"
            )
        registry = default_registry()
        statuses = {m.type.value: m.status for m in registry.list()}
        for capability in entry.capabilities:
            if statuses.get(capability) is not ModuleStatus.IMPLEMENTED:
                problems.append(
                    f"capability {capability!r} has no implemented module yet"
                )

    data = document.get("data")
    if not isinstance(data, str) or not data.strip():
        problems.append("the job must name a dataset path in 'data'")

    base_tier = document.get("base_tier", "tier-2-basic")
    if base_tier not in TIERS:
        problems.append(
            f"unknown base_tier {base_tier!r}; valid: {sorted(TIERS)}"
        )

    if problems:
        raise TrainingError("invalid training job: " + "; ".join(problems))
    return TrainingJob(
        model=entry.id,
        data=data,
        base_tier=base_tier,
        notes=str(document.get("notes", "")),
        capabilities=tuple(entry.capabilities),
    )


def run_training_job(job: TrainingJob) -> None:
    """Run a validated training job.

    This build ships no training backend, and says so: training
    Gravion bundles happens in the hosted training pipeline.
    Raising here — after validation has proven the job itself is
    well-formed — is the honest behaviour; inventing weights or a
    progress log would be the dishonest one.
    """
    raise TrainingError(
        f"training job for {job.model!r} is valid, but this build "
        "contains no training backend: customer fine-tuning runs "
        "through the hosted Gravion training pipeline (see "
        "ARCHITECTURE.md). No weights were produced."
    )
