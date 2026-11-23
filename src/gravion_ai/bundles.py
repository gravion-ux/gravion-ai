"""Tier bundles (Piece 7) — packaging the tiers over the module
system.

A *tier bundle* is the concrete, checkable form of a tier: for
every module in the registry it records which available models
serve that module's capability at the tier, and whether the tier
is **complete** — every implemented module has at least one
available model offered at the tier. A bundle descriptor is a
deterministic JSON document with a SHA-256 hash over its canonical
spec, so two builds of the same tier over the same registry and
catalog produce byte-identical descriptors.

Honesty rules, as everywhere in this project:

- A tier with gaps **cannot be built**: `build_bundle` raises
  `BundleError` naming the missing capabilities instead of writing
  a descriptor that claims coverage it does not have. (At this
  piece, Tier 4 is exactly that case — the Advanced bundles land
  with Gravion Omni in the next piece.)
- A bundle descriptor is a *packaging manifest*, not model
  weights: it records what a deployment at that tier is entitled
  to run, resolved from the same catalog the runtime uses.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import __version__
from .catalog import CAPABILITIES, ModelCatalog
from .errors import BundleError
from .registry import ModuleRegistry, ModuleStatus, default_registry
from .tiers import TIERS, get_tier


def bundle_spec(
    tier_id: str,
    registry: ModuleRegistry | None = None,
    catalog: ModelCatalog | None = None,
) -> dict:
    """Describe one tier bundle: per-module model coverage and
    whether the tier is complete.

    A module type is a *gap* when the module is implemented but no
    available model serves its capability at this tier. The
    all-in-one module adds no capability of its own — it packages
    the nine real ones — so it is listed with its serving models
    (the entries covering every capability) but is never a gap.
    """
    try:
        tier = get_tier(tier_id)
    except ValueError as exc:
        raise BundleError(str(exc)) from exc
    registry = registry or default_registry()
    catalog = catalog or ModelCatalog.load()

    modules = []
    gaps: list[str] = []
    for manifest in sorted(registry.list(), key=lambda m: m.name):
        if manifest.type.value in CAPABILITIES:
            models = sorted(
                entry.id
                for entry in catalog.models_for(
                    manifest.type.value, tier=tier_id
                )
                if entry.status == "available"
            )
            implemented = manifest.status is ModuleStatus.IMPLEMENTED
            if implemented and not models:
                gaps.append(manifest.type.value)
        else:
            # The all-in-one module adds no capability of its own —
            # it packages the nine real ones — so it is never a
            # coverage gap. Its serving models are the available
            # entries that cover every capability (Gravion Omni),
            # listed for information.
            models = sorted(
                entry.id
                for entry in catalog.list()
                if entry.status == "available"
                and entry.offered_in(tier_id)
                and set(CAPABILITIES) <= set(entry.capabilities)
            )
        modules.append(
            {
                "module": manifest.name,
                "type": manifest.type.value,
                "status": manifest.status.value,
                "models": models,
            }
        )
    return {
        "tier": tier.id,
        "tier_label": tier.label,
        "training_level": tier.training_level,
        "modules": modules,
        "complete": not gaps,
        "gaps": gaps,
    }


def bundle_hash(spec: dict) -> str:
    """The SHA-256 of a bundle spec's canonical JSON form."""
    canonical = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_bundle(
    tier_id: str,
    out_path,
    registry: ModuleRegistry | None = None,
    catalog: ModelCatalog | None = None,
) -> dict:
    """Write a tier bundle descriptor to `out_path` and return it.

    Refuses — with the gaps named — when the tier is incomplete:
    a bundle file is a claim about coverage, and this project does
    not write claims it cannot back.
    """
    spec = bundle_spec(tier_id, registry=registry, catalog=catalog)
    if not spec["complete"]:
        raise BundleError(
            f"tier bundle {tier_id!r} is incomplete: no available model "
            f"serves {', '.join(spec['gaps'])} at this tier yet"
        )
    document = {
        "bundle": spec,
        "bundle_hash": bundle_hash(spec),
        "built_by": f"gravion-ai {__version__}",
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(document, indent=2) + "\n", encoding="utf-8"
    )
    return document


def all_bundle_specs(
    registry: ModuleRegistry | None = None,
    catalog: ModelCatalog | None = None,
) -> list[dict]:
    """The bundle spec of every tier, in tier order."""
    registry = registry or default_registry()
    catalog = catalog or ModelCatalog.load()
    return [
        bundle_spec(tier_id, registry=registry, catalog=catalog)
        for tier_id in TIERS
    ]
