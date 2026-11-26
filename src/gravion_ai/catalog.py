"""The Gravion model registry.

Gravion AI is one product family with its own models. This registry is
the structured list of those models: each entry is a Gravion model or
variant, the capabilities it serves (the module types), the tiers it
is offered in, and its build status.

Modules declare the *capability* they need; the registry resolves
which Gravion model entries can serve it. The registry ships as data
(data/models.json). ``status="available"`` means the registry entry
and module/bundle packaging path are available to resolve; it does
not by itself claim that trained weights are included. The separate
``weights_status`` field states the weights position explicitly:
Gravion Core includes compact local trained weights (with a
deterministic fallback), and the specialist entries are
packaging-ready entries with no trained weights included in this
repository. Real specialist model behaviour requires an
operator-supplied endpoint or a future operations-supplied weights
bundle.

The registry is data, not code: a deployment can replace it wholesale
by placing its own `models.json` in the deployment home (same schema,
same validation), which is how private Gravion variants get registered
without touching the platform source.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from .errors import CatalogError
from .registry import ModuleType
from .tiers import TIERS

# Capabilities a model can serve = module types, minus the all-in-one
# bundle (a package of modules, not a model capability).
CAPABILITIES = frozenset(
    t.value for t in ModuleType if t is not ModuleType.ALL_IN_ONE
)
STATUSES = frozenset({"available", "planned"})
WEIGHTS_STATUSES = frozenset(
    {
        "not_applicable",
        "packaging_ready_no_weights",
        "planned_no_weights",
        "operator_supplied",
        "included",
        "unspecified",
    }
)
REQUIRED_FIELDS = ("id", "name", "capabilities", "tiers", "status")


@dataclass(frozen=True)
class ModelEntry:
    id: str
    name: str
    capabilities: tuple[str, ...]
    tiers: tuple[str, ...]
    status: str
    context_window: int | None = None
    notes: str = ""
    weights_status: str = "unspecified"
    parameter_count: int | None = None
    weights_artifact: str = ""
    weights_sha256: str = ""

    @property
    def weights_included(self) -> bool:
        """True only when the registry explicitly says weights ship
        with the entry. ``status="available"`` alone never implies
        trained weights are present."""
        return self.weights_status == "included"

    def serves(self, capability: str) -> bool:
        return capability in self.capabilities

    def offered_in(self, tier_id: str) -> bool:
        return tier_id in self.tiers


def _validate_entry(raw: object, index: int) -> ModelEntry:
    where = f"registry entry #{index}"
    if not isinstance(raw, dict):
        raise CatalogError(f"{where}: entry must be a JSON object")
    if raw.get("id"):
        where = f"registry entry #{index} (id {raw['id']!r})"
    for field_name in REQUIRED_FIELDS:
        if field_name not in raw:
            raise CatalogError(f"{where}: missing required field {field_name!r}")
    if not isinstance(raw["id"], str) or not raw["id"]:
        raise CatalogError(f"{where}: 'id' must be a non-empty string")
    capabilities = raw["capabilities"]
    if not isinstance(capabilities, list) or not capabilities:
        raise CatalogError(f"{where}: 'capabilities' must be a non-empty list")
    unknown_caps = sorted(set(capabilities) - CAPABILITIES)
    if unknown_caps:
        raise CatalogError(
            f"{where}: unknown capabilities {unknown_caps}; "
            f"valid capabilities are {sorted(CAPABILITIES)}"
        )
    tiers = raw["tiers"]
    if not isinstance(tiers, list) or not tiers:
        raise CatalogError(f"{where}: 'tiers' must be a non-empty list")
    unknown_tiers = sorted(set(tiers) - set(TIERS))
    if unknown_tiers:
        raise CatalogError(
            f"{where}: unknown tiers {unknown_tiers}; valid tiers are {sorted(TIERS)}"
        )
    if raw["status"] not in STATUSES:
        raise CatalogError(
            f"{where}: 'status' must be one of {sorted(STATUSES)}, "
            f"got {raw['status']!r}"
        )
    weights_status = raw.get("weights_status", "unspecified")
    if weights_status not in WEIGHTS_STATUSES:
        raise CatalogError(
            f"{where}: 'weights_status' must be one of "
            f"{sorted(WEIGHTS_STATUSES)}, got {weights_status!r}"
        )
    context_window = raw.get("context_window")
    if context_window is not None and (
        not isinstance(context_window, int) or context_window <= 0
    ):
        raise CatalogError(
            f"{where}: 'context_window' must be a positive integer or null"
        )
    parameter_count = raw.get("parameter_count")
    if parameter_count is not None and (
        isinstance(parameter_count, bool)
        or not isinstance(parameter_count, int)
        or parameter_count < 0
    ):
        raise CatalogError(
            f"{where}: 'parameter_count' must be a non-negative integer or null"
        )
    weights_artifact = raw.get("weights_artifact", "")
    if not isinstance(weights_artifact, str):
        raise CatalogError(f"{where}: 'weights_artifact' must be a string")
    weights_sha256 = raw.get("weights_sha256", "")
    if not isinstance(weights_sha256, str):
        raise CatalogError(f"{where}: 'weights_sha256' must be a string")
    if weights_status == "included" and (
        not parameter_count or not weights_artifact or not weights_sha256
    ):
        raise CatalogError(
            f"{where}: included weights require parameter_count, "
            "weights_artifact, and weights_sha256"
        )
    return ModelEntry(
        id=raw["id"],
        name=raw["name"],
        capabilities=tuple(capabilities),
        tiers=tuple(tiers),
        status=raw["status"],
        context_window=context_window,
        notes=raw.get("notes", ""),
        weights_status=weights_status,
        parameter_count=parameter_count,
        weights_artifact=weights_artifact,
        weights_sha256=weights_sha256,
    )


class ModelCatalog:
    """The loaded, validated Gravion model registry."""

    def __init__(self, entries: list[ModelEntry], source: str = "<data>"):
        self._entries = entries
        self._by_id = {e.id: e for e in entries}
        self.source = source

    # -- loading ---------------------------------------------------------
    @classmethod
    def from_data(cls, data: object, source: str = "<data>") -> "ModelCatalog":
        if not isinstance(data, dict) or not isinstance(data.get("models"), list):
            raise CatalogError(
                f"{source}: registry must be a JSON object with a 'models' list"
            )
        entries: list[ModelEntry] = []
        seen: set[str] = set()
        for index, raw in enumerate(data["models"]):
            entry = _validate_entry(raw, index)
            if entry.id in seen:
                raise CatalogError(
                    f"registry entry #{index} (id {entry.id!r}): duplicate id"
                )
            seen.add(entry.id)
            entries.append(entry)
        if not entries:
            raise CatalogError(f"{source}: registry contains no models")
        return cls(entries, source=source)

    @classmethod
    def load(cls, path: Path | None = None) -> "ModelCatalog":
        """Load a registry file, or the bundled Gravion registry."""
        if path is not None:
            path = Path(path)
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise CatalogError(f"cannot read registry at {path}: {exc}") from exc
            return cls.from_data(data, source=str(path))
        resource = resources.files("gravion_ai").joinpath("data", "models.json")
        data = json.loads(resource.read_text(encoding="utf-8"))
        return cls.from_data(data, source="bundled Gravion model registry")

    @classmethod
    def load_for_home(cls, home: Path) -> "ModelCatalog":
        """A deployment's own models.json (in its home) wins over the
        bundled registry — private Gravion variants register there."""
        override = Path(home) / "models.json"
        if override.exists():
            return cls.load(override)
        return cls.load()

    # -- lookup ------------------------------------------------------------
    def list(self) -> list[ModelEntry]:
        return list(self._entries)

    def get(self, entry_id: str) -> ModelEntry | None:
        return self._by_id.get(entry_id)

    def models_for(
        self, capability: str | ModuleType, *, tier: str | None = None
    ) -> list[ModelEntry]:
        """All Gravion models that can serve a capability (optionally
        restricted to one tier), in registry order."""
        value = capability.value if isinstance(capability, ModuleType) else capability
        if value not in CAPABILITIES:
            raise CatalogError(
                f"unknown capability {value!r}; valid: {sorted(CAPABILITIES)}"
            )
        result = [e for e in self._entries if e.serves(value)]
        if tier is not None:
            if tier not in TIERS:
                raise CatalogError(f"unknown tier {tier!r}; valid: {sorted(TIERS)}")
            result = [e for e in result if e.offered_in(tier)]
        return result
