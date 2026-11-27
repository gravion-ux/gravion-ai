"""Generate the Gravion model specification file.

Every measured field — architecture, embedding width, vocabulary and
class counts, storage dtype, quantization method and the parameter
count, split by storage dtype — is computed from the shipped weight
artifacts. The licence, the specification version and the optional
base-model lineage are declared in ``DECLARED`` below, or kept from
the previous file so a hand-set version is not silently lost.

Run it after retraining or adding a model:

    python tools/build_model_specs.py

The test suite fails if the checked-in file drifts away from the
artifacts, so the file and the weights can never disagree.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from gravion_ai.errors import CatalogError
from gravion_ai.model_spec import derive_spec, load_specs

DATA = ROOT / "src" / "gravion_ai" / "data"
REGISTRY = DATA / "models.json"
OUTPUT = DATA / "model_specs.json"

# The licence every bundled Gravion model ships under.
LICENSE_ID = "MIT"
LICENSE_LINK = "https://github.com/gravion-ux/gravion-ai/blob/main/LICENSE"

# The version of the specification itself, not of the software.
VERSION = "1.0.0"

COMMENT = (
    "The Gravion model specifications. Every measured field — architecture, "
    "embedding width, vocabulary and class counts, storage dtype, "
    "quantization method and the parameter count — is computed from the "
    "shipped weight artifacts; the licence, the specification version and "
    "the optional base-model lineage are declared. Regenerate with "
    "`python tools/build_model_specs.py`; the test suite fails if this file "
    "drifts away from the artifacts."
)


def _declared_for(model_id: str, previous: dict) -> dict[str, object]:
    """The curated fields for a model, keeping any hand-set values."""
    declared = {
        "license": LICENSE_ID,
        "license_link": LICENSE_LINK,
        "version": VERSION,
        "base_model": None,
        "base_model_relation": None,
    }
    existing = previous.get(model_id)
    if existing is not None:
        for field in declared:
            if field in existing:
                declared[field] = existing[field]
    return declared


def main() -> int:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    models = registry.get("models")
    if not isinstance(models, list) or not models:
        print("registry has no models")
        return 1

    previous = {}
    if OUTPUT.exists():
        try:
            previous = {k: v for k, v in load_specs(OUTPUT).items()}
        except CatalogError as exc:  # a stale file must not block a rebuild
            print(f"ignoring unreadable {OUTPUT.name}: {exc}")

    specs = []
    for entry in models:
        model_id = entry.get("id")
        artifact_path = entry.get("weights_artifact")
        if not isinstance(model_id, str) or not isinstance(artifact_path, str):
            print(f"registry entry without an id or artifact: {entry!r}")
            return 1
        artifact_file = DATA / Path(artifact_path).name
        if not artifact_file.exists():
            print(f"{model_id}: no weights artifact at {artifact_file}")
            return 1
        artifact = json.loads(artifact_file.read_text(encoding="utf-8"))
        declared = _declared_for(model_id, previous)
        spec = derive_spec(artifact, declared)
        specs.append(spec.to_dict())

    document = {
        "comment": COMMENT,
        "version": registry.get("version", 1),
        "specs": specs,
    }
    OUTPUT.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(specs)} model specs to {OUTPUT}")
    for spec in specs:
        print(
            f"  {spec['id']:<16} {spec['parameter_count']:>9} params  "
            f"{spec['architecture']}  {spec['dtype']}/{spec['quantization']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
