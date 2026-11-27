"""Model specifications for the Gravion model registry.

The weights artifact that ships with a Gravion model already records
what a specification needs: the architecture it was trained with, the
embedding width, the quantized tensor encoding, the number of answer
classes and the vocabulary size. This module *measures* those facts
from the artifact, so the registry never repeats — or guesses — a
number it can count.

A specification adds what an artifact cannot know: the licence the
model is distributed under, the version of the specification, and the
lineage relation to a base model when one is declared.

Storage dtype and quantization method are kept apart on purpose.
``int8`` is how a tensor is stored; ``symmetric-int8`` is how it was
quantized. A model that stores int8 tensors quantized symmetrically
reports both, and its parameters are reported per storage dtype, so a
float32 bias vector is never counted as an int8 weight.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from importlib import resources
from pathlib import Path

from .errors import CatalogError

# How a tensor is stored. A dtype says nothing about how the values
# were produced.
DTYPES = frozenset(
    {"float32", "float16", "bfloat16", "int8", "uint8", "int4"}
)

# How a tensor was quantized. A quantization method is not a dtype.
QUANTIZATIONS = frozenset(
    {"none", "symmetric-int8", "symmetric-int4", "symmetric-fp8"}
)

# Architecture slugs. The artifact records the architecture in prose;
# the slug is the stable name a registry can key on.
ARCHITECTURES = frozenset({"sparse-ngram-linear"})

ARCHITECTURE_DETAILS = {
    "sparse-ngram-linear": (
        "sparse n-gram embedding + direct linear softmax classifier"
    ),
}

# SPDX identifiers. A model that is distributed under a licence outside
# this list is a packaging mistake, not a detail to skip.
LICENSES = frozenset(
    {
        "MIT",
        "Apache-2.0",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "ISC",
        "MPL-2.0",
        "GPL-3.0-only",
        "GPL-3.0-or-later",
        "LGPL-3.0-only",
        "AGPL-3.0-only",
        "CC0-1.0",
        "CC-BY-4.0",
        "Unlicense",
    }
)

# How a model relates to the model it was derived from.
BASE_MODEL_RELATIONS = frozenset(
    {"base", "fine-tuned", "adapted", "quantized-from", "merged"}
)

# The quantized tensor blocks a v3 artifact stores.
TENSOR_BLOCKS = ("direct", "embedding", "classifier")

# Artifact tensor encoding -> (storage dtype, quantization method).
ENCODINGS = {
    "int8-base64": ("int8", "symmetric-int8"),
    "int4-base64": ("int4", "symmetric-int4"),
    "float32": ("float32", "none"),
}

VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")


def format_parameters(count: int | None) -> str:
    """A parameter count as a short human-readable size."""
    if count is None:
        return "-"
    if isinstance(count, bool) or not isinstance(count, int):
        raise CatalogError(f"parameter count must be an integer, got {count!r}")
    if count < 0:
        raise CatalogError(f"parameter count must not be negative, got {count!r}")
    if count >= 1_000_000_000:
        return f"{count / 1_000_000_000:.2f}B"
    if count >= 1_000_000:
        return f"{count / 1_000_000:.2f}M"
    if count >= 1_000:
        return f"{count / 1_000:.2f}K"
    return str(count)


def architecture_slug(detail: str) -> str:
    """The stable architecture slug for an artifact's architecture."""
    for slug, known in ARCHITECTURE_DETAILS.items():
        if detail == known:
            return slug
    raise CatalogError(f"unknown architecture {detail!r}; known: {sorted(ARCHITECTURES)}")


@dataclass(frozen=True)
class ModelSpec:
    """A measured specification of one Gravion model."""

    id: str
    parameter_count: int
    architecture: str
    architecture_detail: str
    hidden_size: int
    vocab_size: int
    feature_count: int
    output_classes: int
    dtype: str
    quantization: str
    parameters_by_dtype: dict[str, int]
    license: str
    license_link: str
    version: str
    base_model: str | None
    base_model_relation: str | None
    trained_from: str
    dataset_sha256: str
    weights_sha256: str
    spec_digest: str

    def declared(self) -> dict[str, object]:
        """The curated fields — the ones an artifact cannot know."""
        return {
            "license": self.license,
            "license_link": self.license_link,
            "version": self.version,
            "base_model": self.base_model,
            "base_model_relation": self.base_model_relation,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "parameter_count": self.parameter_count,
            "architecture": self.architecture,
            "architecture_detail": self.architecture_detail,
            "hidden_size": self.hidden_size,
            "vocab_size": self.vocab_size,
            "feature_count": self.feature_count,
            "output_classes": self.output_classes,
            "dtype": self.dtype,
            "quantization": self.quantization,
            "parameters_by_dtype": dict(self.parameters_by_dtype),
            "license": self.license,
            "license_link": self.license_link,
            "version": self.version,
            "base_model": self.base_model,
            "base_model_relation": self.base_model_relation,
            "trained_from": self.trained_from,
            "dataset_sha256": self.dataset_sha256,
            "weights_sha256": self.weights_sha256,
            "spec_digest": self.spec_digest,
        }


def spec_digest(spec: ModelSpec) -> str:
    """The SHA-256 of a spec's canonical JSON form.

    Same shape of spec in, same digest out; one changed field, one
    changed digest. The digest covers the measured facts *and* the
    declared ones, so a re-trained model and a re-licensed model are
    both visible.
    """
    payload = spec.to_dict()
    payload.pop("spec_digest", None)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _tensor_encoding(artifact: Mapping[str, object], model_id: str) -> tuple[str, str]:
    """The storage dtype and quantization method of the tensor blocks."""
    encodings: set[str] = set()
    for name in TENSOR_BLOCKS:
        block = artifact.get(name)
        if not isinstance(block, Mapping):
            raise CatalogError(f"{model_id}: {name} weights are missing")
        encoding = block.get("encoding")
        if not isinstance(encoding, str) or not encoding:
            raise CatalogError(f"{model_id}: {name} weights have no encoding")
        if encoding not in ENCODINGS:
            raise CatalogError(
                f"{model_id}: quantization encoding {encoding!r} is not supported"
            )
        encodings.add(encoding)
    if len(encodings) != 1:
        raise CatalogError(
            f"{model_id}: tensor blocks use different encodings {sorted(encodings)}"
        )
    return ENCODINGS[encodings.pop()]


def _parameters_by_dtype(
    artifact: Mapping[str, object], model_id: str, dtype: str
) -> dict[str, int]:
    """The counted parameters, split by the dtype they are stored in."""
    dim = artifact["dim"]
    features = len(artifact["features"])
    classes = len(artifact["classes"])
    quantized = {
        "direct": features * classes,
        "embedding": features * dim,
        "classifier": classes * dim,
    }
    stored = sum(quantized[name] for name in TENSOR_BLOCKS)

    floats = 0
    bias = artifact.get("bias")
    if isinstance(bias, list):
        floats += sum(1 for value in bias if _is_number(value))
    transitions = artifact.get("transitions")
    if isinstance(transitions, Mapping):
        for values in transitions.values():
            if isinstance(values, Mapping):
                floats += sum(1 for value in values.values() if _is_number(value))
    unigram = artifact.get("unigram_weights")
    if isinstance(unigram, Mapping):
        floats += sum(1 for value in unigram.values() if _is_number(value))

    by_dtype: dict[str, int] = {}
    by_dtype[dtype] = by_dtype.get(dtype, 0) + stored
    by_dtype["float32"] = by_dtype.get("float32", 0) + floats
    return by_dtype


def derive_spec(
    artifact: Mapping[str, object], declared: Mapping[str, object] | None = None
) -> ModelSpec:
    """Measure a specification from a weights artifact.

    `declared` carries the curated fields the artifact cannot know:
    ``license``, ``license_link``, ``version`` and the optional
    ``base_model`` / ``base_model_relation`` lineage.
    """
    if not isinstance(artifact, Mapping):
        raise CatalogError("weights artifact must be a JSON object")
    declared = dict(declared or {})

    model_id = artifact.get("model_id")
    if not isinstance(model_id, str) or not model_id:
        raise CatalogError("weights artifact has no model_id")

    detail = artifact.get("architecture")
    if not isinstance(detail, str) or not detail:
        raise CatalogError(f"{model_id}: weights artifact has no architecture")

    dim = artifact.get("dim")
    features = artifact.get("features")
    classes = artifact.get("classes")
    vocabulary = artifact.get("vocabulary")
    if not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0:
        raise CatalogError(f"{model_id}: weights artifact has no embedding width")
    if not isinstance(features, list) or not features:
        raise CatalogError(f"{model_id}: weights artifact has no feature vocabulary")
    if not isinstance(classes, list) or not classes:
        raise CatalogError(f"{model_id}: weights artifact has no answer classes")
    if not isinstance(vocabulary, list) or not vocabulary:
        raise CatalogError(f"{model_id}: weights artifact has no vocabulary")

    dtype, quantization = _tensor_encoding(artifact, model_id)
    by_dtype = _parameters_by_dtype(artifact, model_id, dtype)
    parameter_count = sum(by_dtype.values())
    stored_count = artifact.get("parameter_count")
    if stored_count != parameter_count:
        raise CatalogError(
            f"{model_id}: parameter_count does not match the stored weights "
            f"({stored_count!r} != {parameter_count})"
        )

    license_id = declared.get("license")
    if not isinstance(license_id, str) or license_id not in LICENSES:
        raise CatalogError(
            f"{model_id}: license must be one of {sorted(LICENSES)}, "
            f"got {license_id!r}"
        )
    license_link = declared.get("license_link", "")
    if not isinstance(license_link, str):
        raise CatalogError(f"{model_id}: license_link must be a string")

    version = declared.get("version")
    if not isinstance(version, str) or not VERSION_RE.match(version):
        raise CatalogError(
            f"{model_id}: version must be a MAJOR.MINOR.PATCH string, got {version!r}"
        )

    base_model = declared.get("base_model")
    if base_model is not None and (
        not isinstance(base_model, str) or not base_model
    ):
        raise CatalogError(f"{model_id}: base_model must be a non-empty string or null")
    if base_model == model_id:
        raise CatalogError(f"{model_id}: base_model may not reference the model itself")
    relation = declared.get("base_model_relation")
    if base_model is None:
        if relation is not None:
            raise CatalogError(f"{model_id}: base_model_relation requires a base_model")
    elif relation not in BASE_MODEL_RELATIONS:
        raise CatalogError(
            f"{model_id}: base_model_relation must be one of "
            f"{sorted(BASE_MODEL_RELATIONS)}, got {relation!r}"
        )

    trained_from = artifact.get("trained_from")
    source = ""
    dataset_sha256 = ""
    if isinstance(trained_from, Mapping):
        source = trained_from.get("source") or ""
        dataset_sha256 = trained_from.get("dataset_sha256") or ""
    elif isinstance(trained_from, str):
        source = trained_from
    if not isinstance(source, str):
        source = ""
    if not isinstance(dataset_sha256, str):
        dataset_sha256 = ""

    weights_sha256 = artifact.get("weights_sha256", "")
    if not isinstance(weights_sha256, str):
        weights_sha256 = ""

    spec = ModelSpec(
        id=model_id,
        parameter_count=parameter_count,
        architecture=architecture_slug(detail),
        architecture_detail=detail,
        hidden_size=dim,
        vocab_size=len(vocabulary),
        feature_count=len(features),
        output_classes=len(classes),
        dtype=dtype,
        quantization=quantization,
        parameters_by_dtype=by_dtype,
        license=license_id,
        license_link=license_link,
        version=version,
        base_model=base_model,
        base_model_relation=relation,
        trained_from=source,
        dataset_sha256=dataset_sha256,
        weights_sha256=weights_sha256,
        spec_digest="",
    )
    return replace(spec, spec_digest=spec_digest(spec))


def validate_lineage(specs: Mapping[str, ModelSpec]) -> None:
    """Reject a base-model lineage that points at itself or loops."""
    for model_id, spec in specs.items():
        if spec.base_model == model_id:
            raise CatalogError(
                f"{model_id}: base_model may not reference the model itself"
            )
        if spec.base_model is not None and spec.base_model not in specs:
            raise CatalogError(
                f"{model_id}: base_model {spec.base_model!r} is not a known model"
            )
    for model_id in specs:
        chain: list[str] = []
        current: str | None = model_id
        while current is not None:
            if current in chain:
                raise CatalogError(
                    f"{model_id}: base_model lineage contains a cycle: "
                    f"{' -> '.join(chain + [current])}"
                )
            chain.append(current)
            spec = specs.get(current)
            current = spec.base_model if spec is not None else None


def _positive_int(raw: object, where: str, field: str) -> int:
    if not isinstance(raw, int) or isinstance(raw, bool) or raw <= 0:
        raise CatalogError(f"{where}: '{field}' must be a positive integer")
    return raw


def _spec_from_record(raw: Mapping[str, object], index: int) -> ModelSpec:
    where = f"model spec #{index}"
    model_id = raw.get("id")
    if not isinstance(model_id, str) or not model_id:
        raise CatalogError(f"{where}: 'id' must be a non-empty string")
    where = f"model spec #{index} (id {model_id!r})"

    architecture = raw.get("architecture")
    if architecture not in ARCHITECTURES:
        raise CatalogError(
            f"{where}: 'architecture' must be one of {sorted(ARCHITECTURES)}"
        )
    architecture_detail = raw.get("architecture_detail")
    if not isinstance(architecture_detail, str) or not architecture_detail:
        raise CatalogError(f"{where}: 'architecture_detail' must be a non-empty string")
    dtype = raw.get("dtype")
    if dtype not in DTYPES:
        raise CatalogError(f"{where}: 'dtype' must be one of {sorted(DTYPES)}")
    quantization = raw.get("quantization")
    if quantization not in QUANTIZATIONS:
        raise CatalogError(
            f"{where}: 'quantization' must be one of {sorted(QUANTIZATIONS)}"
        )
    license_id = raw.get("license")
    if license_id not in LICENSES:
        raise CatalogError(f"{where}: 'license' must be one of {sorted(LICENSES)}")
    license_link = raw.get("license_link", "")
    if not isinstance(license_link, str) or not license_link:
        raise CatalogError(f"{where}: 'license_link' must be a non-empty string")
    version = raw.get("version")
    if not isinstance(version, str) or not VERSION_RE.match(version):
        raise CatalogError(
            f"{where}: 'version' must be a MAJOR.MINOR.PATCH string, got {version!r}"
        )
    parameter_count = raw.get("parameter_count")
    if (
        not isinstance(parameter_count, int)
        or isinstance(parameter_count, bool)
        or parameter_count < 0
    ):
        raise CatalogError(f"{where}: 'parameter_count' must be a non-negative integer")
    by_dtype = raw.get("parameters_by_dtype")
    if not isinstance(by_dtype, Mapping) or not by_dtype:
        raise CatalogError(f"{where}: 'parameters_by_dtype' must be a non-empty object")
    counted: dict[str, int] = {}
    for name, value in by_dtype.items():
        if not isinstance(name, str) or name not in DTYPES:
            raise CatalogError(f"{where}: '{name}' is not a known storage dtype")
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise CatalogError(
                f"{where}: 'parameters_by_dtype'[{name!r}] must be a non-negative integer"
            )
        counted[name] = value
    if sum(counted.values()) != parameter_count:
        raise CatalogError(
            f"{where}: 'parameters_by_dtype' does not add up to 'parameter_count'"
        )
    base_model = raw.get("base_model")
    if base_model is not None and (
        not isinstance(base_model, str) or not base_model
    ):
        raise CatalogError(f"{where}: 'base_model' must be a non-empty string or null")
    relation = raw.get("base_model_relation")
    if relation is not None and relation not in BASE_MODEL_RELATIONS:
        raise CatalogError(
            f"{where}: 'base_model_relation' must be one of "
            f"{sorted(BASE_MODEL_RELATIONS)}"
        )
    trained_from = raw.get("trained_from", "")
    if not isinstance(trained_from, str):
        raise CatalogError(f"{where}: 'trained_from' must be a string")
    dataset_sha256 = raw.get("dataset_sha256", "")
    if not isinstance(dataset_sha256, str):
        raise CatalogError(f"{where}: 'dataset_sha256' must be a string")
    weights_sha256 = raw.get("weights_sha256", "")
    if not isinstance(weights_sha256, str):
        raise CatalogError(f"{where}: 'weights_sha256' must be a string")

    spec = ModelSpec(
        id=model_id,
        parameter_count=parameter_count,
        architecture=architecture,
        architecture_detail=architecture_detail,
        hidden_size=_positive_int(raw.get("hidden_size"), where, "hidden_size"),
        vocab_size=_positive_int(raw.get("vocab_size"), where, "vocab_size"),
        feature_count=_positive_int(raw.get("feature_count"), where, "feature_count"),
        output_classes=_positive_int(
            raw.get("output_classes"), where, "output_classes"
        ),
        dtype=dtype,
        quantization=quantization,
        parameters_by_dtype=counted,
        license=license_id,
        license_link=license_link,
        version=version,
        base_model=base_model,
        base_model_relation=relation,
        trained_from=trained_from,
        dataset_sha256=dataset_sha256,
        weights_sha256=weights_sha256,
        spec_digest="",
    )
    computed = spec_digest(spec)
    stored = raw.get("spec_digest")
    if stored is not None and stored != computed:
        raise CatalogError(f"{where}: 'spec_digest' does not match the spec contents")
    return replace(spec, spec_digest=computed)


def parse_specs(data: object) -> dict[str, ModelSpec]:
    """Parse and validate a model spec document."""
    if not isinstance(data, Mapping) or not isinstance(data.get("specs"), list):
        raise CatalogError("model spec file must be a JSON object with a 'specs' list")
    specs: dict[str, ModelSpec] = {}
    for index, raw in enumerate(data["specs"]):
        if not isinstance(raw, Mapping):
            raise CatalogError(f"model spec #{index} must be a JSON object")
        spec = _spec_from_record(raw, index)
        if spec.id in specs:
            raise CatalogError(
                f"model spec #{index} (id {spec.id!r}): duplicate id"
            )
        specs[spec.id] = spec
    if not specs:
        raise CatalogError("model spec file contains no specs")
    validate_lineage(specs)
    return specs


def load_specs(path: Path | None = None) -> dict[str, ModelSpec]:
    """Load the bundled model specifications, or a spec file."""
    if path is not None:
        path = Path(path)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CatalogError(f"cannot read model specs at {path}: {exc}") from exc
        return parse_specs(data)
    resource = resources.files("gravion_ai").joinpath("data", "model_specs.json")
    data = json.loads(resource.read_text(encoding="utf-8"))
    return parse_specs(data)
