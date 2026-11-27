"""Model specifications: measured from the shipped weight artifacts.

A Gravion model entry's weights artifact already records the facts a
spec needs. These tests pin the measurement rules, the separation of
storage dtype from quantization method, the spec digest, and the
agreement between the shipped spec file, the registry and the
artifacts themselves.
"""

import json
import re
from importlib import resources

import pytest
from gravion_ai.catalog import ModelCatalog
from gravion_ai.cli import main
from gravion_ai.errors import CatalogError
from gravion_ai.model_spec import (
    BASE_MODEL_RELATIONS,
    DTYPES,
    LICENSES,
    QUANTIZATIONS,
    derive_spec,
    format_parameters,
    load_specs,
    spec_digest,
    validate_lineage,
)


def _artifact(**overrides):
    """A minimal v3 weights artifact, shaped like the shipped ones."""
    artifact = {
        "model_id": "gravion-test",
        "format": "gravion-core-local-v3",
        "architecture": "sparse n-gram embedding + direct linear softmax classifier",
        "dim": 16,
        "features": ["a b", "c d"],
        "classes": ["x", "y", "z"],
        "vocabulary": ["a", "b", "c", "d"],
        "direct": {"data": "", "encoding": "int8-base64", "scale": 0.01},
        "embedding": {"data": "", "encoding": "int8-base64", "scale": 0.01},
        "classifier": {"data": "", "encoding": "int8-base64", "scale": 0.01},
        "bias": [0.1, 0.2, 0.3],
        "transitions": {"<s>": {"a": 0.5}},
        "unigram_weights": {"a": 0.25},
        # 2x3 direct + 2x16 embedding + 3x16 classifier = 86 int8 values,
        # plus 3 bias + 1 transition + 1 unigram float values = 5.
        "parameter_count": 91,
        "trained_from": {
            "source": "data/gravion-test-seed.jsonl",
            "examples": 10,
            "classes": 3,
            "features": 2,
            "dataset_sha256": "ab" * 32,
        },
    }
    artifact.update(overrides)
    return artifact


def _read_artifact(relative_path):
    return json.loads(
        resources.files("gravion_ai")
        .joinpath(relative_path)
        .read_text(encoding="utf-8")
    )


def test_format_parameters_renders_human_readable_counts():
    assert format_parameters(10_586_543) == "10.59M"
    assert format_parameters(43_096) == "43.10K"
    assert format_parameters(1_500) == "1.50K"
    assert format_parameters(999) == "999"
    assert format_parameters(0) == "0"
    assert format_parameters(None) == "-"


def test_storage_dtype_and_quantization_method_are_separate_vocabularies():
    """`int8` is how a tensor is stored; `symmetric-int8` is how it was
    quantized. Keeping them apart stops a quantization method from
    being mistaken for a dtype."""
    assert "int8" in DTYPES
    assert "int8" not in QUANTIZATIONS
    assert "symmetric-int8" in QUANTIZATIONS
    assert "symmetric-int8" not in DTYPES
    assert not DTYPES & QUANTIZATIONS


def test_derive_spec_measures_the_shape_from_the_artifact():
    spec = derive_spec(
        _artifact(), {"license": "MIT", "version": "1.0.0"}
    )
    assert spec.id == "gravion-test"
    assert spec.architecture == "sparse-ngram-linear"
    assert spec.architecture_detail == (
        "sparse n-gram embedding + direct linear softmax classifier"
    )
    assert spec.hidden_size == 16
    assert spec.vocab_size == 4
    assert spec.feature_count == 2
    assert spec.output_classes == 3
    assert spec.dtype == "int8"
    assert spec.quantization == "symmetric-int8"
    assert spec.parameter_count == 91
    assert spec.parameters_by_dtype == {"float32": 5, "int8": 86}
    assert spec.license == "MIT"
    assert spec.version == "1.0.0"
    assert spec.base_model is None
    assert spec.base_model_relation is None
    assert spec.trained_from == "data/gravion-test-seed.jsonl"
    assert spec.dataset_sha256 == "ab" * 32


def test_derive_spec_rejects_a_parameter_count_the_weights_do_not_support():
    with pytest.raises(CatalogError, match="parameter_count"):
        derive_spec(
            _artifact(parameter_count=999),
            {"license": "MIT", "version": "1.0.0"},
        )


def test_derive_spec_rejects_an_unknown_quantization_encoding():
    artifact = _artifact()
    artifact["direct"] = {"data": "", "encoding": "int3-base64", "scale": 0.01}
    with pytest.raises(CatalogError, match="quantization"):
        derive_spec(artifact, {"license": "MIT", "version": "1.0.0"})


def test_derive_spec_rejects_a_license_outside_the_spdx_allowlist():
    with pytest.raises(CatalogError, match="license"):
        derive_spec(
            _artifact(), {"license": "Totally-Made-Up", "version": "1.0.0"}
        )


def test_derive_spec_rejects_a_missing_or_malformed_version():
    with pytest.raises(CatalogError, match="version"):
        derive_spec(_artifact(), {"license": "MIT"})
    with pytest.raises(CatalogError, match="version"):
        derive_spec(_artifact(), {"license": "MIT", "version": "one"})


def test_a_base_model_may_not_reference_the_model_itself():
    with pytest.raises(CatalogError, match="itself"):
        derive_spec(
            _artifact(model_id="gravion-a"),
            {
                "license": "MIT",
                "version": "1.0.0",
                "base_model": "gravion-a",
                "base_model_relation": "fine-tuned",
            },
        )


def test_a_base_model_relation_must_be_a_known_relation():
    with pytest.raises(CatalogError, match="base_model_relation"):
        derive_spec(
            _artifact(),
            {
                "license": "MIT",
                "version": "1.0.0",
                "base_model": "gravion-other",
                "base_model_relation": "invented",
            },
        )


def test_base_model_cycles_are_rejected():
    a = derive_spec(
        _artifact(model_id="gravion-a"),
        {
            "license": "MIT",
            "version": "1.0.0",
            "base_model": "gravion-b",
            "base_model_relation": "fine-tuned",
        },
    )
    b = derive_spec(
        _artifact(model_id="gravion-b"),
        {
            "license": "MIT",
            "version": "1.0.0",
            "base_model": "gravion-a",
            "base_model_relation": "fine-tuned",
        },
    )
    with pytest.raises(CatalogError, match="cycle"):
        validate_lineage({"gravion-a": a, "gravion-b": b})


def test_a_valid_base_model_lineage_is_accepted():
    base = derive_spec(
        _artifact(model_id="gravion-a"), {"license": "MIT", "version": "1.0.0"}
    )
    derived = derive_spec(
        _artifact(model_id="gravion-b"),
        {
            "license": "MIT",
            "version": "1.0.0",
            "base_model": "gravion-a",
            "base_model_relation": "fine-tuned",
        },
    )
    validate_lineage({"gravion-a": base, "gravion-b": derived})


def test_the_spec_digest_is_stable_and_covers_every_field():
    spec = derive_spec(
        _artifact(), {"license": "MIT", "version": "1.0.0"}
    )
    assert spec.spec_digest == spec_digest(spec)
    other = derive_spec(
        _artifact(), {"license": "Apache-2.0", "version": "1.0.0"}
    )
    assert other.spec_digest != spec.spec_digest


def test_the_shipped_specs_cover_every_registry_entry():
    catalog = ModelCatalog.load()
    specs = load_specs()
    assert set(specs) == {entry.id for entry in catalog.list()}


def test_the_shipped_specs_match_the_shipped_weight_artifacts():
    """The spec file is measured from the artifacts; it must not drift
    away from the weights that actually ship."""
    catalog = ModelCatalog.load()
    specs = load_specs()
    for entry in catalog.list():
        artifact = _read_artifact(entry.weights_artifact)
        derived = derive_spec(artifact, specs[entry.id].declared())
        assert derived.spec_digest == specs[entry.id].spec_digest, entry.id
        assert derived.parameter_count == entry.parameter_count, entry.id
        assert derived.weights_sha256 == entry.weights_sha256, entry.id


def test_the_registry_and_the_specs_agree_on_parameter_counts():
    catalog = ModelCatalog.load()
    specs = load_specs()
    for entry in catalog.list():
        assert specs[entry.id].parameter_count == entry.parameter_count, entry.id


def test_every_shipped_spec_uses_an_spdx_license():
    for spec in load_specs().values():
        assert spec.license in LICENSES, spec.id
        assert spec.license_link, spec.id


def test_every_shipped_spec_declares_a_version():
    for spec in load_specs().values():
        assert re.match(r"^\d+\.\d+\.\d+$", spec.version), spec.id


def test_every_shipped_spec_records_its_training_provenance():
    for spec in load_specs().values():
        assert spec.trained_from, spec.id
        assert re.match(r"^[0-9a-f]{64}$", spec.dataset_sha256), spec.id


def test_the_shipped_models_declare_no_base_model():
    """Every bundled Gravion model is trained from its own bundled seed
    corpus, so none of them is a derivative of another."""
    for spec in load_specs().values():
        assert spec.base_model is None, spec.id
        assert spec.base_model_relation is None, spec.id


def test_the_known_base_model_relations_are_spelled_out():
    assert "fine-tuned" in BASE_MODEL_RELATIONS
    assert "base" in BASE_MODEL_RELATIONS


def test_catalog_total_parameters_sums_the_bundled_registry():
    catalog = ModelCatalog.load()
    total = catalog.total_parameters()
    assert total == sum(e.parameter_count for e in catalog.list())
    assert total > 10_000_000


def test_catalog_models_at_least_filters_by_parameter_count():
    catalog = ModelCatalog.load()
    large = {e.id for e in catalog.models_at_least(1_000_000)}
    assert large == {"gravion-core", "gravion-omni"}
    assert catalog.models_at_least(10_000_000_000) == []


def test_cli_models_list_shows_parameters_architecture_and_dtype(capsys):
    assert main(["models", "list"]) == 0
    out = capsys.readouterr().out
    assert "params=" in out
    assert "arch=" in out
    assert "dtype=" in out
    assert "quant=" in out


def test_cli_models_list_min_parameters_filter(capsys):
    assert main(["models", "list", "--min-parameters", "1000000"]) == 0
    out = capsys.readouterr().out
    assert "gravion-core" in out
    assert "gravion-omni" in out
    assert "gravion-chat" not in out


def test_cli_models_list_rejects_a_negative_min_parameters(capsys):
    assert main(["models", "list", "--min-parameters", "-1"]) == 1
    err = capsys.readouterr().err
    assert "min-parameters" in err


def test_cli_models_list_reports_when_no_model_matches(capsys):
    assert main(["models", "list", "--min-parameters", "100000000000"]) == 0
    out = capsys.readouterr().out
    assert "no Gravion model" in out


def test_cli_models_spec_prints_the_full_spec(capsys):
    assert main(["models", "spec", "gravion-core"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["id"] == "gravion-core"
    assert payload["parameter_count"] == 10586543
    assert payload["architecture"] == "sparse-ngram-linear"
    assert payload["dtype"] == "int8"
    assert payload["quantization"] == "symmetric-int8"
    assert payload["license"] == "MIT"
    assert payload["spec_digest"]
    assert payload["parameters_by_dtype"]["int8"] > 0
    assert payload["parameters_by_dtype"]["float32"] > 0


def test_cli_models_spec_rejects_an_unknown_model(capsys):
    assert main(["models", "spec", "gravion-nope"]) == 1
    assert "gravion-nope" in capsys.readouterr().err
