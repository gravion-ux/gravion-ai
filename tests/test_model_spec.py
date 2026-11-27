"""Model specifications: measured from the shipped weight artifacts.

A Gravion model entry's weights artifact already records the facts a
spec needs. These tests pin the measurement rules, the separation of
storage dtype from quantization method, the spec digest, and the
agreement between the shipped spec file, the registry and the
artifacts themselves.
"""

import json
import re
from dataclasses import replace
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
    architecture_slug,
    derive_spec,
    format_parameters,
    load_specs,
    parse_specs,
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


# -- the paths that reject bad input ---------------------------------------
#
# A specification is measured from an artifact and validated against a
# declared record. Every one of those checks has a failure mode, and a
# failure mode nobody exercises is a failure mode nobody notices when it
# changes. The tests below pin each of them.


def _record(**overrides):
    """A valid spec record, taken from the shipped specifications."""
    record = load_specs()["gravion-core"].to_dict()
    record.pop("spec_digest", None)
    record.update(overrides)
    return record


def test_format_parameters_rejects_a_count_that_is_not_an_integer():
    with pytest.raises(CatalogError, match="must be an integer"):
        format_parameters("1000")
    with pytest.raises(CatalogError, match="must be an integer"):
        format_parameters(True)


def test_format_parameters_rejects_a_negative_count():
    with pytest.raises(CatalogError, match="must not be negative"):
        format_parameters(-1)


def test_format_parameters_renders_a_billion_scale_count():
    assert format_parameters(2_500_000_000) == "2.50B"


def test_an_unknown_architecture_is_rejected():
    with pytest.raises(CatalogError, match="unknown architecture"):
        architecture_slug("a transformer with attention")


def test_derive_spec_rejects_a_missing_tensor_block():
    with pytest.raises(CatalogError, match="embedding weights are missing"):
        derive_spec(
            _artifact(embedding=None), {"license": "MIT", "version": "1.0.0"}
        )


def test_derive_spec_rejects_a_tensor_block_without_an_encoding():
    artifact = _artifact()
    artifact["direct"] = {"data": ""}
    with pytest.raises(CatalogError, match="have no encoding"):
        derive_spec(artifact, {"license": "MIT", "version": "1.0.0"})


def test_derive_spec_rejects_tensor_blocks_with_different_encodings():
    artifact = _artifact()
    artifact["classifier"] = {"data": "", "encoding": "float32", "scale": 1.0}
    with pytest.raises(CatalogError, match="different encodings"):
        derive_spec(artifact, {"license": "MIT", "version": "1.0.0"})


def test_derive_spec_rejects_an_artifact_that_is_not_an_object():
    with pytest.raises(CatalogError, match="must be a JSON object"):
        derive_spec(
            ["not", "an", "artifact"], {"license": "MIT", "version": "1.0.0"}
        )


def test_derive_spec_rejects_an_artifact_without_a_model_id():
    with pytest.raises(CatalogError, match="no model_id"):
        derive_spec(_artifact(model_id=""), {"license": "MIT", "version": "1.0.0"})


def test_derive_spec_rejects_an_artifact_without_an_architecture():
    with pytest.raises(CatalogError, match="no architecture"):
        derive_spec(_artifact(architecture=""), {"license": "MIT", "version": "1.0.0"})


def test_derive_spec_rejects_an_artifact_without_an_embedding_width():
    with pytest.raises(CatalogError, match="no embedding width"):
        derive_spec(_artifact(dim=0), {"license": "MIT", "version": "1.0.0"})


def test_derive_spec_rejects_an_artifact_without_features():
    with pytest.raises(CatalogError, match="no feature vocabulary"):
        derive_spec(_artifact(features=[]), {"license": "MIT", "version": "1.0.0"})


def test_derive_spec_rejects_an_artifact_without_classes():
    with pytest.raises(CatalogError, match="no answer classes"):
        derive_spec(_artifact(classes=[]), {"license": "MIT", "version": "1.0.0"})


def test_derive_spec_rejects_an_artifact_without_a_vocabulary():
    with pytest.raises(CatalogError, match="no vocabulary"):
        derive_spec(_artifact(vocabulary=[]), {"license": "MIT", "version": "1.0.0"})


def test_derive_spec_rejects_a_license_link_that_is_not_a_string():
    with pytest.raises(CatalogError, match="license_link"):
        derive_spec(
            _artifact(),
            {"license": "MIT", "license_link": 7, "version": "1.0.0"},
        )


def test_derive_spec_rejects_an_empty_base_model():
    with pytest.raises(CatalogError, match="base_model"):
        derive_spec(
            _artifact(),
            {
                "license": "MIT",
                "version": "1.0.0",
                "base_model": "",
                "base_model_relation": "fine-tuned",
            },
        )


def test_derive_spec_rejects_a_relation_without_a_base_model():
    with pytest.raises(CatalogError, match="requires a base_model"):
        derive_spec(
            _artifact(),
            {
                "license": "MIT",
                "version": "1.0.0",
                "base_model_relation": "fine-tuned",
            },
        )


def test_derive_spec_accepts_a_training_source_given_as_a_string():
    spec = derive_spec(
        _artifact(trained_from="data/seed.jsonl"),
        {"license": "MIT", "version": "1.0.0"},
    )
    assert spec.trained_from == "data/seed.jsonl"
    assert spec.dataset_sha256 == ""


def test_derive_spec_ignores_a_training_source_that_is_not_a_string():
    spec = derive_spec(
        _artifact(trained_from={"source": 7}),
        {"license": "MIT", "version": "1.0.0"},
    )
    assert spec.trained_from == ""


def test_derive_spec_ignores_a_dataset_digest_that_is_not_a_string():
    spec = derive_spec(
        _artifact(trained_from={"source": "data/seed.jsonl", "dataset_sha256": 7}),
        {"license": "MIT", "version": "1.0.0"},
    )
    assert spec.dataset_sha256 == ""


def test_derive_spec_ignores_a_weights_digest_that_is_not_a_string():
    spec = derive_spec(
        _artifact(weights_sha256=7),
        {"license": "MIT", "version": "1.0.0"},
    )
    assert spec.weights_sha256 == ""


def test_validate_lineage_rejects_a_model_that_references_itself():
    spec = derive_spec(_artifact(), {"license": "MIT", "version": "1.0.0"})
    with pytest.raises(CatalogError, match="itself"):
        validate_lineage(
            {"gravion-test": replace(spec, base_model="gravion-test")}
        )


def test_validate_lineage_rejects_a_base_model_that_is_not_known():
    spec = derive_spec(_artifact(), {"license": "MIT", "version": "1.0.0"})
    with pytest.raises(CatalogError, match="not a known model"):
        validate_lineage(
            {"gravion-test": replace(spec, base_model="gravion-missing")}
        )


def test_parse_specs_accepts_a_valid_record():
    specs = parse_specs({"specs": [_record()]})
    assert "gravion-core" in specs


def test_parse_specs_rejects_a_record_without_an_id():
    with pytest.raises(CatalogError, match="'id' must be a non-empty string"):
        parse_specs({"specs": [_record(id="")]})


def test_parse_specs_rejects_an_unknown_architecture():
    with pytest.raises(CatalogError, match="'architecture' must be one of"):
        parse_specs({"specs": [_record(architecture="transformer")]})


def test_parse_specs_rejects_a_missing_architecture_detail():
    with pytest.raises(CatalogError, match="'architecture_detail' must be"):
        parse_specs({"specs": [_record(architecture_detail="")]})


def test_parse_specs_rejects_an_unknown_dtype():
    with pytest.raises(CatalogError, match="'dtype' must be one of"):
        parse_specs({"specs": [_record(dtype="int3")]})


def test_parse_specs_rejects_an_unknown_quantization():
    with pytest.raises(CatalogError, match="'quantization' must be one of"):
        parse_specs({"specs": [_record(quantization="magic-int8")]})


def test_parse_specs_rejects_an_unknown_license():
    with pytest.raises(CatalogError, match="'license' must be one of"):
        parse_specs({"specs": [_record(license="Proprietary")]})


def test_parse_specs_rejects_a_missing_license_link():
    with pytest.raises(CatalogError, match="'license_link' must be a non-empty"):
        parse_specs({"specs": [_record(license_link="")]})


def test_parse_specs_rejects_a_malformed_version():
    with pytest.raises(CatalogError, match="'version' must be a MAJOR"):
        parse_specs({"specs": [_record(version="1.0")]})


def test_parse_specs_rejects_a_negative_parameter_count():
    with pytest.raises(CatalogError, match="'parameter_count' must be"):
        parse_specs({"specs": [_record(parameter_count=-1)]})


def test_parse_specs_rejects_a_parameter_count_that_is_not_an_integer():
    with pytest.raises(CatalogError, match="'parameter_count' must be"):
        parse_specs({"specs": [_record(parameter_count="100")]})


def test_parse_specs_rejects_a_missing_parameter_split():
    with pytest.raises(CatalogError, match="'parameters_by_dtype' must be"):
        parse_specs({"specs": [_record(parameters_by_dtype={})]})


def test_parse_specs_rejects_an_unknown_storage_dtype_in_the_split():
    record = _record()
    record["parameters_by_dtype"] = {"int3": 1}
    with pytest.raises(CatalogError, match="is not a known storage dtype"):
        parse_specs({"specs": [record]})


def test_parse_specs_rejects_a_negative_count_in_the_split():
    record = _record()
    record["parameters_by_dtype"] = {"int8": -1}
    with pytest.raises(CatalogError, match="must be a non-negative integer"):
        parse_specs({"specs": [record]})


def test_parse_specs_rejects_a_split_that_does_not_add_up():
    record = _record()
    record["parameters_by_dtype"] = {"int8": 1, "float32": 1}
    with pytest.raises(CatalogError, match="does not add up"):
        parse_specs({"specs": [record]})


def test_parse_specs_rejects_a_count_that_is_not_positive():
    with pytest.raises(CatalogError, match="'hidden_size' must be a positive"):
        parse_specs({"specs": [_record(hidden_size=0)]})


def test_parse_specs_rejects_an_empty_base_model():
    with pytest.raises(CatalogError, match="'base_model' must be a non-empty"):
        parse_specs({"specs": [_record(base_model="")]})


def test_parse_specs_rejects_an_unknown_base_model_relation():
    record = _record(base_model="gravion-other", base_model_relation="invented")
    with pytest.raises(CatalogError, match="'base_model_relation' must be one of"):
        parse_specs({"specs": [record]})


def test_parse_specs_rejects_a_training_source_that_is_not_a_string():
    with pytest.raises(CatalogError, match="'trained_from' must be a string"):
        parse_specs({"specs": [_record(trained_from=7)]})


def test_parse_specs_rejects_a_dataset_digest_that_is_not_a_string():
    with pytest.raises(CatalogError, match="'dataset_sha256' must be a string"):
        parse_specs({"specs": [_record(dataset_sha256=7)]})


def test_parse_specs_rejects_a_weights_digest_that_is_not_a_string():
    with pytest.raises(CatalogError, match="'weights_sha256' must be a string"):
        parse_specs({"specs": [_record(weights_sha256=7)]})


def test_parse_specs_rejects_a_digest_that_does_not_match_the_spec():
    with pytest.raises(CatalogError, match="'spec_digest' does not match"):
        parse_specs({"specs": [_record(spec_digest="0" * 64)]})


def test_parse_specs_rejects_a_document_without_a_specs_list():
    with pytest.raises(CatalogError, match="must be a JSON object with a 'specs'"):
        parse_specs({"specs": {}})
    with pytest.raises(CatalogError, match="must be a JSON object with a 'specs'"):
        parse_specs([])


def test_parse_specs_rejects_an_entry_that_is_not_an_object():
    with pytest.raises(CatalogError, match="must be a JSON object"):
        parse_specs({"specs": ["gravion-core"]})


def test_parse_specs_rejects_a_duplicate_id():
    with pytest.raises(CatalogError, match="duplicate id"):
        parse_specs({"specs": [_record(), _record()]})


def test_parse_specs_rejects_an_empty_document():
    with pytest.raises(CatalogError, match="contains no specs"):
        parse_specs({"specs": []})


def test_load_specs_reads_a_spec_file(tmp_path):
    path = tmp_path / "specs.json"
    path.write_text(json.dumps({"specs": [_record()]}), encoding="utf-8")
    specs = load_specs(path)
    assert "gravion-core" in specs


def test_load_specs_reports_a_file_it_cannot_read(tmp_path):
    with pytest.raises(CatalogError, match="cannot read model specs"):
        load_specs(tmp_path / "missing.json")


def test_catalog_models_at_least_rejects_a_minimum_that_is_not_an_integer():
    catalog = ModelCatalog.load()
    with pytest.raises(CatalogError, match="must be an integer"):
        catalog.models_at_least("1000")


def test_catalog_models_at_least_rejects_a_negative_minimum():
    catalog = ModelCatalog.load()
    with pytest.raises(CatalogError, match="must not be negative"):
        catalog.models_at_least(-1)


def test_cli_models_list_reports_when_no_model_serves_the_capability(
    tmp_path, capsys
):
    """A deployment with its own registry may serve no model at all for
    the capability that was asked for."""
    entry = ModelCatalog.load().list()[0]
    private = {
        "id": "private-model",
        "name": "Private model",
        "capabilities": ["chat"],
        "tiers": list(entry.tiers),
        "status": "available",
    }
    home = tmp_path / "home"
    home.mkdir()
    (home / "gravion.json").write_text("{}", encoding="utf-8")
    (home / "models.json").write_text(
        json.dumps({"models": [private]}), encoding="utf-8"
    )
    assert main(
        ["--home", str(home), "models", "list", "--capability", "coder"]
    ) == 0
    out = capsys.readouterr().out
    assert "no Gravion models serve capability" in out
