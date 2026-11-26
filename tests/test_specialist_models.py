import json

import pytest

from gravion_ai.catalog import ModelCatalog
from gravion_ai.core_model import (
    SPECIALIST_MODEL_IDS,
    bundled_seed_path,
    count_parameters,
    load_bundled_model,
)
from gravion_ai.providers import GravionCoreProvider
from gravion_ai.registry import ModuleContext, default_registry

MODULE_FOR_MODEL = {
    "gravion-chat": "chat",
    "gravion-support": "support",
    "gravion-review": "coder",
    "gravion-code": "coder",
    "gravion-builder": "website-builder",
    "gravion-vision": "image",
    "gravion-video": "video",
    "gravion-audio": "music",
    "gravion-shield": "cyber",
    "gravion-agent": "agentic",
    "gravion-omni": "all-in-one",
}


def _first_seed_example(model_id):
    first_line = bundled_seed_path(model_id).read_text(encoding="utf-8").splitlines()[0]
    record = json.loads(first_line)
    return record["prompt"], record["completion"]


@pytest.mark.parametrize("model_id", SPECIALIST_MODEL_IDS)
def test_specialist_catalog_artifact_and_seed(model_id):
    entry = ModelCatalog.load().get(model_id)
    assert entry is not None
    assert entry.weights_status == "included"
    assert entry.weights_artifact == f"data/{model_id}-weights.json"

    model = load_bundled_model(model_id)
    assert model.model_id == model_id
    assert model.parameter_count == entry.parameter_count
    assert model.parameter_count == count_parameters(model.to_dict())
    assert model.weights_sha256 == entry.weights_sha256
    assert bundled_seed_path(model_id).exists()


@pytest.mark.parametrize("model_id", SPECIALIST_MODEL_IDS)
def test_specialist_provider_loads_weights_and_falls_back(model_id):
    prompt, completion = _first_seed_example(model_id)
    provider = GravionCoreProvider()
    messages = [{"role": "user", "content": prompt}]
    assert provider.complete(messages, model=model_id) == completion
    assert provider.parameter_count_for(model_id) > 0

    unknown = [{"role": "user", "content": "zxqv unrelated token stream"}]
    assert provider.complete(unknown, model=model_id) == (
        "[gravion-core] You said: zxq v unrelated token stream".replace("zxq v", "zxqv")
    )


def test_sparse_evidence_prompt_falls_back_instead_of_wrong_answer():
    # Regression: margin-only confidence let "a website for our bakery"
    # match a resume-site answer at 0.79 on 3 of 12 known terms.
    # Coverage-scaled confidence must keep such prompts below threshold.
    builder = load_bundled_model("gravion-builder")
    assert builder.predict("a website for our bakery") is None
    agent = load_bundled_model("gravion-agent")
    assert agent.predict("plan an agent for reports") is None


def test_specialist_miss_routes_to_core_before_echo_fallback():
    # Routing: when a specialist has no confident match, Gravion Core
    # still answers its doc-style questions instead of echoing.
    provider = GravionCoreProvider()
    out = provider.complete(
        [{"role": "user", "content": "explain gravion tiers"}],
        model="gravion-builder",
    )
    assert "Tiers are readiness levels" in out


@pytest.mark.parametrize("model_id", SPECIALIST_MODEL_IDS)
def test_specialist_module_uses_its_local_weights(model_id):
    prompt, completion = _first_seed_example(model_id)
    module = default_registry().instantiate(MODULE_FOR_MODEL[model_id])
    ctx = ModuleContext(
        "You are Gravion.",
        GravionCoreProvider(),
        model=model_id,
    )
    assert module.run(ctx, prompt) == completion
