"""Tests for the translation, summarization, retrieval, data-analysis
and document-qa modules, plus a catalog regression test.

Regression context (27 Nov 2026): the five specialist models
(gravion-translate, gravion-summary, gravion-search, gravion-analyst,
gravion-docqa) were registered with weights_status "included" before
their seed corpora were trained, so catalog loading raised
CatalogError and 82 tests failed. The regression test below pins the
invariant that every specialist entry's included weights artifact
exists on disk with a matching hash and parameter count.
"""

from pathlib import Path

from gravion_ai.catalog import ModelCatalog
from gravion_ai.core_model import SPECIALIST_MODEL_IDS, load_bundled_model
from gravion_ai.providers import GravionCoreProvider
from gravion_ai.registry import ModuleContext, default_registry

DATA_DIR = Path(__file__).resolve().parent.parent / "src" / "gravion_ai" / "data"


def _run(module_name: str, model_id: str, prompt: str) -> str:
    module = default_registry().instantiate(module_name)
    ctx = ModuleContext("You are Gravion.", GravionCoreProvider(), model=model_id)
    return module.run(ctx, prompt)


def test_every_specialist_weights_artifact_exists_and_matches_catalog():
    catalog = ModelCatalog.load()
    for model_id in SPECIALIST_MODEL_IDS:
        entry = catalog.get(model_id)
        assert entry is not None, model_id
        assert entry.weights_status == "included", model_id
        assert (DATA_DIR / f"{model_id}-weights.json").exists(), model_id
        assert (DATA_DIR / f"{model_id}-seed.jsonl").exists(), model_id
        loaded = load_bundled_model(model_id)
        assert loaded.weights_sha256 == entry.weights_sha256, model_id
        assert loaded.parameter_count == entry.parameter_count, model_id
        assert entry.parameter_count > 0, model_id


def test_translation_module_phrasebook_paths():
    out = _run("translation", "gravion-translate", "translate to hindi: thank you")
    assert "dhanyavaad" in out
    out = _run("translation", "gravion-translate", "translate to english: shukriya")
    assert out == "thank you"
    # Outside the phrasebook the module marks words, never guesses.
    out = _run("translation", "gravion-translate", "translate to hindi: zxqwv")
    assert "outside my Gravion phrasebook" in out


def test_summarization_module_extractive_path():
    text = (
        "Cats sleep for long hours each day. Cats also groom "
        "themselves carefully. Dogs need daily walks and training. "
        "Birds sing at dawn in spring. Cats hunt small rodents at night."
    )
    out = _run("summarization", "gravion-summary", f"summarize: {text}")
    assert out
    # Extractive: every summary sentence comes from the source.
    for sentence in out.split(". "):
        assert sentence.strip().rstrip(".") in text


def test_retrieval_module_ranks_matching_passage_first():
    prompt = (
        "query: how do cats groom themselves\n"
        "passage: Dogs enjoy long walks in parks.\n"
        "passage: Cats groom themselves with their tongues."
    )
    out = _run("retrieval", "gravion-search", prompt)
    assert "1." in out
    assert out.index("Cats groom") < out.index("Dogs enjoy")


def test_data_analysis_module_computes_statistics():
    out = _run("data-analysis", "gravion-analyst", "analyze these numbers: 4, 8, 15, 16, 23, 42")
    assert "Count: 6" in out
    assert "Sum: 108" in out
    assert "Mean: 18" in out
    assert "Median: 15.5" in out
    assert "Min: 4" in out
    assert "Max: 42" in out


def test_document_qa_module_answers_from_document_only():
    prompt = (
        "document: The library opens at nine in the morning. "
        "It closes at eight in the evening.\n"
        "question: when does the library open"
    )
    out = _run("document-qa", "gravion-docqa", prompt)
    assert "nine in the morning" in out
    missing = _run(
        "document-qa",
        "gravion-docqa",
        "document: The library opens at nine.\nquestion: what is the wifi password",
    )
    assert "does not contain an answer" in missing
