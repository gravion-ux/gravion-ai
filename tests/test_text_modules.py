"""Tests for the text specialist modules: translation, summarization,
retrieval, data-analysis and document-qa. Each module's deterministic
engine is exercised directly, and guidance-style prompts are shown to
fall through to the resolved specialist model."""

from gravion_ai.modules.data_analysis.module import analyze, parse_numbers
from gravion_ai.modules.document_qa.module import answer_question
from gravion_ai.modules.retrieval.module import rank_passages
from gravion_ai.modules.summarization.module import summarize
from gravion_ai.modules.translation.module import translate_phrase
from gravion_ai.providers import GravionCoreProvider
from gravion_ai.registry import ModuleContext, default_registry


def _run(module_name, model_id, prompt):
    module = default_registry().instantiate(module_name)
    ctx = ModuleContext("You are Gravion.", GravionCoreProvider(), model=model_id)
    return module.run(ctx, prompt)


def test_translation_phrasebook_exact_and_word_level():
    assert "dhanyavaad" in translate_phrase("thank you", to_hindi=True)
    assert translate_phrase("shukriya", to_hindi=False) == "thank you"
    # Unknown words are marked, never guessed.
    out = translate_phrase("thank you zxqwv", to_hindi=True)
    assert "[zxqwv]" in out
    assert "outside my Gravion phrasebook" in translate_phrase("zxqwv", to_hindi=True)


def test_translation_module_paths():
    assert "dhanyavaad" in _run("translation", "gravion-translate", "translate to hindi: thank you")
    assert _run("translation", "gravion-translate", "translate to english: shukriya") == "thank you"


def test_summarization_is_extractive():
    text = (
        "Cats sleep for long hours each day. Cats groom themselves with "
        "their tongues. Dogs need daily walks. Cats hunt at night and "
        "sleep again at dawn."
    )
    out = summarize(text, max_sentences=2)
    assert out
    for sentence in out.split(". "):
        assert sentence.strip().rstrip(".") in text
    assert _run("summarization", "gravion-summary", f"summarize: {text}") == summarize(text)


def test_retrieval_ranks_relevant_passage_first():
    ranked = rank_passages(
        "how do cats groom",
        ["Dogs enjoy long walks in parks.", "Cats groom themselves with their tongues."],
    )
    assert ranked[0][1].startswith("Cats groom")
    assert ranked[0][0] > ranked[1][0]
    out = _run(
        "retrieval",
        "gravion-search",
        "query: how do cats groom\n"
        "passage: Dogs enjoy long walks in parks.\n"
        "passage: Cats groom themselves with their tongues.",
    )
    assert out.splitlines()[1].endswith("Cats groom themselves with their tongues.")


def test_data_analysis_computes_statistics():
    numbers = parse_numbers("values: 4, 8, 15, 16, 23, 42")
    assert numbers == [4.0, 8.0, 15.0, 16.0, 23.0, 42.0]
    out = analyze(numbers)
    assert "Count: 6" in out
    assert "Sum: 108" in out
    assert "Mean: 18" in out
    assert "Median: 15.5" in out
    assert "rising" in out
    assert "Count: 6" in _run("data-analysis", "gravion-analyst", "analyze: 4, 8, 15, 16, 23, 42")


def test_document_qa_quotes_evidence_and_admits_gaps():
    document = "The library opens at nine in the morning. It closes at eight in the evening."
    out = answer_question(document, "when does the library open")
    assert "opens at nine" in out
    missing = answer_question(document, "who founded the library")
    assert "does not contain an answer" in missing
    out2 = _run(
        "document-qa",
        "gravion-docqa",
        f"document: {document}\nquestion: when does the library open",
    )
    assert "opens at nine" in out2
