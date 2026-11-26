"""Tests for the text specialist modules: translation, summarization,
retrieval, data-analysis and document-qa. Each module's deterministic
engine is exercised directly, and guidance-style prompts are shown to
fall through to the resolved specialist model."""

from gravion_ai.modules.data_analysis.module import analyze, parse_numbers
from gravion_ai.modules.document_qa.module import answer_question
from gravion_ai.modules.retrieval.module import rank_passages
from gravion_ai.modules.summarization.module import split_sentences, summarize
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


def test_split_sentences_keeps_trailing_fragment_and_decimals():
    # Regression: the old regex splitter dropped any text after the
    # last terminator and split on decimal/abbreviation dots, so a
    # final unpunctuated sentence vanished from summaries and
    # document Q&A could not quote it.
    assert split_sentences("Cats sleep. Dogs bark") == ["Cats sleep.", "Dogs bark"]
    assert split_sentences("Pi is 3.14 approx. Cats sleep.") == [
        "Pi is 3.14 approx.",
        "Cats sleep.",
    ]
    assert split_sentences("Dr. Smith arrived. He sat down.") == [
        "Dr. Smith arrived.",
        "He sat down.",
    ]
    assert split_sentences("") == []
    # Document Q&A can now answer from an unpunctuated final line.
    out = answer_question(
        "It closes at eight in the evening. The library opens at nine in the morning",
        "when does the library open",
    )
    assert "opens at nine" in out


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


def test_data_analysis_parses_grouped_and_scientific_numbers():
    # Regression: the old number regex split grouped thousands into
    # fragments ("1,000" -> 1.0, 0.0) and mangled scientific
    # notation ("1e3" -> 1.0, 3.0), so every statistic was wrong.
    assert parse_numbers("values: 1,000, 2,500, 3,000") == [1000.0, 2500.0, 3000.0]
    assert parse_numbers("values: 1,00,000, 2,50,000") == [100000.0, 250000.0]
    assert parse_numbers("values: 1e3, 2.5e2, .5") == [1000.0, 250.0, 0.5]
    out = analyze(parse_numbers("values: 1,000, 2,500, 3,000"))
    assert "Sum: 6500" in out
    # Honest mode: no repeated value means no mode, ties list all.
    assert "Mode: none" in analyze([4.0, 8.0, 15.0])
    assert "Mode: 2, 5 (each appears 2 times)" in analyze([2.0, 5.0, 2.0, 5.0, 9.0])
    assert analyze([]) == "No numbers to analyze."


def test_data_analysis_range_hyphen_is_not_a_minus_sign():
    # Regression: a hyphen glued to a digit was read as a minus
    # sign, so a range "5-10" parsed as [5, -10] (sum -5) and a
    # date "2026-11-27" contributed two negative values. The hyphen
    # is a separator there; only a free-standing minus is a sign,
    # and digits inside a word are not numbers.
    assert parse_numbers("scores range 5-10 points") == [5.0, 10.0]
    assert parse_numbers("on 2026-11-27 values 4 8 15") == [
        2026.0,
        11.0,
        27.0,
        4.0,
        8.0,
        15.0,
    ]
    assert parse_numbers("temps -5, -3, 2") == [-5.0, -3.0, 2.0]
    assert parse_numbers("version2 build3 has 10 20 30") == [10.0, 20.0, 30.0]
    assert "Sum: 15" in analyze(parse_numbers("range 5-10"))


def test_summarization_never_repeats_a_sentence():
    # Regression: a repeated sentence inflated its own term
    # frequencies, so the summary could be that one sentence
    # repeated max_sentences times.
    text = (
        "Cats sleep all day. Dogs bark at night. Cats sleep all day. "
        "Birds fly south in winter. Fish swim in the pond. "
        "Cats sleep all day."
    )
    out = summarize(text)
    assert out.count("Cats sleep all day.") == 1
    assert "Dogs bark at night." in out or "Birds fly south in winter." in out


def test_retrieval_rare_terms_outweigh_common_words():
    # IDF weighting: the passage carrying the query's distinctive
    # terms beats one that only shares words common to every
    # passage, and a stopword-only passage scores zero.
    ranked = rank_passages(
        "what is the capital of france",
        [
            "the the of of the the of the",
            "france is a country in europe and paris is its capital",
            "the capital city of france is paris",
        ],
    )
    assert ranked[0][1].startswith("the capital city")
    assert ranked[-1][1].startswith("the the")
    assert ranked[-1][0] == 0.0
    assert ranked[1][0] > ranked[-1][0]


def test_document_qa_matches_singular_and_plural_forms():
    # Regression: the old stemmer turned "class" into "clas" but
    # "classes" into "class", so a question about a class never
    # matched a document sentence about classes and the module
    # wrongly reported that the document held no answer.
    out = answer_question(
        "The classes meet on Monday morning. The library opens at nine.",
        "when does the class meet",
    )
    assert "classes meet on Monday" in out
    out = answer_question(
        "The cities listed are Delhi and Pune. The office is in Delhi.",
        "which cities are listed",
    )
    assert "cities listed are Delhi and Pune" in out


def test_retrieval_matches_singular_and_plural_forms():
    # A query for "cats" must score the passage about a "cat":
    # unstemmed, the two forms hash to different buckets and the
    # relevant passage scores zero.
    ranked = rank_passages(
        "cats",
        ["The dog runs in the park.", "The cat sleeps all day."],
    )
    assert ranked[0][1].startswith("The cat")
    assert ranked[0][0] > 0.0


def test_retrieval_keeps_multiline_passages_whole():
    # Regression: passage parsing captured one line only, so a
    # passage wrapped over several lines lost everything after
    # its first line before ranking.
    out = _run(
        "retrieval",
        "gravion-search",
        "query: what do cats need\n"
        "passage: Cats need food\n"
        "and fresh water daily.\n"
        "passage: Dogs need daily walks.",
    )
    assert "Cats need food and fresh water daily." in out


def test_translation_matches_longest_phrase_inside_a_sentence():
    # Regression: greedy phrase matching capped spans at four
    # words, so the five-word phrasebook entry "how much does
    # this cost" never matched inside a longer sentence.
    out = translate_phrase("please tell me how much does this cost today", to_hindi=True)
    assert "ye kitne ka hai" in out


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
