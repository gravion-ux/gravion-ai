"""Tests for the text specialist modules: translation, summarization,
retrieval, data-analysis and document-qa. Each module's deterministic
engine is exercised directly, and guidance-style prompts are shown to
fall through to the resolved specialist model."""

from gravion_ai.modules.data_analysis.module import analyze, parse_numbers
from gravion_ai.modules.document_qa.module import answer_question
from gravion_ai.modules.retrieval.module import rank_passages
from gravion_ai.modules.summarization.module import (
    split_sentences,
    stem_word,
    summarize,
    word_tokens,
)
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
    # Regression: everyday words and the Roman spellings of
    # entries the book carries the other way were refused
    # as outside the phrasebook.
    assert "madad" in translate_phrase("help", to_hindi=True)
    assert "khush" in translate_phrase("happy", to_hindi=True)
    assert "aap kahan hain" in translate_phrase("where are you", to_hindi=True)
    assert translate_phrase("shubh raatri", to_hindi=False) == "good night"
    assert translate_phrase("khana", to_hindi=False) == "food"
    assert translate_phrase("kaise ho", to_hindi=False) == "how are you"
    assert translate_phrase("mera naam hai", to_hindi=False) == "my name is"
    assert translate_phrase("madad", to_hindi=False) == "help"
    assert translate_phrase("मदद", to_hindi=False) == "help"
    # Unknown words are marked, never guessed.
    out = translate_phrase("thank you zxqwv", to_hindi=True)
    assert "[zxqwv]" in out
    assert "outside my Gravion phrasebook" in translate_phrase("zxqwv", to_hindi=True)


def test_translation_danda_and_apostropheless_contractions():
    # Regression: the danda is a sentence terminator, not part
    # of the word — "धन्यवाद।" missed the phrasebook entry it
    # names and was refused as outside the book.
    assert translate_phrase("धन्यवाद।", to_hindi=False) == "thank you"
    assert translate_phrase("नमस्ते।", to_hindi=False) == "hello"
    # Apostrophe-less contractions expand like their
    # apostrophised forms, so casual Hinglish typing reaches
    # the expanded phrasebook keys.
    assert "mujhe samajh nahi aaya" in translate_phrase(
        "i dont understand", to_hindi=True
    )
    assert "chaliye shuru karte hain" in translate_phrase(
        "lets start", to_hindi=True
    )


def test_translation_module_paths():
    assert "dhanyavaad" in _run("translation", "gravion-translate", "translate to hindi: thank you")
    assert _run("translation", "gravion-translate", "translate to english: shukriya") == "thank you"
    # The colon-less shapes name the same input: the
    # trailing-language form is what the all-in-one router
    # sends here on its translate+language signal, and
    # "translate this into hindi: ..." puts the text after
    # the colon instead of translating the pronoun.
    assert "dhanyavaad" in _run(
        "translation", "gravion-translate", "translate thank you into hindi"
    )
    assert "dhanyavaad" in _run(
        "translation", "gravion-translate", "please translate thank you to hindi"
    )
    assert "dhanyavaad" in _run(
        "translation", "gravion-translate", "translate this into hindi: thank you"
    )
    assert _run(
        "translation", "gravion-translate", "translate shukriya into english"
    ) == "thank you"
    # A guidance request names a description, not literal
    # text: the colon-less forms stay with the resolved model
    # instead of refusing text the phrasebook never claimed.
    guidance = _run(
        "translation", "gravion-translate", "translate a greeting into hindi"
    )
    assert "namaste" in guidance
    assert "outside my Gravion phrasebook" not in guidance
    # The say-in-language shape names its text the same way:
    # covered text translates from the phrasebook, while a
    # description stays with the resolved model.
    assert "dhanyavaad" in _run(
        "translation", "gravion-translate", "how do you say thank you in hindi"
    )
    assert _run(
        "translation", "gravion-translate", "how do you say shukriya in english"
    ) == "thank you"
    say_guidance = _run(
        "translation", "gravion-translate", "how do you say a greeting in hindi"
    )
    assert "outside my Gravion phrasebook" not in say_guidance
    assert "how do you say a greeting in hindi" in say_guidance


def test_translation_what_is_and_bare_say_shapes():
    # Regression: "what is hello in hindi?" and "say hello
    # in hindi" name their text like the say-in-language
    # shape, but no pattern matched them, so covered text
    # fell through to the model; "goodbye"/"good luck"/
    # "see you later" and the Devanagari short forms were
    # missing from the book entirely.
    assert "namaste" in _run(
        "translation", "gravion-translate", "what is hello in hindi?"
    )
    assert "dhanyavaad" in _run(
        "translation", "gravion-translate", "what's thank you in hindi"
    )
    assert _run(
        "translation", "gravion-translate", "what is shukriya in english?"
    ) == "thank you"
    assert "namaste" in _run(
        "translation", "gravion-translate", "say hello in hindi"
    )
    assert "alvida" in translate_phrase("goodbye", to_hindi=True)
    assert translate_phrase("alvida", to_hindi=False) == "goodbye"
    assert "shubhkamnayein" in translate_phrase("good luck", to_hindi=True)
    assert "phir milenge" in translate_phrase("see you later", to_hindi=True)
    assert translate_phrase("शुभ प्रभात", to_hindi=False) == "good morning"
    assert translate_phrase("कैसे हो", to_hindi=False) == "how are you"
    assert translate_phrase("मेरा नाम", to_hindi=False) == "my name is"
    # A description stays with the resolved model.
    guidance = _run(
        "translation", "gravion-translate", "what is a greeting in hindi?"
    )
    assert "outside my Gravion phrasebook" not in guidance


def test_translation_accepts_devanagari_hindi_input():
    # Regression: normalisation stripped the Devanagari
    # block, so covered Hindi input ("धन्यवाद") erased to
    # nothing and was refused as outside the phrasebook. The
    # reverse Devanagari entries derive from the same book
    # that prints them.
    assert translate_phrase("धन्यवाद", to_hindi=False) == "thank you"
    assert translate_phrase("आप कैसे हैं", to_hindi=False) == "how are you"
    assert _run(
        "translation", "gravion-translate", "translate to english: धन्यवाद"
    ) == "thank you"
    assert _run("translation", "gravion-translate", "translate: धन्यवाद") == "thank you"


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
    # Regression: the British leading form routed here through
    # all-in-one but the module accepted only "summarize:".
    assert _run("summarization", "gravion-summary", f"summarise: {text}") == summarize(text)
    # Regression: the "summarize this:" form names the same
    # input; refusing it left the instruction inside the text
    # being summarized.
    assert _run("summarization", "gravion-summary", f"summarize this: {text}") == summarize(text)
    # Regression: the "tldr:" prefix asks for the same summary
    # in shorthand; refusing it left the prefix inside the
    # text being summarized.
    assert _run("summarization", "gravion-summary", f"tldr: {text}") == summarize(text)
    assert _run("summarization", "gravion-summary", f"tl;dr: {text}") == summarize(text)
    assert _run("summarization", "gravion-summary", f"summarize the following: {text}") == summarize(text)
    # Regression: the object may carry its noun ("summarize
    # the following text:", "summarize this passage:"); the
    # router sends the passage form here on the verb+noun
    # signal, so refusing it dropped a routed request to chat.
    assert _run("summarization", "gravion-summary", f"summarize the following text: {text}") == summarize(text)
    assert _run("summarization", "gravion-summary", f"summarize this passage: {text}") == summarize(text)
    assert _run("summarization", "gravion-summary", f"summarise the following article: {text}") == summarize(text)


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
    # An ellipsis trailing into a lowercase word is a pause in
    # one sentence, not a boundary; before a capital it still
    # ends the sentence.
    assert split_sentences("Wait... what happened? She left.") == [
        "Wait... what happened?",
        "She left.",
    ]
    assert split_sentences("He paused... Then he left.") == [
        "He paused...",
        "Then he left.",
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


def test_text_modules_handle_hindi_devanagari_text():
    # Regression: the text modules tokenized with an ASCII-only
    # pattern and split sentences only on ".!?", so Hindi text
    # broke end to end — "बिल्ली" split into single letters, a
    # Hindi question matched no Hindi sentence, Hindi retrieval
    # scored every passage zero, and the danda "।" never ended
    # a sentence, so a Hindi summary returned the whole text.
    assert word_tokens("बिल्ली café") == ["बिल्ली", "café"]
    assert split_sentences("बिल्ली सोती है। कुत्ता भौंकता है।") == [
        "बिल्ली सोती है।",
        "कुत्ता भौंकता है।",
    ]
    out = answer_question(
        "बिल्ली चटाई पर बैठी है। कुत्ता बगीचे में खेल रहा है।",
        "बिल्ली कहाँ बैठी है?",
    )
    assert "बिल्ली चटाई पर बैठी है" in out
    assert "कुत्ता" not in out
    ranked = rank_passages(
        "बिल्ली कहाँ है",
        ["कुत्ता बगीचे में है", "बिल्ली चटाई पर है"],
    )
    assert ranked[0][1] == "बिल्ली चटाई पर है"
    assert ranked[0][0] > ranked[1][0]
    summary = summarize(
        "बिल्ली चटाई पर बैठी है। बिल्ली दूध पीती है। "
        "कुत्ता बगीचे में खेलता है। बिल्ली सोती है। पक्षी उड़ता है।",
        max_sentences=2,
    )
    assert summary.count("।") == 2
    assert "बिल्ली" in summary


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
    # sign, so a range "5-10" parsed as [5, -10] (sum -5). The
    # hyphen is a separator there; only a free-standing minus is
    # a sign, and digits inside a word are not numbers. (A date
    # such as "2026-11-27" is a label, not values; see the date
    # regression below.)
    assert parse_numbers("scores range 5-10 points") == [5.0, 10.0]
    assert parse_numbers("temps -5, -3, 2") == [-5.0, -3.0, 2.0]
    assert parse_numbers("version2 build3 has 10 20 30") == [10.0, 20.0, 30.0]
    assert "Sum: 15" in analyze(parse_numbers("range 5-10"))


def test_parse_numbers_ignores_dates_as_labels():
    # Regression: a calendar date beside real values was read
    # as three measurements, so "sales on 2026-11-27 were
    # 10 20 30" reported count 6 with the year, month and day
    # skewing every statistic. Plausible dates are labels and
    # state no values; an implausible chain or a subtraction
    # chain keeps its numbers.
    assert parse_numbers("sales on 2026-11-27 were 10 20 30") == [10.0, 20.0, 30.0]
    assert parse_numbers("sales on 27/11/2026 were 10 20 30") == [10.0, 20.0, 30.0]
    assert parse_numbers("sales on 11/27/2026 were 10 20 30") == [10.0, 20.0, 30.0]
    assert parse_numbers("sales on 11-27-2026 were 10 20 30") == [10.0, 20.0, 30.0]
    assert "Count: 3" in analyze(parse_numbers("sales on 2026-11-27 were 10 20 30"))
    assert parse_numbers("values 13/40/2026 5 10") == [13.0, 40.0, 2026.0, 5.0, 10.0]
    assert parse_numbers("10-5-2 values 1 2 3") == [10.0, 5.0, 2.0, 1.0, 2.0, 3.0]
    # The year-first slash form is the same label: the
    # day-first pattern cannot see it, so it was read as the
    # three values 2026, 11 and 27. An implausible year-first
    # chain keeps its numbers.
    assert parse_numbers("sales on 2026/11/27 were 10 20 30") == [10.0, 20.0, 30.0]
    assert parse_numbers("values 2026/13/40 5 10") == [2026.0, 13.0, 40.0, 5.0, 10.0]
    # The unpadded year-first form ("2026-1-5") is the same
    # label, and the hyphen day-first form takes a two-digit
    # year like its slash twin ("27-11-26"); both were read as
    # phantom measurements beside the real values.
    assert parse_numbers("sales on 2026-1-5 were 10 20 30") == [10.0, 20.0, 30.0]
    assert parse_numbers("sales on 27-11-26 were 10 20 30") == [10.0, 20.0, 30.0]
    assert parse_numbers("sales on 2026-13-40 were 10 20 30") == [
        2026.0,
        13.0,
        40.0,
        10.0,
        20.0,
        30.0,
    ]


def test_parse_numbers_ignores_clock_times_and_month_name_dates():
    # Regression: month-name dates and clock times beside real
    # values were read as measurements, so "meeting at 10:30,
    # scores 5 6 7" reported count 5 and "on 27 Nov 2026 scores
    # were 5 6 7" folded the day and year into every statistic.
    # Plausible clock readings and month-name dates are labels;
    # a ratio, an implausible chain and a modal "may" keep
    # their values.
    assert parse_numbers("meeting at 10:30, scores 5 6 7") == [5.0, 6.0, 7.0]
    assert parse_numbers("call at 3:30 pm about values 5 6 7") == [5.0, 6.0, 7.0]
    assert parse_numbers("duration 1:05:30 and values 5 6 7") == [5.0, 6.0, 7.0]
    assert parse_numbers("on 27 Nov 2026 scores were 5 6 7") == [5.0, 6.0, 7.0]
    assert parse_numbers("Nov 27, 2026 results: 5 6 7") == [5.0, 6.0, 7.0]
    assert parse_numbers("27th November 2026 values 5 6 7") == [5.0, 6.0, 7.0]
    assert parse_numbers("on 5 May 2026 values 5 6 7") == [5.0, 6.0, 7.0]
    assert "Count: 3" in analyze(parse_numbers("meeting at 10:30, scores 5 6 7"))
    assert parse_numbers("ratio 3:1 with values 5 6 7") == [3.0, 1.0, 5.0, 6.0, 7.0]
    assert parse_numbers("implausible 25:99 values 5 6 7") == [25.0, 99.0, 5.0, 6.0, 7.0]
    assert parse_numbers("chapter 5 may contain 5 6 7") == [5.0, 5.0, 6.0, 7.0]
    assert parse_numbers("we march 5 miles with 5 6 7") == [5.0, 5.0, 6.0, 7.0]
    assert parse_numbers("32 Nov 2026 values 5 6 7") == [32.0, 2026.0, 5.0, 6.0, 7.0]
    # The hyphenated month-name forms are the same labels with
    # hyphens instead of spaces; their day and year were read
    # as values. An implausible day keeps its numbers.
    assert parse_numbers("on 27-Nov-2026 scores were 5 6 7") == [5.0, 6.0, 7.0]
    assert parse_numbers("on Nov-27-2026 scores were 5 6 7") == [5.0, 6.0, 7.0]
    assert parse_numbers("32-Nov-2026 values 5 6 7") == [32.0, 2026.0, 5.0, 6.0, 7.0]


def test_parse_numbers_ignores_ordinals_as_labels():
    # Regression: rank labels beside real values were read as
    # measurements, so "ranked 1st, 2nd and 3rd with scores
    # 10 20 30" reported count 6 with the ranks skewing every
    # statistic. Ordinals are labels like dates and clock
    # times; an ordinal inside a month-name date stays a date,
    # and plain values are untouched.
    assert parse_numbers("ranked 1st, 2nd and 3rd with scores 10 20 30") == [10.0, 20.0, 30.0]
    assert parse_numbers("finished 21st with 5 6 7") == [5.0, 6.0, 7.0]
    assert parse_numbers("on 27th Nov 2026 values 1 2 3") == [1.0, 2.0, 3.0]
    assert "Count: 3" in analyze(parse_numbers("ranked 1st with scores 10 20 30"))
    assert parse_numbers("values 5 6 7") == [5.0, 6.0, 7.0]


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


def test_retrieval_accepts_single_line_markers():
    # Regression: the markers written on one line ("query: ...
    # passage: ...") routed here through the all-in-one
    # package but the parser only accepted line-start markers,
    # so the routed request silently fell through to chat.
    out = _run(
        "retrieval",
        "gravion-search",
        "query: how do cats groom passage: Dogs enjoy long walks. "
        "passage: Cats groom themselves with their tongues.",
    )
    assert out.splitlines()[1].endswith("Cats groom themselves with their tongues.")


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
    # Regression: the question-first marker order routed to
    # document-qa through the all-in-one package but the
    # parser only accepted document-first, so the routed
    # request silently fell through to chat.
    out3 = _run(
        "document-qa",
        "gravion-docqa",
        f"question: when does the library open\ndocument: {document}",
    )
    assert "opens at nine" in out3
    # Regression: a document that merely mentions "question:"
    # mid-sentence is content, not the question marker;
    # splitting there truncated the document and corrupted
    # the question. Line-anchored markers win, and the
    # single-line form still parses.
    form_document = (
        "The form has one field. The form asks question: what is "
        "your name? The name field is required."
    )
    out4 = _run(
        "document-qa",
        "gravion-docqa",
        f"document: {form_document}\nquestion: which field is required",
    )
    assert "name field is required" in out4
    out5 = _run(
        "document-qa",
        "gravion-docqa",
        f"document: {document} question: when does the library open",
    )
    assert "opens at nine" in out5


def test_stem_word_matches_irregular_sis_plurals():
    # Regression: the regular "es" rule stemmed "analyses" to
    # "analys" while "analysis" stayed whole, so singular and
    # plural never matched; "news" and "lens" were mangled too.
    assert stem_word("analyses") == stem_word("analysis") == "analysis"
    assert stem_word("crises") == stem_word("crisis") == "crisis"
    assert stem_word("news") == "news"
    assert stem_word("lens") == "lens"
    # Invariant -ies nouns keep their form instead of the
    # ies rule's "sery"/"specy".
    assert stem_word("series") == "series"
    assert stem_word("species") == "species"
    assert stem_word("classes") == stem_word("class") == "class"


def test_stem_word_matches_irregular_past_tense_verbs():
    # Regression: the plural-only stemmer left "sat" and "sit"
    # apart, so a present-tense question never matched a
    # past-tense document sentence.
    assert stem_word("sat") == stem_word("sit") == "sit"
    assert stem_word("ran") == stem_word("run") == "run"
    assert stem_word("went") == stem_word("go") == "go"
    assert stem_word("wrote") == stem_word("write") == "write"
    # Regression: common irregular past tenses and the
    # -ying verbs stayed split from their bases, so a
    # base-form question missed past-tense evidence.
    assert stem_word("had") == stem_word("have") == "have"
    assert stem_word("has") == stem_word("have") == "have"
    assert stem_word("built") == stem_word("build") == "build"
    assert stem_word("woke") == stem_word("wake") == "wake"
    assert stem_word("sold") == stem_word("sell") == "sell"
    assert stem_word("dying") == stem_word("die") == "die"
    assert stem_word("lying") == stem_word("lie") == "lie"
    # The y-base verbs the -ying rule must not mangle.
    assert stem_word("relying") == stem_word("rely") == "rely"
    # Regression: the remaining common irregular past tenses
    # and -en participles stayed split from their bases, so
    # "when did she become a doctor?" missed "she became a
    # doctor" and "where has he go?" missed "he has gone".
    assert stem_word("became") == stem_word("become") == "become"
    assert stem_word("gone") == stem_word("go") == "go"
    assert stem_word("done") == stem_word("do") == "do"
    assert stem_word("flew") == stem_word("fly") == "fly"
    assert stem_word("forgave") == stem_word("forgive") == "forgive"
    assert stem_word("got") == stem_word("get") == "get"
    assert stem_word("lit") == stem_word("light") == "light"
    assert stem_word("wept") == stem_word("weep") == "weep"
    assert stem_word("begun") == stem_word("begin") == "start"
    assert stem_word("thrown") == stem_word("throw") == "throw"
    assert stem_word("worn") == stem_word("wear") == "wear"
    assert stem_word("known") == stem_word("know") == "know"
    # The s-singular animal plurals and classical plurals
    # split from their singulars the same way.
    assert stem_word("platypuses") == stem_word("platypus") == "platypus"
    assert stem_word("rhinoceroses") == stem_word("rhinoceros") == "rhinoceros"
    assert stem_word("genera") == stem_word("genus") == "genus"
    assert stem_word("loci") == stem_word("locus") == "locus"


def test_stem_word_matches_regular_verb_forms():
    # Regression: regular -ing/-ed forms never met their base
    # ("studied" vs "study", "stopped" vs "stop"), so a
    # base-form question missed a derived-form sentence.
    assert stem_word("studied") == stem_word("study") == "study"
    assert stem_word("running") == stem_word("run") == "run"
    assert stem_word("stopped") == stem_word("stop") == "stop"
    assert stem_word("making") == stem_word("make") == "make"
    assert stem_word("hoped") == stem_word("hope") == "hope"
    assert stem_word("hopping") == stem_word("hop") == "hop"
    assert stem_word("used") == stem_word("use") == "use"
    assert stem_word("tried") == stem_word("try") == "try"
    assert stem_word("written") == stem_word("write") == "write"
    # Short non-verbs and -ing/-ed nouns keep their form.
    assert stem_word("king") == "king"
    assert stem_word("seed") == "seed"
    assert stem_word("during") == "during"
    assert stem_word("hundred") == "hundred"


def test_stem_word_matches_remaining_irregulars_and_e_verbs():
    # Regression: "lied"/"arose"/"burnt" and the -u
    # participles stayed split from their bases, the -sis
    # and -mata plurals landed on stems no singular
    # produces, possessive plurals ("childrens") stemmed
    # to the plural stem, and -dge/-ue/-ce verbs lost
    # their e ("judging" -> "judg", "arguing" -> "argu",
    # "forcing" -> "forc"), so base-form questions missed
    # derived-form evidence.
    assert stem_word("lied") == stem_word("lie") == "lie"
    assert stem_word("arose") == stem_word("arise") == "arise"
    assert stem_word("burnt") == stem_word("burn") == "burn"
    assert stem_word("leapt") == stem_word("leap") == "leap"
    assert stem_word("learnt") == stem_word("learn") == "learn"
    assert stem_word("drunk") == stem_word("drink") == "drink"
    assert stem_word("sung") == stem_word("sing") == "sing"
    assert stem_word("fallen") == stem_word("fall") == "fall"
    assert stem_word("forbidden") == stem_word("forbid") == "forbid"
    assert stem_word("swum") == stem_word("swim") == "swim"
    assert stem_word("emphases") == stem_word("emphasis") == "emphasis"
    assert stem_word("syntheses") == stem_word("synthesis") == "synthesis"
    assert stem_word("stigmata") == stem_word("stigma") == "stigma"
    assert stem_word("schemata") == stem_word("schema") == "schema"
    assert stem_word("childrens") == stem_word("child") == "child"
    assert stem_word("mens") == stem_word("man") == "man"
    assert stem_word("womens") == stem_word("woman") == "woman"
    assert stem_word("judging") == stem_word("judge") == "judge"
    assert stem_word("judged") == stem_word("judge") == "judge"
    assert stem_word("bridging") == stem_word("bridge") == "bridge"
    assert stem_word("arguing") == stem_word("argue") == "argue"
    assert stem_word("argued") == stem_word("argue") == "argue"
    assert stem_word("rescuing") == stem_word("rescue") == "rescue"
    assert stem_word("issued") == stem_word("issue") == "issue"
    assert stem_word("forcing") == stem_word("force") == "force"
    assert stem_word("announcing") == stem_word("announce") == "announce"
    assert stem_word("challenging") == stem_word("challenge") == "challenge"
    # Bases the e-restoration must not mangle.
    assert stem_word("hanging") == stem_word("hang") == "hang"
    assert stem_word("syncing") == stem_word("sync") == "sync"
    # Regression: the -en participles "woken"/"drawn" and
    # the s-singular plural "funguses" stayed split from
    # their bases, so base-form questions missed
    # participle evidence.
    assert stem_word("woken") == stem_word("wake") == "wake"
    assert stem_word("drawn") == stem_word("draw") == "draw"
    assert stem_word("funguses") == stem_word("fungus") == "fungus"
    out = answer_question(
        "She lied to him yesterday. He left early.",
        "Who did she lie to?",
    )
    assert "lied to him" in out
    out = answer_question(
        "The sun arose early. Birds chirped.",
        "When did the sun arise?",
    )
    assert "arose early" in out


def test_stem_word_matches_se_and_ve_verb_forms():
    # Regression: -se/-ve verbs lost their e behind a
    # consonant cluster ("licensing" -> "licens",
    # "serving" -> "serv") or a vowel pair ("pleasing"
    # -> "pleas", "receiving" -> "receiv"), so a
    # base-form question missed derived-form evidence
    # and retrieval scored the relevant passage zero;
    # the CVC rule also invented "focuse"/"buse" for
    # the "us" bases "focus"/"bus".
    assert stem_word("licensing") == stem_word("license") == "license"
    assert stem_word("licensed") == stem_word("license") == "license"
    assert stem_word("sensing") == stem_word("sense") == "sense"
    assert stem_word("rinsing") == stem_word("rinse") == "rinse"
    assert stem_word("pleasing") == stem_word("please") == "please"
    assert stem_word("increasing") == stem_word("increase") == "increase"
    assert stem_word("releasing") == stem_word("release") == "release"
    assert stem_word("raising") == stem_word("raise") == "raise"
    assert stem_word("housing") == stem_word("house") == "house"
    assert stem_word("choosing") == stem_word("choose") == "choose"
    assert stem_word("easing") == stem_word("ease") == "ease"
    assert stem_word("analysing") == stem_word("analyse") == "analyse"
    assert stem_word("analyzing") == stem_word("analyze") == "analyze"
    assert stem_word("serving") == stem_word("serve") == "serve"
    assert stem_word("solving") == stem_word("solve") == "solve"
    assert stem_word("receiving") == stem_word("receive") == "receive"
    assert stem_word("believing") == stem_word("believe") == "believe"
    assert stem_word("creating") == stem_word("create") == "create"
    assert stem_word("created") == stem_word("create") == "create"
    assert stem_word("arranging") == stem_word("arrange") == "arrange"
    # Bases the restoration must not mangle.
    assert stem_word("focusing") == stem_word("focus") == "focus"
    assert stem_word("focused") == stem_word("focus") == "focus"
    assert stem_word("busing") == stem_word("bus") == "bus"
    assert stem_word("biasing") == stem_word("bias") == "bias"
    assert stem_word("lensing") == stem_word("lens") == "lens"
    assert stem_word("passing") == stem_word("pass") == "pass"
    assert stem_word("using") == stem_word("use") == "use"
    assert stem_word("eating") == stem_word("eat") == "eat"
    assert stem_word("treating") == stem_word("treat") == "treat"
    assert stem_word("helping") == stem_word("help") == "help"
    # The split reached retrieval and document Q&A: a
    # licensing passage scored zero for "license", and
    # a licensing sentence was no answer to a license
    # question.
    ranked = rank_passages(
        "license terms",
        ["Licensing terms apply here.", "Cooking recipes for dinner."],
    )
    assert ranked[0][1] == "Licensing terms apply here."
    assert ranked[0][0] > 0
    out = answer_question(
        "Licensing terms apply to every module. Prices vary by tier.",
        "What license terms apply?",
    )
    assert "Licensing terms apply" in out


def test_retrieval_module_admits_when_no_passage_matches(make_config):
    # Improvement: a query no passage shares a term
    # with was answered with every passage listed as
    # a "top match" at 0.000, presenting non-matches
    # as evidence; the module now says so plainly.
    from gravion_ai.runtime import GravionRuntime

    runtime = GravionRuntime(make_config(model="gravion-omni"))
    out = runtime.run_module(
        "retrieval",
        "query: quantum chromodynamics\n"
        "passage: Cats groom themselves with their tongues.\n"
        "passage: Dogs enjoy long walks in parks.",
    )
    assert out.startswith("No passage matches the query: quantum chromodynamics")
    assert "Top matches" not in out
    matched = runtime.run_module(
        "retrieval",
        "query: how do cats groom\n"
        "passage: Cats groom themselves with their tongues.\n"
        "passage: Dogs enjoy long walks in parks.",
    )
    assert matched.startswith("Top matches for: how do cats groom")


def test_split_sentences_splits_after_sentence_final_abbreviations():
    # Regression: every abbreviation dot counted as
    # mid-sentence, so "Washington D.C. He was happy." and
    # "met at 5 p.m. They talked." read as one sentence.
    assert split_sentences("Dr. Smith went to Washington D.C. He was happy.") == [
        "Dr. Smith went to Washington D.C.",
        "He was happy.",
    ]
    assert split_sentences("Mr. Jones met at 5 p.m. They talked.") == [
        "Mr. Jones met at 5 p.m.",
        "They talked.",
    ]
    assert split_sentences("He moved to the U.S. The weather was warm.") == [
        "He moved to the U.S.",
        "The weather was warm.",
    ]
    assert split_sentences("She earned her Ph.D. She now teaches.") == [
        "She earned her Ph.D.",
        "She now teaches.",
    ]
    # Titles and capitalised continuations stay whole.
    assert split_sentences("The U.S.A. team won. They celebrated.") == [
        "The U.S.A. team won.",
        "They celebrated.",
    ]
    assert split_sentences("Cats vs. Dogs is a film. It is funny.") == [
        "Cats vs. Dogs is a film.",
        "It is funny.",
    ]
    assert split_sentences("St. Louis is a city. It is in Missouri.") == [
        "St. Louis is a city.",
        "It is in Missouri.",
    ]


def test_document_qa_matches_derived_verb_evidence():
    # Regression: "what did she study?" shares no stem with
    # "she studied mathematics" until "studied" stems to
    # "study", so the module wrongly reported no answer.
    out = answer_question(
        "She studied mathematics at college. The cat sat on the mat.",
        "What did she study?",
    )
    assert "studied mathematics" in out
    out = answer_question(
        "He stopped the car near the gate. Birds fly south.",
        "What did he stop?",
    )
    assert "stopped the car" in out


def test_document_qa_matches_past_tense_evidence():
    # Regression: "where did the cat sit?" shares only "cat"
    # with "the cat sat on the mat" until "sat" stems to
    # "sit", so the module wrongly reported no answer.
    out = answer_question(
        "The cat sat on the mat. Dogs bark loudly.",
        "Where did the cat sit?",
    )
    assert "cat sat on the mat" in out
    # Regression: "had"/"built"/"woke" stayed split from
    # "have"/"build"/"wake", so these questions wrongly
    # reported that the document holds no answer.
    out = answer_question(
        "She had a red bicycle. He sold the car.",
        "What did she have?",
    )
    assert "red bicycle" in out
    out = answer_question(
        "The bridge was built in 1900. It spans the river.",
        "When was the bridge build?",
    )
    assert "built in 1900" in out
    out = answer_question(
        "He woke at dawn. She slept late.",
        "When did he wake?",
    )
    assert "woke at dawn" in out


def test_stem_word_matches_o_and_suppletive_plurals():
    # Regression: "oes" plurals stemmed to a trailing "e" the
    # singular never produces ("heroes" -> "heroe"), and
    # suppletive plurals never met their singulars, so a question
    # about heroes missed a document about a hero.
    assert stem_word("heroes") == stem_word("hero") == "hero"
    assert stem_word("potatoes") == stem_word("potato") == "potato"
    assert stem_word("shoes") == stem_word("shoe") == "shoe"
    assert stem_word("indices") == stem_word("index") == "index"
    assert stem_word("lives") == stem_word("life") == "life"
    assert stem_word("children") == stem_word("child") == "child"
    assert stem_word("women") == stem_word("woman") == "woman"
    out = answer_question(
        "The hero saved the city at dawn. The hero helped people escape.",
        "what did heroes do",
    )
    assert "hero saved the city" in out


def test_stem_word_matches_ves_sis_and_doubled_z_plurals():
    # Regression: f -> ves plurals beyond the first few stemmed
    # to a trailing "e" the singular never produces ("loaves"
    # -> "loave", "thieves" -> "thieve"), -sis plurals lost
    # their ending ("oases" -> "oas", "parentheses" ->
    # "parenthes"), the "oes" rule over-stripped "canoes" to
    # "cano", and doubled-z plurals kept both letters
    # ("quizzes" -> "quizz"), so questions using those plurals
    # missed documents using the singulars.
    assert stem_word("loaves") == stem_word("loaf") == "loaf"
    assert stem_word("thieves") == stem_word("thief") == "thief"
    assert stem_word("calves") == stem_word("calf") == "calf"
    assert stem_word("selves") == stem_word("self") == "self"
    assert stem_word("oases") == stem_word("oasis") == "oasis"
    assert stem_word("parentheses") == stem_word("parenthesis") == "parenthesis"
    assert stem_word("apices") == stem_word("apex") == "apex"
    assert stem_word("canoes") == stem_word("canoe") == "canoe"
    assert stem_word("quizzes") == stem_word("quiz") == "quiz"
    out = answer_question(
        "The thief took the loaf at dawn. The thief hid the loaf.",
        "what did thieves do with loaves",
    )
    assert "thief took the loaf" in out


def test_split_sentences_keeps_numeric_abbreviations_whole():
    # Regression: "No. 5", "Fig. 3", month abbreviations and
    # dotted acronyms split into fragments, so document Q&A
    # quoted "No." as a sentence and summaries counted fragments
    # as sentences.
    assert split_sentences("No. 5 is ready. It works well.") == [
        "No. 5 is ready.",
        "It works well.",
    ]
    assert split_sentences("Fig. 3 shows growth. Sales rose.") == [
        "Fig. 3 shows growth.",
        "Sales rose.",
    ]
    assert split_sentences("Jan. 5 was cold. Snow fell.") == [
        "Jan. 5 was cold.",
        "Snow fell.",
    ]
    assert split_sentences("The U.S.A. team won. They celebrated.") == [
        "The U.S.A. team won.",
        "They celebrated.",
    ]
    # The same word still ends a real sentence without a digit.
    assert split_sentences("He said no. Then he left.") == [
        "He said no.",
        "Then he left.",
    ]


def test_summarize_never_returns_empty_for_stopword_text():
    # Regression: when every sentence was stopwords alone nothing
    # could be scored, and the summary came back empty instead of
    # keeping the source's opening sentences.
    out = summarize("The and or. Is are was. Of at by. For with about.")
    assert out == "The and or. Is are was. Of at by."
    assert summarize("Cats sleep. Dogs bark.", max_sentences=0) == ""


def test_parse_numbers_ignores_version_and_address_labels():
    # Regression: "3.14.15" was read as the pair 3.14 and 15, so
    # a version label inflated every statistic. Dotted sequences
    # with two or more dots are labels, not measurements; plain
    # decimals still parse.
    assert parse_numbers("release 3.14.15 values 5, 10, 15") == [5.0, 10.0, 15.0]
    assert parse_numbers("ip 10.0.0.8 count 5") == [5.0]
    # IPv6 labels are addresses too: their trailing groups are
    # not measurements. Validation keeps invalid colon chains
    # available to the clock/ratio handling below.
    assert parse_numbers("ipv6 fe80::1 values 5 6 7") == [5.0, 6.0, 7.0]
    assert parse_numbers("host 2001:db8:85a3::8a2e:370:7334 values 5 6 7") == [5.0, 6.0, 7.0]
    assert parse_numbers("mapped ::ffff:192.0.2.1 values 5 6 7") == [5.0, 6.0, 7.0]
    assert parse_numbers("chain 1:2:3 values 5 6 7") == [1.0, 2.0, 3.0, 5.0, 6.0, 7.0]
    assert parse_numbers("3.14 and 2.71") == [3.14, 2.71]
    assert parse_numbers("value 3.14. Next 2.71") == [3.14, 2.71]


def test_stem_word_matches_se_ie_and_classical_plurals():
    # Regression: the "es" rule dropped the ending after any
    # s-stem, so -se singulars split from their plurals
    # ("responses" -> "respons" while "response" stayed whole)
    # and a question about responses missed a document about a
    # response. -ie plurals ("movies" -> "movy"), s-singular
    # plurals ("buses") and classical plurals ("criteria")
    # split the same way.
    assert stem_word("cases") == stem_word("case") == "case"
    assert stem_word("responses") == stem_word("response") == "response"
    assert stem_word("databases") == stem_word("database") == "database"
    assert stem_word("houses") == stem_word("house") == "house"
    assert stem_word("classes") == stem_word("class") == "class"
    assert stem_word("buses") == stem_word("bus") == "bus"
    assert stem_word("aliases") == stem_word("alias") == "alias"
    assert stem_word("movies") == stem_word("movie") == "movie"
    assert stem_word("cookies") == stem_word("cookie") == "cookie"
    assert stem_word("criteria") == stem_word("criterion") == "criterion"
    assert stem_word("data") == stem_word("datum") == "datum"
    out = answer_question(
        "The response arrived at dawn. The response helped the team.",
        "what responses arrived",
    )
    assert "response arrived at dawn" in out


def test_parse_numbers_reads_underscore_groups_and_ignores_machine_literals():
    # Regression: "1_000" was read as the phantom pair 1 and 0,
    # and a hex literal "0x10" as the phantom value 0, so code
    # literals skewed every statistic. Underscore grouping now
    # parses as one value and machine literals are not decimal
    # measurements at all.
    assert parse_numbers("values 1_000 2_000 3_000") == [1000.0, 2000.0, 3000.0]
    assert parse_numbers("values 1_000.5 2_000.5") == [1000.5, 2000.5]
    assert parse_numbers("0x10 0x20 0x30") == []
    assert parse_numbers("count 0xFF values 5 10 15") == [5.0, 10.0, 15.0]


def test_parse_numbers_ignores_uuids_and_overflowing_tokens():
    # Regression: a UUID's first group read as the
    # scientific-notation value 123e4567, which overflows to
    # infinity and crashed analyze() inside the standard
    # deviation, and its trailing group read as a huge phantom
    # measurement. UUIDs are identifier labels, and a token
    # whose value overflows float states no finite measurement.
    uuid = "123e4567-e89b-12d3-a456-426614174000"
    assert parse_numbers(f"record {uuid} values 5 10 15") == [5.0, 10.0, 15.0]
    assert "Count: 3" in analyze(parse_numbers(f"record {uuid} values 5 10 15"))
    assert parse_numbers("values 1e999 5 10 15") == [5.0, 10.0, 15.0]
    assert "Count: 3" in analyze(parse_numbers("values 1e999 5 10 15"))


def test_translation_bare_request_picks_direction_by_coverage():
    # Regression: a bare "translate:" request picked its
    # direction from the first word alone, so "how are you
    # today" (whose first word is no phrasebook key) was sent
    # to the Hindi table and refused. Direction now follows
    # whichever table covers more of the text; uncovered text
    # is still refused, never guessed.
    assert "aap kaise hain" in _run(
        "translation", "gravion-translate", "translate: how are you today"
    )
    assert "suprabhaat" in _run(
        "translation", "gravion-translate", "translate: good morning friend"
    )
    assert _run(
        "translation", "gravion-translate", "translate: kal milte hain"
    ) == "see you tomorrow"
    assert "outside my Gravion phrasebook" in _run(
        "translation", "gravion-translate", "translate: zxqwv"
    )


def test_translation_expands_contractions_to_phrasebook_forms():
    # Regression: phrasebook keys are expanded forms, so
    # "i don't understand" normalised to "i don t understand" and
    # missed a covered phrase. Contractions (straight or curly
    # apostrophe) now expand before matching; uncovered text is
    # still refused, never guessed.
    assert translate_phrase("i don't understand", True).startswith("mujhe samajh")
    assert translate_phrase("i don’t understand", True).startswith("mujhe samajh")
    assert translate_phrase("let's start", True).startswith("chaliye shuru")
    assert translate_phrase("i'm fine", True).startswith("main theek")
    assert "outside my Gravion phrasebook" in translate_phrase("don't worry", True)


def test_parse_numbers_reads_fractions_as_single_values():
    # Regression: a fraction "1/2" was read as the phantom pair
    # 1 and 2, so "1/2, 1/4, 3/4" reported count 6 and a mean
    # over numerators and denominators. A fraction is one
    # value, a mixed number is one value, and a zero
    # denominator states no value at all.
    assert parse_numbers("values 1/2, 1/4, 3/4") == [0.5, 0.25, 0.75]
    assert parse_numbers("values 1 1/2, 2 1/2, 3") == [1.5, 2.5, 3.0]
    assert parse_numbers("values 5/4, 3/2") == [1.25, 1.5]
    assert parse_numbers("values 1/0, 5, 10") == [5.0, 10.0]
    assert "Sum: 1.5" in analyze(parse_numbers("values 1/2, 1/4, 3/4"))
    # A digit run past the integer-conversion limit crashed
    # the fraction fold; it now states no value and the real
    # measurements beside it survive.
    assert parse_numbers("values 10 20 " + "9" * 5000 + "/2") == [10.0, 20.0]
    assert parse_numbers("values 10 20 1 " + "9" * 5000 + "/2") == [10.0, 20.0, 1.0]


def test_analyze_large_values_do_not_overflow_the_mean():
    # Regression: the mean was total / count, so values at
    # 1e308 summed to infinity and reported "Mean: inf" with a
    # phantom rising trend, though the mean is plainly 1e308.
    # The exact total also survives cancellation past float
    # range ("1e308 + 1e308 - 1e308" is 1e308, not infinity).
    out = analyze([1e308, 1e308, 1e308])
    assert "Mean: 1e+308" in out
    assert "Trend: flat" in out
    out = analyze([1e308, 1e308, -1e308])
    assert "Sum: 1e+308" in out
    assert "Mean: 3.33333e+307" in out
    # The variance of these values is past float range while
    # the deviation is not; the statistic must still report.
    assert "Standard deviation: 9.42809e+307" in out


def test_analyze_median_does_not_overflow_at_float_range():
    # Regression: the median averaged its two middle values
    # in float arithmetic, so an even-length list at float
    # range reported "Median: inf" though the median is
    # plainly 1e308 — the same phantom infinity the exact
    # mean already avoids.
    out = analyze([1e308, 1e308])
    assert "Median: 1e+308" in out
    out = analyze([1e308, 1.5e308])
    assert "Median: 1.25e+308" in out
    out = analyze([-1e308, 1e308])
    assert "Median: 0" in out


def test_parse_numbers_reads_unicode_minus_as_a_sign():
    # Regression: text pasted from documents writes negatives
    # with the Unicode minus (U+2212), which lost its sign and
    # read "−5" as +5. A Unicode range dash keeps the range
    # reading, like its ASCII twin.
    assert parse_numbers("temps −5, −3, 2") == [-5.0, -3.0, 2.0]
    assert parse_numbers("scores range 5−10 points") == [5.0, 10.0]


def test_stem_word_matches_compound_and_classical_plurals():
    # Regression: compound -man plurals end in no "s", so no
    # suffix rule reached them and a question about firemen
    # missed a document about a fireman. They are listed (a
    # blanket men -> man rule would mangle "specimen"), along
    # with classical -i/-ae plurals and the variant "busses".
    assert stem_word("firemen") == stem_word("fireman") == "fireman"
    assert stem_word("policemen") == stem_word("policeman") == "policeman"
    assert stem_word("grandchildren") == stem_word("grandchild") == "grandchild"
    assert stem_word("stimuli") == stem_word("stimulus") == "stimulus"
    assert stem_word("vertebrae") == stem_word("vertebra") == "vertebra"
    assert stem_word("busses") == stem_word("bus") == "bus"
    assert stem_word("specimen") == "specimen"
    out = answer_question(
        "The fireman rescued the cat from the roof. "
        "The policeman directed traffic at the corner.",
        "what did the firemen rescue from the roof",
    )
    assert "fireman rescued the cat" in out


def test_stem_word_matches_women_children_and_classical_plurals():
    # Regression: compound -women plurals split from their
    # singulars exactly as the -man compounds had, and
    # classical plurals such as "radii" and "errata" landed
    # on stems their singulars never produce, so a question
    # about businesswomen or radii missed a document about a
    # businesswoman or a radius.
    assert stem_word("businesswomen") == stem_word("businesswoman") == "businesswoman"
    assert stem_word("chairwomen") == stem_word("chairwoman") == "chairwoman"
    assert stem_word("stepchildren") == stem_word("stepchild") == "stepchild"
    assert stem_word("salespeople") == stem_word("salesperson") == "salesperson"
    assert stem_word("radii") == stem_word("radius") == "radius"
    assert stem_word("errata") == stem_word("erratum") == "erratum"
    assert stem_word("codices") == stem_word("codex") == "codex"
    assert stem_word("nebulae") == stem_word("nebula") == "nebula"
    assert stem_word("wharves") == stem_word("wharf") == "wharf"
    assert stem_word("lice") == stem_word("louse") == "louse"
    out = answer_question(
        "The businesswoman opened the bakery at dawn. "
        "The library opens at nine in the morning.",
        "when did the businesswomen open the bakery",
    )
    assert "businesswoman opened the bakery" in out
    out = answer_question(
        "The radius of the circle is five units. The square has four sides.",
        "what is the radii of the circle",
    )
    assert "radius of the circle" in out


def test_data_analysis_sum_and_range_past_float_range_stay_finite():
    # Regression: a finite sum or range past float range
    # ("1e308 + 1e308", the range of "-1e308, 1e308") reported
    # as "inf" even though the exact value is known — the same
    # phantom infinity the exact mean and median already avoid.
    out = analyze([1e308, 1e308])
    assert "Sum: 2e+308" in out
    assert "inf" not in out
    out = analyze([-1e308, 1e308])
    assert "Range: 2e+308" in out
    assert "inf" not in out
    out = analyze([1.7e308, 1.7e308, 1.7e308])
    assert "Sum: 5.1e+308" in out
    # Ordinary values keep their float formatting.
    out = analyze([2.0, 4.0, 6.0])
    assert "Sum: 12" in out
    assert "Range: 4" in out


def test_phone_numbers_are_labels_not_values():
    # Regression: a phone number read as three
    # measurements and skewed every statistic.
    assert parse_numbers("phone 555-123-4567 total 100") == [100.0]
    assert parse_numbers("call 555.123.4567 scores 10 20") == [10.0, 20.0]
    # A plain range keeps its reading.
    assert parse_numbers("range 5-10") == [5.0, 10.0]


def test_whole_statistics_print_in_full():
    # Regression: six-significant-digit formatting printed
    # the median of 1000000/1000001/1000002 as "1e+06",
    # the same text as its minimum.
    out = analyze([1000000.0, 1000001.0, 1000002.0])
    assert "Sum: 3000003" in out
    assert "Mean: 1000001" in out
    assert "Median: 1000001" in out
    assert "Min: 1000000" in out
    assert "Max: 1000002" in out
    # Values past the exact-integer range keep the
    # scientific form.
    out = analyze([1e308, 1e308])
    assert "Median: 1e+308" in out


def test_rupee_abbreviation_and_everyday_phrasebook_entries():
    # Regression: "Rs." before an amount split the amount
    # into its own sentence.
    assert split_sentences("It costs Rs. 500. Pay now.") == [
        "It costs Rs. 500.",
        "Pay now.",
    ]
    # The bare everyday verbs beside "come here" were
    # refused as outside the phrasebook.
    assert "jaaiye" in translate_phrase("go", to_hindi=True)
    assert "aaiye" in translate_phrase("come", to_hindi=True)
    assert "mujhe aapse pyaar hai" in translate_phrase(
        "i love you", to_hindi=True
    )
    assert translate_phrase("bahut accha", to_hindi=False) == "very good"
    assert translate_phrase("jaaiye", to_hindi=False) == "go"


def test_stem_word_matches_degree_and_adverb_forms():
    # Regression: base-form questions ("who was happy?",
    # "who worked quick?", "who is fast?") matched no document
    # sentence in a degree form ("happier", "quickly",
    # "faster"), so document Q&A reported no answer and
    # retrieval scored the relevant passage zero.
    assert stem_word("happier") == stem_word("happy") == "happy"
    assert stem_word("happiest") == "happy"
    assert stem_word("happily") == "happy"
    assert stem_word("faster") == stem_word("fast") == "fast"
    assert stem_word("fastest") == "fast"
    assert stem_word("quickly") == stem_word("quick") == "fast"
    assert stem_word("bigger") == stem_word("big") == "big"
    assert stem_word("biggest") == "big"
    # "large" lands on "big" through the synonym stems, so
    # its degree forms land there too.
    assert stem_word("larger") == stem_word("large") == "big"
    assert stem_word("largest") == "big"
    assert stem_word("nicer") == stem_word("nice") == "nice"
    assert stem_word("easily") == stem_word("easy") == "easy"
    assert stem_word("teacher") == stem_word("teach") == "teach"
    assert stem_word("runner") == stem_word("run") == "run"
    # Base words whose ending belongs to the word keep it.
    assert stem_word("river") == "river"
    assert stem_word("family") == "family"
    assert stem_word("never") == "never"
    assert stem_word("water") == "water"
    assert stem_word("hundred") == "hundred"
    out = answer_question(
        "The children were happier at the park. "
        "The team worked quickly. She runs faster than him.",
        "who was happy at the park",
    )
    assert "happier at the park" in out
    out = answer_question(
        "The children were happier at the park. "
        "The team worked quickly. She runs faster than him.",
        "who is fast",
    )
    assert "runs faster" in out


def test_stem_word_matches_suppletive_comparatives():
    # Regression: suppletive comparatives share no stem with
    # their base ("better" landed on "bet", "worse" stayed
    # apart from "bad"), so a base-form question matched no
    # document sentence in a comparative form and document
    # Q&A reported no answer. Positional -er words keep
    # their form instead of landing on "form"/"in"/"up".
    assert stem_word("better") == stem_word("good") == "good"
    assert stem_word("best") == "good"
    assert stem_word("worse") == stem_word("bad") == "bad"
    assert stem_word("worst") == "bad"
    assert stem_word("farther") == stem_word("far") == "far"
    assert stem_word("furthest") == "far"
    assert stem_word("elder") == stem_word("old") == "old"
    assert stem_word("eldest") == "old"
    assert stem_word("less") == stem_word("little") == "little"
    assert stem_word("least") == "little"
    assert stem_word("former") == "former"
    assert stem_word("latter") == "latter"
    assert stem_word("inner") == "inner"
    assert stem_word("outer") == "outer"
    assert stem_word("upper") == "upper"
    out = answer_question(
        "She is better at chess than him. He plays football daily.",
        "who is good at chess",
    )
    assert "better at chess" in out
    out = answer_question(
        "This is the worst result so far. Other results were fine.",
        "what is the bad result",
    )
    assert "worst result" in out


def test_summarize_skips_near_duplicate_sentences():
    # Regression: top-scored sentences restated one point
    # ("Cats sleep." / "Cats sleep a lot.") while a distinct
    # point was dropped from the summary.
    text = (
        "Cats sleep. Cats sleep a lot. Dogs bark loudly. "
        "Birds fly south. Cats sleep all day."
    )
    out = summarize(text, max_sentences=2)
    assert "Cats sleep." in out
    assert "Cats sleep a lot." not in out
    assert "Birds fly south." in out
    out = summarize(text, max_sentences=3)
    assert "Dogs bark loudly." in out
    assert "Birds fly south." in out


def test_synonym_stems_match_document_and_retrieval():
    # Regression: a question using one word ("buy", "car")
    # missed a document sentence using its synonym
    # ("purchased", "automobile"), so document Q&A
    # reported no answer and retrieval scored the
    # relevant passage zero.
    assert stem_word("purchased") == stem_word("buy") == "buy"
    assert stem_word("automobiles") == stem_word("car") == "car"
    assert stem_word("colours") == stem_word("color") == "color"
    assert stem_word("kids") == stem_word("child") == "child"
    out = answer_question(
        "She purchased a bicycle yesterday. He sold a car.",
        "what did she buy",
    )
    assert "purchased a bicycle" in out
    ranked = rank_passages(
        "car", ["The automobile was red", "The house was blue"]
    )
    assert ranked[0][0] > 0
    assert ranked[0][1] == "The automobile was red"
    # Size, speed and everyday verb synonyms state the
    # same matching fact: "which big house" missed "the
    # large house" and a "fast animal" query scored "the
    # quick fox ran" zero.
    assert stem_word("larger") == stem_word("big") == "big"
    assert stem_word("tiny") == stem_word("small") == "small"
    assert stem_word("quickly") == stem_word("fast") == "fast"
    assert stem_word("began") == stem_word("started") == "start"
    assert stem_word("displayed") == stem_word("showed") == "show"
    assert stem_word("finished") == stem_word("completed") == "complete"
    out = answer_question(
        "She began the work early. He ended late.",
        "who started the work",
    )
    assert "began the work" in out
    ranked = rank_passages(
        "fast animal",
        ["The quick fox ran", "The slow turtle walked"],
    )
    assert ranked[0][0] > 0
    assert ranked[0][1] == "The quick fox ran"
