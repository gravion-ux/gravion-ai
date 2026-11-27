"""Tests for the text specialist modules: translation, summarization,
retrieval, data-analysis and document-qa. Each module's deterministic
engine is exercised directly, and guidance-style prompts are shown to
fall through to the resolved specialist model."""

from gravion_ai.modules.data_analysis.module import analyze, parse_numbers
from gravion_ai.modules.document_qa.module import answer_question
from gravion_ai.modules.retrieval.module import rank_passages
from gravion_ai.modules.summarization.module import split_sentences, stem_word, summarize
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
    assert _run("summarization", "gravion-summary", f"summarize the following: {text}") == summarize(text)


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
