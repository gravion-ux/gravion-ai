"""SummarizationModule — extractive summarization, computed locally.

The summary is built only from the source's own sentences: sentences
are scored by term frequency and position, the strongest are kept in
their original order, and nothing is rewritten or invented. Requests
without source text are answered by the resolved model (Gravion
Summary), whose trained weights cover summarization guidance.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

from ...registry import ModuleContext, ModuleManifest

_STOP = frozenset(
    "a an the and or but if then else of at by for with about into "
    "through after over between out against during without before "
    "under around among is are was were be been being have has had "
    "do does did will would shall should can could may might must "
    "i you he she it we they this that these those as in on to from".split()
)

# Accept both spellings: the all-in-one router already treats a
# leading "summarise:" as decisive, so refusing it here would
# drop a correctly routed request to the model fallback. The
# "summarize this:" / "summarize the following:" forms name the
# same input: refusing them left the instruction itself inside
# the text being summarized. The object may carry its noun
# ("summarize the following text:", "summarize this passage:"):
# the router sends "summarize this passage:" here on the
# verb+noun signal, so refusing the noun form dropped a routed
# request to chat whenever the text ran under four sentences.
# The "tldr:" / "tl;dr:" prefix asks for the same summary in
# shorthand; refusing it left the prefix inside the text being
# summarized whenever the text ran to four sentences.
_SUMMARIZE_RE = re.compile(
    r"(?:summar(?:ize|ise)(?:\s+(?:this|the following)"
    r"(?:\s+(?:text|passage|article|document|paragraph|story|report|essay))?"
    r"|\s+the\s+(?:text|passage|article|document|paragraph|story|report|essay))?"
    r"|tl;?dr)"
    r"\s*:\s*(.+)",
    re.IGNORECASE | re.DOTALL,
)

# Abbreviations whose final dot does not end a sentence, plus
# dotted forms (e.g. "e.g", "u.s"). Single uppercase initials
# ("A. Smith") are handled separately in split_sentences.
_ABBREVIATIONS = frozenset(
    "dr mr mrs ms prof sr jr st vs etc e.g i.e u.s u.k a.m p.m "
    "dept est inc ltd corp al cf ed eds gen sen rep "
    "gov sgt capt lt col maj rev hon ph.d m.d b.a m.a".split()
)
# Abbreviations that only abbreviate before a number ("No. 5",
# "Fig. 3", "Jan. 5", "approx. 5"): the same words end real
# sentences ("he said no. Then...", "Pi is 3.14 approx. Cats
# sleep."), so their dot counts as an abbreviation dot only
# when a digit follows.
_NUMERIC_ABBREVIATIONS = frozenset(
    "no fig figs eq vol pp p ch sec approx ca jan feb mar apr "
    "jun jul aug sep sept oct nov dec".split()
)
_CLOSERS = "\"'”’)]"


# Irregular plurals whose regular rules would land on a stem the
# singular never produces: -sis plurals ("analyses" -> "analys"
# while "analysis" stays whole, "oases" -> "oas"), -ices plurals
# ("indices" -> "indice", "apices" -> "apice"), f -> ves plurals
# ("lives" -> "live" while "life" stays whole, "loaves" ->
# "loave"), the -oe plural the "oes" rule over-strips ("canoes"
# -> "cano"), doubled-z plurals ("quizzes" -> "quizz" while
# "quiz" keeps one z) and suppletive plurals ("children",
# "women"), plus two common non-plurals the final-s rule would
# otherwise mangle ("news" -> "new", "lens" -> "len").
# Single-s singulars ("bus", "alias", "virus") spell their
# plurals exactly like -se singulars ("case", "response"), so
# the s-singular plurals live here and the regular "es" rule
# only fires after ss/x/z/ch/sh; -ie singulars ("movie",
# "cookie") likewise spell their plurals like y-singulars
# ("city"), so they are listed too, along with classical
# plurals ("criteria", "phenomena", "data") no suffix rule
# can derive. Singulars the final-s rule would mangle
# ("alias" -> "alia", "canvas" -> "canva") map to themselves.
_IRREGULAR_STEMS = {
    "analyses": "analysis",
    "crises": "crisis",
    "theses": "thesis",
    "diagnoses": "diagnosis",
    "hypotheses": "hypothesis",
    "oases": "oasis",
    "parentheses": "parenthesis",
    "indices": "index",
    "matrices": "matrix",
    "vertices": "vertex",
    "appendices": "appendix",
    "apices": "apex",
    "vortices": "vortex",
    "cacti": "cactus",
    "fungi": "fungus",
    "lives": "life",
    "wives": "wife",
    "leaves": "leaf",
    "wolves": "wolf",
    "knives": "knife",
    "halves": "half",
    "shelves": "shelf",
    "loaves": "loaf",
    "thieves": "thief",
    "calves": "calf",
    "scarves": "scarf",
    "dwarves": "dwarf",
    "selves": "self",
    "elves": "elf",
    "canoes": "canoe",
    "quizzes": "quiz",
    "whizzes": "whiz",
    "fezzes": "fez",
    "children": "child",
    "men": "man",
    "women": "woman",
    "feet": "foot",
    "teeth": "tooth",
    "geese": "goose",
    "mice": "mouse",
    "people": "person",
    "oxen": "ox",
    "goes": "go",
    # Irregular past-tense verbs map to their base form, or a
    # question in the present tense ("where did the cat sit?")
    # never matches a document sentence in the past tense
    # ("the cat sat on the mat") and document Q&A wrongly
    # reports that the document holds no answer.
    "sat": "sit",
    "ran": "run",
    "went": "go",
    "came": "come",
    "saw": "see",
    "ate": "eat",
    "drank": "drink",
    "wrote": "write",
    "spoke": "speak",
    "took": "take",
    "gave": "give",
    "made": "make",
    "said": "say",
    "told": "tell",
    "found": "find",
    "thought": "think",
    "brought": "bring",
    "bought": "buy",
    "taught": "teach",
    "caught": "catch",
    "began": "begin",
    "broke": "break",
    "chose": "choose",
    "drove": "drive",
    "fell": "fall",
    "felt": "feel",
    "forgot": "forget",
    "grew": "grow",
    "kept": "keep",
    "knew": "know",
    "left": "leave",
    "lost": "lose",
    "met": "meet",
    "paid": "pay",
    "sent": "send",
    "slept": "sleep",
    "stood": "stand",
    "understood": "understand",
    "won": "win",
    # More irregular past tenses no suffix rule reaches, plus
    # the -ying verbs whose stripped stem ("dy", "ly") is a
    # form the base ("die", "lie") never produces: without
    # them "what did she have?" missed "she had a bicycle",
    # "when was the bridge build?" missed "built in 1900",
    # and "when did he wake?" missed "he woke at dawn", so
    # document Q&A wrongly reported no answer.
    "had": "have",
    "has": "have",
    "built": "build",
    "sold": "sell",
    "held": "hold",
    "heard": "hear",
    "woke": "wake",
    "stole": "steal",
    "drew": "draw",
    "threw": "throw",
    "fought": "fight",
    "sought": "seek",
    "dug": "dig",
    "hung": "hang",
    "shook": "shake",
    "rode": "ride",
    "sang": "sing",
    "swam": "swim",
    "rang": "ring",
    "sank": "sink",
    "hid": "hide",
    "froze": "freeze",
    "rose": "rise",
    "tore": "tear",
    "wore": "wear",
    "stuck": "stick",
    "led": "lead",
    "spent": "spend",
    "bent": "bend",
    "lent": "lend",
    "dealt": "deal",
    "fed": "feed",
    # The remaining common irregular past tenses and their
    # -en participles: without them "when did she become a
    # doctor?" missed "she became a doctor", "where has he
    # go?" missed "he has gone to Delhi" and "where did the
    # bird fly?" missed "the bird flew south", so document
    # Q&A wrongly reported no answer. "Laid" maps to "lay"
    # (the hen sense); the past of "lie" is also spelled
    # "lay", which no context-free map can hold both ways,
    # so "lay" itself stays unmapped. The s-singular animal
    # plurals ("platypuses", "rhinoceroses") split from
    # their singulars the same way "buses" did, and
    # "genera"/"loci" are the classical plurals of
    # "genus"/"locus" no suffix rule derives.
    "became": "become",
    "bit": "bite",
    "blew": "blow",
    "bore": "bear",
    "bound": "bind",
    "bred": "breed",
    "clung": "cling",
    "crept": "creep",
    "dove": "dive",
    "dreamt": "dream",
    "fled": "flee",
    "flew": "fly",
    "forbade": "forbid",
    "forgave": "forgive",
    "got": "get",
    "ground": "grind",
    "laid": "lay",
    "lit": "light",
    "meant": "mean",
    "shot": "shoot",
    "shrank": "shrink",
    "slid": "slide",
    "spun": "spin",
    "sprang": "spring",
    "stung": "sting",
    "struck": "strike",
    "swore": "swear",
    "swung": "swing",
    "wept": "weep",
    "withdrew": "withdraw",
    "bled": "bleed",
    "knelt": "kneel",
    "awoke": "awake",
    "beaten": "beat",
    "begun": "begin",
    "bitten": "bite",
    "blown": "blow",
    "born": "bear",
    "done": "do",
    "forgiven": "forgive",
    "frozen": "freeze",
    "gone": "go",
    "gotten": "get",
    "grown": "grow",
    "hidden": "hide",
    "known": "know",
    "mistaken": "mistake",
    "ridden": "ride",
    "risen": "rise",
    "shaken": "shake",
    "shown": "show",
    "shrunk": "shrink",
    "stolen": "steal",
    "sprung": "spring",
    "sworn": "swear",
    "thrown": "throw",
    "torn": "tear",
    "withdrawn": "withdraw",
    "worn": "wear",
    "awoken": "awake",
    "genera": "genus",
    "loci": "locus",
    "platypuses": "platypus",
    "hippopotamuses": "hippopotamus",
    "walruses": "walrus",
    "rhinoceroses": "rhinoceros",
    "rhinoceros": "rhinoceros",
    "dying": "die",
    "lying": "lie",
    "tying": "tie",
    "vying": "vie",
    "does": "does",
    "news": "news",
    "lens": "lens",
    "alias": "alias",
    "canvas": "canvas",
    "atlas": "atlas",
    "bias": "bias",
    "buses": "bus",
    "gases": "gas",
    "lenses": "lens",
    "aliases": "alias",
    "viruses": "virus",
    "statuses": "status",
    "canvases": "canvas",
    "octopuses": "octopus",
    "geniuses": "genius",
    "atlases": "atlas",
    "biases": "bias",
    "bonuses": "bonus",
    "censuses": "census",
    "surpluses": "surplus",
    "cactuses": "cactus",
    "focuses": "focus",
    "syllabuses": "syllabus",
    "indexes": "index",
    "appendixes": "appendix",
    "movies": "movie",
    "cookies": "cookie",
    "zombies": "zombie",
    "calories": "calorie",
    "hoodies": "hoodie",
    "brownies": "brownie",
    "genies": "genie",
    "pixies": "pixie",
    "prairies": "prairie",
    "rookies": "rookie",
    "selfies": "selfie",
    "newbies": "newbie",
    "criteria": "criterion",
    "phenomena": "phenomenon",
    "syllabi": "syllabus",
    "foci": "focus",
    "nuclei": "nucleus",
    "strata": "stratum",
    "curricula": "curriculum",
    "memoranda": "memorandum",
    "bacteria": "bacterium",
    "data": "datum",
    "media": "medium",
    "hooves": "hoof",
    "dice": "die",
    # Compound -man plurals and the -child compound: no suffix
    # rule reaches them ("firemen" ends in no "s"), so a
    # question about firemen missed a document about a
    # fireman. Listed, not ruled: a blanket men -> man rule
    # would mangle singulars such as "specimen" and "omen".
    "firemen": "fireman",
    "policemen": "policeman",
    "businessmen": "businessman",
    "chairmen": "chairman",
    "spokesmen": "spokesman",
    "congressmen": "congressman",
    "salesmen": "salesman",
    "workmen": "workman",
    "fishermen": "fisherman",
    "craftsmen": "craftsman",
    "statesmen": "statesman",
    "postmen": "postman",
    "mailmen": "mailman",
    "snowmen": "snowman",
    "gentlemen": "gentleman",
    "sportsmen": "sportsman",
    "guardsmen": "guardsman",
    "servicemen": "serviceman",
    "tradesmen": "tradesman",
    "watchmen": "watchman",
    "laymen": "layman",
    "clergymen": "clergyman",
    "journeymen": "journeyman",
    "grandchildren": "grandchild",
    # Classical -i/-ae plurals and the variant "busses" the
    # ss rule lands on "buss", a stem "bus" never produces.
    "stimuli": "stimulus",
    "alumni": "alumnus",
    "octopi": "octopus",
    "vertebrae": "vertebra",
    "formulae": "formula",
    "larvae": "larva",
    "antennae": "antenna",
    "millennia": "millennium",
    "busses": "bus",
    # Compound -women plurals split from their singulars the
    # same way the -man compounds did ("businesswomen" ends
    # in no "s"), as do the -children/-people compounds the
    # whole-word entries above cannot reach.
    "businesswomen": "businesswoman",
    "chairwomen": "chairwoman",
    "spokeswomen": "spokeswoman",
    "policewomen": "policewoman",
    "congresswomen": "congresswoman",
    "saleswomen": "saleswoman",
    "gentlewomen": "gentlewoman",
    "sportswomen": "sportswoman",
    "servicewomen": "servicewoman",
    "stepchildren": "stepchild",
    "schoolchildren": "schoolchild",
    "salespeople": "salesperson",
    # More classical plurals no suffix rule derives: -i/-ae
    # ("radii", "nebulae"), -ices ("codices", "helices"),
    # -sis ("synopses", "prognoses", "testes"), the -a
    # plurals "errata"/"addenda", the f -> ves plural
    # "wharves" and the suppletive "lice".
    "radii": "radius",
    "alumnae": "alumna",
    "nebulae": "nebula",
    "supernovae": "supernova",
    "personae": "persona",
    "codices": "codex",
    "helices": "helix",
    "synopses": "synopsis",
    "prognoses": "prognosis",
    "testes": "testis",
    "errata": "erratum",
    "addenda": "addendum",
    "wharves": "wharf",
    "lice": "louse",
    # Invariant -ies nouns: singular and plural are the same
    # word, so the ies rule's "sery"/"specy" stems are wrong
    # forms no singular produces.
    "series": "series",
    "species": "species",
    # -en participles and short -ied verbs the verb rules
    # below cannot derive ("written" keeps no -ing/-ed to
    # strip, "died" would lose its vowel to the -ied rule),
    # the -ance/-ange/-ice verbs whose stripped stem ends in
    # a consonant cluster the e-restoration cannot tell from
    # "hang", and non-verbs ending in -ing/-ed that the verb
    # rules would otherwise mangle ("during" -> "dur",
    # "hundred" -> "hundr", "something" -> "someth").
    "written": "write",
    "taken": "take",
    "given": "give",
    "spoken": "speak",
    "broken": "break",
    "chosen": "choose",
    "driven": "drive",
    "forgotten": "forget",
    "eaten": "eat",
    "seen": "see",
    "died": "die",
    "tied": "tie",
    "dancing": "dance",
    "danced": "dance",
    "changing": "change",
    "changed": "change",
    "noticing": "notice",
    "noticed": "notice",
    "during": "during",
    "morning": "morning",
    "evening": "evening",
    "something": "something",
    "nothing": "nothing",
    "everything": "everything",
    "anything": "anything",
    "hundred": "hundred",
    "sacred": "sacred",
    "naked": "naked",
    "wicked": "wicked",
    "hatred": "hatred",
}


def word_tokens(text: str) -> list[str]:
    """Lowercased word tokens for the text modules.

    A token is a run of letters, digits and combining marks,
    so a Devanagari word ("बिल्ली", whose vowel signs are
    combining marks) and an accented word ("café") each stay
    one token. The ASCII-only pattern this replaces split
    "बिल्ली" into single letters and "café" into "caf", so a
    Hindi question matched no Hindi sentence and Hindi
    retrieval scored every passage zero. Punctuation such as
    the Hindi danda "।" ends a token instead of gluing to
    the word before it.
    """
    tokens: list[str] = []
    current: list[str] = []
    for char in text.lower():
        if char.isalnum() or unicodedata.category(char).startswith("M"):
            current.append(char)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tokens


def _verb_stem(word: str) -> str | None:
    """Stem of a regular -ing/-ed verb form, or None.

    A question in the base form ("what did she study?", "what
    did he stop?") must land on the same stem as a document
    sentence in a derived form ("she studied mathematics",
    "he stopped the car"), or document Q&A wrongly reports
    that the document holds no answer and retrieval scores
    the relevant passage zero. The strip is guarded so short
    non-verbs keep their form: the remaining stem needs a
    vowel and at least two letters ("king", "sing", "red"
    and "bed" are not verb forms), and an -ed stem ending in
    a vowel is left alone ("seed", "need"). A doubled final
    consonant undoubles ("running" -> "run", "stopped" ->
    "stop") except l/s/z, which belong to the base ("fall",
    "pass", "buzz"); otherwise a consonant-vowel-consonant
    stem regains its silent e ("making" -> "make", "hoped"
    -> "hope", "using" -> "use").
    """
    if word.endswith("ied") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("eed") and len(word) > 4:
        return word[:-1]
    stem: str | None = None
    if word.endswith("ing"):
        stem = word[:-3]
    elif word.endswith("ed"):
        stem = word[:-2]
        if stem and stem[-1] in "aeiou":
            return None
    if not stem or len(stem) < 2 or not any(c in "aeiouy" for c in stem):
        return None
    if len(stem) >= 2 and stem[-1] == stem[-2] and stem[-1] not in "aeioulsz":
        return stem[:-1]
    if (
        stem[-1] not in "aeiouywx"
        and stem[-2] in "aeiou"
        and (len(stem) == 2 or stem[-3] not in "aeiou")
    ):
        return stem + "e"
    return stem


def stem_word(word: str) -> str:
    """Light plural stemmer shared by the text modules.

    Singular and plural forms must land on the same stem, or a
    question about a "class" never matches a document sentence
    about "classes". The rules only strip a plural ending: "ies"
    becomes "y" ("cities" -> "city"), "oes" drops its "es" in
    longer words ("heroes" -> "hero", while short "shoes" keeps
    its "e" via the final-s rule), "es" drops after ss/x/z/ch/sh
    ("classes" -> "class", "boxes" -> "box"; a single-s stem
    keeps its "e" via the final-s rule instead, so "cases" ->
    "case" and "responses" -> "response", while true
    s-singular plurals such as "buses" are in the irregular
    map), and a final "s"
    drops unless the word ends in "ss", "us" or "is" ("class",
    "status" and "analysis" are not plurals). Regular verb
    forms stem to their base ("studied" -> "study",
    "running" -> "run") for the same reason. A short irregular
    map covers the plurals and non-plurals the regular rules
    would split or mangle; anything else is returned unchanged.
    """
    irregular = _IRREGULAR_STEMS.get(word)
    if irregular is not None:
        return irregular
    verb = _verb_stem(word)
    if verb is not None:
        return verb
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 5 and word.endswith("oes"):
        return word[:-2]
    if len(word) > 4 and word.endswith("es"):
        stem = word[:-2]
        if stem.endswith(("ss", "x", "z", "ch", "sh")):
            return stem
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def _is_abbreviation_dot(text: str, index: int) -> bool:
    """True when the dot at ``index`` belongs to an abbreviation/initial."""
    start = index - 1
    while start >= 0 and (text[start].isalpha() or text[start] == "."):
        start -= 1
    word = text[start + 1 : index]
    lowered = word.lower()
    if lowered in _ABBREVIATIONS:
        return True
    # A single uppercase letter before the dot is a name initial;
    # check it before the numeric rule so "P. Smith" stays whole.
    if len(word) == 1 and word.isupper():
        return True
    if lowered in _NUMERIC_ABBREVIATIONS:
        # "No. 5" abbreviates; "he said no. Then" ends a sentence.
        following = text[index + 1 :].lstrip()
        return bool(following) and following[0].isdigit()
    # A dotted acronym of single letters ("U.S.A.") never ends a
    # sentence mid-text, matching the listed "u.s"/"u.k" forms.
    if "." in word and all(len(part) == 1 for part in word.split(".")):
        return True
    return False


def split_sentences(text: str) -> list[str]:
    """Split ``text`` into sentences, keeping a trailing fragment.

    A dot between two digits is a decimal point, not a sentence
    boundary, and abbreviation dots ("Dr.", "e.g.") do not end a
    sentence either. The Hindi danda ("।", and its double form
    "॥") ends a sentence like a full stop: without it a Hindi
    text read as one sentence, so summarization returned it
    whole and document Q&A quoted all of it as one answer.
    Text after the last terminator is kept as its
    own sentence instead of being silently dropped, so callers that
    quote sentences (document Q&A) can still find an answer in a
    final line that has no closing punctuation.
    """
    stripped = text.strip()
    if not stripped:
        return []
    sentences: list[str] = []
    start = 0
    index = 0
    while index < len(stripped):
        char = stripped[index]
        if char in ".!?।॥":
            if char == ".":
                # Decimal point: "3.14" stays inside its sentence.
                if (
                    0 < index < len(stripped) - 1
                    and stripped[index - 1].isdigit()
                    and stripped[index + 1].isdigit()
                ):
                    index += 1
                    continue
                # Ellipsis: only the last dot can end the sentence.
                if index + 1 < len(stripped) and stripped[index + 1] == ".":
                    index += 1
                    continue
                # An ellipsis trailing into a lowercase word is a
                # pause inside one sentence ("Wait... what
                # happened?"), not a boundary: splitting there
                # handed summarization and document Q&A a
                # fragment as if it were a sentence. An ellipsis
                # before a capital still ends the sentence.
                if index > 0 and stripped[index - 1] == ".":
                    following = stripped[index + 1 :].lstrip(" \t\r\n\"'“”‘’")
                    if following and following[0].islower():
                        index += 1
                        continue
                if _is_abbreviation_dot(stripped, index):
                    index += 1
                    continue
            end = index + 1
            while end < len(stripped) and stripped[end] in _CLOSERS:
                end += 1
            if end == len(stripped) or stripped[end].isspace():
                sentence = stripped[start:end].strip()
                if sentence:
                    sentences.append(sentence)
                start = end
                index = end
                continue
        index += 1
    tail = stripped[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def summarize(text: str, max_sentences: int = 3) -> str:
    """Extractive summary: top-scored sentences in original order."""
    if max_sentences < 1:
        return ""
    # Drop exact duplicate sentences first (keeping the first
    # occurrence): a repeated sentence inflates its own term
    # frequencies, so without this a summary could be the same
    # sentence repeated max_sentences times.
    sentences = list(dict.fromkeys(split_sentences(text)))
    if len(sentences) <= max_sentences:
        return " ".join(sentences)
    # Stem before counting so singular and plural forms of a word
    # ("cat"/"cats") share one frequency instead of splitting it.
    words = [
        stem_word(w) for w in word_tokens(" ".join(sentences)) if w not in _STOP
    ]
    freq = Counter(words)
    scored: list[tuple[float, int]] = []
    for index, sentence in enumerate(sentences):
        terms = [stem_word(w) for w in word_tokens(sentence) if w not in _STOP]
        if not terms:
            continue
        score = sum(freq[w] for w in terms) / len(terms)
        # Mild position bonus: openings and closings carry the thesis.
        if index == 0:
            score *= 1.25
        elif index == len(sentences) - 1:
            score *= 1.1
        scored.append((score, index))
    if not scored:
        # Every sentence is stopwords alone, so nothing can be
        # scored; keep the opening sentences instead of returning
        # an empty summary that silently drops the source.
        return " ".join(sentences[:max_sentences])
    keep = sorted(index for _, index in sorted(scored, reverse=True)[:max_sentences])
    return " ".join(sentences[i] for i in keep)


class SummarizationModule:
    def __init__(self, manifest: ModuleManifest):
        self.manifest = manifest

    def run(self, ctx: ModuleContext, prompt: str) -> str:
        match = _SUMMARIZE_RE.search(prompt)
        if match:
            return summarize(match.group(1))
        if len(split_sentences(prompt)) >= 4:
            return summarize(prompt)
        return ctx.chat(prompt)
