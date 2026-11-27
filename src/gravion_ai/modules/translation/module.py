"""TranslationModule — phrasebook translation (Hindi/English/Hinglish).

Honest scope: this module translates with a Gravion-owned phrasebook
and glossary, not open-domain machine translation. Whole-phrase
matches translate exactly; otherwise known words are translated one
by one and unknown words are marked, never guessed. Requests without
text to translate are answered by the resolved model (Gravion
Translate), whose trained weights cover translation guidance.
"""

from __future__ import annotations

import re

from ...registry import ModuleContext, ModuleManifest

# Gravion-owned phrasebook: English phrase -> Hindi (Roman + Devanagari).
PHRASES_EN_HI = {
    "hello": "namaste (नमस्ते)",
    "thank you": "dhanyavaad (धन्यवाद)",
    "thank you very much": "bahut dhanyavaad (बहुत धन्यवाद)",
    "please": "kripya (कृपया)",
    "sorry": "maaf kijiye (माफ़ कीजिए)",
    "yes": "haan (हाँ)",
    "no": "nahi (नहीं)",
    "good morning": "suprabhaat (सुप्रभात)",
    "good evening": "shubh sandhya (शुभ संध्या)",
    "good night": "shubh raatri (शुभ रात्रि)",
    "how are you": "aap kaise hain (आप कैसे हैं)",
    "i am fine": "main theek hoon (मैं ठीक हूँ)",
    "what is your name": "aapka naam kya hai (आपका नाम क्या है)",
    "my name is": "mera naam hai (मेरा नाम है)",
    "where are you from": "aap kahan se hain (आप कहाँ से हैं)",
    "i do not understand": "mujhe samajh nahi aaya (मुझे समझ नहीं आया)",
    "please speak slowly": "kripya dheere boliye (कृपया धीरे बोलिए)",
    "how much does this cost": "ye kitne ka hai (ये कितने का है)",
    "where is the station": "station kahan hai (स्टेशन कहाँ है)",
    "i need help": "mujhe madad chahiye (मुझे मदद चाहिए)",
    "see you tomorrow": "kal milte hain (कल मिलते हैं)",
    "welcome": "swagat hai (स्वागत है)",
    "congratulations": "badhai ho (बधाई हो)",
    "happy birthday": "janamdin mubarak (जन्मदिन मुबारक)",
    "best of luck": "shubhkamnayein (शुभकामनाएँ)",
    "i agree": "main sahmat hoon (मैं सहमत हूँ)",
    "i disagree": "main sahmat nahi hoon (मैं सहमत नहीं हूँ)",
    "please wait": "kripya intezaar kijiye (कृपया इंतज़ार कीजिए)",
    "let us start": "chaliye shuru karte hain (चलिए शुरू करते हैं)",
    "well done": "bahut badhiya (बहुत बढ़िया)",
    "take care": "apna khayal rakhna (अपना ख्याल रखना)",
    "i am hungry": "mujhe bhookh lagi hai (मुझे भूख लगी है)",
    "i am thirsty": "mujhe pyaas lagi hai (मुझे प्यास लगी है)",
    "what time is it": "kitne baje hain (कितने बजे हैं)",
    "today": "aaj (आज)",
    "tomorrow": "kal (कल)",
    "yesterday": "kal (कल)",
    "water": "paani (पानी)",
    "food": "khaana (खाना)",
    "home": "ghar (घर)",
    "work": "kaam (काम)",
    "friend": "dost (दोस्त)",
    "book": "kitaab (किताब)",
    "school": "vidyalaya (विद्यालय)",
    "money": "paisa (पैसा)",
    "time": "samay (समय)",
    "love": "pyaar (प्यार)",
    "peace": "shaanti (शांति)",
    "strength": "taakat (ताकत)",
    "knowledge": "gyaan (ज्ञान)",
    "success": "safalta (सफलता)",
    "hard work": "mehnat (मेहनत)",
    "truth": "sach (सच)",
    "dream": "sapna (सपना)",
    "future": "bhavishya (भविष्य)",
    "team": "team (टीम)",
    "computer": "computer (कंप्यूटर)",
    "security": "suraksha (सुरक्षा)",
    "password": "password (पासवर्ड)",
    "meeting": "baithak (बैठक)",
    "report": "report (रिपोर्ट)",
}

# Roman Hindi / Hinglish phrase -> English.
PHRASES_HI_EN = {
    "namaste": "hello",
    "dhanyavaad": "thank you",
    "shukriya": "thank you",
    "kripya": "please",
    "maaf kijiye": "sorry",
    "haan": "yes",
    "nahi": "no",
    "aap kaise hain": "how are you",
    "main theek hoon": "i am fine",
    "aapka naam kya hai": "what is your name",
    "mera naam": "my name is",
    "mujhe samajh nahi aaya": "i do not understand",
    "ye kitne ka hai": "how much does this cost",
    "mujhe madad chahiye": "i need help",
    "kal milte hain": "see you tomorrow",
    "swagat hai": "welcome",
    "badhai ho": "congratulations",
    "janamdin mubarak": "happy birthday",
    "main sahmat hoon": "i agree",
    "chaliye shuru karte hain": "let us start",
    "bahut badhiya": "well done",
    "apna khayal rakhna": "take care",
    "mujhe bhookh lagi hai": "i am hungry",
    "mujhe pyaas lagi hai": "i am thirsty",
    "kitne baje hain": "what time is it",
    "aaj": "today",
    "paani": "water",
    "khaana": "food",
    "ghar": "home",
    "kaam": "work",
    "dost": "friend",
    "kitaab": "book",
    "paisa": "money",
    "samay": "time",
    "pyaar": "love",
    "shaanti": "peace",
    "taakat": "strength",
    "gyaan": "knowledge",
    "safalta": "success",
    "mehnat": "hard work",
    "sach": "truth",
    "sapna": "dream",
    "bhavishya": "future",
    "suraksha": "security",
    "baithak": "meeting",
    "theek hai": "alright",
    "koi baat nahi": "no problem",
    "samajh gaya": "understood",
    "phir milenge": "see you again",
    "dhyaan se": "carefully",
    "jaldi karo": "hurry up",
}


def _devanagari_entries() -> dict[str, str]:
    """Devanagari Hindi -> English, derived from the EN->HI book.

    Every EN->HI value carries its Devanagari form in
    parentheses ("dhanyavaad (धन्यवाद)"), so the reverse
    Devanagari entries come from the same single source
    instead of a second hand-kept list. "कल" covers both
    "tomorrow" and "yesterday"; the first entry wins, the
    same ambiguity the Roman side avoids by having no bare
    "kal" key.
    """
    entries: dict[str, str] = {}
    for english, hindi in PHRASES_EN_HI.items():
        match = re.search(r"\(([^)]+)\)\s*$", hindi)
        if match:
            entries.setdefault(match.group(1), english)
    return entries


PHRASES_HI_EN.update(_devanagari_entries())

# The trailing-language forms ("translate thank you into
# hindi") are the shape the all-in-one router already sends
# here on its translate+language signal, so refusing them
# dropped a routed request to the model fallback. The
# colon-less forms are coverage-gated in run(): a guidance
# request ("translate a greeting into hindi") names a
# description, not literal text, and stays with the resolved
# model whose weights cover translation guidance.
_TO_HINDI_RE = re.compile(
    r"translate\s+(?:to|into|in)\s+hindi\s*:\s*(.+)", re.IGNORECASE | re.DOTALL
)
_TO_ENGLISH_RE = re.compile(
    r"translate\s+(?:to|into|in)\s+english\s*:\s*(.+)", re.IGNORECASE | re.DOTALL
)
_TO_HINDI_LEAD_RE = re.compile(
    r"translate\s+(?:to|into|in)\s+hindi\s+(.+)", re.IGNORECASE | re.DOTALL
)
_TO_ENGLISH_LEAD_RE = re.compile(
    r"translate\s+(?:to|into|in)\s+english\s+(.+)", re.IGNORECASE | re.DOTALL
)
_TO_HINDI_TRAILING_RE = re.compile(
    r"translate\s+(.+?)\s+(?:to|into|in)\s+hindi\b", re.IGNORECASE | re.DOTALL
)
_TO_ENGLISH_TRAILING_RE = re.compile(
    r"translate\s+(.+?)\s+(?:to|into|in)\s+english\b", re.IGNORECASE | re.DOTALL
)
# "translate this into hindi: thank you" puts the text after
# the colon; the trailing form alone would capture the
# pronoun "this" as the text to translate.
_TO_HINDI_THIS_RE = re.compile(
    r"translate\s+(?:this|that|it)\s+(?:to|into|in)\s+hindi\s*:\s*(.+)",
    re.IGNORECASE | re.DOTALL,
)
_TO_ENGLISH_THIS_RE = re.compile(
    r"translate\s+(?:this|that|it)\s+(?:to|into|in)\s+english\s*:\s*(.+)",
    re.IGNORECASE | re.DOTALL,
)
_TRANSLATE_RE = re.compile(r"translate\s*:\s*(.+)", re.IGNORECASE | re.DOTALL)
# "How do you say thank you in hindi" names the text between
# "say" and the language; without this shape the request fell
# through to the model even when the phrasebook covers the
# text exactly. Coverage-gated in run() like the other
# colon-less forms, so a description ("how do you say a
# greeting in hindi") stays with the resolved model.
_SAY_IN_HINDI_RE = re.compile(
    r"how do you say\s+(.+?)\s+in\s+hindi\b", re.IGNORECASE | re.DOTALL
)
_SAY_IN_ENGLISH_RE = re.compile(
    r"how do you say\s+(.+?)\s+in\s+english\b", re.IGNORECASE | re.DOTALL
)


# The phrasebook keys are expanded forms ("i do not understand",
# "let us start"), so contractions are expanded before matching:
# without this, "i don't understand" normalised to "i don t
# understand" and missed a phrase the book does cover.
_CONTRACTIONS = {
    "i'm": "i am",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "can't": "can not",
    "won't": "will not",
    "let's": "let us",
    "it's": "it is",
    "that's": "that is",
    "what's": "what is",
    "isn't": "is not",
    "aren't": "are not",
    "i'll": "i will",
    "we're": "we are",
    "you're": "you are",
}
_CONTRACTION_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _CONTRACTIONS) + r")\b"
)


def _normalize(text: str) -> str:
    lowered = text.lower().replace("’", "'")
    expanded = _CONTRACTION_RE.sub(lambda m: _CONTRACTIONS[m.group(1)], lowered)
    # Keep the Devanagari block: stripping it erased Hindi
    # input the phrasebook covers ("धन्यवाद" normalised to
    # nothing and was refused as outside the book).
    return re.sub(r"[^a-z0-9\u0900-\u097f ]+", " ", expanded).strip()


def _greedy_translate(words: list[str], table: dict[str, str]) -> tuple[list[str], int]:
    """Longest-phrase-first greedy matching over the word stream.

    Returns the translated parts (unknown words marked) and how
    many source words the table covered. The span cap comes
    from the phrasebook itself: a fixed cap below the longest
    entry ("how much does this cost", five words) made that
    phrase unmatchable inside a longer sentence.
    """
    out: list[str] = []
    known = 0
    max_span = max(len(phrase.split()) for phrase in table)
    i = 0
    while i < len(words):
        matched = False
        for span in range(min(max_span, len(words) - i), 0, -1):
            candidate = " ".join(words[i : i + span])
            if candidate in table:
                out.append(table[candidate])
                known += span
                i += span
                matched = True
                break
        if not matched:
            out.append(f"[{words[i]}]")
            i += 1
    return out, known


def _detect_to_hindi(text: str) -> bool:
    """Pick the translation direction by phrasebook coverage.

    A bare "translate: ..." request names no direction, so the
    direction is whichever table covers more of the text. The
    earlier first-word check sent "how are you today" to the
    Hindi table (its first word is no English key on its own)
    and refused a phrase the book does cover.
    """
    key = _normalize(text)
    if key in PHRASES_EN_HI:
        return True
    if key in PHRASES_HI_EN:
        return False
    words = key.split()
    _, known_en = _greedy_translate(words, PHRASES_EN_HI)
    _, known_hi = _greedy_translate(words, PHRASES_HI_EN)
    if known_en != known_hi:
        return known_en > known_hi
    if known_en:
        # Equal coverage: follow the first word either table
        # knows on its own, else keep the Hindi direction.
        for word in words:
            if word in PHRASES_EN_HI:
                return True
            if word in PHRASES_HI_EN:
                return False
        return True
    return False


def _phrasebook_covers(text: str, to_hindi: bool) -> bool:
    """True when the phrasebook covers any of ``text``."""
    table = PHRASES_EN_HI if to_hindi else PHRASES_HI_EN
    key = _normalize(text)
    if key in table:
        return True
    _, known = _greedy_translate(key.split(), table)
    return known > 0


def translate_phrase(text: str, to_hindi: bool) -> str:
    """Phrasebook translation; unknown words are marked, not guessed."""
    table = PHRASES_EN_HI if to_hindi else PHRASES_HI_EN
    key = _normalize(text)
    if key in table:
        return table[key]
    out, known = _greedy_translate(key.split(), table)
    if known == 0:
        return (
            "That text is outside my Gravion phrasebook, so I will not "
            "guess a translation. I cover common Hindi, English and "
            "Hinglish phrases and everyday words."
        )
    return " ".join(out)


class TranslationModule:
    def __init__(self, manifest: ModuleManifest):
        self.manifest = manifest

    def run(self, ctx: ModuleContext, prompt: str) -> str:
        match = _TO_HINDI_RE.search(prompt)
        if match:
            return translate_phrase(match.group(1), to_hindi=True)
        match = _TO_ENGLISH_RE.search(prompt)
        if match:
            return translate_phrase(match.group(1), to_hindi=False)
        match = _TO_HINDI_THIS_RE.search(prompt)
        if match:
            return translate_phrase(match.group(1), to_hindi=True)
        match = _TO_ENGLISH_THIS_RE.search(prompt)
        if match:
            return translate_phrase(match.group(1), to_hindi=False)
        # Colon-less forms only intercept text the phrasebook
        # covers; a description ("a greeting") is a guidance
        # request for the resolved model, not literal text.
        match = _TO_HINDI_LEAD_RE.search(prompt)
        if match and _phrasebook_covers(match.group(1), to_hindi=True):
            return translate_phrase(match.group(1), to_hindi=True)
        match = _TO_ENGLISH_LEAD_RE.search(prompt)
        if match and _phrasebook_covers(match.group(1), to_hindi=False):
            return translate_phrase(match.group(1), to_hindi=False)
        match = _TO_HINDI_TRAILING_RE.search(prompt)
        if (
            match
            and _normalize(match.group(1)) not in {"this", "that", "it"}
            and _phrasebook_covers(match.group(1), to_hindi=True)
        ):
            return translate_phrase(match.group(1), to_hindi=True)
        match = _TO_ENGLISH_TRAILING_RE.search(prompt)
        if (
            match
            and _normalize(match.group(1)) not in {"this", "that", "it"}
            and _phrasebook_covers(match.group(1), to_hindi=False)
        ):
            return translate_phrase(match.group(1), to_hindi=False)
        match = _SAY_IN_HINDI_RE.search(prompt)
        if match and _phrasebook_covers(match.group(1), to_hindi=True):
            return translate_phrase(match.group(1), to_hindi=True)
        match = _SAY_IN_ENGLISH_RE.search(prompt)
        if match and _phrasebook_covers(match.group(1), to_hindi=False):
            return translate_phrase(match.group(1), to_hindi=False)
        match = _TRANSLATE_RE.search(prompt)
        if match:
            text = match.group(1)
            return translate_phrase(text, to_hindi=_detect_to_hindi(text))
        return ctx.chat(prompt)
