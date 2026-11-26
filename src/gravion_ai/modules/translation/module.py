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

_TO_HINDI_RE = re.compile(
    r"translate\s+(?:to|into|in)\s+hindi\s*:\s*(.+)", re.IGNORECASE | re.DOTALL
)
_TO_ENGLISH_RE = re.compile(
    r"translate\s+(?:to|into|in)\s+english\s*:\s*(.+)", re.IGNORECASE | re.DOTALL
)
_TRANSLATE_RE = re.compile(r"translate\s*:\s*(.+)", re.IGNORECASE | re.DOTALL)


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).strip()


def translate_phrase(text: str, to_hindi: bool) -> str:
    """Phrasebook translation; unknown words are marked, not guessed."""
    table = PHRASES_EN_HI if to_hindi else PHRASES_HI_EN
    key = _normalize(text)
    if key in table:
        return table[key]
    words = key.split()
    out: list[str] = []
    known = 0
    # Longest-phrase-first greedy matching over the word stream.
    i = 0
    while i < len(words):
        matched = False
        for span in range(min(4, len(words) - i), 0, -1):
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
        match = _TRANSLATE_RE.search(prompt)
        if match:
            text = match.group(1)
            key = _normalize(text)
            to_hindi = key in PHRASES_EN_HI or key.split()[:1] and key.split()[0] in PHRASES_EN_HI
            return translate_phrase(text, to_hindi=bool(to_hindi))
        return ctx.chat(prompt)
