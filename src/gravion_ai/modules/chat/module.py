"""ChatModule — the one module implemented end-to-end in Piece 1.

It is deliberately small: it exists to prove the whole path — config →
system context (name + company policy) → model catalog resolution →
provider → entitlement gate → signed module verification — works, so
every later module piece follows the same shape.

One deterministic addition sits in front of that path, mirroring the
support module's escalation short-circuit: greetings, identity
questions ("who are you?"), capability questions ("what can you
do?") and the other conversational lines a demo opens and closes
with ("how are you?", "who built you?", "thank you", "goodbye") are
answered by the module itself instead of the model. The bundled
local model is trained on Gravion documentation and has no confident
match for these lines, so they used to fall through to the echo
fallback — the first thing a new user (or a demo audience) says came
back as "[gravion-core] You said: ...". The short-circuit is narrow:
only a short greeting, an explicit identity, capability or builder
question, or a whole-message thanks or farewell matches, negated
forms ("what can you not do?") stay with the model, and everything
else follows the reference path unchanged.
"""

from __future__ import annotations

import re

from ...registry import ModuleContext, ModuleManifest

GREETING_RESPONSE = (
    "Hello, I am Gravion. Tell me what you would like to work on — "
    "a question, code, a website, a security review or an agent plan — "
    "and I will help step by step."
)

IDENTITY_RESPONSE = (
    "I am Gravion, a modular AI platform built by Priyanshu Kumar "
    "Thakur. One module system runs chat, support, coding, website "
    "building, image, music, video, cyber security and agentic "
    "tools, and the built-in Gravion models run fully offline."
)

CAPABILITIES_RESPONSE = (
    "I can chat and explain ideas, write and review code, build "
    "websites, create images, music and video briefs, analyse and "
    "triage security alerts, draft agent plans, translate and "
    "summarize text, answer questions from documents, and help with "
    "customer support. Ask for any of these and I will route it to "
    "the right Gravion module."
)

WELLBEING_RESPONSE = (
    "I am doing well and ready to help. How are you? Tell me what "
    "you would like to work on — a question, code, a website, a "
    "security review or an agent plan."
)

BUILDER_RESPONSE = (
    "Gravion was built by Priyanshu Kumar Thakur. It is a modular "
    "AI platform: one module system runs chat, support, coding, "
    "website building, image, music, video, cyber security and "
    "agentic tools, with built-in models that run fully offline."
)

ABOUT_RESPONSE = (
    "Gravion is a modular AI platform built by Priyanshu Kumar "
    "Thakur. Startups can build on its module system — chat, "
    "support, coding, website building, image, music, video, cyber "
    "security and agentic tools — instead of building each AI type "
    "from scratch, and the built-in Gravion models run fully "
    "offline with no API key."
)

AI_IDENTITY_RESPONSE = (
    "Yes — I am Gravion, an AI platform built by Priyanshu Kumar "
    "Thakur, not a human. I answer through Gravion modules for "
    "chat, code, websites, security and more, running fully "
    "offline."
)

LANGUAGE_RESPONSE = (
    "I can work in English and Roman Hindi here, and the "
    "translation module handles Hindi phrases. Ask in either and "
    "I will follow your language."
)

THANKS_RESPONSE = (
    "You are welcome. Ask for anything else — a question, code, a "
    "website, a security review or an agent plan — and I will help."
)

FAREWELL_RESPONSE = (
    "Goodbye, and thank you for trying Gravion. Come back any "
    "time — I will be ready to help."
)

MEET_RESPONSE = (
    "Nice to meet you too. Tell me what you would like to work "
    "on — a question, code, a website, a security review or an "
    "agent plan — and I will help step by step."
)

OFFLINE_RESPONSE = (
    "Yes — the built-in Gravion models run fully offline with "
    "no network and no API key, so chat, code, cyber and the "
    "other modules keep working on your own machine."
)

START_RESPONSE = (
    "Install Gravion, run gravion init to create your "
    "deployment, then gravion status and gravion doctor to "
    "check readiness — after that, gravion chat opens the "
    "conversation and every module is ready to use."
)

FREE_RESPONSE = (
    "Development and evaluation are free, with no agreement "
    "needed. Production use runs under a commercial agreement, "
    "and the tiers set how trained each module is when it "
    "reaches you."
)

STARTUP_RESPONSE = (
    "Gravion gives a startup one module system for chat, "
    "support, coding, website building, image, music, video, "
    "cyber security and agentic tools, with built-in models "
    "that run offline — so a small team ships AI features "
    "without building each AI type from scratch."
)

AUDIENCE_RESPONSE = (
    "Gravion is for startups and companies building AI "
    "features: one module system covers chat, support, "
    "coding, website building, cyber security and agentic "
    "tools, with built-in models that run fully offline, "
    "so a small team ships without building each AI type "
    "from scratch."
)

LAPTOP_RESPONSE = (
    "Yes — Gravion runs on your own machine: the built-in "
    "models work fully offline with no network and no API "
    "key, so a normal laptop is enough for the local modules."
)

DIFFERENT_RESPONSE = (
    "Gravion is one modular platform instead of separate "
    "tools: chat, support, coding, website building, image, "
    "music, video, cyber security and agentic modules share "
    "one system, and the built-in models run fully offline "
    "with no API key."
)

PRICE_RESPONSE = (
    "Development and evaluation are free, with no agreement "
    "needed. Production pricing runs under a commercial "
    "agreement by tier, and there is no per-token or API-key "
    "cost because the built-in models run offline on your "
    "own machine."
)

COMPANY_RESPONSE = (
    "Yes — a company can build on Gravion's module system "
    "for chat, support, coding, website building, cyber "
    "security and agentic tools instead of building each AI "
    "type from scratch. Development and evaluation are free; "
    "production use runs under a commercial agreement."
)

INCIDENT_RESPONSE = (
    "First contain it: preserve logs and evidence, check "
    "whether any login succeeded, block or rate-limit the "
    "source, force password resets for touched accounts, "
    "and review related activity before bringing services "
    "back. The cyber module can triage the alert in detail."
)

TEAM_RESPONSE = (
    "Gravion supports a team rather than replacing it: the "
    "support module answers routine questions from your own "
    "material and escalates to a person on request, so the "
    "team spends its time on the cases that need judgement."
)

MODEL_RESPONSE = (
    "Start with Gravion Core for chat and support. Use Gravion "
    "Omni for every module in one deployment, or a specialist "
    "such as Gravion Code, Gravion Shield, Gravion Agent or "
    "Gravion Builder when you need one capability."
)

CODE_RESPONSE = (
    "Yes — Gravion writes and reviews code through its coder "
    "module, and the same platform covers chat, support, "
    "websites, cyber security and agentic tools."
)

WEBSITE_RESPONSE = (
    "Yes — Gravion builds websites through its website "
    "builder module: describe the site you want and it plans "
    "the pages, sections and contact actions, running fully "
    "offline."
)

TIERS_RESPONSE = (
    "Gravion has four tiers over one module system: Tier 1 "
    "Complete (fully trained, ready to use), Tier 2 Basic, "
    "Tier 3 Intermediate and Tier 4 Advanced — the modules "
    "stay the same and the training level rises by tier."
)

PRIVACY_RESPONSE = (
    "Yes — the built-in Gravion models run fully offline on "
    "your own machine with no network and no API key, so "
    "your prompts and data stay with you instead of being "
    "sent to an outside service."
)

MODELS_COUNT_RESPONSE = (
    "Gravion bundles 17 models: Gravion Core and Gravion "
    "Omni, plus specialists such as Gravion Code, Gravion "
    "Shield, Gravion Agent, Gravion Builder, Gravion Chat "
    "and Gravion Support, each with its own bundled weights."
)

OMNI_RESPONSE = (
    "Gravion Omni is the all-capabilities Gravion model: "
    "one deployment serves chat, support, coding, website "
    "building, image, music, video, cyber security, "
    "agentic tools and the other modules, running fully "
    "offline with no API key."
)

HARDWARE_RESPONSE = (
    "No — Gravion needs no GPU. The built-in models run "
    "fully offline on a normal laptop with no network "
    "and no API key."
)

_GREETING = re.compile(
    r"^(?:hello|hi|hey|namaste|greetings|good (?:morning|afternoon|evening))\b"
)
_IDENTITY = re.compile(
    r"\b(?:who are you|what is your name|what s your name|"
    r"introduce yourself|tell me about yourself|"
    r"your name|"
    # Roman Hindi identity asks (28 Nov 2026 cycle): a demo
    # audience in India opens with "tum kaun ho" as plainly as
    # "who are you", and it used to echo back unanswered.
    r"tum kaun ho|aap kaun hain|tumhara naam kya hai|"
    r"apna naam batao|tum kaun|aap kaun)\b"
)
_CAPABILITIES = re.compile(
    r"\b(?:what can you do|what do you do|your capabilities|"
    r"what are your capabilities|what are you able to do|"
    r"how can you help|what modules do you have|"
    # Roman Hindi capability asks, same reasoning as above:
    # "tum kya kar sakte ho" is "what can you do".
    r"tum kya kar sakte ho|aap kya kar sakte hain|"
    r"kya kar sakte ho|tum kya kya kar sakte ho|"
    # The same ask about Gravion in the third person (28 Nov
    # 2026 cycle, 06:16 run): "Gravion kya kar sakta hai?"
    # echoed while the tum/aap forms answered.
    r"gravion kya kar sakta hai|gravion kya kar sakti hai|"
    # The module-count ask in Roman Hindi (28 Nov 2026 cycle,
    # 08:46 run): "Gravion kitne module hai?" echoed while
    # "how many modules does Gravion have?" answered.
    r"gravion kitne module|kitne module hai|"
    r"kitne modules hai|"
    # Plural + deta-hai form (28 Nov 2026 cycle, 10:16
    # run): "Gravion kitne modules deta hai?" echoed while
    # the singular module-count shapes answered.
    r"gravion kitne modules|kitne modules deta|"
    r"kitne module deta)\b"
)
_NEGATION = re.compile(
    r"\b(?:not|never|no|cannot|can t|don t|nahi|nahin)\b"
)
# A bare help request ("can you help me today?") is the same
# opening as a capability question, but it carries no task, so
# it used to echo while "how can you help" answered. The match
# is anchored to the whole message: "can you help me write
# code" names a task and stays with the model.
_BARE_HELP = re.compile(r"^(?:can you help me|help me|please help me)(?: today| please)?$")
# Conversational lines a demo opens and closes with (28 Nov
# 2026 cycle, 03:16 run): "how are you?", "who built you?",
# "tell me about Gravion", "are you an AI?", "thank you" and
# "goodbye" all echoed under the default model while the
# greeting and identity lines answered. Each pattern stays
# narrow: builder and about asks name Gravion or its maker,
# wellbeing asks are short, and thanks and farewell match the
# whole message so "write a thank you note" stays a task.
_WELLBEING = re.compile(
    r"\b(?:how are you|how are you doing|how is it going|"
    r"kaise ho|aap kaise hain|kya haal hai|haal kaisa hai)\b"
)
_BUILDER = re.compile(
    r"\b(?:who built you|who made you|who created you|"
    r"who built gravion|who made gravion|who created gravion|"
    r"who is your creator|who is your maker|"
    # The founder shape of the same ask (28 Nov 2026 cycle,
    # 05:16 run): "who is the founder of Gravion?" echoed
    # while "who built you?" answered.
    r"who is the founder of gravion|who founded gravion|"
    r"founder of gravion|who is your founder|"
    r"tumhe kisne banaya|gravion kisne banaya|"
    # Behind/founder forms (28 Nov 2026 cycle, 13:46
    # run): "who is behind Gravion?" and "Gravion ka
    # founder kaun hai?" echoed while "who built you?"
    # answered.
    r"who is behind gravion|who is behind you|"
    r"gravion ka founder|founder kaun hai)\b"
)
_ABOUT = re.compile(
    r"\b(?:what is gravion|tell me about gravion|"
    r"about gravion|gravion kya hai|gravion ke baare me batao)\b"
)
_AI_IDENTITY = re.compile(
    r"\b(?:are you an ai|are you a robot|are you human|"
    r"are you a human|are you real|kya tum ai ho|"
    r"tum ai ho kya)\b"
)
_LANGUAGE = re.compile(
    r"\b(?:can you speak hindi|do you speak hindi|"
    r"can you speak hinglish|hindi aati hai|"
    r"hindi bol sakte ho|"
    # The language ask in its list form ("what languages do
    # you support?") is the same opening as the Hindi ask
    # above; with only the Hindi forms listed, it echoed.
    r"what languages do you support|"
    r"which languages do you support|"
    r"what languages can you speak|"
    # The same ask about Gravion in the third person (28 Nov
    # 2026 cycle, 07:16 run): "does Gravion work in Hindi?"
    # echoed while the you-forms answered.
    r"does gravion work in hindi|does gravion speak hindi|"
    r"gravion work in hindi|gravion speak hindi|"
    # The support-Hindi and Roman Hindi forms (28 Nov 2026
    # cycle, 09:46 run): "does Gravion support Hindi?" and
    # "kya Gravion Hindi me kaam karta hai?" echoed while the
    # work/speak forms answered.
    r"does gravion support hindi|gravion support hindi|"
    r"kya gravion hindi|gravion hindi me|"
    r"hindi me kaam karta hai|hindi mein kaam karta hai|"
    # The third-person list form (28 Nov 2026 cycle, 12:31
    # run): "what languages does Gravion support?" echoed
    # while "what languages do you support?" answered.
    r"what languages does gravion support|"
    r"which languages does gravion support|"
    r"languages does gravion support|"
    r"what languages does gravion|"
    r"gravion support languages|"
    # The second-person support form (28 Nov 2026 cycle,
    # 14:46 run): "do you support Hindi?" echoed while
    # "does Gravion support Hindi?" answered.
    r"do you support hindi|can you support hindi|"
    r"you support hindi)\b"
)
# Demo questions about Gravion itself (28 Nov 2026 cycle,
# 04:16 run): "nice to meet you", "do you work offline?",
# "is Gravion free?" and "how do I get started?" all echoed
# while the neighbouring identity and capability lines
# answered. Each pattern stays narrow: the meet reply is a
# short whole-message line, the offline and free asks name
# their subject, and the start ask names getting started.
_MEET = re.compile(
    r"^(?:nice to meet you|pleased to meet you|"
    r"good to meet you|glad to meet you)$"
)
_OFFLINE = re.compile(
    r"\b(?:do you work offline|can you work offline|"
    r"do you need internet|do you need an internet connection|"
    r"work without internet|offline kaam karte ho|"
    # The same ask naming Gravion (28 Nov 2026 cycle, 06:16
    # run): "does Gravion need internet?" echoed while "do
    # you need internet?" answered.
    r"does gravion need internet|gravion need internet|"
    r"does gravion work offline|gravion work offline|"
    # The full "internet connection" form naming Gravion
    # (28 Nov 2026 cycle, 08:46 run): "does Gravion need an
    # internet connection?" echoed while the short forms
    # answered.
    r"does gravion need an internet connection|"
    r"gravion need an internet connection|"
    # The Roman Hindi offline ask (28 Nov 2026 cycle,
    # 11:01 run): "Gravion offline chalta hai kya?" echoed
    # while the English offline forms answered.
    r"gravion offline chalta hai|offline chalta hai|"
    r"bina internet chalta hai|offline kaam karta hai|"
    # The run-without-internet form (28 Nov 2026 cycle,
    # 13:16 run): "can Gravion run without internet?"
    # echoed while the work-offline forms answered; it
    # asks the same offline question.
    r"run without internet|gravion run without internet|"
    r"can gravion run without)\b"
)
# The API-key form of the same ask (28 Nov 2026 cycle,
# 13:16 run): "do I need an API key to use Gravion?"
# echoed beside the offline forms. It is its own
# pattern because the natural question runs to nine
# words, one past the offline guard above; the reply
# is the same offline answer.
_API_KEY = re.compile(
    r"\b(?:need an api key|api key to use gravion|"
    r"does gravion need an api key|gravion api key|"
    r"do i need an api key)\b"
)
_START = re.compile(
    r"\b(?:how do i get started|how to get started|"
    r"get started with gravion|how do i start using gravion|"
    r"how do i start with gravion|"
    # The install form of the same ask (28 Nov 2026 cycle,
    # 09:46 run): "how do I install Gravion?" echoed while
    # the get-started forms answered.
    r"how do i install gravion|how to install gravion|"
    r"install gravion|"
    # Roman Hindi use form (28 Nov 2026 cycle, 13:46
    # run): "Gravion ko kaise use karun?" echoed while
    # the get-started forms answered.
    r"kaise use karun|gravion ko kaise use|"
    r"gravion kaise use|"
    # The Roman Hindi demo form (28 Nov 2026 cycle, 12:01
    # run): "Gravion ka demo kaise milega?" echoed while
    # the get-started forms answered; the demo runs from
    # the same install and init steps.
    r"gravion ka demo|demo kaise milega|"
    r"gravion demo kaise|"
    # The see-a-demo form (28 Nov 2026 cycle, 12:31 run):
    # "can I see a demo of Gravion?" echoed while the
    # Roman Hindi demo form answered; the demo runs from
    # the same install and init steps.
    r"can i see a demo|see a demo|show me a demo|"
    r"demo of gravion|gravion demo)\b"
)
_FREE = re.compile(
    r"\b(?:is gravion free|is it free|gravion free hai kya|"
    # The try-free form (28 Nov 2026 cycle, 09:46 run):
    # "can I try Gravion for free?" echoed while the
    # is-it-free forms answered.
    r"try gravion for free|can i try gravion|"
    r"try gravion free|"
    # Roman Hindi free form (28 Nov 2026 cycle, 13:46
    # run): "kya Gravion free hai?" echoed while the
    # "Gravion free hai kya" order answered.
    r"kya gravion free|gravion free hai|"
    # The second-person free-to-use form (28 Nov 2026
    # cycle, 15:46 run): "are you free to use?" echoed
    # while the is-Gravion-free forms answered.
    r"are you free to use|are you free|free to use)\b"
)
# Investor questions about Gravion itself (28 Nov 2026 cycle,
# 05:16 run): "what can Gravion do for a startup?", "can
# Gravion run on my laptop?" and "how is Gravion different?"
# all echoed while the neighbouring identity and capability
# lines answered. Each pattern stays narrow: the startup ask
# names a startup, the laptop ask names running locally, and
# the difference ask names Gravion.
_STARTUP = re.compile(
    r"\b(?:what can gravion do for (?:a |my )?startup|"
    r"how can gravion help (?:a |my )?startup|"
    r"gravion for startups|why should (?:a |my )?startup use gravion|"
    # The direct and Roman Hindi startup asks (28 Nov 2026
    # cycle, 11:01 run): "can Gravion help my startup?" and
    # "Gravion startup ke liye hai kya?" echoed while the
    # what-can-it-do forms answered.
    r"can gravion help (?:a |my )?startup|"
    r"gravion help (?:a |my )?startup|"
    r"gravion startup ke liye|startup ke liye gravion|"
    r"startup ke liye hai kya|"
    # The mere-startup form (28 Nov 2026 cycle, 13:16
    # run): "Gravion mere startup ke liye kaam karega?"
    # echoed while the startup-ke-liye forms answered.
    r"mere startup ke liye|gravion mere startup|"
    r"startup ke liye kaam karega|"
    # The good-for and apne-startup forms (28 Nov 2026
    # cycle, 16:46 run): "is Gravion good for startups?"
    # and "kya main Gravion apne startup ke liye use kar
    # sakta hoon?" echoed while the startup-ke-liye forms
    # answered.
    r"is gravion good for startups|gravion good for startups|"
    r"good for startups|apne startup ke liye|"
    r"startup ke liye use kar sakta)\b"
)
# The audience ask (28 Nov 2026 cycle, 12:31 run): an
# investor asks "who is Gravion for?" and "Gravion kis ke
# liye hai?" in the same breath as the startup ask, and
# both echoed. The pattern names Gravion's audience
# directly, in English and Roman Hindi.
_AUDIENCE = re.compile(
    r"\b(?:who is gravion for|who can use gravion|"
    r"who is gravion made for|gravion kis ke liye|"
    r"kis ke liye hai gravion|gravion kiske liye|"
    # The team-use form (28 Nov 2026 cycle, 16:46 run):
    # "can my team use Gravion?" echoed while the
    # who-is-it-for forms answered.
    r"can my team use gravion|my team use gravion|"
    r"team use gravion|can our team use gravion|"
    # The built-for form (28 Nov 2026 cycle, 17:46 run):
    # "who is Gravion built for?" was answered with a
    # licensing fragment while the who-is-it-for forms
    # answered.
    r"who is gravion built for|gravion built for|"
    r"who was gravion built for)\b"
)
_LAPTOP = re.compile(
    r"\b(?:can gravion run on my laptop|run on my laptop|"
    r"run on my computer|can i run gravion locally|"
    r"run gravion locally|does gravion run locally|"
    # The indefinite-article form (28 Nov 2026 cycle, 08:46
    # run): "can Gravion run on a laptop?" echoed while the
    # "my laptop" form answered.
    r"can gravion run on a laptop|run on a laptop|"
    r"run on a computer|"
    # The Windows form (28 Nov 2026 cycle, 16:46 run):
    # "does Gravion work on Windows?" echoed while the
    # laptop forms answered.
    r"does gravion work on windows|gravion on windows|"
    r"work on windows|run on windows|gravion windows|"
    # The phone form (28 Nov 2026 cycle, 17:46 run):
    # "can Gravion run on my phone?" echoed while the
    # laptop forms answered; the reply stays honest
    # that a normal laptop is the local path.
    r"can gravion run on my phone|run on my phone|"
    r"gravion on my phone|run on a phone)\b"
)
_DIFFERENT = re.compile(
    r"\b(?:how is gravion different|what makes gravion different|"
    r"why gravion|why choose gravion|"
    # The first-person form (28 Nov 2026 cycle, 11:01
    # run): "why should I use Gravion?" echoed while
    # "why choose Gravion?" answered.
    r"why should i use gravion|why use gravion)\b"
)
# Price and company asks (28 Nov 2026 cycle, 06:16 run): an
# investor asks "what is the price of Gravion?" and "can I
# use Gravion for my company?" in the same breath as the
# startup and free asks, and both echoed. Each pattern stays
# narrow: the price ask names Gravion's price or cost, and
# the company ask names company or business use.
_PRICE = re.compile(
    r"\b(?:price of gravion|how much does gravion cost|"
    r"gravion price|gravion cost|what does gravion cost|"
    r"pricing of gravion|gravion pricing|"
    # The Roman Hindi price ask (28 Nov 2026 cycle, 09:46
    # run): "Gravion ka price kya hai?" echoed while the
    # English price forms answered.
    r"gravion ka price|gravion ki price|gravion ka daam|"
    r"gravion kitne ka hai|gravion ki keemat)\b"
)
_COMPANY = re.compile(
    r"\b(?:use gravion for my company|gravion for my company|"
    r"use gravion in my company|can companies use gravion|"
    r"gravion for companies|use gravion for business|"
    r"gravion for business|"
    # The first-person and Roman Hindi business forms
    # (28 Nov 2026 cycle, 12:01 run): "can I use Gravion
    # for my business?" and "Gravion business ke liye use
    # kar sakta hun kya?" echoed while the bare business
    # forms answered.
    r"use gravion for my business|gravion for my business|"
    r"can i use gravion for my business|"
    r"gravion business ke liye|business ke liye gravion|"
    r"business ke liye use kar|"
    # Small-business form (28 Nov 2026 cycle, 13:46
    # run): "can Gravion help a small business?" echoed
    # while the company forms answered. The pattern
    # names Gravion with the small business, so a task
    # about a small business stays with the model.
    r"gravion[^.?!]*small business|"
    r"small business[^.?!]*gravion)\b"
)
# A live security incident asked in chat (28 Nov 2026 cycle,
# 07:16 run): "someone is trying to break into our server"
# and "our website got hacked" echoed in chat while the
# cyber module answered both. The pattern stays narrow:
# break-in, hacked and under-attack phrasings about the
# asker's own systems, not general security questions.
_INCIDENT = re.compile(
    r"\b(?:trying to break into|break into our|"
    r"got hacked|was hacked|been hacked|"
    r"under attack|someone hacked|hacked our|"
    r"hacked my)\b"
)
# The support-team ask (28 Nov 2026 cycle, 07:16 run): an
# investor asks "can Gravion replace my support team?" in
# the same breath as the company ask, and it echoed. The
# pattern names replacing a support team; the reply is
# honest that Gravion assists the team and escalates.
_TEAM = re.compile(
    r"\b(?:replace (?:my|our|a|the) support team|"
    r"replace support team|"
    r"gravion replace (?:my|our) team)\b"
)
# Model-choice, code-capability and tier asks (28 Nov 2026
# cycle, 09:46 run): an investor asks "which Gravion model
# should I use?", "can Gravion write code?" and "what tiers
# does Gravion have?" in the same breath as the price and
# company asks, and all three echoed. Each pattern stays
# narrow: the model ask names choosing a model, the code
# ask is a short capability question with no task attached,
# and the tier ask names Gravion's tiers.
_MODEL_CHOICE = re.compile(
    r"\b(?:which gravion model|which model should i use|"
    r"what model should i use|which model is best|"
    r"what gravion model should i use)\b"
)
_CODE_CAPABILITY = re.compile(
    r"^(?:can gravion write code|does gravion write code|"
    r"can you write code|can gravion code|"
    r"does gravion code|"
    # Help-with-coding form (28 Nov 2026 cycle, 10:16
    # run): "can you help me with coding?" echoed while
    # "can you write code?" answered. Whole-message only,
    # so a coding task ("help me with coding a parser")
    # still stays with the model.
    r"can you help me with coding|can you help with coding|"
    r"help me with coding|can gravion help with coding|"
    # The for-me form (28 Nov 2026 cycle, 16:46 run):
    # "can Gravion write code for me?" was answered with
    # a Fibonacci completion while "can Gravion write
    # code?" answered. Whole-message only, like the
    # shapes above.
    r"can gravion write code for me|can you write code for me|"
    r"write code for me)$"
)
# The website-capability ask (28 Nov 2026 cycle, 14:46
# run): "Gravion se website ban sakti hai?" echoed while
# the code-capability ask answered. Whole-message only,
# like the code ask, so a website brief ("build a website
# for a bakery") still stays with the model.
_WEBSITE_CAPABILITY = re.compile(
    r"^(?:can gravion build websites|can gravion build a website|"
    r"can gravion make a website|does gravion build websites|"
    r"can you build websites|can you build a website|"
    r"gravion se website ban sakti hai|"
    r"kya gravion website bana sakta hai|"
    r"gravion website bana sakta hai|"
    r"website ban sakti hai kya)$"
)
_TIERS = re.compile(
    r"\b(?:what tiers does gravion have|gravion tiers|"
    r"tiers does gravion|how many tiers does gravion have|"
    r"what are the gravion tiers)\b"
)
# Privacy and model-count asks (28 Nov 2026 cycle, 11:01
# run): an investor asks "is my data private with
# Gravion?" and "how many models does Gravion have?" in
# the same breath as the offline and startup asks, and
# both echoed. Each pattern stays narrow: the privacy
# ask names data privacy or safety with Gravion, and the
# count ask names Gravion's models. The count in the
# reply is checked against the bundled catalog by a
# test, so it cannot drift silently.
_PRIVACY = re.compile(
    r"\b(?:is my data private|data private with gravion|"
    r"is my data safe with gravion|is gravion private|"
    r"gravion data privacy|data privacy in gravion|"
    r"mera data private|data surakshit hai kya|"
    # The safe-to-use form (28 Nov 2026 cycle, 14:46
    # run): "is Gravion safe to use?" echoed while the
    # data-safe forms answered; it asks the same safety
    # question and gets the same offline answer.
    r"is gravion safe|gravion safe to use|"
    r"is it safe to use gravion|gravion safe hai)\b"
)
_MODELS_COUNT = re.compile(
    r"\b(?:how many models does gravion have|"
    r"how many gravion models|models does gravion have|"
    r"how many models are in gravion|"
    r"gravion kitne model|kitne model hai|"
    r"kitne model deta|kitne models hai)\b"
)
# Omni and GPU asks (28 Nov 2026 cycle, 13:46 run):
# "what is Gravion Omni?" was answered with the general
# Gravion about line, and "does Gravion need a GPU?"
# echoed beside the laptop forms. The Omni pattern
# names Gravion Omni directly and is checked before
# the general about pattern; the GPU pattern names
# the hardware need.
_OMNI = re.compile(
    r"\b(?:what is gravion omni|gravion omni kya hai|"
    r"about gravion omni|tell me about gravion omni)\b"
)
_HARDWARE = re.compile(
    r"\b(?:need a gpu|need gpu|gpu chahiye|"
    r"graphics card|does gravion need a gpu)\b"
)
_THANKS = re.compile(
    r"^(?:thank you|thanks|thanks a lot|thank you so much|"
    r"many thanks|dhanyavaad|shukriya|bahut dhanyavaad)"
    r"(?: very much| so much)?$"
)
_FAREWELL = re.compile(
    r"^(?:goodbye|bye|bye bye|see you|see you later|"
    r"good night|phir milenge|alvida)(?: soon| later)?$"
)


def _normalized(prompt: str) -> str:
    text = prompt.lower().replace("'", " ")
    return re.sub(r"[^a-z0-9 ]+", " ", text).strip()


def opening_response(prompt: str) -> str | None:
    """Deterministic reply for greeting/identity/capability openings.

    Returns None for everything else, including negated capability
    questions, so the model keeps answering real requests.
    """

    text = _normalized(prompt)
    if not text or _NEGATION.search(text):
        return None
    # A translation request that quotes an opening line ("translate
    # how are you into hindi") asks for the translation, not the
    # opening reply, so it stays with the model.
    if "translate" in text.split():
        return None
    words = len(text.split())
    if _BUILDER.search(text):
        return BUILDER_RESPONSE
    if _AI_IDENTITY.search(text):
        return AI_IDENTITY_RESPONSE
    if _OMNI.search(text) and words <= 8:
        return OMNI_RESPONSE
    if _ABOUT.search(text) and words <= 8:
        return ABOUT_RESPONSE
    if _IDENTITY.search(text):
        return IDENTITY_RESPONSE
    if _CAPABILITIES.search(text) or _BARE_HELP.match(text):
        return CAPABILITIES_RESPONSE
    if _MEET.match(text):
        return MEET_RESPONSE
    if _OFFLINE.search(text) and words <= 8:
        return OFFLINE_RESPONSE
    if _API_KEY.search(text) and words <= 9:
        return OFFLINE_RESPONSE
    if _START.search(text) and words <= 8:
        return START_RESPONSE
    if _FREE.search(text) and words <= 6:
        return FREE_RESPONSE
    if _STARTUP.search(text) and words <= 12:
        return STARTUP_RESPONSE
    if _AUDIENCE.search(text) and words <= 8:
        return AUDIENCE_RESPONSE
    if _HARDWARE.search(text) and words <= 8:
        return HARDWARE_RESPONSE
    if _LAPTOP.search(text) and words <= 9:
        return LAPTOP_RESPONSE
    if _DIFFERENT.search(text) and words <= 8:
        return DIFFERENT_RESPONSE
    if _PRICE.search(text) and words <= 8:
        return PRICE_RESPONSE
    if _COMPANY.search(text) and words <= 9:
        return COMPANY_RESPONSE
    if _INCIDENT.search(text) and words <= 12:
        return INCIDENT_RESPONSE
    if _TEAM.search(text) and words <= 9:
        return TEAM_RESPONSE
    if _MODEL_CHOICE.search(text) and words <= 8:
        return MODEL_RESPONSE
    if _CODE_CAPABILITY.match(text):
        return CODE_RESPONSE
    if _WEBSITE_CAPABILITY.match(text):
        return WEBSITE_RESPONSE
    if _TIERS.search(text) and words <= 8:
        return TIERS_RESPONSE
    if _PRIVACY.search(text) and words <= 8:
        return PRIVACY_RESPONSE
    if _MODELS_COUNT.search(text) and words <= 8:
        return MODELS_COUNT_RESPONSE
    if _LANGUAGE.search(text):
        return LANGUAGE_RESPONSE
    if _WELLBEING.search(text) and words <= 8:
        return WELLBEING_RESPONSE
    if _THANKS.match(text):
        return THANKS_RESPONSE
    if _FAREWELL.match(text):
        return FAREWELL_RESPONSE
    if _GREETING.match(text) and words <= 6:
        return GREETING_RESPONSE
    return None


class ChatModule:
    def __init__(self, manifest: ModuleManifest):
        self.manifest = manifest

    def run(self, ctx: ModuleContext, prompt: str) -> str:
        opening = opening_response(prompt)
        if opening is not None:
            return opening
        return ctx.chat(prompt)
