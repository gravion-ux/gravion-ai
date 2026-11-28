"""Chat opening short-circuit tests.

Regression context (28 Nov 2026): Roman Hindi openings a demo
audience actually uses ("tum kaun ho", "tum kya kar sakte ho")
fell through to the echo fallback, while their English forms
were answered directly. Negated forms stay with the model in
both languages.
"""

from gravion_ai.modules.chat.module import opening_response


def test_english_openings_still_answer():
    assert "modular AI platform" in opening_response("Hello, who are you?")
    assert "route it to" in opening_response("What can you do?")


def test_hinglish_identity_and_capabilities_answer():
    assert "modular AI platform" in opening_response("Namaste, tum kaun ho?")
    assert "modular AI platform" in opening_response("tumhara naam kya hai")
    assert "route it to" in opening_response("tum kya kar sakte ho?")


def test_negated_capability_questions_stay_with_the_model():
    assert opening_response("what can you not do?") is None
    assert opening_response("tum kya nahi kar sakte ho?") is None


def test_real_requests_are_not_short_circuited():
    assert opening_response("write a poem about rain") is None
    assert opening_response("tum kaun ho is a question in my exam") is not None


def test_demo_conversational_lines_answer():
    # 28 Nov 2026 cycle (03:16 run): the lines a demo opens and
    # closes with echoed under the default model.
    assert "doing well" in opening_response("How are you?")
    assert "doing well" in opening_response("kaise ho?")
    assert "built by Priyanshu Kumar Thakur" in opening_response("Who built you?")
    assert "from scratch" in opening_response("Tell me about Gravion")
    assert "from scratch" in opening_response("What is Gravion?")
    assert "not a human" in opening_response("Are you an AI?")
    assert "Roman Hindi" in opening_response("Can you speak Hindi?")
    assert "welcome" in opening_response("Thank you")
    assert "trying Gravion" in opening_response("Goodbye")


def test_demo_questions_about_gravion_answer():
    # 28 Nov 2026 cycle (04:16 run): the meet line and the
    # questions a demo audience asks about Gravion itself
    # echoed while the identity lines answered.
    assert "Nice to meet you too" in opening_response("Nice to meet you")
    assert "Roman Hindi" in opening_response("What languages do you support?")
    assert "fully offline" in opening_response("Do you work offline?")
    assert "gravion init" in opening_response("How do I get started with Gravion?")
    assert "Development and evaluation are free" in opening_response("Is Gravion free?")


def test_gravion_question_tasks_stay_with_the_model():
    # Tasks that merely mention the same words are real
    # requests, not openings about Gravion itself.
    assert opening_response("write a nice to meet you card for my teacher") is None
    assert opening_response("how do i get started with python programming today please") is None
    assert opening_response("is the museum free on sundays") is None


def test_incident_and_investor_lines_answer():
    # 28 Nov 2026 cycle (07:16 run): a live incident asked in
    # chat, the support-team question and the third-person
    # Hindi ask echoed while the cyber module and the
    # you-forms answered.
    assert "preserve logs" in opening_response(
        "Someone is trying to break into our server, what do we do?"
    )
    assert "preserve logs" in opening_response(
        "Our website got hacked, what should we do?"
    )
    assert "supports a team" in opening_response(
        "Can Gravion replace my support team?"
    )
    assert "Roman Hindi" in opening_response("Does Gravion work in Hindi?")
    # A general security question is not a live incident and
    # stays with the model.
    assert opening_response("how do attackers break into servers in general") is None


def test_laptop_internet_and_module_count_lines_answer():
    # 28 Nov 2026 cycle (08:46 run): "can Gravion run on a
    # laptop?", "does Gravion need an internet connection?"
    # and the Roman Hindi module-count ask echoed while
    # their neighbouring forms answered.
    assert "normal laptop" in opening_response("Can Gravion run on a laptop?")
    assert "fully offline" in opening_response(
        "Does Gravion need an internet connection?"
    )
    assert "route it to" in opening_response("Gravion kitne module hai?")


def test_privacy_and_model_count_lines_answer():
    # 28 Nov 2026 cycle (11:01 run): the startup-help,
    # Roman Hindi offline, data-privacy, model-count and
    # why-use asks echoed while their neighbouring forms
    # answered. The model count in the reply must match
    # the bundled catalog, so it cannot drift silently.
    import json
    from importlib import resources

    assert "module system" in opening_response("Can Gravion help my startup?")
    assert "module system" in opening_response("Gravion startup ke liye hai kya?")
    assert "fully offline" in opening_response("Gravion offline chalta hai kya?")
    assert "stay with you" in opening_response("Is my data private with Gravion?")
    assert "one modular platform" in opening_response("Why should I use Gravion?")
    catalog = json.loads(
        resources.files("gravion_ai.data").joinpath("models.json").read_text(encoding="utf-8")
    )
    count = len(catalog["models"])
    assert f"{count} models" in opening_response("How many models does Gravion have?")
    assert f"{count} models" in opening_response("Gravion kitne model deta hai?")
    # A task that merely mentions models stays with the model.
    assert opening_response("how many models does the museum display on sundays") is None


def test_conversational_tasks_stay_with_the_model():
    # Whole-message and translation guards: tasks that quote an
    # opening line are real requests, not openings.
    assert opening_response("write a thank you note") is None
    assert opening_response("translate how are you into hindi") is None
    assert opening_response("thank you for the report, now summarise it") is None
