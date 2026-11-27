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


def test_conversational_tasks_stay_with_the_model():
    # Whole-message and translation guards: tasks that quote an
    # opening line are real requests, not openings.
    assert opening_response("write a thank you note") is None
    assert opening_response("translate how are you into hindi") is None
    assert opening_response("thank you for the report, now summarise it") is None
