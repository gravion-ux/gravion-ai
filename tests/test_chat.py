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
