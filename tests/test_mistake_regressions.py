import pytest

from gravion_ai.training.mistakes import load_bundled_mistakes, run_regression_case

CASES = [case for case in load_bundled_mistakes() if case.corrected]


@pytest.mark.parametrize("case", CASES, ids=[case.id for case in CASES])
def test_corrected_mistake_stays_fixed(case):
    assert run_regression_case(case)
