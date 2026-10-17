"""Response cleaning and localized final-answer regions."""

from __future__ import annotations

import pytest

from slmforge.eval.extract import clean_response, find_answer_region


def test_clean_response_cuts_leaked_turns() -> None:
    assert clean_response("  70 %<|im_end|>\nUser: next") == "70 %"
    assert clean_response("Answer.\nQuestion: what else?") == "Answer."


@pytest.mark.parametrize(
    ("response", "region"),
    [
        ("The ratio is capped.\nFinal answer: 70%", "70%"),
        ("Le plafond...\n**Réponse finale :** 70 %", "70 %"),
        ("كاين شرط...\nالجواب: سبعين فالمية", "سبعين فالمية"),
        ("l'plafond...\nJawab:\n\n70 f lmya", "70 f lmya"),
        ("Final answer: 60%\nWait, correction.\nFinal answer: 70%", "70%"),  # last marker wins
    ],
)
def test_final_line(response: str, region: str) -> None:
    r = find_answer_region(response)
    assert (r.text, r.method) == (region, "final_line")
    assert response[r.span[0] : r.span[1]] == region


def test_full_text_fallback() -> None:
    r = find_answer_region("  It is 70 percent.  ")
    assert (r.text, r.method) == ("It is 70 percent.", "full_text")
