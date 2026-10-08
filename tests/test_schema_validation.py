"""Schema validation: malformed backend outputs are rejected before routing."""

import pytest

from verdict.backends.base import QuestionKind
from verdict.runtime import MalformedOutputError, TrustedRuntime
from tests.helpers import choice_step, score_step, scripted_backend

GOOD_PROBS = {"a": 0.9, "b": 0.1}


def test_valid_choice_passes():
    rt = TrustedRuntime(scripted_backend(QuestionKind.CHOICE, "a", 0.9, GOOD_PROBS))
    out = rt.execute(choice_step(), "some state")
    assert out.value == "a"
    assert out.confidence == 0.9


def test_choice_value_not_in_options_rejected():
    rt = TrustedRuntime(scripted_backend(QuestionKind.CHOICE, "zzz", 0.9, GOOD_PROBS))
    with pytest.raises(MalformedOutputError):
        rt.execute(choice_step(), "some state")


def test_choice_probability_keys_must_match_options():
    rt = TrustedRuntime(
        scripted_backend(QuestionKind.CHOICE, "a", 0.9, {"a": 1.0})  # missing "b"
    )
    with pytest.raises(MalformedOutputError):
        rt.execute(choice_step(), "some state")


def test_choice_confidence_out_of_range_rejected():
    rt = TrustedRuntime(scripted_backend(QuestionKind.CHOICE, "a", 1.5, GOOD_PROBS))
    with pytest.raises(MalformedOutputError):
        rt.execute(choice_step(), "some state")


def test_answer_kind_must_match_declaration():
    rt = TrustedRuntime(
        scripted_backend(QuestionKind.SCORE, 1, 0.9, {"0": 0.1, "1": 0.9})
    )
    with pytest.raises(MalformedOutputError):
        rt.execute(choice_step(), "some state")


def test_score_out_of_range_rejected():
    rt = TrustedRuntime(
        scripted_backend(QuestionKind.SCORE, 7, 0.9, {"0": 0.0, "1": 0.0, "2": 1.0})
    )
    with pytest.raises(MalformedOutputError):
        rt.execute(score_step(), "some state")


def test_score_probability_keys_must_match_levels():
    rt = TrustedRuntime(
        scripted_backend(QuestionKind.SCORE, 1, 0.9, {"0": 0.5, "1": 0.5})  # missing "2"
    )
    with pytest.raises(MalformedOutputError):
        rt.execute(score_step(), "some state")
