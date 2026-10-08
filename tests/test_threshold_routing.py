"""Threshold routing: confidence decides auto / escalate / human."""

import pytest

from verdict.backends.base import QuestionKind
from verdict.backends.mock import MockBackend
from verdict.runtime import BackendError, RoutingDecision, TrustedRuntime
from verdict.runtime.errors import DecisionError
from verdict import Boolean, DecisionStep, FallbackPolicy, ThresholdPolicy
from tests.helpers import choice_step, scripted_backend

PROBS = {"a": 0.9, "b": 0.1}


def _route_at(confidence: float):
    rt = TrustedRuntime(scripted_backend(QuestionKind.CHOICE, "a", confidence, PROBS))
    return rt.execute(choice_step(), "state").routing


def test_high_confidence_auto_proceeds():
    assert _route_at(0.95) is RoutingDecision.AUTO_PROCEED


def test_mid_confidence_escalates_to_llm():
    assert _route_at(0.70) is RoutingDecision.ESCALATE_TO_LLM


def test_low_confidence_goes_to_human():
    assert _route_at(0.40) is RoutingDecision.HUMAN_REVIEW


def test_boundary_values():
    assert _route_at(0.85) is RoutingDecision.AUTO_PROCEED  # == auto_threshold
    assert _route_at(0.60) is RoutingDecision.ESCALATE_TO_LLM  # == review_threshold


def test_boolean_routes_on_effective_confidence():
    # noul 0.9 -> value True, effective confidence |0.9-0.5|*2 = 0.8 -> escalate
    step = DecisionStep(
        name="b",
        question=Boolean(instructions="is it true?"),
        policy=ThresholdPolicy(auto_threshold=0.85, review_threshold=0.60),
    )
    rt = TrustedRuntime(scripted_backend(QuestionKind.NOUL, 0.9))
    out = rt.execute(step, "state")
    assert out.value is True
    assert out.confidence == pytest.approx(0.8)
    assert out.routing is RoutingDecision.ESCALATE_TO_LLM


def test_backend_error_defaults_to_human_review():
    def exploding(state, qname, question):
        raise DecisionError("boom")

    rt = TrustedRuntime(MockBackend(exploding))
    out = rt.execute(choice_step(), "state")
    assert out.routing is RoutingDecision.HUMAN_REVIEW
    assert out.value is None
    assert out.error is not None


def test_backend_error_fail_closed_raises():
    def exploding(state, qname, question):
        raise DecisionError("boom")

    rt = TrustedRuntime(MockBackend(exploding))
    step = choice_step(on_backend_error=FallbackPolicy.FAIL_CLOSED)
    with pytest.raises(BackendError):
        rt.execute(step, "state")
