"""Shared test fixtures."""

from verdict.backends.base import BackendAnswer, QuestionKind, TokenUsage
from verdict.backends.mock import MockBackend
from verdict import Choice, DecisionStep, Score, ThresholdPolicy


def choice_step(**policy_kw) -> DecisionStep:
    policy_kw.setdefault("auto_threshold", 0.85)
    policy_kw.setdefault("review_threshold", 0.60)
    return DecisionStep(
        name="q",
        question=Choice(
            instructions="pick one", options={"a": "Option A", "b": "Option B"}
        ),
        policy=ThresholdPolicy(**policy_kw),
    )


def score_step() -> DecisionStep:
    return DecisionStep(
        name="s",
        question=Score(instructions="rate it", levels=["bad", "ok", "good"]),
    )


def scripted_backend(kind: QuestionKind, value, confidence=0.9, probabilities=None):
    """Backend returning one canned (possibly malformed) answer."""

    def handler(state, qname, question):
        return BackendAnswer(
            question_name=qname,
            kind=kind,
            value=value,
            probabilities=probabilities,
            confidence=confidence,
            model_version="mock-test",
            usage=TokenUsage(input_tokens=10),
            cost_usd=0.0,
            latency_ms=1.0,
        )

    return MockBackend(handler)
