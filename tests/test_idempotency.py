"""Idempotency: a repeated key never re-invokes the backend."""

from verdict.backends.base import QuestionKind
from verdict.runtime import TrustedRuntime
from tests.helpers import choice_step, scripted_backend

PROBS = {"a": 0.9, "b": 0.1}


def test_repeated_idempotency_key_calls_backend_once():
    backend = scripted_backend(QuestionKind.CHOICE, "a", 0.9, PROBS)
    rt = TrustedRuntime(backend)
    first = rt.execute(choice_step(), "state", idempotency_key="k-1")
    second = rt.execute(choice_step(), "state", idempotency_key="k-1")
    assert len(backend.calls) == 1
    assert second.value == first.value == "a"
    assert second.audit_id == first.audit_id


def test_different_keys_invoke_backend_each_time():
    backend = scripted_backend(QuestionKind.CHOICE, "a", 0.9, PROBS)
    rt = TrustedRuntime(backend)
    rt.execute(choice_step(), "state", idempotency_key="k-1")
    rt.execute(choice_step(), "state", idempotency_key="k-2")
    assert len(backend.calls) == 2
