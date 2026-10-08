"""Audit replay: outcomes are reconstructible from the log, no backend needed."""

from verdict.backends.base import QuestionKind
from verdict.backends.mock import MockBackend
from verdict.runtime import TrustedRuntime
from verdict.runtime.audit import AuditLog
from verdict.runtime.errors import DecisionError
from tests.helpers import choice_step, scripted_backend

PROBS = {"a": 0.9, "b": 0.1}


def test_replay_reconstructs_outcome_without_backend():
    audit = AuditLog()  # in-memory
    rt = TrustedRuntime(scripted_backend(QuestionKind.CHOICE, "a", 0.9, PROBS), audit)
    original = rt.execute(choice_step(), "some dispute text", idempotency_key="case-1")

    def exploding(state, qname, question):
        raise DecisionError("backend is gone")

    replay_rt = TrustedRuntime(MockBackend(exploding), audit)
    replayed = replay_rt.replay(original.audit_id)

    assert replayed.value == original.value
    assert replayed.confidence == original.confidence
    assert replayed.probabilities == original.probabilities
    assert replayed.routing == original.routing
    assert replayed.model_version == original.model_version
    assert replayed.audit_id == original.audit_id


def test_audit_log_records_routing_inputs(tmp_path):
    log_path = tmp_path / "audit.jsonl"
    audit = AuditLog(log_path)
    rt = TrustedRuntime(scripted_backend(QuestionKind.CHOICE, "b", 0.7, {"a": 0.3, "b": 0.7}), audit)
    out = rt.execute(choice_step(), "state", idempotency_key="k")
    rec = audit.get(out.audit_id)
    assert rec["value"] == "b"
    assert rec["routing"] == "escalate_to_llm"
    assert rec["idempotency_key"] == "k"
    assert "state_sha256" in rec
    assert log_path.exists()  # file-backed log persisted
