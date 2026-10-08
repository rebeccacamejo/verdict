"""TrustedRuntime: the only component allowed to execute a DecisionStep.

Execution pipeline (every step enforced here, in platform-owned code):

1. Validate the *declaration* (SDK output is never trusted).
2. Idempotency check -- a repeated key returns the stored outcome,
   no backend call, no double side effects.
3. Translate the SDK declaration into a backend question.
4. Call the backend through the adapter interface (timed).
5. Schema-validate the backend's raw answer. Anything malformed raises
   MalformedOutputError *before* it can influence routing -- this is the
   "malformed outputs impossible by construction" guarantee.
6. Route on confidence per the declared ThresholdPolicy.
7. Append the audit record (input, model version, output, routing).

Backend *failures* (as opposed to malformed outputs) follow the declared
FallbackPolicy: human review, escalate to LLM, or fail closed.
"""

from __future__ import annotations

import hashlib
import time
from typing import Mapping

from verdict.backends.base import BackendAnswer, BackendQuestion, DecisionBackend, QuestionKind
from verdict.sdk import Boolean, Choice, DecisionStep, FallbackPolicy, Score

from .audit import AuditLog
from .errors import (
    BackendError,
    DeclarationError,
    DecisionError,
    MalformedOutputError,
    PolicyViolationError,
)
from .types import DecisionOutcome, RoutingDecision


class TrustedRuntime:
    def __init__(
        self, backend: DecisionBackend, audit_log: AuditLog | None = None
    ) -> None:
        self._backend = backend
        # NOTE: `is None` check is deliberate -- AuditLog defines __len__,
        # so an empty log is falsy and `or` would silently discard it.
        self._audit = audit_log if audit_log is not None else AuditLog()
        self._idempotency: dict[str, DecisionOutcome] = {}

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def execute(
        self, step: DecisionStep, state: str, *, idempotency_key: str | None = None
    ) -> DecisionOutcome:
        self._validate_declaration(step, state)

        if idempotency_key is not None and idempotency_key in self._idempotency:
            cached = self._idempotency[idempotency_key]
            self._audit.append(
                {
                    "step_name": step.name,
                    "event": "idempotency_hit",
                    "idempotency_key": idempotency_key,
                    "replayed_audit_id": cached.audit_id,
                }
            )
            return cached

        question = self._to_backend_question(step)
        t0 = time.perf_counter()
        try:
            answers = self._backend.evaluate(state, {step.name: question})
        except DecisionError as exc:
            return self._on_backend_error(step, state, idempotency_key, exc)
        except Exception as exc:  # noqa: BLE001 -- normalize unexpected backend failures
            return self._on_backend_error(
                step, state, idempotency_key, BackendError(str(exc))
            )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        raw = answers.get(step.name)
        if raw is None:
            raise MalformedOutputError(
                f"backend returned no answer for question {step.name!r}"
            )
        value, probabilities, confidence = self._validate_answer(step, raw)
        routing = self._route(step, confidence)

        outcome = DecisionOutcome(
            step_name=step.name,
            value=value,
            probabilities=probabilities,
            confidence=confidence,
            routing=routing,
            model_version=raw.model_version,
            latency_ms=raw.latency_ms or latency_ms,
            cost_usd=raw.cost_usd,
            idempotency_key=idempotency_key,
        )
        outcome.audit_id = self._audit.append(self._record(step, state, raw, outcome))
        if idempotency_key is not None:
            self._idempotency[idempotency_key] = outcome
        return outcome

    def replay(self, audit_id: int) -> DecisionOutcome:
        """Reconstruct an outcome from the audit log -- no backend call."""
        rec = self._audit.get(audit_id)
        if rec.get("event") == "idempotency_hit":
            return self.replay(rec["replayed_audit_id"])
        return DecisionOutcome(
            step_name=rec["step_name"],
            value=rec["value"],
            probabilities=rec["probabilities"],
            confidence=rec["confidence"],
            routing=RoutingDecision(rec["routing"]),
            model_version=rec["model_version"],
            latency_ms=rec["latency_ms"],
            cost_usd=rec["cost_usd"],
            idempotency_key=rec.get("idempotency_key"),
            audit_id=audit_id,
            error=rec.get("error"),
        )

    # ------------------------------------------------------------------
    # declaration validation (the SDK is never trusted)
    # ------------------------------------------------------------------

    def _validate_declaration(self, step: DecisionStep, state: str) -> None:
        if not isinstance(step.name, str) or not step.name:
            raise DeclarationError("step name must be a non-empty string")
        if not isinstance(state, str) or not state.strip():
            raise DeclarationError("state must be a non-empty string")

        q = step.question
        if isinstance(q, Choice):
            if not q.instructions or not q.instructions.strip():
                raise DeclarationError("Choice instructions must be non-empty")
            if not isinstance(q.options, Mapping) or len(q.options) < 2:
                raise DeclarationError("Choice needs at least 2 options")
            if len(q.options) > 255:
                raise DeclarationError(
                    f"Choice supports at most 255 options, got {len(q.options)}"
                )
            for key, desc in q.options.items():
                if not isinstance(key, str) or not key:
                    raise DeclarationError("Choice option keys must be non-empty strings")
                if not isinstance(desc, str) or not desc.strip():
                    raise DeclarationError(f"Choice option {key!r} needs a description")
        elif isinstance(q, Score):
            if not q.instructions or not q.instructions.strip():
                raise DeclarationError("Score instructions must be non-empty")
            if not 2 <= len(q.levels) <= 10:
                raise DeclarationError(
                    f"Score needs 2..10 levels, got {len(q.levels)}"
                )
        elif isinstance(q, Boolean):
            if not q.instructions or not q.instructions.strip():
                raise DeclarationError("Boolean instructions must be non-empty")
        else:
            raise DeclarationError(f"unknown question type: {type(q).__name__}")

        p = step.policy
        for attr in ("auto_threshold", "review_threshold"):
            v = getattr(p, attr)
            if not 0.0 <= v <= 1.0:
                raise PolicyViolationError(f"{attr} must be in [0, 1], got {v}")
        if p.review_threshold > p.auto_threshold:
            raise PolicyViolationError(
                "review_threshold must be <= auto_threshold "
                f"({p.review_threshold} > {p.auto_threshold})"
            )

    # ------------------------------------------------------------------
    # backend invocation + output schema validation (the hard guarantee)
    # ------------------------------------------------------------------

    def _to_backend_question(self, step: DecisionStep) -> BackendQuestion:
        q = step.question
        if isinstance(q, Choice):
            return BackendQuestion(
                QuestionKind.CHOICE, q.instructions, options=dict(q.options)
            )
        if isinstance(q, Score):
            return BackendQuestion(
                QuestionKind.SCORE, q.instructions, levels=tuple(q.levels)
            )
        return BackendQuestion(QuestionKind.NOUL, q.instructions)

    def _validate_answer(
        self, step: DecisionStep, ans: BackendAnswer
    ) -> tuple[str | int | bool, dict[str, float], float]:
        q = step.question
        expected = {
            Choice: QuestionKind.CHOICE,
            Score: QuestionKind.SCORE,
            Boolean: QuestionKind.NOUL,
        }[type(q)]
        if ans.kind is not expected:
            raise MalformedOutputError(
                f"question {step.name!r} declared {expected.value}, "
                f"backend answered {ans.kind.value}"
            )

        if isinstance(q, Choice):
            options = set(q.options)
            if not isinstance(ans.value, str) or ans.value not in options:
                raise MalformedOutputError(
                    f"choice value {ans.value!r} not in declared options {sorted(options)}"
                )
            probs = self._validate_probabilities(ans.probabilities, options, step.name)
            conf = self._validate_confidence(ans.confidence, step.name)
            return ans.value, probs, conf

        if isinstance(q, Score):
            n = len(q.levels)
            level = ans.value
            if isinstance(level, float) and level.is_integer():
                level = int(level)
            if not isinstance(level, int) or not 0 <= level < n:
                raise MalformedOutputError(
                    f"score {ans.value!r} out of declared range 0..{n - 1}"
                )
            expected_keys = {str(i) for i in range(n)}
            probs = self._validate_probabilities(ans.probabilities, expected_keys, step.name)
            conf = self._validate_confidence(ans.confidence, step.name)
            return level, probs, conf

        # Boolean <- Jev noul: 0..1 truth value.
        noul = ans.value
        if not isinstance(noul, (int, float)) or not 0.0 <= noul <= 1.0:
            raise MalformedOutputError(f"noul {ans.value!r} not in [0, 1]")
        value = bool(noul >= 0.5)
        confidence = abs(float(noul) - 0.5) * 2.0  # 0 at 0.5, 1.0 at the extremes
        return value, {}, confidence

    @staticmethod
    def _validate_probabilities(
        probs: Mapping[str, float] | None, expected_keys: set[str], step_name: str
    ) -> dict[str, float]:
        if not isinstance(probs, Mapping):
            raise MalformedOutputError(
                f"question {step_name!r}: probabilities missing or not a mapping"
            )
        if set(probs.keys()) != expected_keys:
            raise MalformedOutputError(
                f"question {step_name!r}: probability keys {sorted(probs)} "
                f"!= declared {sorted(expected_keys)}"
            )
        for k, v in probs.items():
            if not isinstance(v, (int, float)) or not 0.0 <= v <= 1.0:
                raise MalformedOutputError(
                    f"question {step_name!r}: probability {k!r}={v!r} not in [0, 1]"
                )
        return {k: float(v) for k, v in probs.items()}

    @staticmethod
    def _validate_confidence(conf: float | None, step_name: str) -> float:
        if not isinstance(conf, (int, float)) or not 0.0 <= conf <= 1.0:
            raise MalformedOutputError(
                f"question {step_name!r}: confidence {conf!r} not in [0, 1]"
            )
        return float(conf)

    # ------------------------------------------------------------------
    # routing + fallback
    # ------------------------------------------------------------------

    def _route(self, step: DecisionStep, confidence: float) -> RoutingDecision:
        if confidence >= step.policy.auto_threshold:
            return RoutingDecision.AUTO_PROCEED
        if confidence >= step.policy.review_threshold:
            return RoutingDecision.ESCALATE_TO_LLM
        return RoutingDecision.HUMAN_REVIEW

    def _on_backend_error(
        self, step: DecisionStep, state: str, idempotency_key: str | None, exc: Exception
    ) -> DecisionOutcome:
        policy = step.policy.on_backend_error
        if policy is FallbackPolicy.FAIL_CLOSED:
            raise BackendError(f"backend failed and policy is fail_closed: {exc}") from exc
        routing = (
            RoutingDecision.HUMAN_REVIEW
            if policy is FallbackPolicy.HUMAN_REVIEW
            else RoutingDecision.ESCALATE_TO_LLM
        )
        outcome = DecisionOutcome(
            step_name=step.name,
            value=None,
            probabilities={},
            confidence=0.0,
            routing=routing,
            model_version=f"{self._backend.name} (error)",
            latency_ms=0.0,
            cost_usd=0.0,
            idempotency_key=idempotency_key,
            error=f"{type(exc).__name__}: {exc}",
        )
        outcome.audit_id = self._audit.append(self._record(step, state, None, outcome))
        if idempotency_key is not None:
            self._idempotency[idempotency_key] = outcome
        return outcome

    # ------------------------------------------------------------------
    # audit
    # ------------------------------------------------------------------

    @staticmethod
    def _record(
        step: DecisionStep,
        state: str,
        raw: BackendAnswer | None,
        outcome: DecisionOutcome,
    ) -> dict:
        q = step.question
        return {
            "event": "executed",
            "step_name": step.name,
            "question_kind": (
                q.__class__.__name__
            ),
            "policy": {
                "auto_threshold": step.policy.auto_threshold,
                "review_threshold": step.policy.review_threshold,
                "on_backend_error": step.policy.on_backend_error.value,
            },
            "state_sha256": hashlib.sha256(state.encode()).hexdigest(),
            "state": state,  # POC: full state stored; production would store a reference
            "model_version": outcome.model_version,
            "value": outcome.value,
            "probabilities": dict(outcome.probabilities),
            "confidence": outcome.confidence,
            "routing": outcome.routing.value,
            "latency_ms": outcome.latency_ms,
            "cost_usd": outcome.cost_usd,
            "idempotency_key": outcome.idempotency_key,
            "error": outcome.error,
            "raw_backend_kind": raw.kind.value if raw else None,
        }
