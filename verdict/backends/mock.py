"""MockBackend: offline, deterministic-ish backend for dev and tests.

Driven by a caller-supplied handler so examples and tests can script
exact answers (including edge cases like malformed outputs). Token usage
is a rough estimate; cost is always 0.
"""

from __future__ import annotations

import time
from typing import Callable, Mapping

from .base import (
    BackendAnswer,
    BackendQuestion,
    DecisionBackend,
    TokenUsage,
)

# handler(state, question_name, question) -> BackendAnswer
Handler = Callable[[str, str, BackendQuestion], BackendAnswer]


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _default_handler(state: str, qname: str, q: BackendQuestion) -> BackendAnswer:
    """Zero-config neutral answers: exercises the plumbing, decides nothing.

    Choice -> first option at 0.5 confidence (uniform probabilities);
    score -> middle level; noul -> 0.5 (maximum uncertainty). Everything
    routes to human review / LLM escalation, which is exactly right for
    a backend that knows nothing.
    """
    from .base import QuestionKind

    if q.kind is QuestionKind.CHOICE:
        options = list((q.options or {}).keys())
        value = options[0] if options else ""
        n = max(1, len(options))
        probs = {k: 1.0 / n for k in options}
        confidence = 0.5
    elif q.kind is QuestionKind.SCORE:
        levels = list(q.levels or [])
        value = len(levels) // 2
        probs = {str(i): 1.0 / max(1, len(levels)) for i in range(len(levels))}
        confidence = 0.5
    else:
        value, probs, confidence = 0.5, None, None
    return BackendAnswer(
        question_name=qname, kind=q.kind, value=value,
        probabilities=probs, confidence=confidence,
        model_version="", usage=TokenUsage(), cost_usd=0.0, latency_ms=0.0,
    )


class MockBackend(DecisionBackend):
    name = "mock"

    def __init__(
        self,
        handler: Handler | None = None,
        model_version: str = "mock-1",
        latency_ms: float = 5.0,
    ) -> None:
        self._handler = handler or _default_handler
        self._model_version = model_version
        self._latency_ms = latency_ms
        self.calls: list[tuple[str, dict[str, BackendQuestion]]] = []

    def evaluate(
        self, state: str, questions: Mapping[str, BackendQuestion]
    ) -> Mapping[str, BackendAnswer]:
        t0 = time.perf_counter()
        self.calls.append((state, dict(questions)))
        out = {}
        for qname, q in questions.items():
            ans = self._handler(state, qname, q)
            # stamp mock metadata (handler may leave these blank)
            out[qname] = BackendAnswer(
                question_name=qname,
                kind=ans.kind,
                value=ans.value,
                probabilities=ans.probabilities,
                confidence=ans.confidence,
                model_version=ans.model_version or self._model_version,
                usage=ans.usage or TokenUsage(input_tokens=_estimate_tokens(state)),
                cost_usd=0.0,
                latency_ms=self._latency_ms,
            )
        _ = time.perf_counter() - t0
        return out

    # -- convenience factories -------------------------------------------

    @classmethod
    def from_script(
        cls,
        script: Mapping[str, tuple],
        model_version: str = "mock-1",
    ) -> "MockBackend":
        """Script canned answers: {question_name: (value, confidence, probabilities?)}."""

        def handler(state, qname, question):
            if qname not in script:
                raise AssertionError(f"no scripted answer for {qname!r}")
            spec = script[qname]
            value, confidence = spec[0], spec[1]
            probs = spec[2] if len(spec) > 2 else None
            return BackendAnswer(
                question_name=qname, kind=question.kind, value=value,
                probabilities=probs, confidence=confidence,
                model_version=model_version, usage=TokenUsage(),
                cost_usd=0.0, latency_ms=0.0,
            )

        return cls(handler, model_version=model_version)
