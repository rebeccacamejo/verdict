"""verdict: typed deterministic decision steps for agent toolkits.

The one-line version::

    from verdict import decide, Choice

    result = decide(
        "Order #4812 shows two charges of $184.00 three minutes apart.",
        Choice(
            instructions="Pick the dispute reason code.",
            options={"10.4": "Duplicate processing", "13.1": "Merchandise not received"},
        ),
        backend="kev",
    )
    result.value        # "10.4"
    result.confidence   # 0.93
    result.routing      # <RoutingDecision.AUTO_PROCEED>

Design rule: the SDK *declares*, the trusted runtime *enforces*.
``decide()`` always executes through ``TrustedRuntime`` -- declarations
are re-validated, backend outputs are schema-validated before they can
influence routing, and every execution is audit-logged with an
idempotency key. See ``verdict.sdk`` and ``verdict.runtime``.

For the declarative path (``verdict.yaml``)::

    from verdict import load_file

    plan = load_file("verdict.yaml")
    plan.decide("dispute_reason_code", "I was charged twice.")
"""

from __future__ import annotations

from typing import Mapping

from verdict.backends import (
    BackendInfo,
    DecisionBackend,
    get_backend,
    list_backends,
)
from verdict.runtime import (
    BackendError,
    DecisionError,
    DecisionOutcome,
    DeclarationError,
    MalformedOutputError,
    PolicyViolationError,
    RoutingDecision,
    TrustedRuntime,
)
from verdict.sdk import (
    Boolean,
    Choice,
    DecisionStep,
    FallbackPolicy,
    Score,
    ThresholdPolicy,
)

__all__ = [
    "BackendError",
    "BackendInfo",
    "Boolean",
    "Choice",
    "DecisionBackend",
    "DecisionError",
    "DecisionOutcome",
    "DecisionPlan",
    "DecisionStep",
    "DeclarationError",
    "FallbackPolicy",
    "MalformedOutputError",
    "PolicyViolationError",
    "RoutingDecision",
    "Score",
    "ThresholdPolicy",
    "TrustedRuntime",
    "decide",
    "decide_many",
    "get_backend",
    "list_backends",
    "load_file",
]


def __getattr__(name: str):
    # Lazy: importing yaml only when the declarative path is used.
    if name in ("load_file", "DecisionPlan"):
        from verdict.declarative import DecisionPlan, load_file

        return {"load_file": load_file, "DecisionPlan": DecisionPlan}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__version__ = "0.2.0"


def decide(
    state: str,
    question: Choice | Score | Boolean,
    backend: str | DecisionBackend = "mock",
    policy: ThresholdPolicy | None = None,
    name: str = "decision",
    description: str = "",
    idempotency_key: str | None = None,
    **backend_kwargs,
) -> DecisionOutcome:
    """Execute one typed decision step and return the trusted outcome.

    Args:
        state: The case text the decision is about (e.g. a dispute record).
        question: A ``Choice``, ``Score``, or ``Boolean`` declaration.
        backend: A registered backend name (``"jev"``, ``"kev"``, ``"semif"``,
            ``"laya"``, ``"mock"``) or an already-constructed
            ``DecisionBackend``. Extra ``backend_kwargs`` (e.g.
            ``base_url=...``) are forwarded to the constructor.
        policy: Confidence routing thresholds. Defaults to
            ``ThresholdPolicy(auto_threshold=0.85, review_threshold=0.60)``.
        name: Step name used in audit records and idempotency scoping.
        description: Free-text description stored on the step.
        idempotency_key: Repeat the call with the same key to get the
            stored outcome back with no backend call.

    Returns:
        A ``DecisionOutcome`` with ``value``, ``probabilities``,
        ``confidence``, ``routing`` (``AUTO_PROCEED`` / ``ESCALATE_TO_LLM`` /
        ``HUMAN_REVIEW``), ``model_version``, ``latency_ms``, ``cost_usd``,
        and ``audit_id``. On backend failure the declared
        ``FallbackPolicy`` applies instead of raising (unless
        ``FAIL_CLOSED``).
    """
    backend_obj = (
        backend if isinstance(backend, DecisionBackend)
        else get_backend(backend, **backend_kwargs)
    )
    step = DecisionStep(
        name=name,
        question=question,
        policy=policy or ThresholdPolicy(),
        description=description,
    )
    runtime = TrustedRuntime(backend_obj)
    return runtime.execute(step, state, idempotency_key=idempotency_key)


def decide_many(
    state: str,
    questions: Mapping[str, Choice | Score | Boolean],
    backend: str | DecisionBackend = "mock",
    policy: ThresholdPolicy | None = None,
    idempotency_key: str | None = None,
    **backend_kwargs,
) -> dict[str, DecisionOutcome]:
    """Execute several named decision steps against the same state.

    Each question runs as its own ``DecisionStep`` through the same
    trusted runtime (and the same backend instance), so audit records
    stay per-question while the backend connection is shared.
    """
    backend_obj = (
        backend if isinstance(backend, DecisionBackend)
        else get_backend(backend, **backend_kwargs)
    )
    runtime = TrustedRuntime(backend_obj)
    outcomes: dict[str, DecisionOutcome] = {}
    for qname, question in questions.items():
        step = DecisionStep(
            name=qname, question=question, policy=policy or ThresholdPolicy()
        )
        key = f"{idempotency_key}:{qname}" if idempotency_key else None
        outcomes[qname] = runtime.execute(step, state, idempotency_key=key)
    return outcomes
