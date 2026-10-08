"""Shared runtime types (kept separate to avoid import cycles)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping


class RoutingDecision(str, Enum):
    AUTO_PROCEED = "auto_proceed"
    ESCALATE_TO_LLM = "escalate_to_llm"
    HUMAN_REVIEW = "human_review"


@dataclass
class DecisionOutcome:
    """The trusted result of executing one DecisionStep."""

    step_name: str
    value: str | int | bool | None
    probabilities: Mapping[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    routing: RoutingDecision = RoutingDecision.HUMAN_REVIEW
    model_version: str = ""
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    idempotency_key: str | None = None
    audit_id: int | None = None
    error: str | None = None  # set when a fallback policy produced this outcome
