"""Backend adapter interface.

A backend turns (state, typed questions) into typed answers. Backends are
dumb by design: they return what the model said. *Trust* is added by the
runtime, which schema-validates every answer before it can influence a
workflow.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Mapping


class QuestionKind(str, Enum):
    CHOICE = "choice"
    SCORE = "score"
    NOUL = "noul"  # Jev's 0..1 truth-value primitive; SDK Boolean maps here


@dataclass(frozen=True)
class BackendQuestion:
    """Normalized question passed to a backend (runtime translates SDK -> this)."""

    kind: QuestionKind
    instructions: str
    options: Mapping[str, str] | None = None  # choice: key -> description
    levels: tuple[str, ...] | None = None  # score: ordered level descriptions


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0  # Jev output is unmetered; kept for interface parity


@dataclass(frozen=True)
class BackendAnswer:
    """One raw answer from a backend. NOT yet trusted -- runtime validates."""

    question_name: str
    kind: QuestionKind
    value: str | int | float | bool
    probabilities: Mapping[str, float] | None
    confidence: float | None  # None for noul (runtime derives it)
    model_version: str
    usage: TokenUsage
    cost_usd: float
    latency_ms: float


class DecisionBackend(ABC):
    """Adapter interface. Implementations: JevBackend, MockBackend."""

    name: str = "base"

    @abstractmethod
    def evaluate(
        self, state: str, questions: Mapping[str, BackendQuestion]
    ) -> Mapping[str, BackendAnswer]:
        """Evaluate questions against state in one parallel pass."""
        raise NotImplementedError
