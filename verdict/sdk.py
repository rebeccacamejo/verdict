"""Thin SDK declaration layer for typed decisions.

DESIGN RULE -- the whole point of this module
---------------------------------------------
The SDK *declares* what a decision step is. It *enforces* nothing.

- No schema validation of backend outputs happens here.
- No threshold or policy enforcement happens here.
- No credentials, rate limits, or audit writes happen here.

All enforcement lives in the trusted runtime (``runtime/``), which the
application developer cannot modify. A ``DecisionStep`` that never passes
through the runtime is just a data structure -- it cannot execute a
decision. This is deliberate: in the platform this POC targets, hard
guarantees must come from platform-owned code, not from
developer-modifiable SDK calls.

Consequently this module intentionally contains **zero validation logic**:
even "obvious" checks (e.g. option-count limits) are re-done by the runtime,
which is the authority. The SDK does cheap construction only, so misuse
fails fast at declaration time but is never *trusted* at execution time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Sequence


class FallbackPolicy(str, Enum):
    """What the trusted runtime does when the decision backend itself fails.

    Declared by the developer, enforced by the runtime.
    """

    HUMAN_REVIEW = "human_review"
    ESCALATE_TO_LLM = "escalate_to_llm"
    FAIL_CLOSED = "fail_closed"  # raise instead of returning an untrusted decision


@dataclass(frozen=True)
class Choice:
    """Choose one option from a fixed set.

    Maps to the Jev ``choice`` primitive. ``options`` maps an option key
    (what your code branches on) to a human description (what the model
    reads). The runtime enforces the 2..255 option limit.
    """

    instructions: str
    options: Mapping[str, str]


@dataclass(frozen=True)
class Score:
    """Score the state against an ordinal rubric.

    Maps to the Jev ``score`` primitive. ``levels`` are the ordered level
    descriptions; the returned value is the winning level index
    (0 .. len(levels)-1). The runtime enforces the 2..10 level limit.
    """

    instructions: str
    levels: Sequence[str]


@dataclass(frozen=True)
class Boolean:
    """A yes/no determination.

    Maps to the Jev ``noul`` primitive, which returns a 0..1 truth value.
    The runtime converts it to a boolean (``noul >= 0.5``) and derives an
    effective confidence of ``abs(noul - 0.5) * 2`` for threshold routing.
    """

    instructions: str


@dataclass(frozen=True)
class ThresholdPolicy:
    """Confidence routing thresholds for one decision step.

    Routing semantics (enforced by the runtime, not the SDK):

    - ``confidence >= auto_threshold``   -> AUTO_PROCEED
    - ``confidence >= review_threshold`` -> ESCALATE_TO_LLM
    - otherwise                          -> HUMAN_REVIEW

    In platform terms these thresholds are policy: they would be expressed
    in the control plane (e.g. as ACS rules like
    "decision X may auto-proceed only above confidence C and below amount A")
    and enforced by the trusted runtime.
    """

    auto_threshold: float = 0.85
    review_threshold: float = 0.60
    on_backend_error: FallbackPolicy = FallbackPolicy.HUMAN_REVIEW


@dataclass(frozen=True)
class DecisionStep:
    """A named, reusable, typed decision contract.

    This is the unit developers share in the toolkit: a self-describing
    question plus its routing policy. Executing it requires a
    ``runtime.TrustedRuntime``.
    """

    name: str
    question: Choice | Score | Boolean
    policy: ThresholdPolicy = field(default_factory=ThresholdPolicy)
    description: str = ""
