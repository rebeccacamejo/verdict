"""Shared fixtures: dispute reason-code step, evidence step, mock resolvers.

Used by both the example pipeline and the eval CLI so the numbers stay
comparable. The resolvers are keyword heuristics -- deliberately crude;
they stand in for the Jev backend until API access is wired up.
"""

from __future__ import annotations

from verdict.backends.base import BackendAnswer, BackendQuestion, QuestionKind, TokenUsage
from verdict import Boolean, Choice, DecisionStep, Score, ThresholdPolicy

REASON_CODES = {
    "unauthorized": "Cardholder states they did not make or authorize the charge.",
    "not_received": "Merchandise or service paid for was never received.",
    "defective": "Merchandise arrived damaged, defective, or not as described.",
    "duplicate": "Cardholder was charged more than once for the same transaction.",
    "subscription": "Recurring charge continued after the subscription was cancelled.",
    "atm": "ATM did not dispense the requested cash (or dispensed a partial amount).",
}

EVIDENCE_LEVELS = [
    "No usable evidence attached.",
    "Weak evidence: cardholder statement only, nothing corroborating.",
    "Partial evidence: one supporting document (receipt, photo, or tracking).",
    "Good evidence: multiple supporting documents corroborating the claim.",
    "Conclusive evidence: documents fully prove the claim as stated.",
]


def reason_code_step() -> DecisionStep:
    return DecisionStep(
        name="dispute_reason_code",
        description="Classify a dispute into a reason code for downstream processing.",
        question=Choice(
            instructions=(
                "Given the cardholder's dispute statement, choose the single "
                "reason code that best describes the claim."
            ),
            options=REASON_CODES,
        ),
        policy=ThresholdPolicy(auto_threshold=0.85, review_threshold=0.60),
    )


def evidence_step() -> DecisionStep:
    return DecisionStep(
        name="evidence_completeness",
        description="Gate case progression on evidence sufficiency.",
        question=Score(
            instructions=(
                "Given the dispute statement and the evidence described as "
                "attached, score how complete the evidence is."
            ),
            levels=EVIDENCE_LEVELS,
        ),
        policy=ThresholdPolicy(auto_threshold=0.80, review_threshold=0.55),
    )


# ----------------------------------------------------------------------
# mock resolvers (stand-ins for Jev until the live backend is wired up)
# ----------------------------------------------------------------------

_KEYWORDS = {
    "unauthorized": ["didn't make", "did not make", "never made", "fraud",
                     "unauthorized", "don't recognize", "do not recognize",
                     "stolen", "wasn't me", "not me"],
    "not_received": ["never arrived", "never received", "didn't arrive",
                     "not delivered", "no delivery", "tracking", "never shipped"],
    "defective": ["broken", "defective", "damaged", "not as described",
                  "doesn't work", "won't power", "cracked", "missing parts"],
    "duplicate": ["twice", "two times", "duplicate", "double charged",
                  "charged again", "two identical", "charged twice"],
    "subscription": ["cancelled", "canceled", "subscription", "recurring",
                     "still charging", "free trial"],
    "atm": ["atm", "cash machine", "dispense", "no cash"],
}

_EVIDENCE_KEYWORDS = ["receipt", "photo", "tracking", "statement", "email", "screenshot"]


def _peaked(code: str, confidence: float) -> dict[str, float]:
    rest = (1.0 - confidence) / max(1, len(REASON_CODES) - 1)
    return {k: (confidence if k == code else rest) for k in REASON_CODES}


def keyword_resolver(state: str, qname: str, question: BackendQuestion) -> BackendAnswer:
    text = state.lower()
    if question.kind is QuestionKind.CHOICE:
        for code, words in _KEYWORDS.items():
            if any(w in text for w in words):
                return BackendAnswer(qname, question.kind, code,
                                     _peaked(code, 0.92), 0.92,
                                     "mock-keyword", TokenUsage(), 0.0, 5.0)
        # no keyword hit: low-confidence guess -> exercises HUMAN_REVIEW routing
        return BackendAnswer(qname, question.kind, "unauthorized",
                             _peaked("unauthorized", 0.35), 0.35,
                             "mock-keyword", TokenUsage(), 0.0, 5.0)
    if question.kind is QuestionKind.SCORE:
        hits = sum(1 for w in _EVIDENCE_KEYWORDS if w in text)
        level, conf = (4, 0.90) if hits >= 2 else (2, 0.70) if hits == 1 else (1, 0.55)
        n = len(EVIDENCE_LEVELS)
        probs = {str(i): (conf if i == level else (1 - conf) / (n - 1)) for i in range(n)}
        return BackendAnswer(qname, question.kind, level, probs, conf,
                             "mock-keyword", TokenUsage(), 0.0, 5.0)
    raise AssertionError(f"keyword_resolver: unsupported kind {question.kind}")


def weak_resolver(state: str, qname: str, question: BackendQuestion) -> BackendAnswer:
    """Majority-class baseline: always guesses 'unauthorized', middling confidence."""
    if question.kind is not QuestionKind.CHOICE:
        raise AssertionError("weak_resolver only supports the reason-code Choice")
    probs = _peaked("unauthorized", 0.55)
    return BackendAnswer(qname, question.kind, "unauthorized", probs, 0.55,
                         "mock-baseline", TokenUsage(), 0.0, 5.0)
