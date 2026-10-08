"""Dispute processing pipeline example: reason-code classification + evidence gate.

Wires the full loop end to end on the MockBackend (no API key needed):

  dispute text
      -> DecisionStep "dispute_reason_code" (Choice, 6 options)
      -> DecisionStep "evidence_completeness" (Score, 5 levels)
      -> threshold routing: AUTO_PROCEED / ESCALATE_TO_LLM / HUMAN_REVIEW

Run:  python examples/dispute_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verdict.backends.mock import MockBackend
from verdict.eval.presets import evidence_step, keyword_resolver, reason_code_step
from verdict import RoutingDecision, TrustedRuntime

DISPUTES = [
    (
        "case-101",
        "I was charged twice for the same $120 hotel stay on May 2nd. "
        "The hotel confirmed the error by email but has not reversed the duplicate charge. "
        "I have attached the email confirmation and my card statement.",
    ),
    (
        "case-102",
        "There's a weird charge on my card and I'm not sure what it is.",
    ),
    (
        "case-103",
        "The ATM debited $200 but no cash came out. I filed the ATM log reference "
        "number with my branch and attached a photo of the machine's error screen.",
    ),
]

EVIDENCE_GATE_LEVEL = 3  # score >= 3 means "enough evidence to proceed"


def process_case(runtime: TrustedRuntime, case_id: str, text: str) -> None:
    print(f"\n=== {case_id} ===")
    print(f"statement: {text[:80]}...")

    reason = runtime.execute(
        reason_code_step(), text, idempotency_key=f"{case_id}-reason"
    )
    print(f"reason code : {reason.value} (confidence {reason.confidence:.2f})")
    route_reason(reason.routing, "classification")

    evidence = runtime.execute(
        evidence_step(), text, idempotency_key=f"{case_id}-evidence"
    )
    print(f"evidence    : level {evidence.value} (confidence {evidence.confidence:.2f})")
    if evidence.value is not None and evidence.value >= EVIDENCE_GATE_LEVEL:
        print("gate        : PASS -- case may proceed to settlement review")
    else:
        print("gate        : HOLD -- request more evidence from cardholder")
    route_reason(evidence.routing, "evidence scoring")


def route_reason(routing: RoutingDecision, what: str) -> None:
    if routing is RoutingDecision.AUTO_PROCEED:
        print(f"routing     : AUTO_PROCEED -- {what} accepted, no human needed")
    elif routing is RoutingDecision.ESCALATE_TO_LLM:
        print(f"routing     : ESCALATE_TO_LLM -- {what} needs a second opinion")
    else:
        print(f"routing     : HUMAN_REVIEW -- {what} queued for an analyst")


def main() -> None:
    backend = MockBackend(keyword_resolver)
    runtime = TrustedRuntime(backend)  # in-memory audit log
    print("Dispute pipeline: mock backend (swap MockBackend -> JevBackend for live)")
    for case_id, text in DISPUTES:
        process_case(runtime, case_id, text)
    print(f"\naudit records: {len(runtime._audit)} "
          f"(every decision replayable via runtime.replay(audit_id))")


if __name__ == "__main__":
    main()
