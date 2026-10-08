"""Runtime error hierarchy.

Failures are typed so the platform can distinguish "the model said
something malformed" (never trusted, never acted on) from "the backend
is down" (fallback policy applies) from "the declaration itself is bad"
(developer bug, fail fast).
"""


class DecisionError(Exception):
    """Base class for all decision-step failures."""


class DeclarationError(DecisionError):
    """The DecisionStep contract itself is invalid (developer bug)."""


class BackendError(DecisionError):
    """The backend call failed or returned no usable answer."""


class MalformedOutputError(DecisionError):
    """Backend output violated the declared schema.

    This is the load-bearing guarantee: a malformed value can never flow
    into a workflow, because the runtime rejects it here, before routing.
    """


class PolicyViolationError(DecisionError):
    """A declared policy is internally inconsistent (e.g. inverted thresholds)."""
