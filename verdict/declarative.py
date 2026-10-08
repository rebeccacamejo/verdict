"""Declarative decision definitions: describe decisions as data, not code.

A ``verdict.yaml`` file declares backends and decision contracts. The
platform owns the file; developers just call it:

.. code-block:: yaml

    version: 1
    backends:
      kev:
        base_url: http://localhost:8000/v1/systemone
    decisions:
      - name: dispute_reason_code
        kind: choice
        instructions: "Pick the dispute reason code."
        options:
          "10.4": "Duplicate processing"
          "13.1": "Merchandise not received"
        policy:
          backend: kev
          auto_threshold: 0.85
          review_threshold: 0.60
          on_backend_error: human_review

Why declarative fits these models: the output space is fully enumerated
up front (options/levels), the model generates no text, and the routing
policy is static. The declaration *is* the complete contract -- there is
nothing to prompt-engineer at runtime. That makes decision definitions
versionable, reviewable, and evaluable in CI: ``verdict validate``,
``verdict eval -f verdict.yaml``, promote.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from verdict import (
    Boolean,
    Choice,
    DecisionStep,
    FallbackPolicy,
    Score,
    ThresholdPolicy,
    get_backend,
)
from verdict.backends import DecisionBackend, list_backends
from verdict.runtime import DecisionOutcome, TrustedRuntime

VALID_KINDS = ("choice", "score", "boolean")
VALID_FALLBACKS = ("human_review", "escalate_to_llm", "fail_closed")


def _load_yaml(path: str | Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "verdict.yaml needs PyYAML: pip install pyyaml  (or use JSON)"
        ) from exc
    text = Path(path).read_text()
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    return data


def _load_json(path: str | Path) -> dict:
    import json

    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    return data


def load_file(path: str | Path) -> "DecisionPlan":
    """Load a verdict.yaml (or .json) file into a DecisionPlan."""
    path = Path(path)
    if path.suffix in (".yaml", ".yml"):
        data = _load_yaml(path)
    elif path.suffix == ".json":
        data = _load_json(path)
    else:
        raise ValueError(f"{path}: expected .yaml, .yml, or .json")
    errors = validate_data(data, source=str(path))
    if errors:
        raise ValueError(f"{path}: invalid:\n" + "\n".join(f"  - {e}" for e in errors))
    return DecisionPlan.from_data(data)


def validate_data(data: Mapping, source: str = "verdict.yaml") -> list[str]:
    """Validate declarative data; returns a list of error strings (empty = ok)."""
    errors: list[str] = []
    if data.get("version") != 1:
        errors.append("version must be 1")
    decisions = data.get("decisions")
    if not isinstance(decisions, list) or not decisions:
        errors.append("decisions must be a non-empty list")
        return errors
    seen: set[str] = set()
    backends_cfg = data.get("backends") or {}
    if not isinstance(backends_cfg, dict):
        errors.append("backends must be a mapping of name -> kwargs")
        backends_cfg = {}
    known = set(list_backends())
    for bname in backends_cfg:
        if bname not in known:
            errors.append(f"backends.{bname}: unknown backend (choose from {sorted(known)})")
    for i, d in enumerate(decisions):
        where = f"decisions[{i}]"
        if not isinstance(d, dict):
            errors.append(f"{where}: must be a mapping")
            continue
        name = d.get("name")
        if not name or not isinstance(name, str):
            errors.append(f"{where}: name must be a non-empty string")
            continue
        if name in seen:
            errors.append(f"{where}: duplicate decision name {name!r}")
        seen.add(name)
        where = f"decisions.{name}"
        kind = d.get("kind")
        if kind not in VALID_KINDS:
            errors.append(f"{where}: kind must be one of {VALID_KINDS}")
        if not (d.get("instructions") or "").strip():
            errors.append(f"{where}: instructions must be non-empty")
        if kind == "choice":
            options = d.get("options")
            if not isinstance(options, dict) or len(options) < 2:
                errors.append(f"{where}: choice needs options mapping with >= 2 entries")
        if kind == "score":
            levels = d.get("levels")
            if not isinstance(levels, list) or not 2 <= len(levels) <= 10:
                errors.append(f"{where}: score needs levels list with 2..10 entries")
        policy = d.get("policy") or {}
        if not isinstance(policy, dict):
            errors.append(f"{where}.policy: must be a mapping")
            continue
        backend = policy.get("backend", "mock")
        if backend not in known:
            errors.append(f"{where}.policy.backend: unknown {backend!r}")
        for attr in ("auto_threshold", "review_threshold"):
            v = policy.get(attr, 0.85 if attr == "auto_threshold" else 0.60)
            if not isinstance(v, (int, float)) or not 0.0 <= v <= 1.0:
                errors.append(f"{where}.policy.{attr}: must be in [0, 1]")
        auto = policy.get("auto_threshold", 0.85)
        review = policy.get("review_threshold", 0.60)
        if isinstance(auto, (int, float)) and isinstance(review, (int, float)):
            if review > auto:
                errors.append(f"{where}.policy: review_threshold must be <= auto_threshold")
        fb = policy.get("on_backend_error", "human_review")
        if fb not in VALID_FALLBACKS:
            errors.append(f"{where}.policy.on_backend_error: must be one of {VALID_FALLBACKS}")
    return errors


@dataclass
class _CompiledDecision:
    name: str
    step: DecisionStep
    backend_name: str
    backend_kwargs: dict = field(default_factory=dict)


@dataclass
class DecisionPlan:
    """A loaded verdict.yaml: named decisions, ready to execute or evaluate."""

    decisions: dict[str, _CompiledDecision]
    backends_config: dict[str, dict] = field(default_factory=dict)

    @classmethod
    def from_data(cls, data: Mapping) -> "DecisionPlan":
        backends_cfg = {k: dict(v or {}) for k, v in (data.get("backends") or {}).items()}
        compiled: dict[str, _CompiledDecision] = {}
        for d in data["decisions"]:
            kind = d["kind"]
            if kind == "choice":
                question: Choice | Score | Boolean = Choice(
                    instructions=d["instructions"], options=dict(d["options"])
                )
            elif kind == "score":
                question = Score(instructions=d["instructions"], levels=list(d["levels"]))
            else:
                question = Boolean(instructions=d["instructions"])
            policy_cfg = d.get("policy") or {}
            step = DecisionStep(
                name=d["name"],
                question=question,
                policy=ThresholdPolicy(
                    auto_threshold=policy_cfg.get("auto_threshold", 0.85),
                    review_threshold=policy_cfg.get("review_threshold", 0.60),
                    on_backend_error=FallbackPolicy(
                        policy_cfg.get("on_backend_error", "human_review")
                    ),
                ),
                description=d.get("description", ""),
            )
            backend_name = policy_cfg.get("backend", "mock")
            compiled[d["name"]] = _CompiledDecision(
                name=d["name"],
                step=step,
                backend_name=backend_name,
                backend_kwargs=dict(backends_cfg.get(backend_name, {})),
            )
        return cls(decisions=compiled, backends_config=backends_cfg)

    def _runtime_for(self, name: str) -> TrustedRuntime:
        compiled = self.decisions[name]
        backend: DecisionBackend = get_backend(
            compiled.backend_name, **compiled.backend_kwargs
        )
        return TrustedRuntime(backend)

    def decide(
        self, name: str, state: str, idempotency_key: str | None = None
    ) -> DecisionOutcome:
        """Execute one named decision through the trusted runtime."""
        if name not in self.decisions:
            raise KeyError(
                f"unknown decision {name!r}; choose from {sorted(self.decisions)}"
            )
        compiled = self.decisions[name]
        return self._runtime_for(name).execute(
            compiled.step, state, idempotency_key=idempotency_key
        )

    def decision_names(self) -> list[str]:
        return sorted(self.decisions)
