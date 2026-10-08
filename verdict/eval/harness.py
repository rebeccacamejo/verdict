"""Eval harness: run a DecisionStep over a labeled JSONL dataset.

Dataset rows: {"id": str, "state": str, "label": <expected value>}
  - Choice  -> label is the expected option key (str)
  - Score   -> label is the expected level index (int)
  - Boolean -> label is the expected boolean

Reports: accuracy, expected calibration error (ECE), latency p50/p99,
total + per-case cost, routing mix. Optionally runs a baseline
(step, backend) pair over the same rows and reports the accuracy delta.

Backends resolve through the registry (``verdict.backends.get_backend``),
so adding a backend is a one-line registry entry -- the harness, the
CLI, and ``verdict decide`` all pick it up. Backends that are not
installed or not reachable report "unavailable" instead of crashing.

Environment overrides (honored unless the kwarg is passed explicitly):
    KEV_BASE_URL   override the Kev server URL
    LAYA_BASE_URL  override the Laya server URL
    SEMIF_MODEL    override the SemIf model id
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from verdict import DecisionStep, TrustedRuntime
from verdict.backends import MockBackend, get_backend, list_backends
from verdict.runtime import DecisionOutcome
from verdict.runtime.errors import DecisionError


@dataclass
class CaseResult:
    id: str
    predicted: object
    expected: object
    correct: bool
    confidence: float
    routing: str
    latency_ms: float
    cost_usd: float


@dataclass
class EvalReport:
    step_name: str
    backend_name: str
    n: int
    accuracy: float
    ece: float
    latency_p50_ms: float
    latency_p99_ms: float
    total_cost_usd: float
    cost_per_case_usd: float
    routing_counts: dict[str, int] = field(default_factory=dict)
    cases: list[CaseResult] = field(default_factory=list)
    baseline_accuracy: float | None = None
    error_count: int = 0  # cases where the fallback policy handled a backend failure
    first_error: str | None = None

    @property
    def accuracy_delta_vs_baseline(self) -> float | None:
        if self.baseline_accuracy is None:
            return None
        return self.accuracy - self.baseline_accuracy


@dataclass
class BackendComparison:
    backend: str
    available: bool
    note: str = ""
    report: EvalReport | None = None


def make_backend(name: str, **kwargs):
    """Construct a backend by registry name. May raise (no key / not installed)."""
    if name == "mock" and not kwargs:
        # The eval's mock uses the keyword heuristic, not the neutral default.
        from verdict.eval.presets import keyword_resolver

        return MockBackend(keyword_resolver)
    return get_backend(name, **kwargs)


def load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path) as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def expected_calibration_error(results: list[CaseResult], n_bins: int = 10) -> float:
    """Mean |accuracy - confidence| weighted by bin size (standard ECE)."""
    if not results:
        return 0.0
    bins: list[list[CaseResult]] = [[] for _ in range(n_bins)]
    for r in results:
        b = min(int(r.confidence * n_bins), n_bins - 1)
        bins[b].append(r)
    ece = 0.0
    for b in bins:
        if not b:
            continue
        acc = sum(1 for r in b if r.correct) / len(b)
        conf = sum(r.confidence for r in b) / len(b)
        ece += abs(acc - conf) * (len(b) / len(results))
    return ece


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    ordered = sorted(xs)
    k = min(len(ordered) - 1, int(p / 100 * len(ordered)))
    return ordered[k]


def _normalize(value: object) -> object:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def run_eval(
    step: DecisionStep,
    dataset: list[dict],
    backend,
    *,
    baseline: tuple[DecisionStep, object] | None = None,
) -> EvalReport:
    runtime = TrustedRuntime(backend)
    cases: list[CaseResult] = []
    first_error: str | None = None
    error_count = 0
    for i, row in enumerate(dataset):
        outcome: DecisionOutcome = runtime.execute(
            step, row["state"], idempotency_key=f"eval-{row.get('id', i)}"
        )
        expected = _normalize(row["label"])
        predicted = _normalize(outcome.value)
        if outcome.error is not None:
            error_count += 1
            if first_error is None:
                first_error = outcome.error
        cases.append(
            CaseResult(
                id=str(row.get("id", i)),
                predicted=predicted,
                expected=expected,
                correct=(predicted == expected) and outcome.error is None,
                confidence=outcome.confidence,
                routing=outcome.routing.value,
                latency_ms=outcome.latency_ms,
                cost_usd=outcome.cost_usd,
            )
        )

    latencies = [c.latency_ms for c in cases]
    routing_counts: dict[str, int] = {}
    for c in cases:
        routing_counts[c.routing] = routing_counts.get(c.routing, 0) + 1

    report = EvalReport(
        step_name=step.name,
        backend_name=getattr(backend, "name", "?"),
        n=len(cases),
        accuracy=sum(1 for c in cases if c.correct) / len(cases) if cases else 0.0,
        ece=expected_calibration_error(cases),
        latency_p50_ms=_percentile(latencies, 50),
        latency_p99_ms=_percentile(latencies, 99),
        total_cost_usd=sum(c.cost_usd for c in cases),
        cost_per_case_usd=(sum(c.cost_usd for c in cases) / len(cases)) if cases else 0.0,
        routing_counts=routing_counts,
        cases=cases,
        error_count=error_count,
        first_error=first_error,
    )
    if baseline is not None:
        base_step, base_backend = baseline
        base_report = run_eval(base_step, dataset, base_backend)
        report.baseline_accuracy = base_report.accuracy
    return report


def run_comparison(
    step: DecisionStep,
    dataset: list[dict],
    backend_names: list[str],
    *,
    baseline_accuracy: float | None = None,
) -> list[BackendComparison]:
    """Evaluate each backend; failures become 'unavailable', never a crash."""
    out: list[BackendComparison] = []
    for name in backend_names:
        try:
            backend = make_backend(name)
        except Exception as exc:  # noqa: BLE001 - surfaced as "unavailable"
            out.append(BackendComparison(
                backend=name, available=False,
                note=f"construction failed: {exc}"))
            continue
        try:
            report = run_eval(step, dataset, backend)
        except Exception as exc:  # noqa: BLE001
            out.append(BackendComparison(
                backend=name, available=False,
                note=f"eval failed: {exc}"))
            continue
        # A backend whose every call failed (fallback policy handled it)
        # produced no measurements -- report it as unavailable, not as a
        # zero-accuracy result.
        if report.n > 0 and report.error_count == report.n:
            out.append(BackendComparison(
                backend=name, available=False,
                note=f"all {report.n} calls failed: "
                     f"{(report.first_error or 'unknown error')[:120]}"))
            continue
        # Each run_eval builds a fresh TrustedRuntime (and audit log),
        # so idempotency keys don't collide across backends.
        report.baseline_accuracy = baseline_accuracy
        out.append(BackendComparison(backend=name, available=True, report=report))
    return out


def _short_routing(counts: dict[str, int]) -> str:
    short = {"auto_proceed": "auto", "escalate_to_llm": "esc",
             "human_review": "human"}
    return " ".join(f"{short.get(k, k)}:{v}" for k, v in sorted(counts.items()))


def print_comparison(comparisons: list[BackendComparison]) -> None:
    width = max([len(c.backend) for c in comparisons] + [8])
    print(f"{'backend':<{width}} {'status':<34} {'acc':>6} {'ECE':>6} "
          f"{'p50':>8} {'p99':>8} {'$/case':>10}  routing")
    print("-" * (width + 102))
    for c in comparisons:
        if not c.available or c.report is None:
            print(f"{c.backend:<{width}} unavailable: {c.note[:33]:<33}")
            continue
        r = c.report
        base = ""
        if r.baseline_accuracy is not None and r.accuracy_delta_vs_baseline is not None:
            base = f" (Δbase {r.accuracy_delta_vs_baseline:+.3f})"
        print(f"{c.backend:<{width}} {'ok':<34} {r.accuracy:>6.3f} {r.ece:>6.3f} "
              f"{r.latency_p50_ms:>7.1f}m {r.latency_p99_ms:>7.1f}m "
              f"${r.cost_per_case_usd:>9.6f}  {_short_routing(r.routing_counts)}{base}")
    missing = [c for c in comparisons if not c.available]
    if missing:
        print()
        for c in missing:
            print(f"  {c.backend}: unavailable -- {c.note}")


def print_report(report: EvalReport) -> None:
    print(f"step:    {report.step_name}  (backend: {report.backend_name})")
    print(f"cases:   {report.n}")
    print(f"accuracy: {report.accuracy:.3f}", end="")
    if report.baseline_accuracy is not None:
        print(
            f"  (baseline {report.baseline_accuracy:.3f}, "
            f"delta {report.accuracy_delta_vs_baseline:+.3f})"
        )
    else:
        print()
    print(f"ECE:      {report.ece:.3f}  (lower is better-calibrated)")
    print(f"latency:  p50 {report.latency_p50_ms:.1f}ms  p99 {report.latency_p99_ms:.1f}ms")
    print(f"cost:     ${report.total_cost_usd:.6f} total, "
          f"${report.cost_per_case_usd:.6f} / case")
    print(f"routing:  {report.routing_counts}")


def comparison_to_json(comparisons: list[BackendComparison],
                       baseline_accuracy: float | None) -> dict:
    rows = []
    for c in comparisons:
        row: dict = {"backend": c.backend, "available": c.available, "note": c.note}
        if c.available and c.report is not None:
            r = c.report
            row.update({
                "n": r.n, "accuracy": r.accuracy, "ece": r.ece,
                "latency_p50_ms": r.latency_p50_ms, "latency_p99_ms": r.latency_p99_ms,
                "total_cost_usd": r.total_cost_usd,
                "cost_per_case_usd": r.cost_per_case_usd,
                "routing_counts": r.routing_counts,
                "error_count": r.error_count,
                "baseline_accuracy": r.baseline_accuracy,
                "accuracy_delta_vs_baseline": r.accuracy_delta_vs_baseline,
            })
        rows.append(row)
    return {"backends": rows, "baseline_accuracy": baseline_accuracy}
