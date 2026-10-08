"""The ``verdict`` command line.

Quick tour::

    verdict quickstart                              # scaffold + real model + sample decision

Or step by step::

    verdict init                                    # scaffold verdict.yaml + sample labels
    verdict up von                                  # install + start server + wire YAML
    verdict doctor                                  # what's reachable right now
    verdict decide -f verdict.yaml --decision dispute_reason_code --state "..."
    verdict eval -f verdict.yaml --dataset eval/labels.jsonl --baseline

Or skip the file and go inline::

    verdict backends
    verdict decide --backend kev --kind choice --instructions "..." \\
        --options '{"10.4": "Duplicate processing"}' --state "..."
    verdict eval --dataset labels.jsonl --backends auto   # every reachable backend
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import verdict
from verdict import (
    Boolean,
    Choice,
    Score,
    ThresholdPolicy,
    get_backend,
    list_backends,
    load_file,
)

NEXT = {
    "backends": 'verdict decide --backend <name> --kind choice --instructions "..." '
                "--options '{...}' --state \"...\"",
    "decide": "verdict eval --dataset <labels.jsonl> --backends <backend> --baseline",
    "decide_file": "verdict eval -f verdict.yaml --dataset <labels.jsonl>",
    "eval": "pin the winner per decision in verdict.yaml (policy.backend), then: "
            "verdict validate -f verdict.yaml",
    "init": "verdict validate -f verdict.yaml",
    "validate": 'verdict decide -f verdict.yaml --decision <name> --state "..."',
    "up": 'verdict decide --backend <name> --kind choice --instructions "..." '
          "--options '{...}' --state \"...\"   # no URL flags needed anymore",
    "down": "verdict ps",
    "ps": "verdict up <backend>",
    "doctor": "verdict up <backend>   # start what's missing",
    "quickstart": "verdict eval -f verdict.yaml --dataset eval/labels.jsonl --baseline",
}


def _next(key: str) -> None:
    print(f"\n→ next: {NEXT[key]}")


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="verdict",
        description="Typed deterministic decision steps: declare a question, "
        "get a calibrated answer with confidence routing.",
    )
    sub = ap.add_subparsers(dest="command", required=True)

    # -- verdict init ----------------------------------------------------
    i = sub.add_parser("init", help="Scaffold a verdict.yaml + sample labels.")
    i.add_argument("-o", "--output", default="verdict.yaml",
                   help="Where to write the scaffold (default: verdict.yaml).")

    # -- verdict validate -------------------------------------------------
    v = sub.add_parser("validate",
                       help="Validate a verdict.yaml without running anything.")
    v.add_argument("-f", "--file", required=True, help="Path to verdict.yaml.")

    # -- verdict decide ---------------------------------------------------
    d = sub.add_parser("decide", help="Execute one decision step.")
    src = d.add_mutually_exclusive_group()
    src.add_argument("-f", "--file", default=None,
                     help="verdict.yaml with declared decisions.")
    src.add_argument("--backend", default=None,
                     help="Backend name for inline mode (default: mock). "
                          "See `verdict backends`.")
    d.add_argument("--decision", default=None,
                   help="Decision name from the file (with --file).")
    d.add_argument("--kind", choices=["choice", "score", "boolean"], default=None)
    d.add_argument("--instructions", default=None)
    d.add_argument("--options",
                   help="JSON object for --kind choice: {key: description}.")
    d.add_argument("--levels",
                   help="JSON array for --kind score: [level descriptions].")
    d.add_argument("--state", required=True, help="The case text to decide about.")
    d.add_argument("--name", default="decision", help="Step name for audit records.")
    d.add_argument("--auto", type=float, default=0.85,
                   help="Auto-proceed confidence threshold (default: 0.85).")
    d.add_argument("--review", type=float, default=0.60,
                   help="LLM-escalation confidence threshold (default: 0.60).")
    d.add_argument("--idempotency-key", default=None)
    d.add_argument("--backend-opt", action="append", default=[],
                   metavar="KEY=VALUE",
                   help="Extra backend constructor kwarg (repeatable), "
                        "e.g. --backend-opt base_url=http://localhost:8000/v1/systemone")
    d.add_argument("--json", action="store_true", help="Emit the outcome as JSON.")

    # -- verdict backends --------------------------------------------------
    sub.add_parser("backends", help="List available decision backends.")

    # -- verdict up / down / ps -------------------------------------------
    u = sub.add_parser("up", help="Install + start a backend server, wire verdict.yaml.")
    u.add_argument("backend", help="Backend name (e.g. von, kev, laya).")
    u.add_argument("--no-install", action="store_true",
                   help="Skip install; just launch and wire.")
    u.add_argument("--no-wire", action="store_true",
                   help="Don't touch verdict.yaml.")
    u.add_argument("-f", "--file", default="verdict.yaml",
                   help="verdict.yaml to wire (default: verdict.yaml).")

    dn = sub.add_parser("down", help="Stop a `verdict up`-managed server.")
    dn.add_argument("backend", help="Backend name.")

    sub.add_parser("ps", help="Show `verdict up`-managed servers.")

    # -- verdict doctor -----------------------------------------------------
    sub.add_parser("doctor", help="Probe every backend: what's reachable right now.")

    # -- verdict quickstart --------------------------------------------------
    q = sub.add_parser("quickstart",
                       help="One command: scaffold + start a real model + sample decision.")
    q.add_argument("--backend", default="von",
                   help="Backend to start (default: von, the tiny CPU one).")
    q.add_argument("-o", "--output", default="verdict.yaml",
                   help="Where to write the scaffold (default: verdict.yaml).")

    # -- verdict eval ------------------------------------------------------
    e = sub.add_parser("eval", help="Evaluate backends head-to-head on labeled data.")
    e.add_argument("--dataset", required=True, help="JSONL: {id, state, label} rows.")
    esrc = e.add_mutually_exclusive_group()
    esrc.add_argument("--backends", default=None,
                      help="Comma-separated backend names, or 'auto' for every "
                           "reachable backend (default: mock).")
    esrc.add_argument("-f", "--file", default=None,
                      help="verdict.yaml: eval each decision with its pinned backend.")
    e.add_argument("--decision", default=None,
                   help="With --file: only eval this decision.")
    e.add_argument("--baseline", action="store_true",
                   help="Also run a weak majority-class baseline.")
    e.add_argument("--json", action="store_true", help="Emit the report as JSON.")
    return ap


def _parse_backend_opts(pairs: list[str]) -> dict:
    out: dict = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--backend-opt expects KEY=VALUE, got {pair!r}")
        key, value = pair.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def _print_outcome(outcome, as_json: bool) -> None:
    if as_json:
        print(json.dumps({
            "value": outcome.value,
            "probabilities": dict(outcome.probabilities),
            "confidence": outcome.confidence,
            "routing": outcome.routing.value,
            "model_version": outcome.model_version,
            "latency_ms": outcome.latency_ms,
            "cost_usd": outcome.cost_usd,
            "audit_id": outcome.audit_id,
            "error": outcome.error,
        }, indent=2))
        return
    print(f"value:      {outcome.value}")
    print(f"confidence: {outcome.confidence:.3f}")
    print(f"routing:    {outcome.routing.value}")
    if outcome.probabilities:
        top = sorted(outcome.probabilities.items(),
                     key=lambda kv: kv[1], reverse=True)[:5]
        print("probs:      " + ", ".join(f"{k}={v:.3f}" for k, v in top))
    print(f"backend:    {outcome.model_version} "
          f"({outcome.latency_ms:.0f}ms, ${outcome.cost_usd:.6f})")
    if outcome.error:
        print(f"note:       fallback engaged -- {outcome.error}")


def _cmd_init(args: argparse.Namespace) -> int:
    out = Path(args.output)
    if out.exists():
        raise SystemExit(f"{out} already exists -- delete it or use -o <other.yaml>")
    out.write_text(_SCAFFOLD_YAML)
    labels = out.parent / "eval" / "labels.jsonl"
    labels.parent.mkdir(parents=True, exist_ok=True)
    if not labels.exists():
        src = Path(__file__).parent / "eval" / "data" / "dispute_reason_codes.jsonl"
        rows = src.read_text().splitlines()[:6]
        labels.write_text("\n".join(rows) + "\n")
    print(f"wrote {out}")
    print(f"wrote {labels} (6 sample rows)")
    _next("init")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    from verdict.declarative import validate_data, _load_yaml, _load_json

    path = Path(args.file)
    data = _load_yaml(path) if path.suffix in (".yaml", ".yml") else _load_json(path)
    errors = validate_data(data, source=str(path))
    if errors:
        print(f"{path}: INVALID")
        for e in errors:
            print(f"  - {e}")
        return 1
    names = [d["name"] for d in data["decisions"]]
    print(f"{path}: valid -- {len(names)} decision(s): {', '.join(names)}")
    _next("validate")
    return 0


def _cmd_decide(args: argparse.Namespace) -> int:
    if args.file:
        if not args.decision:
            raise SystemExit("with --file you also need --decision <name>")
        try:
            plan = load_file(args.file)
            outcome = plan.decide(args.decision, args.state,
                                  idempotency_key=args.idempotency_key)
        except Exception as exc:  # noqa: BLE001 -- CLI surfaces config errors plainly
            print(f"verdict: error: {exc}", file=sys.stderr)
            return 1
        _print_outcome(outcome, args.json)
        _next("decide_file")
        return 0

    if not args.kind:
        raise SystemExit("need --kind choice|score|boolean (or use --file)")
    if not args.instructions:
        raise SystemExit("--kind mode needs --instructions")
    if args.kind == "choice":
        if not args.options:
            raise SystemExit("--kind choice requires --options '{key: description}'")
        try:
            options = json.loads(args.options)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"--options is not valid JSON: {exc}")
        question: Choice | Score | Boolean = Choice(
            instructions=args.instructions, options=options
        )
    elif args.kind == "score":
        if not args.levels:
            raise SystemExit("--kind score requires --levels '[...]'")
        try:
            levels = json.loads(args.levels)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"--levels is not valid JSON: {exc}")
        question = Score(instructions=args.instructions, levels=levels)
    else:
        question = Boolean(instructions=args.instructions)

    policy = ThresholdPolicy(auto_threshold=args.auto, review_threshold=args.review)
    try:
        outcome = verdict.decide(
            args.state,
            question,
            backend=args.backend or "mock",
            policy=policy,
            name=args.name,
            idempotency_key=args.idempotency_key,
            **_parse_backend_opts(args.backend_opt),
        )
    except Exception as exc:  # noqa: BLE001 -- CLI surfaces backend/config errors plainly
        print(f"verdict: error: {exc}", file=sys.stderr)
        return 1
    _print_outcome(outcome, args.json)
    _next("decide")
    return 0


def _cmd_backends(args: argparse.Namespace) -> int:
    backends = list_backends()
    print(f"{'name':<12} {'title':<38} {'auth'}")
    print("-" * 80)
    for name, info in backends.items():
        print(f"{name:<12} {info.title:<38} {info.auth}")
    print("\nPick one, then run it:")
    _next("backends")
    return 0


def _cmd_up(args: argparse.Namespace) -> int:
    from verdict.servers import RECIPES, ServerManager, wire_yaml

    backend = args.backend
    if backend == "mock":
        print("mock is in-process -- nothing to start.")
        return 0
    recipe = RECIPES.get(backend)
    if recipe is None:
        raise SystemExit(
            f"unknown backend {backend!r}; choose from {sorted(RECIPES)}")
    if not recipe.auto:
        print(f"no automatic setup for {backend}:")
        print(recipe.manual)
        return 1
    print(f"verdict up {backend}")
    mgr = ServerManager(backend)
    try:
        url = mgr.start(no_install=args.no_install)
    except RuntimeError as exc:
        print(f"verdict: error: {exc}", file=sys.stderr)
        return 1
    print(f"ready: {url}")
    print(f"logs:  {mgr.log_file}")
    if recipe.notes:
        print(f"note:  {recipe.notes}")
    if not args.no_wire:
        if Path(args.file).exists():
            wire_yaml(Path(args.file), backend, url)
            print(f"wired: {args.file} -> backends.{backend}.base_url = {url}")
        else:
            print(f"no {args.file} here -- run `verdict init` first, or pass -f.")
    _next("up")
    return 0


def _cmd_down(args: argparse.Namespace) -> int:
    from verdict.servers import ServerManager

    mgr = ServerManager(args.backend)
    if mgr.stop():
        print(f"stopped {args.backend}")
    else:
        print(f"{args.backend}: not running")
    _next("down")
    return 0


def _cmd_ps(args: argparse.Namespace) -> int:
    import json
    import os
    from verdict.servers import home, probe

    rows = []
    for d in sorted(home().iterdir()):
        f = d / "server.json"
        if not f.exists():
            continue
        try:
            info = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        try:
            os.kill(info["pid"], 0)
            alive = True
        except OSError:
            alive = False
        reachable = probe(info["url"]) if alive else False
        rows.append((d.name, info["pid"], info["url"],
                     "up" if reachable else ("stale" if alive else "dead")))
    if not rows:
        print("no managed servers -- `verdict up <backend>` to start one")
    else:
        print(f"{'backend':<12} {'pid':<8} {'url':<45} {'status'}")
        print("-" * 80)
        for name, pid, url, status in rows:
            print(f"{name:<12} {pid:<8} {url:<45} {status}")
    _next("ps")
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    import os
    from verdict.servers import RECIPES, managed_url, probe, resolve_url

    print(f"{'backend':<12} {'url':<52} {'status'}")
    print("-" * 90)
    for name in list_backends():
        if name == "mock":
            print(f"{name:<12} {'in-process':<52} ok")
            continue
        if name == "jev":
            key = os.environ.get("TYPESAFE_API_KEY")
            print(f"{name:<12} {'https://api.typesafe.ai/v1/systemone':<52} "
                  f"{'key set' if key else 'needs TYPESAFE_API_KEY'}")
            continue
        if name == "semif":
            import shutil
            has_cli = shutil.which("semif-score") is not None
            print(f"{name:<12} {'local harness':<52} "
                  f"{'semif-score found' if has_cli else 'needs semif-score CLI'}")
            continue
        url = resolve_url(name)
        recipe = RECIPES.get(name)
        if url and probe(url):
            status = "reachable"
        elif managed_url(name):
            status = "managed but not answering (check logs)"
        elif recipe and recipe.auto:
            status = f"not running -- `verdict up {name}`"
        else:
            status = "manual setup -- `verdict up " + name + "` for steps"
        print(f"{name:<12} {(url or '-'): <52} {status}")
    _next("doctor")
    return 0


def _cmd_quickstart(args: argparse.Namespace) -> int:
    out = Path(args.output)
    if not out.exists():
        rc = main(["init", "-o", str(out)])
        if rc != 0:
            return rc
    rc = main(["up", args.backend, "-f", str(out)])
    if rc != 0:
        return rc
    from verdict.servers import set_policy_backend

    n = set_policy_backend(out, args.backend)
    print(f"pinned {n} decision(s) to backend: {args.backend}")
    plan = load_file(str(out))
    first = plan.decision_names()[0]
    print(f"\nsample decision: {first}")
    outcome = plan.decide(first, "I was charged twice for the same order.")
    _print_outcome(outcome, False)
    _next("quickstart")
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from verdict.eval.harness import (
        load_jsonl,
        print_comparison,
        comparison_to_json,
        run_comparison,
        run_eval,
    )
    from verdict.eval.presets import reason_code_step, weak_resolver
    from verdict.backends import MockBackend

    dataset = load_jsonl(args.dataset)

    if args.file:
        from verdict.declarative import DecisionPlan  # noqa: F401 -- re-export check
        plan = load_file(args.file)
        names = [args.decision] if args.decision else plan.decision_names()
        unknown = [n for n in names if n not in plan.decisions]
        if unknown:
            raise SystemExit(f"unknown decision(s): {unknown}")
        all_rows = []
        for dname in names:
            compiled = plan.decisions[dname]
            print(f"== {dname} (backend: {compiled.backend_name}) ==")
            comps = run_comparison(compiled.step, dataset, [compiled.backend_name])
            if args.json:
                all_rows.append({dname: comparison_to_json(comps, None)})
            else:
                print_comparison(comps)
                print()
        if args.json:
            print(json.dumps(all_rows, indent=2))
        _next("eval")
        return 0

    if (args.backends or "mock") == "auto":
        from verdict.servers import probe, resolve_url

        names = ["mock"] + [
            n for n in list_backends()
            if n not in ("mock", "jev", "semif")
            and (u := resolve_url(n)) and probe(u)
        ]
        print(f"auto: evaluating reachable backends: {', '.join(names)}")
    else:
        names = [n.strip() for n in (args.backends or "mock").split(",") if n.strip()]
    step = reason_code_step()

    baseline_accuracy = None
    if args.baseline:
        base_report = run_eval(
            reason_code_step(), dataset,
            MockBackend(weak_resolver, model_version="mock-baseline"),
        )
        baseline_accuracy = base_report.accuracy

    comparisons = run_comparison(step, dataset, names,
                                 baseline_accuracy=baseline_accuracy)
    if args.json:
        print(json.dumps(comparison_to_json(comparisons, baseline_accuracy), indent=2))
    else:
        print_comparison(comparisons)
    _next("eval")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "init":
        return _cmd_init(args)
    if args.command == "validate":
        return _cmd_validate(args)
    if args.command == "decide":
        return _cmd_decide(args)
    if args.command == "backends":
        return _cmd_backends(args)
    if args.command == "up":
        return _cmd_up(args)
    if args.command == "down":
        return _cmd_down(args)
    if args.command == "ps":
        return _cmd_ps(args)
    if args.command == "doctor":
        return _cmd_doctor(args)
    if args.command == "quickstart":
        return _cmd_quickstart(args)
    if args.command == "eval":
        return _cmd_eval(args)
    raise AssertionError(f"unhandled command {args.command!r}")


_SCAFFOLD_YAML = """\
# verdict.yaml -- declarative decision definitions.
# Validate:  verdict validate -f verdict.yaml
# Run one:   verdict decide -f verdict.yaml --decision dispute_reason_code --state "..."
# Eval all:  verdict eval -f verdict.yaml --dataset eval/labels.jsonl --baseline
version: 1

backends:
  # Uncomment and point at your server to use a real model:
  # kev:
  #   base_url: http://localhost:8000/v1/systemone

decisions:
  - name: dispute_reason_code
    kind: choice
    description: "Classify a dispute into a reason code."
    instructions: "Given the cardholder's dispute statement, choose the single reason code that best describes the claim."
    options:
      unauthorized: "Cardholder states they did not make or authorize the charge."
      not_received: "Merchandise or service paid for was never received."
      defective: "Merchandise arrived damaged, defective, or not as described."
      duplicate: "Cardholder was charged more than once for the same transaction."
      subscription: "Recurring charge continued after the subscription was cancelled."
      atm: "ATM did not dispense the requested cash."
    policy:
      backend: mock
      auto_threshold: 0.85
      review_threshold: 0.60
      on_backend_error: human_review

  - name: evidence_completeness
    kind: score
    description: "Gate case progression on evidence sufficiency."
    instructions: "Score how complete the attached evidence is."
    levels:
      - "No usable evidence attached."
      - "Weak evidence: cardholder statement only."
      - "Partial evidence: one supporting document."
      - "Good evidence: multiple supporting documents."
      - "Conclusive evidence: documents fully prove the claim."
    policy:
      backend: mock
      auto_threshold: 0.80
      review_threshold: 0.55
      on_backend_error: human_review
"""


if __name__ == "__main__":
    sys.exit(main())
