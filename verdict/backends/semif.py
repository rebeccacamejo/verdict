"""SemIf backend adapter (Theodore Lee's open "semantic if" decision tool).

SemIf (https://github.com/TheoLeeCJ/SemIf, MIT; formerly "OpenJev") reads
typed option probabilities straight out of a frozen open model (best
reported: Qwen3.5-4B, 0.813 balanced accuracy) in a single forward pass --
no text generation, no JSON to repair. Independent project; not affiliated
with TypeSafe. Decisions are defined at runtime (criteria + options
arrive with the request), not baked into a fine-tune.

INTEGRATION STATUS: real request-shape builder + CLI driver, graceful
"unavailable" fallback. SemIf is a CLI-driven tool, not a Python client
library, and it needs real hardware: a CUDA GPU that holds a 4B BF16
model (published timings: one RTX 3090), or the MLX backend on Apple
Silicon, or the llamacpp CPU backend, or the WebGPU browser build --
plus a ~3GB weight download. Far too heavy to bundle or exercise in this
environment, so:

- ``evaluate()`` builds SemIf's documented JSONL input rows (verified
  format from the repo README -- see ``_build_rows``) and shells out to
  ``semif-score`` when it is on PATH.
- If the CLI is missing, ``evaluate()`` raises ``BackendError`` with
  install instructions; the eval harness reports this as "unavailable",
  never a crash.

What a real integration needs
------------------------------
1. ``pip install -e '.[test]'`` from the SemIf repo (add ``[mlx]`` on
   Apple Silicon or ``[llamacpp]`` for CPU-only), ``export
   HF_HOME=/path/to/large-drive/huggingface``, and let it fetch
   Qwen3.5-4B (pin the revision, e.g. the README's
   ``851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a``).
2. Run once and verify the result-JSONL key names, then tighten
   ``_parse_result_row`` (marked UNCERTAIN below).
3. Choose ``--mode direct`` (fresh scoring, the audited path) vs
   ``--mode shared`` (prefill state once; faster, but upstream marks the
   reuse paths experimental -- BF16 changed 5-6 of 777 argmaxes).
4. Fit SemIf's per-workload temperature on labeled data (upstream:
   raw ECE 0.068 -> 0.038 calibrated on authored decisions) and pass it
   as ``temperature=`` here. Upstream's own warning applies: "Returned
   probabilities are conditional on the supplied options. Calibrate and
   validate them on the workload where they will make decisions."

UNCERTAIN (not verified -- no GPU/weights in this environment)
--------------------------------------------------------------
- Exact key names in semif-score's result JSONL rows. The README says
  each result "contains typed option scores, timing, the exact model
  revision, and a prompt hash"; ``_parse_result_row`` tries
  documented-plausible keys defensively and fails loudly otherwise.
- Exact CLI flags beyond the documented ``--mode/--model/--revision/
  --input/--output`` (e.g. whether temperature can be passed through).
- The score/noul mappings below are OUR adapter's choice (SemIf is
  natively option-based): score levels become ordered options,
  noul becomes a true/false option pair.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Mapping

from verdict.runtime.errors import BackendError

from .base import (
    BackendAnswer,
    BackendQuestion,
    DecisionBackend,
    QuestionKind,
    TokenUsage,
)
from .calibration import apply_temperature

DEFAULT_MODEL = "Qwen/Qwen3.5-4B"
# Pinned revision from the SemIf README's quickstart (2026-09-18).
PINNED_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"

INSTALL_HINT = (
    "semif-score CLI not found. Install SemIf per its README "
    "(https://github.com/TheoLeeCJ/SemIf): pip install -e '.[test]' "
    "(add [mlx] on Apple Silicon, [llamacpp] for CPU-only), "
    "export HF_HOME=..., and let it fetch the model weights "
    "(~3GB download; needs a 4B-capable GPU, MLX, or llamacpp)."
)


class SemIfBackend(DecisionBackend):
    """Drives SemIf's `semif-score` CLI over a JSONL row per question."""

    name = "semif"

    def __init__(
        self,
        model: str | None = None,
        model_revision: str | None = None,
        mode: str = "direct",  # "direct" (audited) or "shared" (experimental reuse)
        temperature: float | None = None,  # post-hoc; fit per workload
        timeout_s: float = 600.0,  # model inference, not an API call
        cli: str = "semif-score",
    ) -> None:
        self.model = model or DEFAULT_MODEL
        self.model_revision = model_revision
        self.mode = mode
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.cli = cli

    # -- public API -----------------------------------------------------

    def evaluate(
        self, state: str, questions: Mapping[str, BackendQuestion]
    ) -> Mapping[str, BackendAnswer]:
        if shutil.which(self.cli) is None:
            raise BackendError(INSTALL_HINT)
        rows = self._build_rows(state, questions)
        t0 = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="semif-") as tmp:
            in_path = Path(tmp) / "input.jsonl"
            out_path = Path(tmp) / "output.jsonl"
            in_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            self._run_cli(in_path, out_path)
            results = self._read_results(out_path)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        out: dict[str, BackendAnswer] = {}
        for row_spec, result_row in zip(rows, results):
            out[row_spec["id"]] = self._parse_result_row(
                row_spec, result_row, questions[row_spec["id"]], latency_ms
            )
        return out

    # -- CLI ------------------------------------------------------------

    def _run_cli(self, in_path: Path, out_path: Path) -> None:
        cmd = [
            self.cli,
            "--mode", self.mode,
            "--model", self.model,
            "--input", str(in_path),
            "--output", str(out_path),
        ]
        if self.model_revision:
            cmd += ["--revision", self.model_revision]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.timeout_s
            )
        except Exception as exc:  # noqa: BLE001
            raise BackendError(f"semif-score execution failed: {exc}") from exc
        if proc.returncode != 0:
            raise BackendError(
                f"semif-score exited {proc.returncode}: "
                f"{proc.stderr.strip()[:500]}"
            )

    @staticmethod
    def _read_results(out_path: Path) -> list[dict]:
        try:
            text = out_path.read_text()
        except OSError as exc:
            raise BackendError(f"semif-score produced no output file: {exc}") from exc
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        if not rows:
            raise BackendError("semif-score produced an empty result set")
        return rows

    # -- request rows (VERIFIED input format from the SemIf README) ------

    def _build_rows(
        self, state: str, questions: Mapping[str, BackendQuestion]
    ) -> list[dict]:
        """One JSONL row per question: {"id","state","question","options"}.

        This shape is documented in the SemIf README (examples/decisions.jsonl).
        """
        rows = []
        for name, q in questions.items():
            if q.kind is QuestionKind.CHOICE:
                options = [
                    {"id": key, "description": desc}
                    for key, desc in (q.options or {}).items()
                ]
            elif q.kind is QuestionKind.SCORE:
                # OUR mapping (SemIf is natively option-based): ordered
                # levels become options with integer ids; the winning id
                # is the level index.
                options = [
                    {"id": str(i), "description": level}
                    for i, level in enumerate(q.levels or [])
                ]
            else:  # NOUL -> true/false option pair (OUR mapping)
                options = [
                    {"id": "true", "description": f"Yes: {q.instructions}"},
                    {"id": "false", "description": f"No: {q.instructions}"},
                ]
            rows.append({
                "id": name,
                "state": state,
                "question": q.instructions,
                "options": options,
            })
        return rows

    # -- result parsing (UNCERTAIN key names -- see module docstring) -----

    def _parse_result_row(
        self, row_spec: dict, result: dict, q: BackendQuestion, latency_ms: float
    ) -> BackendAnswer:
        probs = self._extract_probabilities(result, row_spec["id"])
        if self.temperature is not None:
            probs = apply_temperature(probs, self.temperature)
        total = sum(probs.values()) or 1.0
        probs = {k: v / total for k, v in probs.items()}

        model_version = str(
            result.get("model_revision")
            or result.get("revision")
            or result.get("model")
            or self.model
        )
        usage = TokenUsage()  # SemIf reports timings, not token counts

        if q.kind is QuestionKind.CHOICE:
            best = max(probs, key=probs.get)
            return BackendAnswer(
                question_name=row_spec["id"], kind=q.kind, value=best,
                probabilities=probs, confidence=probs[best],
                model_version=model_version, usage=usage,
                cost_usd=0.0, latency_ms=latency_ms,
            )
        if q.kind is QuestionKind.SCORE:
            best = max(probs, key=probs.get)
            return BackendAnswer(
                question_name=row_spec["id"], kind=q.kind, value=int(best),
                probabilities=probs, confidence=probs[best],
                model_version=model_version, usage=usage,
                cost_usd=0.0, latency_ms=latency_ms,
            )
        # NOUL
        p_true = probs.get("true", 0.0)
        return BackendAnswer(
            question_name=row_spec["id"], kind=q.kind, value=p_true,
            probabilities=None, confidence=None,  # runtime derives it
            model_version=model_version, usage=usage,
            cost_usd=0.0, latency_ms=latency_ms,
        )

    @staticmethod
    def _extract_probabilities(result: dict, row_id: str) -> dict[str, float]:
        """Find the option->probability mapping in a result row.

        UNCERTAIN: the README only promises "typed option scores" per
        row. We try documented-plausible keys and fail loudly so an
        operator can tighten this against a real run instead of silently
        misreading the output.
        """
        for key in ("probabilities", "option_probabilities", "scores", "option_scores"):
            raw = result.get(key)
            if isinstance(raw, Mapping) and raw:
                probs: dict[str, float] = {}
                for opt_id, v in raw.items():
                    if isinstance(v, Mapping):
                        # e.g. {"probability": 0.7, ...}
                        v = v.get("probability", v.get("p", v.get("score")))
                    try:
                        probs[str(opt_id)] = float(v)
                    except (TypeError, ValueError):
                        continue
                if probs:
                    return probs
        raise BackendError(
            f"semif-score result row for {row_id!r} has no recognizable "
            f"option-probability field (tried probabilities/option_probabilities/"
            f"scores/option_scores; row keys: {sorted(result.keys())}). "
            f"Inspect the JSONL output and tighten "
            f"SemIfBackend._extract_probabilities."
        )
