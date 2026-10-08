"""Laya backend adapter (Convai Innovations' open System-1 decision model).

Laya (model weights: Apache 2.0, Hugging Face org ``convaiinnovations``;
checkpoints include a ~421M English ModernBERT-large and a ~322M
multilingual variant; ~33ms per forward pass, <1GB RAM) is an open
Jev-compatible decision model. Its request/response shape matches
TypeSafe Jev's ``system_one`` API -- verified from the
https://github.com/receptron/laya README ("The request/response shape is
the same as the Python reference implementation (``RLAgent.system_one``)
and as TypeSafe Jev's ``system_one`` API"). This backend therefore
reuses the shared ``SystemOneHttpBackend`` wire logic with a different
endpoint/config.

CRITICAL CAVEATS -- read before trusting this backend's probabilities
---------------------------------------------------------------------
- Laya is "a fast base to specialise, not a zero-shot decision engine":
  the model card reports zero-shot typed-decision accuracy near random
  (0.362). The 0.766 headline number comes from a checkpoint fine-tuned
  on that benchmark's own training split. EXPECT TO FINE-TUNE on labeled
  domain data before trusting its decisions.
- Calibration ships poor: mean ECE 0.466, reduced to 0.081 only after a
  per-question-type temperature refit. This backend therefore exposes
  ``temperatures={"choice": T, "score": T, "noul": T}`` and applies them
  post-hoc to returned probabilities (approximate -- a proper refit needs
  logits, which the HTTP shape does not expose; see
  backends/calibration.py). Fit T on YOUR labeled data; the defaults
  below are neutral (no scaling).

UNCERTAIN (not verified in this environment)
-------------------------------------------
- No hosted Laya HTTP endpoint was found. Known-good paths are local:
  the ``@receptron/laya`` Node/ONNX package, or the Python reference
  (``RLAgent.system_one``) from the Hugging Face checkpoint. The default
  ``base_url`` is a localhost placeholder -- self-host an HTTP wrapper
  around one of those and point ``LayaBackend`` at it.
- The ``"model"`` field value a self-hosted wrapper expects
  (``"laya"`` below is a guess).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping

from .base import BackendAnswer, QuestionKind
from .calibration import apply_temperature, apply_temperature_binary
from .systemone import SystemOneHttpBackend

# UNCERTAIN: placeholder -- no verified hosted endpoint; self-host and override.
LAYA_LOCAL_DEFAULT = "http://localhost:8000/v1/systemone"


class LayaBackend(SystemOneHttpBackend):
    """System One wire client configured for a (self-hosted) Laya server."""

    name = "laya"
    default_base_url = LAYA_LOCAL_DEFAULT
    api_key_env = None
    default_model = "laya"  # UNCERTAIN: guess; your wrapper may ignore it
    input_usd_per_mtok = 0.0  # self-hosted: no per-call charge
    _use_sdk = False  # raw HTTP only, by design

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout_s: float = 15.0,
        base_url: str | None = None,
        temperatures: Mapping[str, float] | None = None,
    ) -> None:
        # Per-question-type temperatures, e.g. {"choice": 1.4, "noul": 2.1}.
        # Fit on labeled data; absent entries mean "no scaling".
        self.temperatures = dict(temperatures or {})
        super().__init__(api_key=api_key, model=model, timeout_s=timeout_s,
                         base_url=base_url)

    def evaluate(self, state, questions):
        answers = super().evaluate(state, questions)
        if not self.temperatures:
            return answers
        out = {}
        for name, ans in answers.items():
            out[name] = self._rescale_answer(ans)
        return out

    def _rescale_answer(self, ans: BackendAnswer) -> BackendAnswer:
        t = self.temperatures.get(ans.kind.value)
        if t is None or t == 1.0:
            return ans
        if ans.kind is QuestionKind.NOUL:
            # Binary trick: scale P(true) against its complement. The
            # runtime derives noul confidence from the value, so it
            # follows automatically.
            return replace(ans, value=apply_temperature_binary(float(ans.value), t))
        if not ans.probabilities:
            return ans
        new_probs = apply_temperature(ans.probabilities, t)
        # Keep confidence consistent with the rescaled distribution.
        return replace(ans, probabilities=new_probs,
                       confidence=max(new_probs.values()))
