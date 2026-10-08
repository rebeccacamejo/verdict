"""Shared System One wire logic for Jev-protocol HTTP decision servers.

This module defines ``SystemOneHttpBackend``: the request/response handling
for any server speaking TypeSafe's System One protocol
(``POST /v1/systemone`` with ``{"state", "model", "questions"}``).

Subclasses configure endpoint / auth / model / pricing; the payload shape
and response parsing are inherited unchanged. Backend-specific adapters
(``jev.py``, ``kev.py``, ``laya.py``, ...) are thin configurations of
this class.

Wire format (verified against https://docs.typesafe.ai, 2026-09-23):
- Request:  {"state": str, "model": str,
              "questions": {name: {"type": "choice"|"score"|"noul",
                                   "instructions": str, "criteria": {...} | [...]}}}
- Response: {"model": str,
              "answers": {name: {"type": ..., "choice": key, "confidence": f,
                                 "probabilities": {key: f}}},   # choice
              "usage": {"input_tokens": int, "output_tokens": int}}

Pricing: set ``input_usd_per_mtok`` per subclass (0.0 = self-hosted/unmetered).
"""


from __future__ import annotations

import json
import os
import time
import urllib.request
from typing import Mapping

from verdict.runtime.errors import BackendError

from .base import (
    BackendAnswer,
    BackendQuestion,
    DecisionBackend,
    QuestionKind,
    TokenUsage,
)

API_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
API_KEY_ENV = "TYPESAFE_API_KEY"

# Vendor-published pricing (docs.typesafe.ai, 2026-09-23).
INPUT_USD_PER_MTOK = 0.042


def _get(obj, name, default=None):
    """Defensive accessor: works for SDK response objects AND plain dicts."""
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


class SystemOneHttpBackend(DecisionBackend):
    """Shared wire logic for System One-protocol HTTP decision servers.

    Subclasses set class attributes; everything else (payload shape,
    response parsing, cost accounting) is inherited:

    - ``default_base_url``: POST target (includes the path, e.g.
      ``.../v1/systemone``).
    - ``api_key_env``: env var for the bearer token, or None for keyless
      (self-hosted) servers.
    - ``default_model``: value sent in the ``"model"`` request field.
    - ``input_usd_per_mtok``: 0.0 for self-hosted (unmetered) servers.
    - ``_use_sdk``: only True for TypeSafe's own endpoint. The vendor SDK
      is never pointed at a third-party/self-hosted URL: besides being
      unverified, that would risk sending case data to the wrong party.
    """

    name = "systemone-http"
    default_base_url: str = API_ENDPOINT
    api_key_env: str | None = None
    default_model: str = "systemone-latest"
    input_usd_per_mtok: float = 0.0
    _use_sdk: bool = False

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout_s: float = 15.0,
        base_url: str | None = None,
    ) -> None:
        self.base_url = base_url or self.default_base_url
        self.model = model or self.default_model
        self.timeout_s = timeout_s
        self.api_key = api_key or (
            os.environ.get(self.api_key_env) if self.api_key_env else None
        )
        if self._requires_api_key() and not self.api_key:
            hint = (f"pass api_key= or set {self.api_key_env}."
                    if self.api_key_env else "pass api_key=.")
            raise BackendError(
                f"{type(self).__name__} needs an API key for {self.base_url}: {hint}"
            )
        self._sdk = self._try_import_sdk() if self._use_sdk else None

    def _requires_api_key(self) -> bool:
        return bool(self.api_key_env)

    @staticmethod
    def _try_import_sdk():
        try:
            import typesafe_sdk  # type: ignore

            return typesafe_sdk
        except ImportError:
            return None

    # -- public API -----------------------------------------------------

    def evaluate(
        self, state: str, questions: Mapping[str, BackendQuestion]
    ) -> Mapping[str, BackendAnswer]:
        if self._sdk is not None:
            return self._evaluate_via_sdk(state, questions)
        return self._evaluate_via_http(state, questions)

    # -- via typesafe_sdk (TypeSafe endpoint only) -----------------------

    def _sdk_question(self, q: BackendQuestion):
        sdk = self._sdk
        if q.kind is QuestionKind.CHOICE:
            return sdk.Choice(instructions=q.instructions, criteria=dict(q.options or {}))
        if q.kind is QuestionKind.SCORE:
            return sdk.Score(instructions=q.instructions, criteria=list(q.levels or []))
        return sdk.Noul(instructions=q.instructions)

    def _evaluate_via_sdk(self, state, questions):
        client = self._sdk.TypeSafeClient()
        t0 = time.perf_counter()
        try:
            # UNCERTAIN: whether the SDK client accepts a per-call model override;
            # the documented default is jev-latest, matching our default.
            resp = client.system_one(
                state=state,
                questions={n: self._sdk_question(q) for n, q in questions.items()},
            )
        except Exception as exc:  # noqa: BLE001 - wrap vendor errors uniformly
            raise BackendError(f"System One SDK call failed: {exc}") from exc
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return self._parse_response(resp, questions, latency_ms)

    # -- via raw HTTPS (stdlib fallback; the self-hosted path) -----------

    def _http_payload(self, state, questions) -> dict:
        payload_questions = {}
        for name, q in questions.items():
            entry: dict = {"type": q.kind.value, "instructions": q.instructions}
            if q.kind is QuestionKind.CHOICE:
                entry["criteria"] = dict(q.options or {})
            elif q.kind is QuestionKind.SCORE:
                entry["criteria"] = list(q.levels or [])
            payload_questions[name] = entry
        return {"state": state, "model": self.model, "questions": payload_questions}

    def _evaluate_via_http(self, state, questions):
        body = json.dumps(self._http_payload(state, questions)).encode()
        req = urllib.request.Request(
            self.base_url,
            data=body,
            headers={"Content-Type": "application/json"}
            | ({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}),
            method="POST",
        )
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as http_resp:
                raw = json.loads(http_resp.read().decode())
        except Exception as exc:  # noqa: BLE001
            raise BackendError(
                f"System One HTTP call to {self.base_url} failed: {exc}"
            ) from exc
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return self._parse_response(raw, questions, latency_ms)

    # -- response parsing -------------------------------------------------

    def _parse_response(self, resp, questions, latency_ms) -> dict[str, BackendAnswer]:
        model_version = str(_get(resp, "model", self.model))
        usage = _get(resp, "usage", {}) or {}
        input_tokens = int(_get(usage, "input_tokens", 0) or 0)
        output_tokens = int(_get(usage, "output_tokens", 0) or 0)
        cost_usd = input_tokens * self.input_usd_per_mtok / 1_000_000

        raw_answers = _get(resp, "answers", {}) or {}
        out: dict[str, BackendAnswer] = {}
        for name, q in questions.items():
            raw = _get(raw_answers, name)
            if raw is None:
                raise BackendError(
                    f"System One response missing answer for question {name!r}"
                )
            out[name] = self._parse_answer(
                name, q, raw, model_version, input_tokens, output_tokens, cost_usd, latency_ms
            )
        return out

    def _parse_answer(
        self, name, q: BackendQuestion, raw, model_version,
        input_tokens, output_tokens, cost_usd, latency_ms,
    ) -> BackendAnswer:
        usage = TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens)
        if q.kind is QuestionKind.CHOICE:
            return BackendAnswer(
                question_name=name, kind=q.kind,
                value=_get(raw, "choice"),
                probabilities=dict(_get(raw, "probabilities", {}) or {}),
                confidence=_get(raw, "confidence"),
                model_version=model_version, usage=usage,
                cost_usd=cost_usd, latency_ms=latency_ms,
            )
        if q.kind is QuestionKind.SCORE:
            # UNCERTAIN: docs show float scores (e.g. 1.0); coerce to level index.
            score_raw = _get(raw, "score")
            return BackendAnswer(
                question_name=name, kind=q.kind,
                value=int(round(float(score_raw))),
                probabilities={str(k): float(v) for k, v in
                               ((_get(raw, "probabilities", {}) or {}).items())},
                confidence=_get(raw, "confidence"),
                model_version=model_version, usage=usage,
                cost_usd=cost_usd, latency_ms=latency_ms,
            )
        # NOUL
        return BackendAnswer(
            question_name=name, kind=q.kind,
            value=float(_get(raw, "noul")),
            probabilities=None,
            confidence=None,  # runtime derives effective confidence from noul
            model_version=model_version, usage=usage,
            cost_usd=cost_usd, latency_ms=latency_ms,
        )


