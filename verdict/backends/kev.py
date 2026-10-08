"""Kev backend adapter (Jared Palmer's open System One decision models).

Kev (https://github.com/jaredpalmer/kev) is a family of small decision
models (0.8B / 4B / 9B) built on Qwen3.5, released under Apache 2.0 with
training code and evaluation data published.

Key property: Kev speaks TypeSafe's System One wire protocol exactly --
its FastAPI server implements ``POST /v1/systemone`` with the same
request/response shape -- so this backend is a thin configuration of the
shared ``SystemOneHttpBackend`` wire logic with a different base URL.
No TypeSafe API key is involved when pointed at a local server, and no
case data leaves your infrastructure.

Usage:
    # Against TypeSafe's endpoint (default; same wire protocol, needs key):
    KevBackend()
    # Against a local Kev server (self-hosted, keyless):
    KevBackend(base_url="http://localhost:8000/v1/systemone")

UNCERTAIN (not verified against a live Kev server in this environment)
----------------------------------------------------------------------
- The local server's default host/port. Kev's repo ships ``kev/serve.py``
  (FastAPI, ``POST /v1/systemone`` -- this path IS documented); the
  uvicorn default ``http://localhost:8000`` is assumed below.
- The exact ``kev serve`` CLI entrypoint / module invocation.
- The model id string a Kev server expects in the ``"model"`` field
  (``"kev-latest"`` below is a guess; the server may ignore it).
- Whether the TypeSafe Python SDK accepts a base-url override pointing
  at Kev (Kev's README says the SDK "can point at your local server",
  but the exact client config was not verified). This backend therefore
  uses raw HTTP only -- deliberately, never the vendor SDK: besides
  being unverified, SDK use would risk sending case data to the wrong
  party if misconfigured.
"""

from __future__ import annotations

from .systemone import API_ENDPOINT, SystemOneHttpBackend

# UNCERTAIN: uvicorn default host/port assumed; verify against your deploy.
KEV_LOCAL_DEFAULT = "http://localhost:8000/v1/systemone"


class KevBackend(SystemOneHttpBackend):
    """System One wire client configured for a Kev server."""

    name = "kev"
    default_base_url = API_ENDPOINT  # TypeSafe's; override to point at a local Kev server
    api_key_env = None  # keyless unless talking to TypeSafe's endpoint (see below)
    default_model = "kev-latest"  # UNCERTAIN: guess; the server may ignore it
    input_usd_per_mtok = 0.0  # self-hosted: no per-call charge
    _use_sdk = False  # raw HTTP only, by design (see module docstring)

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout_s: float = 15.0,
        base_url: str | None = None,
    ) -> None:
        resolved = base_url or self.default_base_url
        # Only TypeSafe's hosted endpoint needs an API key; a local Kev
        # server is keyless.
        self._key_required = resolved.rstrip("/") == API_ENDPOINT.rstrip("/")
        super().__init__(api_key=api_key, model=model, timeout_s=timeout_s,
                         base_url=resolved)

    def _requires_api_key(self) -> bool:
        return self._key_required
