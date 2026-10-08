"""Jev backend adapter (TypeSafe's hosted System One decision model).

Jev (https://docs.typesafe.ai) is the reference System One implementation:
typed choice/score/noul decisions with calibrated probabilities, ~100ms,
no text generation. Hosted API, early access, proprietary -- the backend
to beat in the eval harness.

Usage:
    JevBackend()  # reads TYPESAFE_API_KEY

Pricing (vendor-published, 2026-09-23): $0.042 / 1M input tokens,
output unmetered.
"""

from __future__ import annotations

from verdict.runtime.errors import BackendError

from .systemone import (
    API_ENDPOINT,
    API_KEY_ENV,
    INPUT_USD_PER_MTOK,
    SystemOneHttpBackend,
)

# Re-exported for backwards compatibility (they live in .systemone now).
__all__ = ["API_ENDPOINT", "JevBackend", "SystemOneHttpBackend"]

class JevBackend(SystemOneHttpBackend):
    """Live adapter for the TypeSafe Jev decisions endpoint."""

    name = "jev"
    default_base_url = API_ENDPOINT
    api_key_env = API_KEY_ENV
    default_model = "jev-latest"  # documented default alias; pin e.g. "jev-1.13.0" for reproducibility
    input_usd_per_mtok = INPUT_USD_PER_MTOK
    _use_sdk = True

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "jev-latest",
        timeout_s: float = 15.0,
        base_url: str | None = None,
    ) -> None:
        # NOTE: pointing JevBackend at a non-TypeSafe base_url keeps the
        # API-key requirement and never uses the vendor SDK (see base).
        super().__init__(api_key=api_key, model=model, timeout_s=timeout_s, base_url=base_url)
