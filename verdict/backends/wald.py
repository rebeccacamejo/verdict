"""WaldBackend -- Wald-4B (org2ai, Apache-2.0): 4B decision model on Qwen3.5-4B-Base..

Wald-4B (org2ai, Apache-2.0): 4B decision model on Qwen3.5-4B-Base.
Weights: ``org2ai/Wald-4B`` on HuggingFace (v1.2 is the robustness
release -- holds answers under distracting/fake instructions).

Run it self-hosted; serves Jev-compatible ``POST /v1/systemone``.

UNCERTAIN (not verified against a live server in this environment)
----------------------------------------------------------------------
- The default port below is a placeholder to avoid colliding with other
  local servers; override ``base_url`` (or ``WALD_BASE_URL``) with your deploy.
- The model id string your server expects in the ``"model"`` field
  (``"wald-4b"`` below is a guess; the server may ignore it).
"""

from __future__ import annotations

from .systemone import SystemOneHttpBackend

# UNCERTAIN: placeholder port; point at your actual server.
WALD_LOCAL_DEFAULT = "http://localhost:8015/v1/systemone"


class WaldBackend(SystemOneHttpBackend):
    """System One wire client configured for a (self-hosted) Wald-4B (org2ai, Apache-2.0): 4B decision model on Qwen3.5-4B-Base. server."""

    name = "wald"
    default_base_url = WALD_LOCAL_DEFAULT
    api_key_env = None  # keyless self-hosted
    default_model = "wald-4b"  # UNCERTAIN: guess; the server may ignore it
    input_usd_per_mtok = 0.0  # self-hosted: no per-call charge
    _use_sdk = False  # raw HTTP only, by design
