"""OpenJevBackend -- OpenJev-4B (Junhao Shen, MIT): 4B decision model (Qwen3.5-4B.

OpenJev-4B (Junhao Shen, MIT): 4B decision model (Qwen3.5-4B
backbone + option-set decision head, non-autoregressive).
Code: ``github.com/ejhshen/OpenJev``.

Run it: the repo's local server speaks ``POST /v1/systemone``.

UNCERTAIN (not verified against a live server in this environment)
----------------------------------------------------------------------
- The default port below is a placeholder to avoid colliding with other
  local servers; override ``base_url`` (or ``OPENJEV_BASE_URL``) with your deploy.
- The model id string your server expects in the ``"model"`` field
  (``"openjev-4b"`` below is a guess; the server may ignore it).
"""

from __future__ import annotations

from .systemone import SystemOneHttpBackend

# UNCERTAIN: placeholder port; point at your actual server.
OPENJEV_LOCAL_DEFAULT = "http://localhost:8016/v1/systemone"


class OpenJevBackend(SystemOneHttpBackend):
    """System One wire client configured for a (self-hosted) OpenJev-4B (Junhao Shen, MIT): 4B decision model (Qwen3.5-4B server."""

    name = "openjev"
    default_base_url = OPENJEV_LOCAL_DEFAULT
    api_key_env = None  # keyless self-hosted
    default_model = "openjev-4b"  # UNCERTAIN: guess; the server may ignore it
    input_usd_per_mtok = 0.0  # self-hosted: no per-call charge
    _use_sdk = False  # raw HTTP only, by design
