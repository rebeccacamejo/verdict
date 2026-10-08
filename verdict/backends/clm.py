"""ClmBackend -- CLM (Contrastive-LM, Apache-2.0): dual state/action encoders.

CLM (Contrastive-LM, Apache-2.0): dual state/action encoders
(frozen Qwen3-8B + 20M heads); candidates embed once and cache.
TypeSafe wire-compatible per community reports.

Caveat: the thinnest documentation in this registry -- Jev parity
and latency claims are vendor/community-reported, treat as
unverified until you run the eval harness against it.

UNCERTAIN (not verified against a live server in this environment)
----------------------------------------------------------------------
- The default port below is a placeholder to avoid colliding with other
  local servers; override ``base_url`` (or ``CLM_BASE_URL``) with your deploy.
- The model id string your server expects in the ``"model"`` field
  (``"clm"`` below is a guess; the server may ignore it).
"""

from __future__ import annotations

from .systemone import SystemOneHttpBackend

# UNCERTAIN: placeholder port; point at your actual server.
CLM_LOCAL_DEFAULT = "http://localhost:8019/v1/systemone"


class ClmBackend(SystemOneHttpBackend):
    """System One wire client configured for a (self-hosted) CLM (Contrastive-LM, Apache-2.0): dual state/action encoders server."""

    name = "clm"
    default_base_url = CLM_LOCAL_DEFAULT
    api_key_env = None  # keyless self-hosted
    default_model = "clm"  # UNCERTAIN: guess; the server may ignore it
    input_usd_per_mtok = 0.0  # self-hosted: no per-call charge
    _use_sdk = False  # raw HTTP only, by design
