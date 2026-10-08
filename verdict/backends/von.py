"""VonBackend -- Von (wfzyx, Apache-2.0): 395M decision model (ModernBERT-large +.

Von (wfzyx, Apache-2.0): 395M decision model (ModernBERT-large +
Option-Marker head), 8k context. Weights: ``wfzyx/von`` (v1.3).

Run it: ``pip install von-sdk`` then ``von serve --port 8000``.
CPU (OpenVINO) / CUDA / ROCm / MPS.

Wire: exact ``POST /v1/systemone`` server. The tiny-but-mighty
option for CPU-only POCs.

UNCERTAIN (not verified against a live server in this environment)
----------------------------------------------------------------------
- The default port below is a placeholder to avoid colliding with other
  local servers; override ``base_url`` (or ``VON_BASE_URL``) with your deploy.
- The model id string your server expects in the ``"model"`` field
  (``"von"`` below is a guess; the server may ignore it).
"""

from __future__ import annotations

from .systemone import SystemOneHttpBackend

# UNCERTAIN: placeholder port; point at your actual server.
VON_LOCAL_DEFAULT = "http://localhost:8017/v1/systemone"


class VonBackend(SystemOneHttpBackend):
    """System One wire client configured for a (self-hosted) Von (wfzyx, Apache-2.0): 395M decision model (ModernBERT-large + server."""

    name = "von"
    default_base_url = VON_LOCAL_DEFAULT
    api_key_env = None  # keyless self-hosted
    default_model = "von"  # UNCERTAIN: guess; the server may ignore it
    input_usd_per_mtok = 0.0  # self-hosted: no per-call charge
    _use_sdk = False  # raw HTTP only, by design
