"""MapikaBackend -- Mapika's Decider family (Apache-2.0): 2B / 4B / 35B-A3B..

Mapika's Decider family (Apache-2.0): 2B / 4B / 35B-A3B.
Weights: ``Mapika/decider-2b`` (etc.) on HuggingFace.

Run it: ``pip install git+https://github.com/Mapika/decider`` then
``scripts/serve.sh``. CUDA graph capture for low latency.

Wire: exact (``TYPESAFE_BASE_URL`` works); confidence follows the
TypeSafe definition since 1.3.0.

UNCERTAIN (not verified against a live server in this environment)
----------------------------------------------------------------------
- The default port below is a placeholder to avoid colliding with other
  local servers; override ``base_url`` (or ``MAPIKA_BASE_URL``) with your deploy.
- The model id string your server expects in the ``"model"`` field
  (``"decider-2b"`` below is a guess; the server may ignore it).
"""

from __future__ import annotations

from .systemone import SystemOneHttpBackend

# UNCERTAIN: placeholder port; point at your actual server.
MAPIKA_LOCAL_DEFAULT = "http://localhost:8018/v1/systemone"


class MapikaBackend(SystemOneHttpBackend):
    """System One wire client configured for a (self-hosted) Mapika's Decider family (Apache-2.0): 2B / 4B / 35B-A3B. server."""

    name = "mapika"
    default_base_url = MAPIKA_LOCAL_DEFAULT
    api_key_env = None  # keyless self-hosted
    default_model = "decider-2b"  # UNCERTAIN: guess; the server may ignore it
    input_usd_per_mtok = 0.0  # self-hosted: no per-call charge
    _use_sdk = False  # raw HTTP only, by design
