"""StrandsBackend -- AWS Strands Labs' 2B decision model (Qwen3.5-2B + rank-16 LoRA +.

AWS Strands Labs' 2B decision model (Qwen3.5-2B + rank-16 LoRA +
" Hobson" pointer head), Apache-2.0. Weights:
``StrandsAgents/strands-decider-2B-hobson-v19`` on HuggingFace --
training data and scripts are open too (first in class on that).

Run it: ``pip install strands-decider`` then ``strands-decider serve``
(binds 127.0.0.1, no auth -- add your own). 115ms median on RTX 3090,
~150ms on M3 MacBook, runs on laptop CPU.

Wire: exact ``POST /v1/systemone``, independently confirmed.

UNCERTAIN (not verified against a live server in this environment)
----------------------------------------------------------------------
- The default port below is a placeholder to avoid colliding with other
  local servers; override ``base_url`` (or ``STRANDS_BASE_URL``) with your deploy.
- The model id string your server expects in the ``"model"`` field
  (``"strands-decider-2b"`` below is a guess; the server may ignore it).
"""

from __future__ import annotations

from .systemone import SystemOneHttpBackend

# UNCERTAIN: placeholder port; point at your actual server.
STRANDS_LOCAL_DEFAULT = "http://localhost:8013/v1/systemone"


class StrandsBackend(SystemOneHttpBackend):
    """System One wire client configured for a (self-hosted) AWS Strands Labs' 2B decision model (Qwen3.5-2B + rank-16 LoRA + server."""

    name = "strands"
    default_base_url = STRANDS_LOCAL_DEFAULT
    api_key_env = None  # keyless self-hosted
    default_model = "strands-decider-2b"  # UNCERTAIN: guess; the server may ignore it
    input_usd_per_mtok = 0.0  # self-hosted: no per-call charge
    _use_sdk = False  # raw HTTP only, by design
