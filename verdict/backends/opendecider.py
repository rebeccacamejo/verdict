"""OpenDeciderBackend -- Manjunath Shiva's OpenDecider family (Apache-2.0): nano ~400M,.

Manjunath Shiva's OpenDecider family (Apache-2.0): nano ~400M,
small 4B (LoRA on Qwen3-4B-Instruct), small-td, medium 30B MoE.
Weights: ``manjunathshiva/opendecider-{nano,small,small-GGUF}``.

Run it: ``pip install opendecider`` then ``opendecider serve``; also
runnable via vLLM, Ollama, LM Studio, MLX -- the broadest runtime
matrix in this registry.

Wire: exact via ``opendecider serve``. Note: the Python ``score``
returns the most-likely level while HTTP follows Jev's expected-score
convention -- use HTTP for eval parity.

UNCERTAIN (not verified against a live server in this environment)
----------------------------------------------------------------------
- The default port below is a placeholder to avoid colliding with other
  local servers; override ``base_url`` (or ``OPENDECIDER_BASE_URL``) with your deploy.
- The model id string your server expects in the ``"model"`` field
  (``"opendecider-small"`` below is a guess; the server may ignore it).
"""

from __future__ import annotations

from .systemone import SystemOneHttpBackend

# UNCERTAIN: placeholder port; point at your actual server.
OPENDECIDER_LOCAL_DEFAULT = "http://localhost:8014/v1/systemone"


class OpenDeciderBackend(SystemOneHttpBackend):
    """System One wire client configured for a (self-hosted) Manjunath Shiva's OpenDecider family (Apache-2.0): nano ~400M, server."""

    name = "opendecider"
    default_base_url = OPENDECIDER_LOCAL_DEFAULT
    api_key_env = None  # keyless self-hosted
    default_model = "opendecider-small"  # UNCERTAIN: guess; the server may ignore it
    input_usd_per_mtok = 0.0  # self-hosted: no per-call charge
    _use_sdk = False  # raw HTTP only, by design
