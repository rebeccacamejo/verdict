"""ClefFlashBackend -- Cloudflare's 9B decision model (Qwen3.5-9B), Apache-2.0, weights at.

Cloudflare's 9B decision model (Qwen3.5-9B), Apache-2.0, weights at
``Cloudflare/clef-flash`` on HuggingFace. Same Jev-compatible wire
format as Clef (choice/score/noul, confidence, probabilities, usage)
plus image/video fields. The practical self-hosted Cloudflare option.

Run it:
- Local: ``open-jevlike-infer`` (vLLM + custom scoring head), or the
  repo's ``joint_schema_model.py``. Community quants run smaller:
  ``mlx-community/clef-flash-4bit`` (6.2GB, fits a 16GB Mac).
- Hosted: Workers AI ``@cf/cloudflare/clef-flash`` ($0.09/M input).

Caveats: full-context needs ~41GB VRAM (press figure, not the model
card); one independent 42-case test had it agreeing with Jev 83% but
trailing on nuanced triage.

UNCERTAIN (not verified against a live server in this environment)
----------------------------------------------------------------------
- The default port below is a placeholder to avoid colliding with other
  local servers; override ``base_url`` (or ``CLEF_FLASH_BASE_URL``) with your deploy.
- The model id string your server expects in the ``"model"`` field
  (``"clef-flash"`` below is a guess; the server may ignore it).
"""

from __future__ import annotations

from .systemone import SystemOneHttpBackend

# UNCERTAIN: placeholder port; point at your actual server.
CLEF_FLASH_LOCAL_DEFAULT = "http://localhost:8012/v1/systemone"


class ClefFlashBackend(SystemOneHttpBackend):
    """System One wire client configured for a (self-hosted) Cloudflare's 9B decision model (Qwen3.5-9B), Apache-2.0, weights at server."""

    name = "clef-flash"
    default_base_url = CLEF_FLASH_LOCAL_DEFAULT
    api_key_env = None  # keyless self-hosted
    default_model = "clef-flash"  # UNCERTAIN: guess; the server may ignore it
    input_usd_per_mtok = 0.0  # self-hosted: no per-call charge
    _use_sdk = False  # raw HTTP only, by design
