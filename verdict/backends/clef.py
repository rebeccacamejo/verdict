"""ClefBackend -- Cloudflare's 27B decision model (post-trained from Qwen3.5-27B),.

Cloudflare's 27B decision model (post-trained from Qwen3.5-27B),
Apache-2.0, weights at ``Cloudflare/clef`` on HuggingFace. Keeps the
backbone vision encoder (up to 4 images/request, plus video) and 64k
context. Wire-compatible with Jev, plus ``images``/``videos`` fields.

Run it:
- Local: the repo's ``joint_schema_model.py`` (PyTorch + transformers;
  tested on torch 2.11 / transformers 5.10.2, H200 rig -- not a stock
  transformers architecture, needs the repo's code on the path), or the
  community ``open-jevlike-infer`` vLLM server.
- Hosted: Workers AI ``@cf/cloudflare/clef`` ($0.24/M input tokens,
  output free).

Caveats: heavy (~85GB VRAM at full context per press coverage of
Cloudflare's guidance); Cloudflare's benchmarks are self-reported.

UNCERTAIN (not verified against a live server in this environment)
----------------------------------------------------------------------
- The default port below is a placeholder to avoid colliding with other
  local servers; override ``base_url`` (or ``CLEF_BASE_URL``) with your deploy.
- The model id string your server expects in the ``"model"`` field
  (``"clef"`` below is a guess; the server may ignore it).
"""

from __future__ import annotations

from .systemone import SystemOneHttpBackend

# UNCERTAIN: placeholder port; point at your actual server.
CLEF_LOCAL_DEFAULT = "http://localhost:8011/v1/systemone"


class ClefBackend(SystemOneHttpBackend):
    """System One wire client configured for a (self-hosted) Cloudflare's 27B decision model (post-trained from Qwen3.5-27B), server."""

    name = "clef"
    default_base_url = CLEF_LOCAL_DEFAULT
    api_key_env = None  # keyless self-hosted
    default_model = "clef"  # UNCERTAIN: guess; the server may ignore it
    input_usd_per_mtok = 0.0  # self-hosted: no per-call charge
    _use_sdk = False  # raw HTTP only, by design
