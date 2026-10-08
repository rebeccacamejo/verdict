"""Backend registry: every decision backend behind one name.

``get_backend("kev")`` is the programmatic equivalent of
``verdict decide --backend kev`` and ``verdict eval --backends kev``.
New backends register by adding one entry to ``BACKENDS`` -- the CLI,
the eval harness, and ``verdict.list_backends()`` pick it up with no
further changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from verdict.runtime.errors import BackendError

from .base import (
    BackendAnswer,
    BackendQuestion,
    DecisionBackend,
    QuestionKind,
    TokenUsage,
)
from .calibration import apply_temperature, apply_temperature_binary
from .clef import ClefBackend
from .clef_flash import ClefFlashBackend
from .clm import ClmBackend
from .jev import JevBackend
from .kev import KevBackend
from .laya import LayaBackend
from .mapika import MapikaBackend
from .mock import MockBackend
from .opendecider import OpenDeciderBackend
from .openjev import OpenJevBackend
from .semif import SemIfBackend
from .strands import StrandsBackend
from .systemone import SystemOneHttpBackend
from .von import VonBackend
from .wald import WaldBackend


@dataclass(frozen=True)
class BackendInfo:
    """Human-facing metadata for one registered backend."""

    name: str
    title: str
    description: str
    hosting: str
    license: str
    auth: str
    env: tuple[str, ...] = ()
    note: str = ""


BACKENDS: dict[str, BackendInfo] = {
    "jev": BackendInfo(
        name="jev",
        title="Jev (TypeSafe, hosted)",
        description="The reference System One implementation. Typed choice/score/noul "
        "decisions with calibrated probabilities, ~100ms, no text generation.",
        hosting="Hosted API (api.typesafe.ai), early access",
        license="Proprietary",
        auth="TYPESAFE_API_KEY",
        env=("TYPESAFE_API_KEY",),
        note="The backend to beat. PII-bearing data needs a data-sharing agreement.",
    ),
    "kev": BackendInfo(
        name="kev",
        title="Kev (Jared Palmer, self-hosted)",
        description="Open Qwen3.5-based decision models (0.8B / 4B / 9B / 27B) speaking "
        "the System One wire protocol exactly. Honest evals with hashes and "
        "paired bootstraps in the repo. No data leaves your infrastructure.",
        hosting="Self-hosted (`kev serve`), keyless; MLX backend for Apple silicon",
        license="Apache 2.0",
        auth="None for local servers",
        env=("KEV_BASE_URL",),
        note="Default base URL is TypeSafe's endpoint; point KEV_BASE_URL at your "
        "local server for the self-hosted path. 27B is data-center class; "
        "0.8B runs on a laptop.",
    ),
    "semif": BackendInfo(
        name="semif",
        title="SemIf (Theodore Lee, research harness)",
        description="MIT-licensed System One research harness (frozen Qwen3.5-4B), "
        "not a serving model: it reads logits over lettered options via the "
        "`semif-score` CLI. The reference implementation others fork from.",
        hosting="Local harness, GPU/MLX, ~3GB weights",
        license="MIT (code; weights/datasets have separate terms)",
        auth="None (local weights)",
        env=("SEMIF_MODEL",),
        note="Great for study and evals; not a drop-in server for a platform "
        "capability without building your own serving layer around it.",
    ),
    "laya": BackendInfo(
        name="laya",
        title="Laya (Convai Innovations, self-hosted)",
        description="ModernBERT-based decision models: laya (421M English), "
        "laya-multilingual (322M, 100+ langs, 8k ctx), laya-typed-decisions "
        "(fine-tuned, 0.766 on the typed-decisions benchmark).",
        hosting="Self-hosted (`pip install \"laya[serve]\"` -> laya-serve)",
        license="Apache 2.0",
        auth="None (local weights)",
        env=("LAYA_BASE_URL",),
        note="Zero-shot works (0.783 on MASSIVE-intent EN) but fine-tuning on "
        "your domain is where accuracy jumps; free Kaggle notebook provided. "
        "Weak on 20+ option sets; English checkpoint collapses on non-Latin "
        "scripts (use the multilingual checkpoint).",
    ),
    "mock": BackendInfo(
        name="mock",
        title="Mock (offline test double)",
        description="Scriptable handler-driven backend for development, examples, "
        "and tests. No network, no keys, cost always zero.",
        hosting="In-process",
        license="Apache 2.0 (this repo)",
        auth="None",
    ),
    "clef": BackendInfo(
        name="clef",
        title="Clef (Cloudflare, 27B)",
        description="Cloudflare's 27B decision model (Qwen3.5-27B), multimodal "
        "(images/video in), 64k context. Jev wire-compatible plus media fields.",
        hosting="Self-hosted (heavy) or Workers AI @cf/cloudflare/clef",
        license="Apache 2.0",
        auth="None (local) / Cloudflare API token (hosted)",
        env=("CLEF_BASE_URL",),
        note="~85GB VRAM at full context; local needs the repo's custom model "
        "code or open-jevlike-infer. Cloudflare's benchmarks are self-reported.",
    ),
    "clef-flash": BackendInfo(
        name="clef-flash",
        title="Clef-flash (Cloudflare, 9B)",
        description="Cloudflare's 9B decision model (Qwen3.5-9B). Same Jev-compatible "
        "wire format as Clef. The practical self-hosted Cloudflare option.",
        hosting="Self-hosted or Workers AI @cf/cloudflare/clef-flash ($0.09/M in)",
        license="Apache 2.0",
        auth="None (local) / Cloudflare API token (hosted)",
        env=("CLEF_FLASH_BASE_URL",),
        note="4-bit MLX quant (6.2GB) runs on a 16GB Mac. One independent test: "
        "83% agreement with Jev, trailing on nuanced triage.",
    ),
    "strands": BackendInfo(
        name="strands",
        title="Strands Decider 2B (AWS Strands Labs)",
        description="2B decision model (Qwen3.5-2B + LoRA + pointer head). Training "
        "data and scripts open. 115ms on RTX 3090, runs on laptop CPU.",
        hosting="Self-hosted (`strands-decider serve`, localhost)",
        license="Apache 2.0",
        auth="None (local)",
        env=("STRANDS_BASE_URL",),
    ),
    "opendecider": BackendInfo(
        name="opendecider",
        title="OpenDecider (Manjunath Shiva)",
        description="Decision model family: nano 400M / small 4B / 30B MoE. Broadest "
        "runtime matrix here: vLLM, Ollama, LM Studio, MLX.",
        hosting="Self-hosted (`opendecider serve`)",
        license="Apache 2.0",
        auth="None (local)",
        env=("OPENDECIDER_BASE_URL",),
        note="Use the HTTP server (not the Python API) for Jev parity on scores.",
    ),
    "wald": BackendInfo(
        name="wald",
        title="Wald-4B (org2ai)",
        description="4B decision model (Qwen3.5-4B-Base). v1.2 robustness release "
        "holds answers under distracting/fake instructions.",
        hosting="Self-hosted",
        license="Apache 2.0",
        auth="None (local)",
        env=("WALD_BASE_URL",),
    ),
    "openjev": BackendInfo(
        name="openjev",
        title="OpenJev-4B (Junhao Shen)",
        description="4B decision model (Qwen3.5-4B + option-set head, "
        "non-autoregressive). Local /v1/systemone server.",
        hosting="Self-hosted",
        license="MIT",
        auth="None (local)",
        env=("OPENJEV_BASE_URL",),
    ),
    "von": BackendInfo(
        name="von",
        title="Von (wfzyx, 395M)",
        description="395M ModernBERT decision model, 8k context. The tiny-but-mighty "
        "CPU-only option.",
        hosting="Self-hosted (`von serve --port 8000`)",
        license="Apache 2.0",
        auth="None (local)",
        env=("VON_BASE_URL",),
    ),
    "mapika": BackendInfo(
        name="mapika",
        title="Decider (Mapika, 2B/4B/35B)",
        description="Decider family: 2B / 4B / 35B-A3B MoE. CUDA graph capture "
        "for low latency.",
        hosting="Self-hosted (`scripts/serve.sh`)",
        license="Apache 2.0",
        auth="None (local)",
        env=("MAPIKA_BASE_URL",),
    ),
    "clm": BackendInfo(
        name="clm",
        title="CLM (Contrastive-LM)",
        description="Dual state/action encoders (frozen Qwen3-8B + 20M heads); "
        "candidates embed once and cache. Wire-compatible per community reports.",
        hosting="Self-hosted",
        license="Apache 2.0",
        auth="None (local)",
        env=("CLM_BASE_URL",),
        note="Thinnest documentation in this registry -- run the eval harness "
        "before trusting it.",
    ),
}

_BACKEND_CLASSES: dict[str, type[DecisionBackend]] = {
    "jev": JevBackend,
    "kev": KevBackend,
    "semif": SemIfBackend,
    "laya": LayaBackend,
    "mock": MockBackend,
    "clef": ClefBackend,
    "clef-flash": ClefFlashBackend,
    "strands": StrandsBackend,
    "opendecider": OpenDeciderBackend,
    "wald": WaldBackend,
    "openjev": OpenJevBackend,
    "von": VonBackend,
    "mapika": MapikaBackend,
    "clm": ClmBackend,
}


def list_backends() -> dict[str, BackendInfo]:
    """All registered backends, keyed by short name."""
    return dict(BACKENDS)


def get_backend(name: str, **kwargs) -> DecisionBackend:
    """Construct a backend by short name.

    ``kwargs`` are forwarded to the backend constructor (e.g.
    ``get_backend("kev", base_url="http://localhost:8000/v1/systemone")``).
    Environment-variable overrides from the registry are honored unless
    the same kwarg is passed explicitly.
    """
    if name not in _BACKEND_CLASSES:
        raise BackendError(
            f"unknown backend {name!r}; choose from {sorted(_BACKEND_CLASSES)}"
        )
    info = BACKENDS[name]
    for env_var in info.env:
        if env_var in os.environ:
            kwarg = _ENV_TO_KWARG.get(env_var)
            if kwarg and kwarg not in kwargs:
                kwargs[kwarg] = os.environ[env_var]
    # Automatic discovery: a server started by `verdict up` is picked up
    # with no flags, no env vars, no YAML edits.
    if "base_url" not in kwargs:
        from verdict.servers import RECIPES, managed_url

        recipe = RECIPES.get(name)
        if recipe is not None and recipe.auto:
            url = managed_url(name)
            if url:
                kwargs["base_url"] = url
    return _BACKEND_CLASSES[name](**kwargs)


_ENV_TO_KWARG = {
    "KEV_BASE_URL": "base_url",
    "LAYA_BASE_URL": "base_url",
    "SEMIF_MODEL": "model",
    "CLEF_BASE_URL": "base_url",
    "CLEF_FLASH_BASE_URL": "base_url",
    "STRANDS_BASE_URL": "base_url",
    "OPENDECIDER_BASE_URL": "base_url",
    "WALD_BASE_URL": "base_url",
    "OPENJEV_BASE_URL": "base_url",
    "VON_BASE_URL": "base_url",
    "MAPIKA_BASE_URL": "base_url",
    "CLM_BASE_URL": "base_url",
}


__all__ = [
    "BACKENDS",
    "BackendAnswer",
    "BackendInfo",
    "BackendQuestion",
    "ClefBackend",
    "ClefFlashBackend",
    "ClmBackend",
    "DecisionBackend",
    "JevBackend",
    "KevBackend",
    "LayaBackend",
    "MapikaBackend",
    "MockBackend",
    "OpenDeciderBackend",
    "OpenJevBackend",
    "QuestionKind",
    "SemIfBackend",
    "StrandsBackend",
    "SystemOneHttpBackend",
    "TokenUsage",
    "VonBackend",
    "WaldBackend",
    "apply_temperature",
    "apply_temperature_binary",
    "get_backend",
    "list_backends",
]
