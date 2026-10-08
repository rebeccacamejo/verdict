"""Post-hoc temperature scaling for decision probabilities.

Upstream projects (SemIf, Laya) both ship per-workload temperature
calibration: raw option probabilities are often miscalibrated, and a
fitted temperature T applied as softmax(logits / T) pulls confidence
toward observed accuracy. The HTTP decision APIs expose probabilities,
not logits, so we invert approximately (logits ~= log p) and re-softmax.

This is a legitimate approximation for fit-on-your-own-data workflows --
it is NOT a substitute for fitting T on labeled data, and it cannot fix
a model whose argmax is wrong (temperature never changes the top choice).
"""

from __future__ import annotations

import math
from typing import Mapping


def apply_temperature(
    probabilities: Mapping[str, float], temperature: float
) -> dict[str, float]:
    """Rescale a probability distribution with temperature T.

    T < 1 sharpens (more confident), T > 1 flattens (less confident),
    T == 1 is the identity. Output always sums to 1.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature!r}")
    if temperature == 1.0:
        return dict(probabilities)
    logits = [math.log(max(float(p), 1e-12)) for p in probabilities.values()]
    scaled = [lp / temperature for lp in logits]
    m = max(scaled)
    exps = [math.exp(s - m) for s in scaled]
    total = sum(exps)
    return {k: e / total for k, e in zip(probabilities.keys(), exps)}


def apply_temperature_binary(p_true: float, temperature: float) -> float:
    """Temperature-scale a single P(true) via its binary complement."""
    scaled = apply_temperature({"true": p_true, "false": 1.0 - p_true}, temperature)
    return scaled["true"]
