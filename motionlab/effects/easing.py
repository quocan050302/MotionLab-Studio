"""Deterministic bounded easing curves and smooth attack/release envelopes."""

from __future__ import annotations

import math


def clamp01(t: float) -> float:
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    return float(t)


def ease_out_cubic(t: float) -> float:
    x = clamp01(t)
    return 1.0 - (1.0 - x) ** 3


def ease_in_out_cubic(t: float) -> float:
    x = clamp01(t)
    if x < 0.5:
        return 4.0 * x * x * x
    return 1.0 - ((-2.0 * x + 2.0) ** 3) / 2.0


def spring_damped(t: float) -> float:
    """Bounded damped spring settling cleanly to 1.0 without oscillation overload."""
    x = clamp01(t)
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0
    # Slight controlled overshoot (~1.06 max) settling smoothly to 1.0
    val = 1.0 - math.exp(-5.8 * x) * math.cos(7.8 * x)
    return min(1.08, max(0.0, val))


def linear_bounded(t: float) -> float:
    return clamp01(t)


def eval_easing(name: str, t: float) -> float:
    if name == "ease_out_cubic":
        return ease_out_cubic(t)
    if name == "ease_in_out_cubic":
        return ease_in_out_cubic(t)
    if name == "spring_damped":
        return spring_damped(t)
    if name == "linear_bounded":
        return linear_bounded(t)
    return ease_in_out_cubic(t)


def smooth_envelope(t: float, attack: float = 0.2, release: float = 0.2) -> float:
    """Smooth 0 -> 1 -> 0 cubic envelope so effects enter/exit seamlessly without pops."""
    x = clamp01(t)
    a = max(0.02, min(0.49, attack))
    r = max(0.02, min(0.49, release))
    if x < a:
        return ease_out_cubic(x / a)
    if x > 1.0 - r:
        return ease_out_cubic((1.0 - x) / r)
    return 1.0


def bell_curve(t: float) -> float:
    """Smooth 0 -> 1 -> 0 bell peaking cleanly at t=0.5 (for transitions & pulses)."""
    x = clamp01(t)
    return math.sin(math.pi * x) ** 2
