"""Deterministic mathematical easing curves (strictly finite, bounded overshoot)."""

from __future__ import annotations

import math


def clamp01(t: float) -> float:
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    return float(t)


def ease_out_cubic(t: float) -> float:
    u = clamp01(t)
    return 1.0 - (1.0 - u) ** 3


def ease_in_out_cubic(t: float) -> float:
    u = clamp01(t)
    if u < 0.5:
        return 4.0 * u * u * u
    return 1.0 - ((-2.0 * u + 2.0) ** 3) / 2.0


def ease_out_quint(t: float) -> float:
    u = clamp01(t)
    return 1.0 - (1.0 - u) ** 5


def smoothstep(t: float) -> float:
    u = clamp01(t)
    return u * u * (3.0 - 2.0 * u)


def bell_envelope(t: float, ramp_frac: float = 0.25) -> float:
    """Smooth rise (0->1), hold (1), smooth fall (1->0) using cubic easing."""
    u = clamp01(t)
    r = max(0.05, min(0.49, ramp_frac))
    if u < r:
        return ease_in_out_cubic(u / r)
    if u > 1.0 - r:
        return ease_in_out_cubic((1.0 - u) / r)
    return 1.0


def spring_damped(t: float, damping: float = 0.72, freq: float = 2.2) -> float:
    """Bounded under-damped harmonic spring reaching 1.0 smoothly without excessive oscillation."""
    u = clamp01(t)
    if u <= 0.0:
        return 0.0
    if u >= 1.0:
        return 1.0
    zeta = max(0.55, min(0.98, damping))
    omega = max(1.0, min(4.0, freq)) * math.pi
    wd = omega * math.sqrt(max(1e-6, 1.0 - zeta * zeta))
    env = math.exp(-zeta * omega * u)
    val = 1.0 - env * (math.cos(wd * u) + (zeta / math.sqrt(max(1e-6, 1.0 - zeta * zeta))) * math.sin(wd * u))
    # Smoothly pin final 5% to exact 1.0
    if u > 0.92:
        blend = (u - 0.92) / 0.08
        val = val * (1.0 - blend) + 1.0 * blend
    return max(0.0, min(1.12, val))


def evaluate_easing(name: str, t: float) -> float:
    if name == "ease_out_cubic":
        return ease_out_cubic(t)
    if name == "ease_in_out_cubic":
        return ease_in_out_cubic(t)
    if name == "ease_out_quint":
        return ease_out_quint(t)
    if name == "smoothstep":
        return smoothstep(t)
    if name == "spring_damped":
        return spring_damped(t)
    if name == "bell":
        return bell_envelope(t)
    return ease_in_out_cubic(t)
