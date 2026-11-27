"""Phase G4 — Unit/Smoke Verification of ALL 40 Distinct Effects (C01–C10, L01–L10, T01–T08, V01–V06, G01–G06).

Verifies for every single registered effect:
  1. Start, mid, and end frames render cleanly with exact shape, uint8 dtype, and finite values.
  2. No accidental all-black/corrupt frame.
  3. Measurable visual change vs unmodified control frame.
  4. Intensity == 0.0 yields exact 0.0 difference (true no-op control).
  5. Outputs across effects are distinct (not 40 aliases!).
"""

from __future__ import annotations

import hashlib

import cv2
import numpy as np
import pytest

from motionlab.effects.implementations import EFFECT_FUNCTIONS, FrameRenderContext, apply_effect_frame
from motionlab.registry import load_effects
from motionlab.schemas import EffectEvent


def _make_test_frames() -> tuple[np.ndarray, np.ndarray, FrameRenderContext]:
    w, h = 240, 426
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    shot_a = np.zeros((h, w, 3), dtype=np.uint8)
    shot_a[:, :, 0] = np.clip(45 + 140 * (xx / w), 0, 255).astype(np.uint8)
    shot_a[:, :, 1] = np.clip(65 + 120 * (yy / h), 0, 255).astype(np.uint8)
    shot_a[:, :, 2] = np.clip(110 + 90 * np.sin(xx * 0.03), 0, 255).astype(np.uint8)
    cv2.circle(shot_a, (w // 2, int(h * 0.42)), 52, (235, 198, 172), -1)
    cv2.rectangle(shot_a, (w // 2 - 44, int(h * 0.56)), (w // 2 + 44, int(h * 0.78)), (48, 82, 130), -1)
    cv2.circle(shot_a, (int(w * 0.22), int(h * 0.18)), 20, (250, 246, 232), -1)
    for gx in range(0, w, 28):
        cv2.line(shot_a, (gx, 0), (gx, h), (220, 220, 230), 1)

    shot_b = np.zeros((h, w, 3), dtype=np.uint8)
    shot_b[:, :, 0] = np.clip(160 - 95 * (yy / h), 0, 255).astype(np.uint8)
    shot_b[:, :, 1] = np.clip(95 + 100 * (xx / w), 0, 255).astype(np.uint8)
    shot_b[:, :, 2] = np.clip(55 + 145 * (yy / h), 0, 255).astype(np.uint8)
    cv2.rectangle(shot_b, (32, 90), (w - 32, h - 90), (242, 238, 226), 4)

    ctx = FrameRenderContext(
        width=w,
        height=h,
        fps=30.0,
        seed=42,
        audio_envelope_by_frame={f: 0.35 + 0.55 * abs(np.sin(f * 0.25)) for f in range(0, 40)},
        subject_track_by_frame={f: (0.50 + 0.05 * np.sin(f * 0.15), 0.42) for f in range(0, 40)},
        recent_frames={f: np.roll(shot_a, shift=4, axis=1) for f in range(0, 40)},
    )
    return shot_a, shot_b, ctx


ALL_SPECS = load_effects().all_specs()


@pytest.mark.parametrize("spec", ALL_SPECS, ids=[f"{s.code}_{s.id}" for s in ALL_SPECS])
def test_effect_renders_valid_visible_change_and_zero_intensity_noop(spec) -> None:
    assert spec.id in EFFECT_FUNCTIONS
    reg = load_effects()
    shot_a, shot_b, ctx = _make_test_frames()
    params = reg.validate_params(spec.id, dict(spec.default_params))
    ev = EffectEvent(
        id=f"fx_{spec.code.lower()}",
        effect_id=spec.id,
        effect_version=spec.version,
        engine="ffmpeg",
        start_frame=0,
        end_frame_exclusive=30,
        layer=spec.layer,
        region={"cx": 0.5, "cy": 0.42, "w": 1.0, "h": 1.0}
        if spec.family != "motion_graphics"
        else {"cx": 0.84, "cy": 0.86, "w": 0.24, "h": 0.18},
        parameters=params,
        easing="ease_in_out_cubic",
        priority=10,
        reason_codes=["unit_test"],
        evidence_refs=["fixture"],
    )
    ctx.cut_anchor_frames[ev.id] = shot_b.copy()

    # Render start, mid, and end frames
    outs = [apply_effect_frame(shot_a.copy(), f_idx, ev, ctx) for f_idx in (0, 15, 29)]
    for out in outs:
        assert out.shape == shot_a.shape
        assert out.dtype == np.uint8
        assert float(np.mean(out)) > 5.0, f"{spec.id} produced unexpected dark/black frame"

    # Verify visible difference at midpoint (either global MAD or localized crisp graphic delta)
    diff = np.abs(outs[1].astype(np.float32) - shot_a.astype(np.float32))
    mad = float(np.mean(diff))
    max_d = float(np.max(diff))
    changed_px = int(np.sum(np.max(diff, axis=2) >= 6.0))
    assert (mad >= 0.15) or (changed_px >= 60 and max_d >= 12.0), (
        f"{spec.id} had negligible visual delta (mad={mad:.4f}, changed_px={changed_px}, max_d={max_d:.2f})"
    )

    # Zero-intensity control MUST return identical frame
    zero_params = dict(params)
    zero_params["intensity"] = 0.0
    ev_zero = EffectEvent(**{**ev.to_dict(), "id": f"fx_{spec.code.lower()}_z", "parameters": zero_params})
    zero_out = apply_effect_frame(shot_a.copy(), 15, ev_zero, ctx)
    assert np.array_equal(zero_out, shot_a), f"{spec.id} intensity=0.0 was not an exact no-op!"


def test_all_40_effects_are_visually_distinct_not_aliases() -> None:
    """Ensure every one of the 40 registered effects produces a unique rendered midpoint signature."""
    reg = load_effects()
    shot_a, shot_b, ctx = _make_test_frames()
    hashes: dict[str, list[str]] = {}
    for spec in reg.all_specs():
        params = reg.validate_params(spec.id, dict(spec.default_params))
        ev = EffectEvent(
            id=f"uniq_{spec.code.lower()}",
            effect_id=spec.id,
            effect_version=spec.version,
            engine="ffmpeg",
            start_frame=0,
            end_frame_exclusive=30,
            layer=spec.layer,
            region={"cx": 0.5, "cy": 0.42, "w": 1.0, "h": 1.0}
            if spec.family != "motion_graphics"
            else {"cx": 0.84, "cy": 0.86, "w": 0.24, "h": 0.18},
            parameters=params,
            easing="ease_in_out_cubic",
            priority=10,
            reason_codes=["distinctness"],
            evidence_refs=["fixture"],
        )
        ctx.cut_anchor_frames[ev.id] = shot_b.copy()
        out = apply_effect_frame(shot_a.copy(), 15, ev, ctx)
        digest = hashlib.sha256(out.tobytes()).hexdigest()
        hashes.setdefault(digest, []).append(f"{spec.code}:{spec.id}")

    duplicates = {k: v for k, v in hashes.items() if len(v) > 1}
    assert not duplicates, f"Duplicate/alias effect outputs detected: {duplicates}"
    assert len(hashes) == 40
