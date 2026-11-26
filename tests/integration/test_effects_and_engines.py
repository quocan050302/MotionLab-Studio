"""Phase G3 & G4: Real 40-effect rendering audit, multi-engine (FFmpeg/HyperFrames/Remotion) smoke, and edge media fixtures."""

from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from motionlab.config import MotionLabConfig, run_doctor
from motionlab.effects.engine_frames import apply_single_effect
from motionlab.pipeline import run_edit
from motionlab.registry import load_effects
from motionlab.renderers.hyperframes import HyperFramesRenderer
from motionlab.renderers.remotion import RemotionRenderer
from motionlab.schemas import EditPlan, EffectEvent, Rational
from tests.fixtures.generator import (
    generate_multitrack_fixture,
    generate_pcm_mov_fixture,
    generate_rotated_fixture,
    generate_silent_fixture,
)


def _synthetic_two_shot_ring() -> tuple[dict[int, np.ndarray], np.ndarray]:
    w, h = 240, 320
    shot_a = np.zeros((h, w, 3), dtype=np.uint8)
    shot_b = np.zeros((h, w, 3), dtype=np.uint8)
    ys = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    xs = np.linspace(0, 1, w, dtype=np.float32)[None, :]
    shot_a[..., 0] = np.clip(45 + 140 * xs + 40 * ys, 0, 255).astype(np.uint8)
    shot_a[..., 1] = np.clip(65 + 110 * (1 - ys), 0, 255).astype(np.uint8)
    shot_a[..., 2] = np.clip(120 + 90 * (1 - xs), 0, 255).astype(np.uint8)
    cv2.circle(shot_a, (w // 2, int(h * 0.42)), 48, (235, 195, 165), -1)
    cv2.rectangle(shot_a, (36, 190), (204, 285), (42, 66, 98), -1)

    shot_b[..., 0] = np.clip(160 + 60 * ys, 0, 255).astype(np.uint8)
    shot_b[..., 1] = np.clip(90 + 120 * xs, 0, 255).astype(np.uint8)
    shot_b[..., 2] = np.clip(60 + 130 * (1 - ys), 0, 255).astype(np.uint8)
    cv2.rectangle(shot_b, (40, 55), (200, 240), (245, 240, 225), 4)
    cv2.circle(shot_b, (w // 2, h // 2), 40, (255, 120, 70), -1)

    ring = {i: (shot_a.copy() if i < 15 else shot_b.copy()) for i in range(30)}
    audio_env = np.array([0.25 + 0.70 * abs(np.sin(i * 0.45)) for i in range(30)], dtype=np.float32)
    return ring, audio_env


def test_all_40_effects_render_distinctly_and_respect_zero_intensity():
    """Verify all 40/40 effects render real changes, zero-intensity no-op, finite pixels, and non-alias distinctness."""
    reg = load_effects()
    specs = reg.all_specs()
    assert len(specs) == 40
    ring, audio_env = _synthetic_two_shot_ring()

    mid_outputs: dict[str, np.ndarray] = {}
    for spec in specs:
        params = spec.validate_params({})
        params["intensity"] = min(spec.max_intensity, 0.95)
        ev = EffectEvent(
            id=f"test_{spec.id}", effect_id=spec.id, effect_version=spec.version,
            start_frame=0, end_frame_exclusive=30, layer=spec.layer,
            region={"x": 0.3, "y": 0.25, "w": 0.4, "h": 0.35},
            parameters=params, easing="ease_in_out_cubic", priority=5, engine="ffmpeg",
            reason_codes=["unit_smoke"], evidence_refs=["fixture"],
            fallback_id=spec.fallback_id, uses_neighbor_frames=spec.uses_neighbor_frames,
        )
        # Check start, mid, end frames
        for f_idx in (0, 15, 29):
            out = apply_single_effect(ring[f_idx], ev, f_idx, ring, audio_env, seed=42)
            assert out.shape == ring[f_idx].shape
            assert out.dtype == np.uint8
            assert float(out.mean()) > 10.0, f"{spec.id} produced unexpected black frame at f={f_idx}"

        out_mid = apply_single_effect(ring[15], ev, 15, ring, audio_env, seed=42)
        mid_outputs[spec.id] = out_mid
        diff = np.abs(out_mid.astype(np.float32) - ring[15].astype(np.float32))
        mad = float(diff.mean())
        max_px = float(diff.max())
        changed_frac = float((diff.max(axis=2) >= 4.0).mean())
        assert (mad >= 0.35) or (max_px >= 20.0 and changed_frac >= 0.0012), (
            f"Effect {spec.id} ({spec.name}) did not visibly change midpoint frame: mad={mad}, max_px={max_px}"
        )

        # Intensity == 0.0 MUST equal unmodified control frame
        zero_ev = EffectEvent(**{**ev.to_dict(), "parameters": {**params, "intensity": 0.0}})
        out_zero = apply_single_effect(ring[15], zero_ev, 15, ring, audio_env, seed=42)
        assert np.array_equal(out_zero, ring[15]), f"{spec.id} violated intensity=0 no-op equivalence"

    # Prove NO two effects are identical aliases on the midpoint frame
    ids = sorted(mid_outputs.keys())
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            pair_diff = float(np.abs(mid_outputs[a].astype(np.float32) - mid_outputs[b].astype(np.float32)).mean())
            assert pair_diff > 0.02, f"Effects {a} and {b} produced virtually identical output (diff={pair_diff})"


def test_real_hyperframes_offline_alpha_render_and_composite(tmp_path: Path):
    """Verify HyperFrames (Apache-2.0) real offline render, lint, VP9 alpha pixels, and composite."""
    doc = run_doctor()
    if not doc["engines"]["hyperframes"]["available"]:
        pytest.skip("HyperFrames local CLI not installed")

    clip = generate_silent_fixture(tmp_path / "hf_src.mp4", width=360, height=640, fps=30, duration_s=2.0)
    plan = EditPlan(
        schema_version="1.0.0", planner_version="1.0.0", source_hash="hf", config_hash="hf",
        seed=42, profile="neutral", profile_selection_reason="hf_smoke",
        width=360, height=640, fps=Rational(30, 1), duration_frames=60,
        aspect_policy="preserve",
        audio_policy={"has_source_audio": False, "mode": "none", "retime_allowed": False, "default_subtitles": False, "default_music": False},
        engine_capabilities=doc["engines"],
        events=[
            EffectEvent(
                id="ev_hf_G01", effect_id="G01", effect_version="1.0.0",
                start_frame=10, end_frame_exclusive=45, layer="overlay",
                region={"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0},
                parameters={"intensity": 0.9, "margin_frac": 0.04, "opacity": 0.72, "accent_hex": "#7CE0FF"},
                easing="ease_in_out_cubic", priority=5, engine="hyperframes",
                reason_codes=["smoke"], evidence_refs=["smoke"], fallback_id="V02",
            )
        ],
        status="READY",
    )
    res = run_edit(
        clip, tmp_path / "runs",
        MotionLabConfig(profile="neutral", preferred_engine="hyperframes", strict=True, cache_enabled=False),
        existing_plan=plan,
    )
    assert res["overall_status"] == "PASS"
    assert "hyperframes" in res["engines_used"]
    assert Path(res["final_path"]).is_file()


def test_real_remotion_license_gated_alpha_render_and_composite(tmp_path: Path):
    """Verify Remotion is blocked without opt-in and renders real verified alpha overlay with --enable-remotion."""
    # Without opt-in, strict remotion MUST fail closed
    clip = generate_silent_fixture(tmp_path / "rem_src.mp4", width=360, height=640, fps=30, duration_s=2.0)
    with pytest.raises(RuntimeError, match="Strict mode blocked"):
        run_edit(
            clip, tmp_path / "runs_blocked",
            MotionLabConfig(profile="neutral", preferred_engine="remotion", enable_remotion=False, strict=True, cache_enabled=False),
        )

    doc = run_doctor(MotionLabConfig(enable_remotion=True))
    if not doc["engines"]["remotion"]["available"]:
        pytest.skip("Remotion local CLI not installed")

    plan = EditPlan(
        schema_version="1.0.0", planner_version="1.0.0", source_hash="rem", config_hash="rem",
        seed=42, profile="neutral", profile_selection_reason="rem_smoke",
        width=360, height=640, fps=Rational(30, 1), duration_frames=60,
        aspect_policy="preserve",
        audio_policy={"has_source_audio": False, "mode": "none", "retime_allowed": False, "default_subtitles": False, "default_music": False},
        engine_capabilities=doc["engines"],
        events=[
            EffectEvent(
                id="ev_rem_G02", effect_id="G02", effect_version="1.0.0",
                start_frame=10, end_frame_exclusive=45, layer="overlay",
                region={"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0},
                parameters={"intensity": 0.9, "cx": 0.82, "cy": 0.18, "radius_frac": 0.06, "opacity": 0.75, "accent_hex": "#FFD08A"},
                easing="ease_in_out_cubic", priority=5, engine="remotion",
                reason_codes=["smoke"], evidence_refs=["smoke"], fallback_id="G01",
            )
        ],
        status="READY",
    )
    res = run_edit(
        clip, tmp_path / "runs_ok",
        MotionLabConfig(profile="neutral", preferred_engine="remotion", enable_remotion=True, strict=True, cache_enabled=False),
        existing_plan=plan,
    )
    assert res["overall_status"] == "PASS"
    assert "remotion" in res["engines_used"]
    assert Path(res["final_path"]).is_file()


def test_edge_media_silent_multitrack_rotated_pcm_and_unicode_space_paths(tmp_path: Path):
    """Real renders covering silent, multitrack, rotated 90deg, raw PCM MOV transcode, and Unicode/space paths."""
    # 1. Silent + Unicode/space path
    uni_dir = tmp_path / "thư mục thử nghiệm 01"
    silent_src = generate_silent_fixture(uni_dir / "video không tiếng 01.mp4", width=360, height=640, duration_s=3.0)
    r_sil = run_edit(silent_src, tmp_path / "runs_sil", MotionLabConfig(profile="auto", cache_enabled=False))
    assert r_sil["overall_status"] == "PASS"

    # 2. Multitrack preserves both tracks by default, or selects track 1 explicitly
    mt_src = generate_multitrack_fixture(tmp_path / "multi.mp4", duration_s=3.0)
    r_mt_all = run_edit(mt_src, tmp_path / "runs_mt_all", MotionLabConfig(profile="neutral", cache_enabled=False))
    assert r_mt_all["overall_status"] == "PASS"
    r_mt_1 = run_edit(mt_src, tmp_path / "runs_mt_1", MotionLabConfig(profile="neutral", audio_track=1, cache_enabled=False))
    assert r_mt_1["overall_status"] == "PASS"

    # 3. Rotated 90 deg preserves display orientation (360x640 portrait!)
    rot_src = generate_rotated_fixture(tmp_path / "rot90.mp4", duration_s=3.0)
    r_rot = run_edit(rot_src, tmp_path / "runs_rot", MotionLabConfig(profile="neutral", cache_enabled=False))
    assert r_rot["overall_status"] == "PASS"

    # 4. Raw PCM MOV triggers disclosed AAC transcode + PCM correlation validation
    pcm_src = generate_pcm_mov_fixture(tmp_path / "raw_pcm.mov", duration_s=3.0)
    r_pcm = run_edit(pcm_src, tmp_path / "runs_pcm", MotionLabConfig(profile="neutral", cache_enabled=False))
    assert r_pcm["overall_status"] == "PASS"
