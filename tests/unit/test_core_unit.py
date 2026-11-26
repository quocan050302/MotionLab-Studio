"""Phase G2: Unit tests for schemas, rational time, registry cycles, planner policies, cache, and resource governance."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from motionlab.cache import AnalysisCache
from motionlab.config import MotionLabConfig
from motionlab.effects.catalog import build_effect_catalog
from motionlab.planner import build_edit_plan, resolve_layout, validate_plan
from motionlab.policies import select_profile
from motionlab.preflight import PreflightError, probe_media
from motionlab.registry import EffectRegistry, load_effects
from motionlab.resource import (
    BudgetExceededError,
    compute_resource_budget,
    preflight_budget,
    run_managed_subprocess,
    safe_cleanup_run_temp,
)
from motionlab.schemas import (
    AnalysisResult,
    AudioStreamInfo,
    EditPlan,
    EffectEvent,
    EffectSpec,
    MediaInfo,
    MotionWindow,
    ParameterBounds,
    Rational,
    SceneCut,
    SchemaValidationError,
    atomic_write_json,
    ensure_contained_path,
    ensure_finite,
)


def _dummy_media(w: int = 1080, h: int = 1920, frames: int = 450, fps_n: int = 30, has_audio: bool = True) -> MediaInfo:
    astreams = (
        [
            AudioStreamInfo(
                stream_index=1, audio_track_index=0, codec_name="aac",
                sample_rate=48000, channels=2, channel_layout="stereo",
                bit_rate=192000, start_pts=0, start_time_s=0.0,
                duration_s=frames / fps_n, language="und", mp4_copy_compatible=True,
            )
        ]
        if has_audio
        else []
    )
    return MediaInfo(
        path="/tmp/dummy.mp4",
        input_hash="a" * 64,
        file_size_bytes=102400,
        container_format="mov,mp4",
        duration_pts=frames * 1000,
        duration_s=frames / fps_n,
        duration_frames=frames,
        width=w,
        height=h,
        display_width=w,
        display_height=h,
        rotation=0,
        sar=Rational(1, 1),
        fps=Rational(fps_n, 1),
        timebase=Rational(1, 30000),
        vfr=False,
        stream_start_pts=0,
        video_start_time_s=0.0,
        video_codec="h264",
        pixel_format="yuv420p",
        color_metadata={},
        interlaced=False,
        has_alpha=False,
        is_hdr=False,
        audio_streams=astreams,
    )


def _dummy_analysis(frames: int = 450, fps_n: int = 30, caps: list[str] | None = None, flags: list[str] | None = None) -> AnalysisResult:
    return AnalysisResult(
        schema_version="1.0.0",
        analyzer_version="1.0.0",
        source_hash="a" * 64,
        timebase=Rational(1, 30000),
        fps=Rational(fps_n, 1),
        duration_frames=frames,
        cuts=[SceneCut(frame=150, confidence=0.92, boundary_type="detected", hist_diff=0.72, luma_diff=0.25)],
        motion_windows=[
            MotionWindow(0, 150, 0.03, 0.94, True, "static", 0.5, 0.45),
            MotionWindow(150, 300, 0.04, 0.92, True, "static", 0.5, 0.5),
            MotionWindow(300, 450, 0.03, 0.93, True, "static", 0.5, 0.48),
        ],
        audio_envelope=[],
        energy_peaks=[],
        pauses=[],
        subject_regions=[],
        protected_zones=[],
        quality_flags=flags or [],
        capabilities=caps or ["video_basic", "scene_cuts", "sdr_color"],
        summary_stats={"mean_motion": 0.03, "ui_edge_density": 0.02, "face_coverage_frames": 0},
    )


def test_rational_time_precision_and_validation():
    r = Rational(30000, 1001)
    assert r.to_ffmpeg() == "30000/1001"
    sec = r.frame_to_seconds(300)
    assert abs(sec - 10.01) < 1e-4
    assert r.seconds_to_frame(sec) == 300

    with pytest.raises(SchemaValidationError):
        Rational(0, 1)
    with pytest.raises(SchemaValidationError):
        Rational(30, 0)
    with pytest.raises(SchemaValidationError):
        r.seconds_to_frame(float("nan"))


def test_schema_rejects_nan_inf_and_path_traversal(tmp_path: Path):
    with pytest.raises(SchemaValidationError):
        ensure_finite({"a": float("nan")})
    with pytest.raises(SchemaValidationError):
        ensure_finite([1.0, float("inf")])

    root = tmp_path / "run_001"
    root.mkdir()
    inside = ensure_contained_path(root / "temp" / "clip.mp4", root)
    assert str(inside).startswith(str(root.resolve()))

    with pytest.raises(SchemaValidationError):
        ensure_contained_path(root / ".." / "escape.txt", root)


def test_registry_has_40_unique_effects_and_detects_fallback_cycle():
    reg = load_effects()
    specs = reg.all_specs()
    assert len(specs) == 40
    assert reg.count_by_family() == {
        "camera": 10,
        "layout": 10,
        "transition": 8,
        "color": 6,
        "graphics": 6,
    }

    # Construct synthetic cycle A -> B -> A and ensure fail-closed rejection
    a = EffectSpec(
        id="X1", name="x1", family="color", version="1.0.0", milestone="core16",
        description="test", engine="ffmpeg", supported_engines=("ffmpeg",),
        required_features=(), supported_profiles=("neutral",),
        parameter_schema={"intensity": ParameterBounds("float", 0.5, 0.0, 1.0)},
        min_duration_s=1.0, max_duration_s=2.0, max_intensity=1.0,
        layer="color", safety_class="safe_all", incompatibilities=(), fallback_id="X2", estimated_cost=1.0,
    )
    b = EffectSpec(
        id="X2", name="x2", family="color", version="1.0.0", milestone="core16",
        description="test", engine="ffmpeg", supported_engines=("ffmpeg",),
        required_features=(), supported_profiles=("neutral",),
        parameter_schema={"intensity": ParameterBounds("float", 0.5, 0.0, 1.0)},
        min_duration_s=1.0, max_duration_s=2.0, max_intensity=1.0,
        layer="color", safety_class="safe_all", incompatibilities=(), fallback_id="X1", estimated_cost=1.0,
    )
    with pytest.raises(SchemaValidationError, match="Fallback cycle"):
        EffectRegistry([a, b])


def test_planner_determinism_and_no_safe_effects():
    media = _dummy_media()
    analysis = _dummy_analysis()
    cfg = MotionLabConfig(profile="neutral", seed=42)
    caps = {"ffmpeg": {"available": True, "license": "ok"}}

    plan1, _ = build_edit_plan(media, analysis, cfg, caps)
    plan2, _ = build_edit_plan(media, analysis, cfg, caps)
    assert plan1.plan_hash == plan2.plan_hash
    assert len(plan1.events) >= 2
    assert plan1.status == "READY"

    # When zero engines are available -> NO_SAFE_EFFECTS (never forces invalid event)
    empty_plan, _ = build_edit_plan(media, analysis, cfg, {"ffmpeg": {"available": False}})
    assert empty_plan.status == "NO_SAFE_EFFECTS"
    assert len(empty_plan.events) == 0


def test_plan_validator_rejects_bad_bounds_and_unknown_effect():
    reg = load_effects()
    media = _dummy_media(frames=60)
    bad_ev = EffectEvent(
        id="ev_bad", effect_id="C01", effect_version="1.0.0",
        start_frame=50, end_frame_exclusive=99, layer="transform",
        region={}, parameters={"intensity": 0.5, "max_zoom": 1.05, "anchor_x": 0.5, "anchor_y": 0.5},
        easing="ease_in_out_cubic", priority=5, engine="ffmpeg",
        reason_codes=[], evidence_refs=[], fallback_id="V02",
    )
    plan = EditPlan(
        schema_version="1.0.0", planner_version="1.0.0", source_hash="a" * 64, config_hash="b" * 64,
        seed=42, profile="neutral", profile_selection_reason="test",
        width=1080, height=1920, fps=Rational(30, 1), duration_frames=60,
        aspect_policy="preserve",
        audio_policy={"retime_allowed": False, "default_subtitles": False, "default_music": False},
        engine_capabilities={}, events=[bad_ev], status="READY",
    )
    with pytest.raises(SchemaValidationError, match="Invalid frame bounds"):
        validate_plan(plan, media, reg)

    bad_ev.end_frame_exclusive = 58
    bad_ev.effect_id = "DOES_NOT_EXIST"
    with pytest.raises(SchemaValidationError, match="Unknown effect_id"):
        validate_plan(plan, media, reg)


def test_profile_auto_selection_and_layout_policy():
    ana_ui = _dummy_analysis(flags=["busy_ui_or_text_detected"])
    ana_ui.summary_stats["ui_edge_density"] = 0.12
    pol, _ = select_profile(ana_ui, "auto")
    assert pol.name == "tutorial"

    ana_fast = _dummy_analysis(flags=["high_existing_motion"])
    ana_fast.summary_stats["mean_motion"] = 0.55
    pol2, _ = select_profile(ana_fast, "auto")
    assert pol2.name == "gameplay"

    # Landscape -> 9:16 defaults to pad_blur_9x16 (no blind crop!)
    land_media = _dummy_media(w=1920, h=1080)
    w, h, policy, _ = resolve_layout(land_media, _dummy_analysis(), MotionLabConfig(aspect="9:16", quality="preview"))
    assert (w, h) == (540, 960)
    assert policy == "pad_blur_9x16"


def test_resource_budget_timeout_and_safe_cleanup(tmp_path: Path):
    budget = compute_resource_budget(max_workers=2, max_temp_gb=2.0, min_free_gb=999999.0)
    with pytest.raises(BudgetExceededError):
        preflight_budget(tmp_path, tmp_path, 15.0, 1080, 1920, 30.0, budget)

    # Child timeout kills own subprocess cleanly
    with pytest.raises(BudgetExceededError, match="timed out"):
        run_managed_subprocess([sys.executable, "-c", "import time; time.sleep(10)"], timeout_s=0.4)

    # Safe cleanup deletes only files under run_root/temp and refuses escape
    run_root = tmp_path / "run_safe"
    temp_dir = run_root / "temp"
    temp_dir.mkdir(parents=True)
    outside = tmp_path / "keep_me.txt"
    outside.write_text("do not delete", encoding="utf-8")
    (temp_dir / "scratch.tmp").write_text("temp", encoding="utf-8")
    # Even a symlink pointing outside must only unlink the symlink, never the target!
    sym = temp_dir / "link_out.txt"
    sym.symlink_to(outside)
    safe_cleanup_run_temp(temp_dir, run_root)
    assert outside.is_file()
    assert not (temp_dir / "scratch.tmp").exists()
    assert not sym.exists()


def test_cache_hit_invalidation_and_corruption_recovery(tmp_path: Path):
    cache = AnalysisCache(cache_dir=tmp_path / "cache")
    media = _dummy_media()
    cfg1 = MotionLabConfig(analysis_fps=6.0)
    cfg2 = MotionLabConfig(analysis_fps=4.0)
    k1 = cache.make_key(media, cfg1, "ffmpeg-6.0")
    k2 = cache.make_key(media, cfg2, "ffmpeg-6.0")
    assert k1 != k2

    ana = _dummy_analysis()
    entry_path = cache.put(k1, ana)
    loaded = cache.get(k1)
    assert loaded is not None
    assert loaded.source_hash == ana.source_hash

    # Corrupt cache payload -> integrity check rejects & deletes corrupted file
    raw = json.loads(entry_path.read_text(encoding="utf-8"))
    raw["payload"]["duration_frames"] = 999999
    entry_path.write_text(json.dumps(raw), encoding="utf-8")
    assert cache.get(k1) is None
    assert not entry_path.exists()


def test_preflight_rejects_missing_empty_and_corrupt_files(tmp_path: Path):
    with pytest.raises(PreflightError):
        probe_media(tmp_path / "nonexistent.mp4")

    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    with pytest.raises(PreflightError, match="empty"):
        probe_media(empty)

    corrupt = tmp_path / "corrupt.mp4"
    corrupt.write_bytes(b"not a real mp4 file header 1234567890")
    with pytest.raises(PreflightError):
        probe_media(corrupt)
