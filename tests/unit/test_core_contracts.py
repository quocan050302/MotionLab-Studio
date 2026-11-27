"""Phase G2 — Unit tests for Preflight, Schemas, Registry, Planner, Policies, Cache, and Resource Governance."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from motionlab.cache import LocalAnalysisCache
from motionlab.config import EditConfig, ResourceConfig, canonical_json_dumps, ensure_safe_subpath, validate_no_path_traversal
from motionlab.planner import build_edit_plan, validate_plan
from motionlab.policies import load_all_profiles, select_profile
from motionlab.preflight import PreflightError, parse_rational, probe_media
from motionlab.registry import EffectRegistry, load_effects
from motionlab.resource import ResourceBudget, ResourceBudgetError, run_subprocess_guarded
from motionlab.schemas import (
    AnalysisResult,
    CandidateEvent,
    ColorMetadata,
    EditPlan,
    EffectEvent,
    EffectSpec,
    MediaInfo,
    MotionWindow,
)
from tests.fixtures.procedural import make_multishot_demo, make_multitrack_rotated, make_pcm_mkv_and_offset, make_short_clip


def _dummy_media(duration_s: float = 15.0, fps: int = 30, w: int = 1080, h: int = 1920, has_audio: bool = True) -> MediaInfo:
    from motionlab.schemas import AudioStreamInfo

    audio = (
        [
            AudioStreamInfo(
                index=1,
                audio_index=0,
                codec_name="aac",
                sample_rate=44100,
                channels=1,
                channel_layout="mono",
                duration_s=duration_s,
                start_pts=0,
                start_time_s=0.0,
                time_base="1/44100",
                mp4_copy_compatible=True,
            )
        ]
        if has_audio
        else []
    )
    return MediaInfo(
        source_path="dummy.mp4",
        input_hash="a" * 64,
        file_size_bytes=123456,
        duration_pts=int(duration_s * 12288),
        duration_s=duration_s,
        time_base_num=1,
        time_base_den=12288,
        width=w,
        height=h,
        display_width=w,
        display_height=h,
        rotation=0,
        sar_num=1,
        sar_den=1,
        fps_num=fps,
        fps_den=1,
        vfr=False,
        nb_frames=int(round(duration_s * fps)),
        stream_start_pts=0,
        stream_start_time_s=0.0,
        video_codec="h264",
        pixel_format="yuv420p",
        field_order="progressive",
        is_interlaced=False,
        has_alpha=False,
        color_metadata=ColorMetadata(),
        audio_streams=audio,
    )


def _dummy_analysis(media: MediaInfo) -> AnalysisResult:
    cands = [
        CandidateEvent("c1", 30, 120, "sustained_static", 0.85, "static_test"),
        CandidateEvent("c2", 150, 165, "scene_boundary", 0.92, "cut_test"),
        CandidateEvent("c3", 200, 260, "audio_onset", 0.88, "audio_test"),
        CandidateEvent("c4", 300, 390, "safe_hold", 0.80, "hold_test"),
    ]
    return AnalysisResult(
        source_hash=media.input_hash,
        config_hash="cfg12345",
        analyzer_version="1.0.0",
        timebase="1/12288",
        fps_num=media.fps_num,
        fps_den=media.fps_den,
        duration_frames=media.nb_frames,
        motion_windows=[MotionWindow(0, media.nb_frames, 0.05, 0.9, "near_static", "unknown", 0.5, 0.5)],
        capabilities=["video_stream", "audio", "scene_cut", "clean_edges"] if media.has_audio else ["video_stream", "scene_cut", "clean_edges"],
        candidates=cands,
    )


class TestPreflightAndBoundaries:
    def test_rational_parsing(self) -> None:
        assert str(parse_rational("30000/1001")) == "30000/1001"
        assert str(parse_rational("1:1")) == "1"
        assert str(parse_rational("0/0", default=(24, 1))) == "24"

    def test_rejects_missing_empty_and_corrupt_files(self, tmp_path: Path) -> None:
        with pytest.raises(PreflightError):
            probe_media(tmp_path / "does_not_exist.mp4")

        empty = tmp_path / "empty.mp4"
        empty.write_bytes(b"")
        with pytest.raises(PreflightError):
            probe_media(empty)

        corrupt = tmp_path / "corrupt.mp4"
        corrupt.write_bytes(b"NOT_A_VALID_MP4_STREAM_1234567890")
        with pytest.raises(PreflightError):
            probe_media(corrupt)

    def test_probes_rotation_multitrack_and_offset(self, tmp_path: Path) -> None:
        rot_mp4 = make_multitrack_rotated(tmp_path / "rot_multi.mp4", duration_s=2.0)
        info = probe_media(rot_mp4)
        assert len(info.audio_streams) == 2
        assert info.rotation in (90, 270)
        assert info.display_height > info.display_width  # rotated 640x360 -> 360x640

        mkv = make_pcm_mkv_and_offset(tmp_path / "offset_pcm.mkv", duration_s=2.0, ts_offset_s=0.45)
        mkv_info = probe_media(mkv)
        assert mkv_info.audio_streams[0].codec_name == "pcm_s16le"
        assert mkv_info.audio_streams[0].mp4_copy_compatible is False
        assert any("nonzero_video_start_pts" in w for w in mkv_info.warnings)

    def test_probes_silent_short_clip(self, tmp_path: Path) -> None:
        clip = make_short_clip(tmp_path / "tiny.mp4", frames=6, fps=24)
        info = probe_media(clip)
        assert info.has_audio is False
        assert info.nb_frames >= 1


class TestSchemasAndSecurity:
    def test_rejects_nan_and_inf_in_json_and_events(self) -> None:
        with pytest.raises(ValueError):
            canonical_json_dumps({"x": float("nan")})
        with pytest.raises(ValueError):
            canonical_json_dumps({"x": float("inf")})

        ev = EffectEvent(
            id="e1",
            effect_id="subtle_push_in",
            effect_version="1.0.0",
            engine="ffmpeg",
            start_frame=0,
            end_frame_exclusive=30,
            layer="transform",
            region={"cx": 0.5, "cy": 0.5},
            parameters={"intensity": math.nan},
            easing="ease_out_cubic",
            priority=10,
            reason_codes=[],
            evidence_refs=[],
        )
        with pytest.raises(ValueError):
            ev.validate(100)

    def test_rejects_path_traversal_and_escape(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            validate_no_path_traversal("../etc/passwd")
        with pytest.raises(ValueError):
            validate_no_path_traversal("bad\x00path.mp4")
        with pytest.raises(ValueError):
            ensure_safe_subpath(tmp_path / "run", tmp_path / "other_project" / "file.txt")

    def test_rejects_invalid_bounds_unknown_effect_and_forbidden_audio_policy(self) -> None:
        plan = EditPlan(
            schema_version="1.0.0",
            source_hash="a" * 64,
            config_hash="b" * 16,
            seed=42,
            profile="neutral",
            requested_profile="neutral",
            width=720,
            height=1280,
            fps_num=30,
            fps_den=1,
            duration_frames=100,
            aspect_mode="source",
            aspect_bg_mode="blur",
            audio_policy={"retime_allowed": True},
            engine_capabilities={},
            events=[],
        )
        with pytest.raises(ValueError, match="Forbidden audio policy"):
            validate_plan(plan)

        plan.audio_policy = {"retime_allowed": False, "filters_allowed": False}
        plan.events = [
            EffectEvent(
                id="bad",
                effect_id="nonexistent_fx",
                effect_version="1.0.0",
                engine="ffmpeg",
                start_frame=0,
                end_frame_exclusive=20,
                layer="transform",
                region={"cx": 0.5, "cy": 0.5},
                parameters={"intensity": 0.5},
                easing="ease_out_cubic",
                priority=10,
                reason_codes=[],
                evidence_refs=[],
            )
        ]
        with pytest.raises(KeyError):
            validate_plan(plan)


class TestRegistryAndPlanner:
    def test_registry_has_40_effects_16_core_and_rejects_fallback_cycle(self) -> None:
        reg = load_effects()
        assert len(reg.all_specs()) == 40
        assert len(reg.core_specs()) == 16
        with pytest.raises(ValueError, match="Unknown parameter"):
            reg.validate_params("subtle_push_in", {"unknown_param": 1})
        with pytest.raises(ValueError, match="out of bounds"):
            reg.validate_params("subtle_push_in", {"max_zoom": 5.0})

        # Verify cycle detection works
        s1 = EffectSpec(
            id="a", code="X01", version="1", family="camera", description="", engine="ffmpeg",
            supported_engines=("ffmpeg",), layer="transform", required_features=(),
            supported_profiles=("neutral",), parameter_schema={}, default_params={},
            min_duration_s=1.0, max_duration_s=2.0, max_intensity=1.0, safety_class="low_risk",
            incompatibilities=(), fallback_id="b", estimated_cost=1.0,
        )
        s2 = EffectSpec(
            id="b", code="X02", version="1", family="camera", description="", engine="ffmpeg",
            supported_engines=("ffmpeg",), layer="transform", required_features=(),
            supported_profiles=("neutral",), parameter_schema={}, default_params={},
            min_duration_s=1.0, max_duration_s=2.0, max_intensity=1.0, safety_class="low_risk",
            incompatibilities=(), fallback_id="a", estimated_cost=1.0,
        )
        with pytest.raises(ValueError, match="Fallback cycle"):
            EffectRegistry([s1, s2])

    def test_planner_seed_reproducibility_and_policies(self) -> None:
        media = _dummy_media()
        analysis = _dummy_analysis(media)
        caps = {"ffmpeg": {"available": True}, "remotion": {"available": False}, "hyperframes": {"available": False}}
        cfg = EditConfig(profile="auto", seed=42)

        p1, _ = build_edit_plan(media, analysis, cfg, caps)
        p2, _ = build_edit_plan(media, analysis, cfg, caps)
        assert p1.plan_hash() == p2.plan_hash()
        assert p1.to_dict() == p2.to_dict()
        assert len(p1.events) >= 2

        # All 7 concrete profiles load and build valid plans
        profiles = load_all_profiles()
        assert len(profiles) == 7
        for pname in profiles:
            p_prof, _ = build_edit_plan(media, analysis, EditConfig(profile=pname, seed=42), caps)
            validate_plan(p_prof)

        # Density=0 produces explicit NO_SAFE_EFFECTS (never forces fake effects)
        p_empty, _ = build_edit_plan(media, analysis, EditConfig(profile="neutral", density_scale=0.0), caps)
        assert p_empty.status == "NO_SAFE_EFFECTS"
        assert len(p_empty.events) == 0


class TestCacheAndResourceGovernance:
    def test_cache_key_invalidation_and_corruption_recovery(self, tmp_path: Path) -> None:
        cache = LocalAnalysisCache(tmp_path / "cache")
        k1 = cache.build_key("sha1", "cfg1", None, "dep1")
        k2 = cache.build_key("sha1", "cfg2", None, "dep1")
        assert k1 != k2

        cache.put(k1, {"hello": "world", "cuts": [1, 2]})
        assert cache.get(k1) == {"hello": "world", "cuts": [1, 2]}

        # Corrupt entry payload -> must invalidate and return None
        entry_file = tmp_path / "cache" / f"analysis_{k1}.json"
        raw = json.loads(entry_file.read_text(encoding="utf-8"))
        raw["payload"]["cuts"] = [999]
        entry_file.write_text(json.dumps(raw), encoding="utf-8")
        assert cache.get(k1) is None

    def test_resource_budget_rejects_impossible_disk_reserve_and_enforces_timeout(self, tmp_path: Path) -> None:
        # Impossible min_free_gb (999999 GiB) must fail closed with actionable error
        b_fail = ResourceBudget(
            ResourceConfig(min_free_gb=999999.0),
            dest_dir=tmp_path / "dest",
            temp_dir=tmp_path / "temp",
        )
        with pytest.raises(ResourceBudgetError, match="Insufficient"):
            b_fail.preflight_budget(15.0, 1080, 1920, 30.0)

        # Impossible max_temp_gb (0.11 GiB for 4K 1 hour video) must fail closed
        b_temp = ResourceBudget(
            ResourceConfig(max_temp_gb=0.11, min_free_gb=0.5),
            dest_dir=tmp_path / "dest2",
            temp_dir=tmp_path / "temp2",
        )
        with pytest.raises(ResourceBudgetError, match="exceeds --max-temp-gb"):
            b_temp.preflight_budget(3600.0, 3840, 2160, 60.0)

        # Subprocess timeout must terminate child and raise ResourceBudgetError
        b_to = ResourceBudget(
            ResourceConfig(subprocess_timeout_s=5),
            dest_dir=tmp_path / "dest3",
            temp_dir=tmp_path / "temp3",
        )
        with pytest.raises(ResourceBudgetError, match="timed out"):
            run_subprocess_guarded([sys.executable, "-c", "import time; time.sleep(10)"], budget=b_to, timeout_s=1)
