"""Phase G3 & G4 — Real Integration Tests for Renderers, Alpha Overlays, Fallbacks, and Audio Integrity."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from motionlab.analyzer import analyze_media
from motionlab.config import AnalysisConfig, EditConfig, ResourceConfig
from motionlab.pipeline import run_edit_pipeline
from motionlab.planner import build_edit_plan
from motionlab.preflight import find_binary, probe_media
from motionlab.renderers.base import RendererError, probe_all_engine_capabilities, probe_artifact_ffprobe
from motionlab.renderers.ffmpeg import FFmpegRenderer
from motionlab.renderers.hyperframes import HyperFramesRenderer
from motionlab.renderers.remotion import RemotionRenderer
from motionlab.schemas import AnalysisResult, EditPlan, EffectEvent
from tests.fixtures.procedural import (
    make_multishot_demo,
    make_multitrack_rotated,
    make_pcm_mkv_and_offset,
)


def _extract_frame(video: Path, time_s: float, w: int, h: int) -> np.ndarray:
    ff = find_binary("ffmpeg")
    assert ff
    cmd = [
        ff, "-hide_banner", "-loglevel", "error",
        "-ss", f"{time_s:.3f}", "-i", str(video),
        "-frames:v", "1", "-vf", f"scale={w}:{h}", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=True)
    return np.frombuffer(proc.stdout[: w * h * 3], dtype=np.uint8).reshape((h, w, 3))


class TestAudioIntegrityAndLayouts:
    def test_stream_copy_aac_audio_exact_packet_hash_and_visible_effect_vs_control(self, tmp_path: Path) -> None:
        src = make_multishot_demo(tmp_path / "demo_6s.mp4", duration_s=6.0, width=360, height=640, fps=24, with_audio=True)
        runs = tmp_path / "runs"

        # 1. Edited run (FFmpeg engine)
        cfg_edit = EditConfig(
            profile="neutral",
            seed=42,
            quality="preview",
            preferred_engine="ffmpeg",
            allow_remotion=False,
            allow_hyperframes=False,
            analysis=AnalysisConfig(use_cache=False),
            resources=ResourceConfig(min_free_gb=1.0),
        )
        res_edit = run_edit_pipeline(src, runs, cfg_edit)
        assert res_edit["status"] == "TECHNICAL_PASS"
        assert res_edit["audio_mode"] == "stream_copy"
        assert res_edit["events"] >= 1
        val = json.loads((Path(res_edit["run_dir"]) / "validation.json").read_text(encoding="utf-8"))
        checks = {c["name"]: c["status"] for c in val["checks"]}
        assert checks["audio_stream_copy_packet_hash_exact"] == "PASS"
        assert checks["audio_pcm_sync_and_fidelity"] == "PASS"
        assert checks["ffmpeg_full_decode_zero_errors"] == "PASS"
        assert checks["planned_effects_measurably_rendered"] == "PASS"
        assert val["aesthetic_status"] == "AESTHETIC_PENDING"

        # 2. Control run (density_scale=0 -> 0 effects) under identical codec settings
        cfg_ctrl = EditConfig(
            profile="neutral",
            seed=42,
            quality="preview",
            density_scale=0.0,
            preferred_engine="ffmpeg",
            allow_remotion=False,
            allow_hyperframes=False,
            analysis=AnalysisConfig(use_cache=False),
            resources=ResourceConfig(min_free_gb=1.0),
        )
        res_ctrl = run_edit_pipeline(src, runs, cfg_ctrl)
        assert res_ctrl["events"] == 0
        plan_edit = json.loads((Path(res_edit["run_dir"]) / "edit_plan.json").read_text(encoding="utf-8"))
        ev0 = plan_edit["events"][0]
        mid_t = ((ev0["start_frame"] + ev0["end_frame_exclusive"]) / 2.0) / 24.0
        f_edit = _extract_frame(Path(res_edit["final_path"]), mid_t, 360, 640)
        f_ctrl = _extract_frame(Path(res_ctrl["final_path"]), mid_t, 360, 640)
        diff = np.abs(f_edit.astype(np.float32) - f_ctrl.astype(np.float32))
        assert float(np.mean(diff)) > 0.2 or int(np.sum(np.max(diff, axis=2) >= 8.0)) >= 60

    def test_pcm_mkv_disclosed_transcode_and_nonzero_pts_offset(self, tmp_path: Path) -> None:
        mkv = make_pcm_mkv_and_offset(tmp_path / "pcm_offset.mkv", duration_s=3.0, ts_offset_s=0.35)
        cfg = EditConfig(
            profile="neutral",
            seed=42,
            quality="preview",
            allow_remotion=False,
            allow_hyperframes=False,
            analysis=AnalysisConfig(use_cache=False),
            resources=ResourceConfig(min_free_gb=1.0),
        )
        res = run_edit_pipeline(mkv, tmp_path / "runs", cfg)
        assert res["status"] == "TECHNICAL_PASS"
        assert res["audio_mode"] == "transcode_compat"
        val = json.loads((Path(res["run_dir"]) / "validation.json").read_text(encoding="utf-8"))
        checks = {c["name"]: c["status"] for c in val["checks"]}
        assert checks["audio_pcm_sync_and_fidelity"] == "PASS"

    def test_multitrack_selection_and_rotation_handling(self, tmp_path: Path) -> None:
        src = make_multitrack_rotated(tmp_path / "rot_2tracks.mp4", duration_s=3.0)
        # Default preserves both audio tracks
        res_all = run_edit_pipeline(
            src,
            tmp_path / "runs",
            EditConfig(
                profile="neutral",
                quality="preview",
                allow_remotion=False,
                allow_hyperframes=False,
                analysis=AnalysisConfig(use_cache=False),
                resources=ResourceConfig(min_free_gb=1.0),
            ),
        )
        assert res_all["status"] == "TECHNICAL_PASS"
        probe_all = probe_artifact_ffprobe(Path(res_all["final_path"]))
        assert len([s for s in probe_all["streams"] if s["codec_type"] == "audio"]) == 2
        v0 = [s for s in probe_all["streams"] if s["codec_type"] == "video"][0]
        assert int(v0["height"]) > int(v0["width"])  # portrait preserved after autorotation

        # Explicit --audio-track 1 preserves only track 1
        res_t1 = run_edit_pipeline(
            src,
            tmp_path / "runs",
            EditConfig(
                profile="neutral",
                quality="preview",
                audio_track=1,
                allow_remotion=False,
                allow_hyperframes=False,
                analysis=AnalysisConfig(use_cache=False),
                resources=ResourceConfig(min_free_gb=1.0),
            ),
        )
        assert res_t1["status"] == "TECHNICAL_PASS"
        probe_t1 = probe_artifact_ffprobe(Path(res_t1["final_path"]))
        assert len([s for s in probe_t1["streams"] if s["codec_type"] == "audio"]) == 1


class TestMultiEngineRenderersAndFallbacks:
    def test_remotion_offline_alpha_render_and_ffmpeg_composite(self, tmp_path: Path) -> None:
        rem = RemotionRenderer(license_allowed=True)
        cap = rem.probe_capability()
        assert cap["available"] is True, f"Remotion expected available locally: {cap}"

        plan = EditPlan(
            schema_version="1.0.0",
            source_hash="a" * 64,
            config_hash="b" * 16,
            seed=42,
            profile="neutral",
            requested_profile="neutral",
            width=360,
            height=640,
            fps_num=24,
            fps_den=1,
            duration_frames=48,
            aspect_mode="9:16",
            aspect_bg_mode="blur",
            audio_policy={"mode": "none"},
            engine_capabilities={},
            events=[],
        )
        ana = AnalysisResult(
            source_hash="a" * 64,
            config_hash="b" * 16,
            analyzer_version="1.0.0",
            timebase="1/12288",
            fps_num=24,
            fps_den=1,
            duration_frames=48,
        )
        ev = EffectEvent(
            id="rem_smoke_01",
            effect_id="edge_lines",
            effect_version="1.0.0",
            engine="remotion",
            start_frame=0,
            end_frame_exclusive=18,
            layer="overlay",
            region={"cx": 0.5, "cy": 0.88, "w": 0.9, "h": 0.14},
            parameters={"intensity": 0.85, "margin_ratio": 0.045, "opacity": 0.75},
            easing="ease_out_cubic",
            priority=40,
            reason_codes=["integration_smoke"],
            evidence_refs=["smoke"],
        )
        out_webm = tmp_path / "rem_overlay.webm"
        res = rem.render_overlay_event(ev, plan, ana, out_webm, tmp_path)
        assert out_webm.exists() and out_webm.stat().st_size > 200
        assert res["engine"] == "remotion"

        # Composite onto a real video and prove alpha transparency survived (background NOT blacked out!)
        src = make_multishot_demo(tmp_path / "base.mp4", duration_s=2.0, width=360, height=640, fps=24, with_audio=False)
        comp_mp4 = tmp_path / "composited.mp4"
        FFmpegRenderer().composite_alpha_overlays(
            base_video=src,
            overlays=[(ev, out_webm)],
            out_video=comp_mp4,
            plan=plan,
            config=EditConfig(quality="preview"),
        )
        f_src = _extract_frame(src, 0.35, 360, 640)
        f_cmp = _extract_frame(comp_mp4, 0.35, 360, 640)
        # Center must stay nearly identical to source (alpha=0 there -> transparent!)
        center_diff = float(np.mean(np.abs(f_cmp[150:480, 80:280].astype(np.float32) - f_src[150:480, 80:280].astype(np.float32))))
        # Top edge must show crisp rendered white/accent lines (alpha>0 there!)
        top_diff = float(np.max(np.abs(f_cmp[0:60, :].astype(np.float32) - f_src[0:60, :].astype(np.float32))))
        assert center_diff < 5.0, f"Alpha did not preserve center transparency (center_diff={center_diff})"
        assert top_diff > 20.0, f"Remotion overlay did not composite visibly at top margin (top_diff={top_diff})"

    def test_hyperframes_offline_render_and_strict_rejection(self, tmp_path: Path) -> None:
        hf = HyperFramesRenderer(enabled=True)
        cap = hf.probe_capability()
        assert cap["available"] is True, f"HyperFrames CLI expected locally installed: {cap}"

        # Real offline HyperFrames + local GSAP render smoke test
        plan = EditPlan(
            schema_version="1.0.0",
            source_hash="a" * 64,
            config_hash="b" * 16,
            seed=42,
            profile="neutral",
            requested_profile="neutral",
            width=360,
            height=640,
            fps_num=24,
            fps_den=1,
            duration_frames=24,
            aspect_mode="9:16",
            aspect_bg_mode="blur",
            audio_policy={"mode": "none"},
            engine_capabilities={},
            events=[],
        )
        ana = AnalysisResult(
            source_hash="a" * 64,
            config_hash="b" * 16,
            analyzer_version="1.0.0",
            timebase="1/12288",
            fps_num=24,
            fps_den=1,
            duration_frames=24,
        )
        ev = EffectEvent(
            id="hf_smoke_01",
            effect_id="foreground_frame_accent",
            effect_version="1.0.0",
            engine="hyperframes",
            start_frame=0,
            end_frame_exclusive=12,
            layer="overlay",
            region={"cx": 0.5, "cy": 0.5, "w": 1.0, "h": 1.0},
            parameters={"intensity": 0.85, "margin_ratio": 0.045, "stroke_px": 3},
            easing="ease_out_cubic",
            priority=40,
            reason_codes=["integration_smoke"],
            evidence_refs=["smoke"],
        )
        out_webm = tmp_path / "hf_overlay.webm"
        res = hf.render_overlay_event(ev, plan, ana, out_webm, tmp_path)
        assert out_webm.exists() and out_webm.stat().st_size > 200
        assert res["engine"] == "hyperframes"
        assert len(res["ffprobe"].get("streams", [])) >= 1

        # Strict-mode rejection when explicitly requested engine is disabled
        src = make_multishot_demo(tmp_path / "fb.mp4", duration_s=2.0, width=360, height=640, fps=24, with_audio=False)
        with pytest.raises(RendererError, match="Strict mode rejected"):
            run_edit_pipeline(
                src,
                tmp_path / "runs",
                EditConfig(
                    strict=True,
                    preferred_engine="hyperframes",
                    allow_hyperframes=False,
                    resources=ResourceConfig(min_free_gb=1.0),
                ),
            )
