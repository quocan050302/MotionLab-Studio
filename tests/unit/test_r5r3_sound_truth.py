"""Phase A7 (ML-01.R5.R3): 4-Tier Animation Markers (D07), Floor-Free Counterfactual Tamper (D08) & AudioAssetIntelligence."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from motionlab.audio.assets import build_audio_asset_intelligence, inspect_foley_library
from motionlab.audio.mixer import mix_preserving_master, validate_final_audio_mix
from motionlab.audio.semantic_sound import (
    align_sound_to_marker,
    compile_animation_markers,
    extract_realized_animation_markers,
    extract_sonic_opportunities,
    observe_pixel_animation_markers,
    plan_sonic_score,
)
from motionlab.composition.generative_schema import GENERATIVE_SCHEMA_VERSION, LayerSpec, SceneSpec, StoryboardSpec
from motionlab.preflight import probe_media
from motionlab.semantic.eval_harness import _make_media, _make_transcript
from motionlab.semantic.story_graph import build_story_graph


def _make_voice_video(out_mp4: Path, duration_s: float = 4.0, fps: int = 24, sr: int = 44100) -> Path:
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    subprocess.run(
        [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", f"color=c=0x28342e:s=360x640:r={fps}:d={duration_s}",
            "-f", "lavfi", "-i", f"sine=frequency=240:sample_rate={sr}:duration={duration_s}",
            "-af", "volume=0.14",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
            "-shortest", str(out_mp4),
        ],
        check=True, timeout=45, shell=False,
    )
    return out_mp4


class TestR5R3SoundTruth(unittest.TestCase):
    def test_d07_4tier_marker_truth_never_fakes_observed_pixel(self) -> None:
        m = _make_media(6.0, 24, "m4tier")
        tr = _make_transcript("Start with $500 in your account. $300 is for rent. You have $200 left.", 6.0)
        sg = build_story_graph(tr, None, m)
        sc = SceneSpec(
            "sc_01_hook", 0, 72, "hook_contrast", "live_clean_bridge", "hook",
            persistent_ledger_step_id=sg.ledger.steps[0].step_id,
            layers=[LayerSpec("ly_hf", "hf_apple_money_count", "hyperframes", "foreground_graphics", 30, {"from_cents": 50000, "to_cents": 50000})],
        )
        sb = StoryboardSpec(
            GENERATIVE_SCHEMA_VERSION, "AUTO_GENERATIVE", m.input_hash, tr.transcript_hash, sg.story_graph_hash,
            "finance", 0.9, "v", 720, 1280, 24, 1, 144, 42, {}, [sc],
        )
        props = {"sc_01_hook": {"durationInFrames": 72, "globalStartFrame": 0}}

        # Props only -> MUST stay COMPILED, NEVER OBSERVED_PIXEL!
        m_comp = extract_realized_animation_markers(sb, sg, compiled_scene_props=props)
        self.assertTrue(len(m_comp) >= 1)
        self.assertTrue(all(x.marker_tier == "COMPILED" for x in m_comp))

        # Static rendered video (identical solid frames) -> MUST NOT falsely upgrade to OBSERVED_PIXEL!
        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            static_mp4 = td_p / "static.mp4"
            vw = cv2.VideoWriter(str(static_mp4), cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (180, 320))
            fr = np.full((320, 180, 3), 80, dtype=np.uint8)
            for _ in range(72):
                vw.write(fr)
            vw.release()
            m_static = extract_realized_animation_markers(
                sb, sg, compiled_scene_props=props,
                rendered_scenes={"sc_01_hook": {"artifact_path": str(static_mp4), "artifact_sha256": "deadbeef12345678"}},
                observe_pixels=True,
            )
            self.assertTrue(all(x.marker_tier == "RENDER_EMITTED" for x in m_static))

            # Animated rendered video (real pixel change around markers) -> upgrades cleanly to OBSERVED_PIXEL!
            anim_mp4 = td_p / "anim.mp4"
            vw2 = cv2.VideoWriter(str(anim_mp4), cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (180, 320))
            for i in range(72):
                f2 = np.full((320, 180, 3), int((i * 11) % 240), dtype=np.uint8)
                vw2.write(f2)
            vw2.release()
            m_obs = extract_realized_animation_markers(
                sb, sg, compiled_scene_props=props,
                rendered_scenes={"sc_01_hook": {"artifact_path": str(anim_mp4), "artifact_sha256": "cafebabe12345678"}},
                observe_pixels=True,
            )
            self.assertTrue(any(x.marker_tier == "OBSERVED_PIXEL" for x in m_obs))

    def test_d08_floor_free_counterfactual_fails_closed_when_final_lacks_cue(self) -> None:
        """P0 Negative Control D08: Replacing final.mp4 with source-only (no SFX) while mixed_reference.wav is intact MUST FAIL!"""
        ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            src_mp4 = _make_voice_video(td_p / "src.mp4", duration_s=4.0, fps=24, sr=44100)
            media = probe_media(src_mp4)
            tr = _make_transcript("Start with $500 in your account. $300 is for bills. You have $200 left.", 4.0)
            sg = build_story_graph(tr, None, media)
            b_alloc = next(b for b in sg.beats if b.speech_act == "allocation_step")
            sc = SceneSpec(
                "sc_01_bills", b_alloc.start_frame, b_alloc.end_frame_exclusive, "allocation_step", "live_clean_bridge", "bills",
                persistent_ledger_step_id=b_alloc.ledger_step_id,
                layers=[LayerSpec("ly_hf", "hf_mk_progress_stat", "hyperframes", "foreground_graphics", 30, {"amount_cents": 30000, "denom_cents": 50000})],
            )
            sb = StoryboardSpec(
                GENERATIVE_SCHEMA_VERSION, "AUTO_GENERATIVE", media.input_hash, tr.transcript_hash, sg.story_graph_hash,
                "finance", 0.95, "v", media.display_width, media.display_height, 24, 1, media.duration_frames, 42, {}, [sc],
            )
            markers = compile_animation_markers(sb, sg, {"sc_01_bills": {"durationInFrames": b_alloc.end_frame_exclusive - b_alloc.start_frame, "globalStartFrame": b_alloc.start_frame}})
            opps = extract_sonic_opportunities(sg, tr, observed_actions=None, storyboard=sb)
            score = plan_sonic_score(sb, opps, sg, markers, niche_style="finance", seed=42, sr=44100)
            tl, _, _ = align_sound_to_marker(
                score, fps_num=24, fps_den=1, sample_rate=44100, total_frames=media.duration_frames,
                source_hash=media.input_hash, audio_mode="mix", sfx_enabled=True, seed=42,
            )
            self.assertGreaterEqual(len(tl.events), 1)

            run_dir = td_p / "run"
            run_dir.mkdir()
            mix_info = mix_preserving_master(media, tl, tr, run_dir, seed=42, ffmpeg_bin=ffmpeg)

            # 1. Genuine mixed final MP4 -> MUST PASS with reference_floor_used=False!
            good_final = run_dir / "final_good.mp4"
            subprocess.run(
                [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(src_mp4), "-i", mix_info["mixed_reference_path"],
                 "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(good_final)],
                check=True, timeout=45, shell=False,
            )
            _, qa_good = validate_final_audio_mix(media, good_final, tl, mix_info, run_dir, ffmpeg_bin=ffmpeg)
            self.assertEqual(qa_good["status"], "PASS", f"Good mix unexpectedly failed: {qa_good['failure_reasons']}")
            self.assertTrue(all(not e["counterfactual_details"]["reference_floor_used"] for e in qa_good["events"]))
            self.assertTrue(all(e["counterfactual_details"]["final_presence_verified"] for e in qa_good["events"]))

            # 2. Tampered final MP4 replaced with source-only (NO SFX), while mixed_reference.wav stays INTACT -> MUST FAIL!
            tampered_final = run_dir / "final_tampered_nosfx.mp4"
            shutil.copy2(src_mp4, tampered_final)
            _, qa_bad = validate_final_audio_mix(media, tampered_final, tl, mix_info, run_dir, ffmpeg_bin=ffmpeg)
            self.assertEqual(qa_bad["status"], "FAIL")
            self.assertTrue(any(not e["single_cue_ablation_ok"] for e in qa_bad["events"]))
            self.assertTrue(any(not e["counterfactual_details"]["final_presence_verified"] for e in qa_bad["events"]))

    def test_a7_audio_asset_intelligence_deduplicates_raw_wav_and_measures_descriptors(self) -> None:
        lib = inspect_foley_library()
        if lib["status"] != "RECORDED_FOLEY_VERIFIED":
            self.skipTest("Recorded CC0 Foley library not provisioned in this environment")
        intel = build_audio_asset_intelligence()
        self.assertFalse(intel["audio_embedding_model_used"])
        self.assertGreaterEqual(intel["unique_raw_wav_count"], 20)
        self.assertLessEqual(intel["unique_raw_wav_count"], intel["total_bindings"])
        sample = intel["unique_assets"][0]
        self.assertIn("measured_descriptors", sample)
        self.assertIn("human_tags", sample)
        self.assertGreater(sample["measured_descriptors"]["spectral_centroid_hz"], 0.0)


if __name__ == "__main__":
    unittest.main()
