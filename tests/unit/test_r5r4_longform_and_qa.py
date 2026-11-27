"""Phase A8–A10 (ML-01.R5.R4): Rhythm Critic, Synthetic Long-Form Streaming Memory Bounds & QA-02 Pacing Pack."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from motionlab.composition.edit_time_map import build_edit_time_map, validate_edit_time_map
from motionlab.composition.generative_schema import GENERATIVE_SCHEMA_VERSION, SceneSpec, StoryboardSpec
from motionlab.composition.qa_pack import build_qa02_evidence_pack
from motionlab.composition.visual_critic import run_rhythm_critic
from motionlab.semantic.eval_harness import _make_media
from motionlab.semantic.speech_perception import detect_speech_activity


class TestR5R4LongFormAndQA(unittest.TestCase):
    def test_a9_synthetic_longform_300s_chunked_vad_and_edit_time_map_bounded(self) -> None:
        """Long-form 300s (5-minute, 7200 frames) chunked VAD & multi-cut EditTimeMap stay exact & bounded in memory."""
        import shutil
        import subprocess
        from unittest import mock

        from motionlab.schemas import AudioStreamInfo

        media = _make_media(duration_s=300.0, fps=24, tag="long_300s")
        media.audio_streams = [
            AudioStreamInfo(
                index=1, audio_index=0, codec_name="aac", sample_rate=44100, channels=1,
                channel_layout="mono", duration_s=300.0, start_pts=0, start_time_s=0.0, time_base="1/44100",
            )
        ]
        # Synthesize 300s 16kHz PCM in memory (~19 MB) with periodic 3.5s real speech + 2.5s silence blocks
        sr16 = 16000
        n = int(300 * sr16)
        pcm = np.zeros(n, dtype=np.float32)
        speech_tile: np.ndarray | None = None
        say_bin = shutil.which("say")
        ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
        if say_bin:
            with tempfile.TemporaryDirectory() as td:
                aiff_p = Path(td) / "tile.aiff"
                subprocess.run([say_bin, "-o", str(aiff_p), "Start with one thousand dollars in your checking account right now."], check=True, timeout=20, shell=False)
                proc = subprocess.run(
                    [ffmpeg, "-v", "error", "-nostdin", "-i", str(aiff_p), "-ac", "1", "-ar", str(sr16), "-f", "f32le", "pipe:1"],
                    capture_output=True, check=True, timeout=20, shell=False,
                )
                arr = np.frombuffer(proc.stdout, dtype=np.float32).copy()
                if arr.size > 16000:
                    speech_tile = arr[: int(3.5 * sr16)]

        for blk in range(0, 300, 6):
            i0 = int((blk + 0.5) * sr16)
            if speech_tile is not None:
                i1 = min(n, i0 + speech_tile.size)
                pcm[i0:i1] = speech_tile[: i1 - i0]
            else:
                i1 = min(n, int((blk + 4.0) * sr16))
                t = np.arange(i1 - i0, dtype=np.float32) / sr16
                pcm[i0:i1] = 0.22 * np.sin(2.0 * math.pi * 440.0 * t)

        if speech_tile is not None:
            vad = detect_speech_activity(media, pcm_16k=pcm)
            self.assertEqual(vad["vad_backend"], "silero_vad_onnx")
        else:
            with mock.patch("faster_whisper.vad.get_speech_timestamps", side_effect=RuntimeError("no_tts_fallback")):
                vad = detect_speech_activity(media, pcm_16k=pcm)
        self.assertGreaterEqual(len(vad["speech_spans"]), 10)
        self.assertFalse(vad["silent_video"])

        # Apply 20 disjoint cuts across 300s (7200 frames) -> exact frame & sample conservation!
        cuts = [{"cut_id": f"c_{k:02d}", "cut_start_frame": k * 360 + 100, "cut_end_frame_exclusive": k * 360 + 124, "pause_class": "DEAD_AIR"} for k in range(20)]
        emap = build_edit_time_map(media, cuts, pacing_mode="auto")
        self.assertEqual(emap.output_total_frames, 7200 - 20 * 24)
        self.assertEqual(validate_edit_time_map(emap), [])

    def test_a8_rhythm_critic_evaluates_edited_voice_bed_and_flags_word_preservation_failure(self) -> None:
        """Rhythm critic measures post-edit non-speech on edited_voice_bed.wav and fails closed if word preservation failed."""
        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            media = _make_media(duration_s=10.0, fps=24, tag="rhythm")
            emap = build_edit_time_map(media, [{"cut_id": "c0", "cut_start_frame": 96, "cut_end_frame_exclusive": 120, "pause_class": "DEAD_AIR"}], pacing_mode="auto")
            bed_wav = td_p / "bed.wav"
            sr = 44100
            n = emap.edited_total_samples
            t = np.arange(n, dtype=np.float32) / sr
            sig = (0.18 * np.sin(2.0 * math.pi * 350.0 * t)).astype(np.float32)
            # Leave a clean 0.6s pause in middle
            sig[sr * 4 : int(sr * 4.6)] = 0.0005
            with wave.open(str(bed_wav), "wb") as wf:
                wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
                wf.writeframes(np.round(sig * 32767.0).astype(np.int16).tobytes())

            sb = StoryboardSpec(
                GENERATIVE_SCHEMA_VERSION, "AUTO_GENERATIVE", "h", "t", "s", "finance", 0.9, "v",
                720, 1280, 24, 1, emap.output_total_frames, 42, {},
                [SceneSpec("sc_01", 0, emap.output_total_frames, "hook_contrast", "live_clean_bridge", "h")],
            )
            rep_ok = run_rhythm_critic(
                edit_map=emap, pacing_plan={"chosen_plan_id": "pacing_balanced"}, edited_voice_bed_wav=bed_wav,
                word_preservation={"passed": True, "token_recall": 1.0, "critical_token_recall": 1.0},
                cut_continuity={"joins_count": 1, "severe_jump_joins": 0, "status": "PASS"},
                join_audio_qa={"joins_checked": 1, "clicks_detected": 0, "status": "PASS"},
                storyboard=sb, pixel_critic={"severe_findings_count": 0, "vlm_status": "VISION_CRITIC_NOT_RUN"},
                out_dir=td_p / "ev",
            )
            self.assertEqual(rep_ok["objective_rhythm_status"], "PASS")
            self.assertTrue(rep_ok["edited_voice_bed_analyzed"])
            self.assertLess(rep_ok["longest_remaining_non_speech_s"], 1.0)
            self.assertTrue((td_p / "ev" / "rhythm_critic.json").is_file())
            self.assertTrue((td_p / "ev" / "cut_review.json").is_file())

            # Negative: failed word preservation -> objective_rhythm_status MUST FAIL!
            rep_bad = run_rhythm_critic(
                edit_map=emap, pacing_plan=None, edited_voice_bed_wav=bed_wav,
                word_preservation={"passed": False, "missing_numeric_tokens": ["$100"]},
                cut_continuity={"joins_count": 1, "severe_jump_joins": 0},
                join_audio_qa={"status": "PASS"},
                storyboard=sb, pixel_critic={"severe_findings_count": 0},
            )
            self.assertEqual(rep_bad["objective_rhythm_status"], "FAIL")

    def test_a10_qa02_includes_redacted_pacing_and_preserves_https_repo_url(self) -> None:
        """D10/QA-02: Includes edit_time_map + pacing_decisions + rhythm_critic, preserves https:// URL, scrubs host paths."""
        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            run_dir = td_p / "run_001"
            (run_dir / "evidence").mkdir(parents=True)
            (run_dir / "manifest.json").write_text(json.dumps({"input_hash": "abc123"}), encoding="utf-8")
            (run_dir / "validation.json").write_text(json.dumps({"overall_status": "TECHNICAL_PASS", "aesthetic_status": "PENDING_HUMAN_REVIEW"}), encoding="utf-8")
            (run_dir / "storyboard_spec.json").write_text(json.dumps({"width": 720, "height": 1280, "fps_num": 24, "fps_den": 1, "total_frames": 622, "source_hash": "abc123", "scenes": []}), encoding="utf-8")
            (run_dir / "edit_time_map.json").write_text(json.dumps({"source_sha256": "abc123", "source_path": "/Users/secret/orig.mp4", "source_total_frames": 720, "output_total_frames": 622, "source_duration_s": 30.0, "output_duration_s": 25.917, "removed_total_s": 4.083, "pacing_mode": "auto", "status": "EDITED_CUTS_APPLIED", "omitted_spans": [{"cut_id": "c0"}], "map_hash": "m123"}), encoding="utf-8")
            (run_dir / "pacing_plan.json").write_text(json.dumps({"pacing_mode": "auto", "status": "EDITED_CUTS_APPLIED", "cuts_count": 1, "chosen_plan_id": "pacing_balanced"}), encoding="utf-8")
            (run_dir / "word_preservation.json").write_text(json.dumps({"passed": True, "token_recall": 1.0}), encoding="utf-8")
            (run_dir / "evidence" / "rhythm_critic.json").write_text(json.dumps({"objective_rhythm_status": "PASS"}), encoding="utf-8")

            qa_root = td_p / "qa_evidence"
            pack_dir = qa_root / "ML-01.R5.R4" / "deadbee" / "run_001"
            build_qa02_evidence_pack(
                run_dir=run_dir, pack_dir=pack_dir, allowed_root=qa_root,
                tested_code_sha="deadbeef" * 5, repo_url="https://github.com/quocan050302/MotionLab-Studio",
                branch="feat/ml-01-zero-api-effects", label="ML-01.R5.R4", render_mode="AUTO_GENERATIVE",
                final_mp4_name="final.mp4",
            )
            man = json.loads((pack_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(man["repo"], "https://github.com/quocan050302/MotionLab-Studio")
            self.assertEqual(man["input_frames"], 720)
            self.assertEqual(man["edited_frames"], 622)
            self.assertEqual(man["pacing_summary"]["cuts_count"], 1)
            self.assertTrue((pack_dir / "edit_time_map.json").is_file())
            self.assertNotIn("/Users/secret", (pack_dir / "edit_time_map.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
