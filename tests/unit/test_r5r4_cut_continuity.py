"""Phase A4 (ML-01.R5.R4): Cut Continuity, Safe Reframe Crop, L/J-Cut Lip-Sync Guard & Audio Join Click/Drift QA."""

from __future__ import annotations

import math
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from motionlab.audio.mixer import build_edited_voice_bed
from motionlab.composition.edit_continuity import (
    compute_safe_reframe_crop,
    evaluate_audio_joins,
    validate_lj_bridge_safety,
)
from motionlab.composition.edit_time_map import build_edit_time_map
from motionlab.preflight import probe_media
from motionlab.semantic.eval_harness import _make_media
from tests.unit.test_r5r4_alignment_pacing import _make_speech_gap_clip


class TestR5R4CutContinuity(unittest.TestCase):
    def test_lj_bridge_rejects_audio_shift_on_visible_speaking_lips(self) -> None:
        """L/J-cut safety guard: NEVER shift speaking dialogue relative to visible talking lips!"""
        bad = validate_lj_bridge_safety(audio_offset_frames=4, lips_visible_in_window=True, is_broll_or_cutaway=False)
        self.assertFalse(bad["safe"])
        ok_broll = validate_lj_bridge_safety(audio_offset_frames=4, lips_visible_in_window=False, is_broll_or_cutaway=True)
        self.assertTrue(ok_broll["safe"])
        ok_same = validate_lj_bridge_safety(audio_offset_frames=0, lips_visible_in_window=True, is_broll_or_cutaway=False)
        self.assertTrue(ok_same["safe"])

    def test_safe_reframe_crop_caps_zoom_on_720p_and_avoids_clipping_face_or_hands(self) -> None:
        """Safe reframe stays <= 1.06x on 720p and falls back to 1.0x if cropping would clip face or task ROI."""
        good = compute_safe_reframe_crop(
            {"x": 0.35, "y": 0.18, "w": 0.30, "h": 0.25, "conf": 0.9},
            {"x": 0.32, "y": 0.62, "w": 0.36, "h": 0.22, "conf": 0.8},
            width=720, height=1280,
        )
        self.assertTrue(good["safe"])
        self.assertLessEqual(good["zoom"], 1.06)

        # Center-based {cx, cy, w, h} ROI tracks (real Olivia schema) must also work cleanly!
        good_center = compute_safe_reframe_crop(
            {"cx": 0.50, "cy": 0.35, "w": 0.28, "h": 0.16, "conf": 0.95},
            {"cx": 0.52, "cy": 0.72, "w": 0.40, "h": 0.24, "conf": 0.90},
            width=720, height=1280,
        )
        self.assertTrue(good_center["safe"])
        self.assertLessEqual(good_center["zoom"], 1.06)

        # Edge-pinned face + bottom-pinned task -> must fall back safely to 1.0x!
        edge = compute_safe_reframe_crop(
            {"x": 0.01, "y": 0.01, "w": 0.40, "h": 0.30, "conf": 0.9},
            {"x": 0.60, "y": 0.80, "w": 0.39, "h": 0.19, "conf": 0.9},
            width=720, height=1280,
        )
        self.assertEqual(edge["zoom"], 1.0)

    def test_audio_join_qa_passes_smooth_crossfaded_bed_and_fails_injected_click_or_sample_drift(self) -> None:
        """Real crossfaded edited_voice_bed.wav PASSES; injected click spike or truncated sample count FAILS closed!"""
        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            src_mp4 = _make_speech_gap_clip(td_p / "src.mp4", speech_blocks=[(0.3, 2.0), (4.5, 6.2)], duration_s=7.0)
            media = probe_media(src_mp4)
            emap = build_edit_time_map(
                media,
                [{"cut_id": "c_mid", "cut_start_frame": 60, "cut_end_frame_exclusive": 96, "pause_class": "DEAD_AIR"}],
                pacing_mode="auto",
            )
            bed_wav = td_p / "edited_voice_bed.wav"
            bed_info = build_edited_voice_bed(src_mp4, emap, bed_wav)
            self.assertTrue(bed_info["sample_exact"])

            qa_good = evaluate_audio_joins(bed_wav, emap)
            self.assertEqual(qa_good["status"], "PASS", qa_good)
            self.assertEqual(qa_good["clicks_detected"], 0)
            self.assertEqual(qa_good["sample_count_diff"], 0)

            # Negative 1: Inject loud digital click right at the splice sample -> MUST FAIL!
            click_wav = td_p / "bed_click.wav"
            with wave.open(str(bed_wav), "rb") as wf:
                sr, nch, nf = wf.getframerate(), wf.getnchannels(), wf.getnframes()
                arr = np.frombuffer(wf.readframes(nf), dtype=np.int16).copy().reshape((-1, nch))
            js = emap.kept_spans[1].output_start_sample
            arr[js - 2 : js, :] = -24000
            arr[js : js + 2, :] = 24000
            with wave.open(str(click_wav), "wb") as wf:
                wf.setnchannels(nch); wf.setsampwidth(2); wf.setframerate(sr)
                wf.writeframes(arr.tobytes())
            qa_click = evaluate_audio_joins(click_wav, emap)
            self.assertEqual(qa_click["status"], "FAIL")
            self.assertGreaterEqual(qa_click["clicks_detected"], 1)

            # Negative 2: Truncate sample count (A/V drift) -> MUST FAIL!
            drift_wav = td_p / "bed_drift.wav"
            with wave.open(str(drift_wav), "wb") as wf:
                wf.setnchannels(nch); wf.setsampwidth(2); wf.setframerate(sr)
                wf.writeframes(arr[:-2000].tobytes())
            qa_drift = evaluate_audio_joins(drift_wav, emap)
            self.assertEqual(qa_drift["status"], "FAIL")
            self.assertTrue(any("sample_count_drift" in r for r in qa_drift["failure_reasons"]))


if __name__ == "__main__":
    unittest.main()
