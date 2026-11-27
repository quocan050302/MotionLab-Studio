"""Phase A3 (ML-01.R5.R4): Canonical EditTimeMap, Exact Rational Clocks, Tamper/Unsafe-Cut Rejection & Domain Mapping."""

from __future__ import annotations

import unittest

from motionlab.composition.edit_time_map import (
    EditTimeMap,
    build_edit_time_map,
    build_identity_edit_time_map,
    exact_rational_samples,
    map_audio_interval,
    map_source_interval,
    map_word_spans,
    output_frame_to_source,
    remap_roi_tracks_to_edited_domain,
    remap_transcript_to_edited_domain,
    source_frame_to_output,
    validate_edit_time_map,
)
from motionlab.semantic.eval_harness import _make_media, _make_transcript


class TestR5R4EditTimeMap(unittest.TestCase):
    def test_disjoint_cuts_exact_24fps_44100hz_zero_drift_and_removed_maps_to_none(self) -> None:
        """Exact 24fps @ 44.1kHz (1837.5 samples/frame) multi-cut map: zero rounding drift, removed frames -> None."""
        media = _make_media(duration_s=30.0, fps=24, tag="olivia_like")
        cuts = [
            {"cut_id": "c0", "cut_start_frame": 192, "cut_end_frame_exclusive": 228, "pause_class": "DEAD_AIR"},  # 36f
            {"cut_id": "c1", "cut_start_frame": 420, "cut_end_frame_exclusive": 480, "pause_class": "COGNITIVE"},  # 60f
        ]
        emap = build_edit_time_map(media, cuts, pacing_mode="auto")
        self.assertEqual(emap.output_total_frames, 720 - 36 - 60)  # 624 frames = 26.000s
        self.assertEqual(emap.edited_total_samples, 624 * 44100 // 24)  # 1,146,600 exact
        self.assertEqual(validate_edit_time_map(emap), [])

        # Removed frames MUST map strictly to None (never invented nearest frame!)
        self.assertIsNone(source_frame_to_output(emap, 192))
        self.assertIsNone(source_frame_to_output(emap, 210))
        self.assertIsNone(source_frame_to_output(emap, 450))
        # Retained frames map bijectively
        self.assertEqual(source_frame_to_output(emap, 191), 191)
        self.assertEqual(source_frame_to_output(emap, 228), 192)
        self.assertEqual(output_frame_to_source(emap, 192), 228)
        self.assertEqual(source_frame_to_output(emap, 480), 480 - 36 - 60)
        self.assertEqual(output_frame_to_source(emap, 623), 719)

        # Interval crossing a cut splits/trims accurately without ghost frames
        mapped_iv = map_source_interval(emap, 180, 240)
        self.assertEqual(mapped_iv, [(180, 204)])  # 12 before cut + 12 after cut contiguous!
        # Audio interval mapping matches sample clock
        a_iv = map_audio_interval(emap, 0, emap.kept_spans[0].source_end_sample)
        self.assertEqual(a_iv, [(0, emap.kept_spans[0].output_end_sample)])

    def test_rational_ntsc_24000_1001_and_30000_1001_exact_sample_clock(self) -> None:
        """Exact rational NTSC FPS (24000/1001 and 30000/1001) allocates cumulative samples with <=1 sample exactness."""
        for fps_num, fps_den, n_f in ((24000, 1001, 240), (30000, 1001, 300)):
            m = _make_media(duration_s=10.0, fps=24, tag=f"ntsc_{fps_num}")
            m = type(m).from_dict({**m.to_dict(), "fps_num": fps_num, "fps_den": fps_den, "nb_frames": n_f, "duration_s": n_f * fps_den / fps_num})
            cuts = [{"cut_id": "c_ntsc", "cut_start_frame": 60, "cut_end_frame_exclusive": 90, "pause_class": "DEAD_AIR"}]
            emap = build_edit_time_map(m, cuts, pacing_mode="auto")
            self.assertEqual(emap.output_total_frames, n_f - 30)
            exp_samp = exact_rational_samples(n_f - 30, fps_num, fps_den, 44100)
            self.assertEqual(emap.edited_total_samples, exp_samp)
            self.assertEqual(validate_edit_time_map(emap), [])

    def test_negative_forged_frame_count_tampered_hash_overlapping_and_protected_cuts_fail_closed(self) -> None:
        """Negative Controls: forged output_total_frames, tampered map_hash, overlapping cuts, and protected word intersection MUST FAIL."""
        media = _make_media(duration_s=10.0, fps=24, tag="neg_map")
        # 1. Cut intersecting protected word interval -> raises ValueError!
        with self.assertRaises(ValueError):
            build_edit_time_map(
                media,
                [{"cut_id": "bad_prot", "cut_start_s": 2.0, "cut_end_s": 3.5}],
                protected_intervals_s=[(1.5, 2.5)],
                pacing_mode="auto",
            )
        # 2. Overlapping cuts -> raises ValueError!
        with self.assertRaises(ValueError):
            build_edit_time_map(
                media,
                [
                    {"cut_id": "c1", "cut_start_frame": 40, "cut_end_frame_exclusive": 80},
                    {"cut_id": "c2", "cut_start_frame": 70, "cut_end_frame_exclusive": 100},
                ],
                pacing_mode="auto",
            )
        # 3. Forged output_total_frames (attempting to derive expected from output!) -> validate_edit_time_map FAILS!
        valid_map = build_edit_time_map(media, [{"cut_id": "ok", "cut_start_frame": 48, "cut_end_frame_exclusive": 72}], pacing_mode="auto")
        forged_d = valid_map.to_dict()
        forged_d["output_total_frames"] = 999
        forged_map = EditTimeMap.from_dict(forged_d)
        errs = validate_edit_time_map(forged_map)
        self.assertTrue(any("forged_output_total_frames" in e for e in errs))
        self.assertTrue(any("map_hash_tamper_mismatch" in e for e in errs))

    def test_roi_and_transcript_remap_guards_against_double_remap_and_word_loss(self) -> None:
        """Remapping ROI & Transcript is idempotent (no double-remap) and flags any word dropped by a bad cut."""
        media = _make_media(duration_s=10.0, fps=24, tag="remap_guard")
        emap = build_edit_time_map(media, [{"cut_id": "c_mid", "cut_start_frame": 96, "cut_end_frame_exclusive": 144}], pacing_mode="auto")
        roi = {
            "has_face_track": True,
            "keyframes": [
                {"frame": 24, "face": {"x": 0.3, "y": 0.2, "w": 0.3, "h": 0.3, "conf": 0.9}},
                {"frame": 120, "face": {"x": 0.3, "y": 0.2, "w": 0.3, "h": 0.3, "conf": 0.9}},  # Inside cut -> dropped
                {"frame": 168, "face": {"x": 0.3, "y": 0.2, "w": 0.3, "h": 0.3, "conf": 0.9}},  # Maps to 120
            ],
        }
        r1 = remap_roi_tracks_to_edited_domain(roi, emap)
        self.assertEqual([k["frame"] for k in r1["keyframes"]], [24, 120])
        # Calling a second time MUST NOT double-remap!
        r2 = remap_roi_tracks_to_edited_domain(r1, emap)
        self.assertEqual([k["frame"] for k in r2["keyframes"]], [24, 120])

        # Word inside omitted cut [4.0s..6.0s] MUST be flagged as error by map_word_spans!
        bad_words = [{"word_id": 0, "word": "secret", "source_start_s": 4.5, "source_end_s": 5.0}]
        mapped, w_errs = map_word_spans(emap, bad_words)
        self.assertEqual(len(mapped), 0)
        self.assertEqual(len(w_errs), 1)


if __name__ == "__main__":
    unittest.main()
