"""Multi-niche unit, negative & causal tests for A1/A2/A3 reusable modules:
  - motionlab/semantic/speech_dedup.py
  - motionlab/composition/multi_source_timeline.py
  - motionlab/composition/calendar_editorial.py

Includes finance, tutorial, comedy, gameplay, storytelling & silent-art fixtures
so speech repair logic is verified NOT hardcoded to a single video or niche.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from motionlab.composition.calendar_editorial import build_new_wave_layer_html
from motionlab.composition.multi_source_timeline import (
    build_multi_source_edit_map,
    evaluate_watermark_authorization,
    exact_rational_samples,
    map_source_time_to_output,
    remap_multi_source_words,
    validate_multi_source_edit_map,
)
from motionlab.semantic.speech_dedup import (
    choose_clean_speech_path,
    detect_cross_source_overlap,
    detect_disfluency_spans,
    normalize_token,
    validate_semantic_preservation,
)


def _words(tokens: list[tuple[str, float, float, float]]) -> list[dict]:
    return [
        {"word_id": i, "word": w, "source_start_s": s0, "source_end_s": s1, "confidence": c}
        for i, (w, s0, s1, c) in enumerate(tokens)
    ]


class TestV6SpeechDedupAndMultiSource(unittest.TestCase):
    def test_exact_and_restart_stutter_dedup_competes_keep_first_vs_keep_last_causally(self) -> None:
        """Detects partial restart stutter ('what you can act what you can actually spend') + 'mental math' repeat and competes paths causally."""
        story = "It shows what you can actually spend. Imagine shopping without doing mental math the entire time. I left the link in the caption."
        wb = _words([
            ("It", 1.00, 1.15, 0.99), ("shows", 1.15, 1.40, 0.99),
            ("what", 1.40, 1.55, 1.0), ("you", 1.55, 1.70, 1.0), ("can", 1.70, 1.85, 1.0),
            ("actually", 1.85, 2.20, 1.0), ("spend", 2.20, 2.55, 1.0),
            ("and", 2.95, 3.15, 0.85), ("what", 3.20, 3.35, 0.95), ("you", 3.35, 3.48, 0.99),
            ("can", 3.48, 3.60, 0.99), ("act", 3.60, 3.78, 0.92),
            ("what", 3.80, 3.95, 0.95), ("you", 3.95, 4.08, 1.0), ("can", 4.08, 4.20, 1.0),
            ("actually", 4.20, 4.55, 1.0), ("spend.", 4.55, 4.95, 1.0),
            ("Imagine", 5.60, 6.00, 0.99), ("shopping", 6.00, 6.40, 1.0),
            ("without", 6.40, 6.75, 1.0), ("doing", 6.75, 7.00, 1.0),
            ("mental", 7.00, 7.25, 1.0), ("math", 7.25, 7.55, 1.0),
            ("the", 7.65, 7.80, 0.80), ("mental", 7.80, 8.05, 0.95), ("math", 8.05, 8.35, 1.0),
            ("the", 8.50, 8.65, 1.0), ("entire", 8.65, 8.95, 1.0), ("time.", 8.95, 9.30, 1.0),
            ("I", 9.90, 10.10, 0.99), ("know.", 10.10, 10.35, 0.99),
            ("I", 10.70, 10.85, 1.0), ("left", 10.85, 11.05, 1.0), ("the", 11.05, 11.18, 1.0),
            ("link", 11.18, 11.38, 1.0), ("in", 11.38, 11.50, 1.0), ("the", 11.50, 11.62, 1.0), ("caption.", 11.62, 12.00, 1.0),
        ])
        dis = detect_disfluency_spans(wb, intended_story=story, source_id="B", duration_s=12.4)
        self.assertEqual(len(dis["repeat_groups"]), 2)
        self.assertEqual(len(dis["filler_candidates"]), 1)
        self.assertEqual(dis["filler_candidates"][0]["phrase"], ["i", "know"])

        # Causal override proof: forcing keep_first vs keep_last changes chosen_plan_id and plan_hash!
        srcs = [{"source_id": "B", "duration_s": 12.4}]
        plan_first = choose_clean_speech_path(srcs, None, {"B": dis}, candidate_score_overrides={"clean_keep_first": 40.0, "clean_keep_last": -40.0})
        plan_last = choose_clean_speech_path(srcs, None, {"B": dis}, candidate_score_overrides={"clean_keep_first": -40.0, "clean_keep_last": 40.0})
        self.assertEqual(plan_first["chosen_plan_id"], "clean_keep_first")
        self.assertEqual(plan_last["chosen_plan_id"], "clean_keep_last")
        self.assertNotEqual(plan_first["chosen_plan"]["plan_hash"], plan_last["chosen_plan"]["plan_hash"])

    def test_non_finance_intentional_repetition_preserved_in_tutorial_and_comedy(self) -> None:
        """Multi-niche safety: Intentional tutorial instruction & comedy callback repetition MUST NOT be deleted!"""
        # Tutorial intentional repeat present in intended story
        tut_story = "Click export once to verify. Then click export once again to lock."
        wt = _words([
            ("Click", 0.2, 0.5, 0.99), ("export", 0.5, 0.9, 0.99), ("once", 0.9, 1.2, 0.99),
            ("to", 1.2, 1.4, 0.99), ("verify.", 1.4, 1.9, 0.99), ("Then", 2.2, 2.5, 0.99),
            ("click", 2.5, 2.8, 0.99), ("export", 2.8, 3.2, 0.99), ("once", 3.2, 3.5, 0.99),
            ("again", 3.5, 3.9, 0.99), ("to", 3.9, 4.1, 0.99), ("lock.", 4.1, 4.6, 0.99),
        ])
        dis_tut = detect_disfluency_spans(wt, intended_story=tut_story, source_id="A", niche="tutorial", duration_s=5.0)
        self.assertEqual(len(dis_tut["repeat_groups"]), 0)
        self.assertGreaterEqual(len(dis_tut["protected_intentional"]), 1)

        # Comedy refrain without script -> preserved!
        wc = _words([
            ("Never", 0.2, 0.5, 0.99), ("trust", 0.5, 0.9, 0.99), ("free", 0.9, 1.2, 0.99),
            ("airport", 1.2, 1.6, 0.99), ("sushi.", 1.6, 2.1, 0.99),
            ("Seriously", 2.5, 3.0, 0.99), ("guys", 3.0, 3.3, 0.99),
            ("never", 3.4, 3.7, 0.99), ("trust", 3.7, 4.1, 0.99), ("free", 4.1, 4.4, 0.99), ("sushi.", 4.4, 4.9, 0.99),
        ])
        dis_com = detect_disfluency_spans(wc, intended_story=None, source_id="A", niche="comedy", duration_s=5.2)
        self.assertEqual(len(dis_com["repeat_groups"]), 0)

    def test_cross_source_garble_tail_truncated_prefix_and_missing_dialogue_detected_honestly(self) -> None:
        """Detects garbled AI tail in A, orphaned truncated prefix in B, and honestly flags MISSING_SOURCE_DIALOGUE!"""
        story = "Rent is due Tuesday and payday is Friday. That is why I built Focus Plan. It tracks your dates clearly."
        wa = _words([
            ("Rent", 0.2, 0.5, 0.99), ("is", 0.5, 0.65, 0.99), ("due", 0.65, 0.85, 0.99),
            ("Tuesday", 0.85, 1.25, 0.99), ("and", 1.25, 1.4, 0.99), ("payday", 1.4, 1.8, 0.99),
            ("is", 1.8, 1.95, 0.99), ("Friday.", 1.95, 2.4, 0.99),
            ("How", 3.4, 3.7, 0.9), ("morning", 3.7, 4.1, 0.8), ("spawns", 4.1, 4.5, 0.6), ("banana?", 4.5, 5.0, 0.5),
        ])
        wb = _words([
            ("plan.", 0.1, 0.45, 0.75),
            ("It", 0.95, 1.1, 0.99), ("tracks", 1.1, 1.45, 0.99), ("your", 1.45, 1.65, 0.99),
            ("dates", 1.65, 2.0, 0.99), ("clearly.", 2.0, 2.45, 0.99),
        ])
        bnd = detect_cross_source_overlap(wa, wb, intended_story=story, duration_a_s=5.5, duration_b_s=3.0)
        self.assertEqual(bnd["status"], "MISSING_SOURCE_DIALOGUE")
        self.assertIsNotNone(bnd["garbled_tail_a"])
        self.assertIsNotNone(bnd["truncated_prefix_b"])
        self.assertGreaterEqual(bnd["missing_dialogue_count"], 1)

    def test_semantic_preservation_fails_on_dropped_negation_day_or_residual_repeat(self) -> None:
        """Negative test: dropping negation ('don't') or date ('Tuesday') or leaving stutter MUST FAIL preservation!"""
        target = ["rent", "is", "due", "tuesday", "and", "i", "don't", "get", "paid", "until", "friday"]
        ok = validate_semantic_preservation([], target, target)
        self.assertTrue(ok["passed"])
        self.assertEqual(ok["critical_token_recall"], 1.0)

        # Drop negation "don't" -> FAIL
        no_neg = [t for t in target if t != "don't"]
        bad_neg = validate_semantic_preservation([], target, no_neg)
        self.assertFalse(bad_neg["passed"])
        self.assertIn("don't", bad_neg["missing_critical_tokens"])

        # Residual stutter -> FAIL
        stut = target[:3] + ["due", "tuesday", "due", "tuesday"] + target[4:]
        bad_stut = validate_semantic_preservation([], target, stut)
        self.assertFalse(bad_stut["passed"])
        self.assertGreaterEqual(len(bad_stut["residual_repeats"]), 1)

    def test_multi_source_edit_map_mixed_24_30fps_48_44k_exact_and_tamper_fails_closed(self) -> None:
        """Mixed 24fps/48kHz + 30fps/44.1kHz unified onto 30fps/48kHz CFR with zero drift; tampered hash fails closed."""
        srcs = [
            {"source_id": "A", "fps_num": 24, "fps_den": 1, "sample_rate": 48000, "duration_s": 15.0},
            {"source_id": "B", "fps_num": 30, "fps_den": 1, "sample_rate": 44100, "duration_s": 19.9},
        ]
        kept = [
            {"source_id": "A", "kind": "source_media", "start_s": 0.0, "end_s": 10.2},
            {"source_id": "bridge_editorial", "kind": "editorial_bridge", "start_s": 0.0, "end_s": 1.2},
            {"source_id": "B", "kind": "source_media", "start_s": 0.9, "end_s": 6.3},
            {"source_id": "B", "kind": "source_media", "start_s": 9.5, "end_s": 12.6},
        ]
        emap = build_multi_source_edit_map(srcs, kept, [], fps_num=30, fps_den=1, sample_rate=48000)
        self.assertEqual(validate_multi_source_edit_map(emap), [])
        self.assertEqual(emap.output_total_samples, exact_rational_samples(emap.output_total_frames, 30, 1, 48000))
        # Cut region in B (7.5s) MUST map to None!
        self.assertIsNone(map_source_time_to_output(emap, "B", 7.5))
        self.assertIsNotNone(map_source_time_to_output(emap, "B", 10.0))

        # Tamper hash -> fail closed!
        emap.map_hash = "0" * 64
        self.assertIn("map_hash_tampered", validate_multi_source_edit_map(emap))

    def test_watermark_authorization_fails_closed_without_verified_clean_export_license(self) -> None:
        """Unauthorized provenance watermark MUST report BLOCKED_WATERMARK_AUTHORIZATION and clean_release_eligible=False."""
        roi = {"A": {"detected_bright_static_mark": True, "native_bbox_xyxy": [864, 1704, 935, 1775]}}
        blocked = evaluate_watermark_authorization(roi, authorization_doc=None)
        self.assertEqual(blocked["status"], "BLOCKED_WATERMARK_AUTHORIZATION")
        self.assertFalse(blocked["clean_release_eligible"])
        self.assertTrue(blocked["ai_generated_disclosure_required"])

        # Explicit non-mandatory authorized clean export passes
        ok = evaluate_watermark_authorization(roi, authorization_doc={"clean_export_authorized": True, "license_basis": "owner_clean_export", "mandatory_provenance_mark": False})
        self.assertEqual(ok["status"], "AUTHORIZED_CLEANUP_ALLOWED")
        self.assertTrue(ok["clean_release_eligible"])

    def test_new_wave_calendar_editorial_html_zero_v5_finance_jargon_or_fake_currency(self) -> None:
        """New Wave HTML templates contain zero consumer QA labels ('CONCEPTUAL SCHEMATIC') and zero fabricated dollar balances."""
        for kind in ("editorial_hook", "calendar_gap_strip", "question_focus", "editorial_bridge_scene", "progressive_benefits", "cta_caption"):
            h = build_new_wave_layer_html(kind, {}, width=1080, height=1920, duration_s=2.0)
            self.assertNotIn("CONCEPTUAL", h.upper())
            self.assertNotIn("LIVE DETAIL", h.upper())
            self.assertNotIn("$1,200", h)
            self.assertIn("gsap.timeline", h)


if __name__ == "__main__":
    unittest.main()

