"""ML-01.R5.R5 Unit, Negative & Mutation Tests — Part 1:
  - Production CLI multi-source --job / --creative-mode director validation (D01)
  - Generalized arbitrary N>=3 sources, mixed FPS/sample-rates, strict end-exclusive mapping (D04)
  - Zero finance defaults & fail-closed unknown semantic kinds (D03)
  - Generic domain-agnostic TemporalFactLedger rejecting D08 silent rollbacks, premature states & weekday inversions (D08)
  - Strict semantic word preservation & fail-closed Rhythm Critic / unbiased Pacing Scorer (D07, D09)
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from motionlab.cli import build_parser
from motionlab.composition.calendar_editorial import build_new_wave_layer_html
from motionlab.composition.director_pipeline import validate_and_load_job_manifest
from motionlab.composition.generative_schema import SceneSpec, StoryboardSpec
from motionlab.composition.multi_source_timeline import (
    build_multi_source_edit_map,
    evaluate_watermark_authorization,
    exact_rational_samples,
    map_source_time_to_output,
    validate_multi_source_edit_map,
)
from motionlab.composition.scene_graph_v2 import compile_scene_v2_html, solve_dynamic_safe_layout
from motionlab.composition.visual_critic import run_rhythm_critic
from motionlab.semantic.pacing_director import score_pacing_candidate
from motionlab.semantic.speech_dedup import validate_semantic_preservation
from motionlab.semantic.temporal_ledger import (
    TemporalEntity,
    TemporalFactLedger,
    TemporalStateEvent,
    validate_temporal_fact_ledger,
)


class TestR5R5MultiSourceCliAndTemporalLedger(unittest.TestCase):
    def test_cli_parses_multisource_job_and_creative_mode_director(self) -> None:
        """D01: Production CLI exposes --job and --creative-mode director while keeping --input intact."""
        parser = build_parser()
        args_job = parser.parse_args(["edit", "--job", "jobs/sample.json", "--creative-mode", "director", "--dry-run"])
        self.assertEqual(args_job.job, "jobs/sample.json")
        self.assertEqual(args_job.creative_mode, "director")
        self.assertTrue(args_job.dry_run)
        self.assertIsNone(args_job.input)

        args_legacy = parser.parse_args(["edit", "--input", "clip.mp4", "--pacing", "preserve"])
        self.assertEqual(args_legacy.input, "clip.mp4")
        self.assertEqual(args_legacy.pacing, "preserve")
        self.assertEqual(args_legacy.creative_mode, "legacy")

    def test_job_manifest_fail_closed_unknown_keys_traversal_and_sha_mismatch(self) -> None:
        """D01/A0 security: Reject unknown top-level keys, path traversal, and SHA-256 tampering."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dummy = root / "clip.mp4"
            dummy.write_bytes(b"dummy-video-bytes-for-hash-check")

            # Unknown field -> fail closed
            j_unk = root / "bad_unknown.json"
            j_unk.write_text(json.dumps({"sources": [{"source_id": "S1", "path": str(dummy)}], "shell_cmd": "whoami"}), encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_and_load_job_manifest(j_unk)

            # Path traversal -> fail closed
            j_trav = root / "bad_trav.json"
            j_trav.write_text(json.dumps({"sources": [{"source_id": "S1", "path": "../../etc/passwd"}]}), encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_and_load_job_manifest(j_trav)

            # SHA-256 mismatch -> fail closed
            j_sha = root / "bad_sha.json"
            j_sha.write_text(json.dumps({"sources": [{"source_id": "S1", "path": str(dummy), "sha256": "0" * 64}]}), encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_and_load_job_manifest(j_sha)

            # Valid manifest loads cleanly with exact SHA
            j_ok = root / "ok.json"
            j_ok.write_text(json.dumps({"sources": [{"source_id": "S1", "path": str(dummy)}]}), encoding="utf-8")
            loaded = validate_and_load_job_manifest(j_ok)
            self.assertEqual(len(loaded["sources"]), 1)
            self.assertEqual(loaded["sources"][0]["source_id"], "S1")

    def test_three_arbitrary_sources_mixed_fps_sr_and_strict_end_exclusive_mapping(self) -> None:
        """D04: 3 arbitrary source IDs (24, 30, 30000/1001 fps; 44.1/48kHz) + strict [start, end) cut boundary."""
        srcs = [
            {"source_id": "S_INTRO", "fps_num": 24, "fps_den": 1, "sample_rate": 48000, "duration_s": 5.0},
            {"source_id": "S_DEMO", "fps_num": 30000, "fps_den": 1001, "sample_rate": 44100, "duration_s": 8.0},
            {"source_id": "S_OUTRO", "fps_num": 30, "fps_den": 1, "sample_rate": 48000, "duration_s": 4.0},
        ]
        kept = [
            {"source_id": "S_INTRO", "kind": "source_media", "start_s": 0.0, "end_s": 4.0},
            {"source_id": "S_DEMO", "kind": "source_media", "start_s": 1.0, "end_s": 4.0},
            {"source_id": "S_DEMO", "kind": "source_media", "start_s": 6.0, "end_s": 7.5},
            {"source_id": "S_OUTRO", "kind": "source_media", "start_s": 0.5, "end_s": 3.5},
        ]
        emap = build_multi_source_edit_map(srcs, kept, [], fps_num=30, fps_den=1, sample_rate=48000)
        self.assertEqual(validate_multi_source_edit_map(emap), [])
        self.assertEqual(emap.output_total_samples, exact_rational_samples(emap.output_total_frames, 30, 1, 48000))

        # Strict end-exclusive [start_s, end_s): cut boundary 4.000s and 4.01s inside cut span MUST map to None!
        self.assertIsNotNone(map_source_time_to_output(emap, "S_DEMO", 1.0))
        self.assertIsNotNone(map_source_time_to_output(emap, "S_DEMO", 3.99))
        self.assertIsNone(map_source_time_to_output(emap, "S_DEMO", 4.0))
        self.assertIsNone(map_source_time_to_output(emap, "S_DEMO", 4.02))
        self.assertIsNone(map_source_time_to_output(emap, "S_DEMO", 5.98))
        self.assertIsNotNone(map_source_time_to_output(emap, "S_DEMO", 6.0))

        # Watermark authorization reports arbitrary detected_sources dynamically (zero hardcoded A/B text)
        roi = {
            "S_INTRO": {"detected_bright_static_mark": True},
            "S_DEMO": {"detected_bright_static_mark": False},
            "S_OUTRO": {"detected_bright_static_mark": True},
        }
        wm = evaluate_watermark_authorization(roi, authorization_doc=None)
        self.assertEqual(wm["status"], "BLOCKED_WATERMARK_AUTHORIZATION")
        self.assertEqual(wm["detected_sources"], ["S_INTRO", "S_OUTRO"])

    def test_d03_zero_finance_defaults_and_fail_closed_unknown_kinds(self) -> None:
        """D03: Generic compilers MUST NOT embed finance/Olivia defaults and MUST fail closed on unknown kind."""
        forbidden_defaults = ("CHECKING AGAIN", "RENT DUE", "PAYDAY", "THE NEXT PAYCHECK PLAN", "THE PAYDAY GAP")
        for kind in ("editorial_hook", "calendar_gap_strip", "question_focus", "editorial_bridge_scene", "progressive_benefits", "cta_caption"):
            html_out = build_new_wave_layer_html(kind, {}, width=1080, height=1920, duration_s=2.0)
            for term in forbidden_defaults:
                self.assertNotIn(term, html_out.upper(), f"Hardcoded finance default {term!r} leaked in {kind}")

        # Unknown semantic kind raises ValueError instead of silently falling back to CTA!
        with self.assertRaises(ValueError):
            build_new_wave_layer_html("nonexistent_kind_xyz", {}, width=1080, height=1920, duration_s=2.0)

        with self.assertRaises(ValueError):
            compile_scene_v2_html({"scene_id": "bad", "treatment_mode": "unknown_operator_xyz", "duration_s": 2.0})

        # XSS / arbitrary HTML injection is safely escaped in SceneGraph v2
        safe_html, _ = compile_scene_v2_html(
            {
                "scene_id": "xss_test",
                "treatment_mode": "spatial_editorial_anchor",
                "duration_s": 2.0,
                "params": {"headline": "<script>alert(1)</script>"},
            }
        )
        self.assertNotIn("<script>alert(1)</script>", safe_html)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", safe_html)

    def test_d08_temporal_fact_ledger_rejects_silent_rollback_premature_and_weekday_inversion(self) -> None:
        """D08: Rejects R5.R4 $1200->$100->$1200 silent rollback, premature dependency & tutorial step rollback; allows explicit recap."""
        # 1. Finance decreasing sequence with silent rollback ($1200 -> $400 -> $100 -> $1200) -> FAIL
        led_bad = TemporalFactLedger(time_domain="edited")
        led_bad.add_entity(TemporalEntity("balance_remaining", "numeric", "Remaining Balance", unit="$", monotonic_direction="decreasing"))
        led_bad.add_event(TemporalStateEvent("e0", "balance_remaining", "s0", 0, 60, 1200, "$1,200", "$", "initial"))
        led_bad.add_event(TemporalStateEvent("e1", "balance_remaining", "s1", 60, 120, 400, "$400", "$", "delta", depends_on_event_id="e0"))
        led_bad.add_event(TemporalStateEvent("e2", "balance_remaining", "s2", 120, 180, 100, "$100", "$", "resolve", depends_on_event_id="e1"))
        led_bad.add_event(TemporalStateEvent("e3", "balance_remaining", "s3", 180, 240, 1200, "$1,200", "$", "set"))  # Silent rollback!
        errs_bad = validate_temporal_fact_ledger(led_bad)
        self.assertTrue(any("rollback" in e or "after_resolution" in e for e in errs_bad), f"Expected rollback violation, got {errs_bad}")

        # 2. Same replay WITH explicit is_recap=True and recap_label -> PASS
        led_recap = TemporalFactLedger(time_domain="edited")
        led_recap.add_entity(TemporalEntity("balance_remaining", "numeric", "Remaining Balance", unit="$", monotonic_direction="decreasing"))
        led_recap.add_event(TemporalStateEvent("e0", "balance_remaining", "s0", 0, 60, 1200, "$1,200", "$", "initial"))
        led_recap.add_event(TemporalStateEvent("e1", "balance_remaining", "s1", 60, 120, 100, "$100", "$", "resolve", depends_on_event_id="e0"))
        led_recap.add_event(TemporalStateEvent("e2", "balance_remaining", "s2", 120, 180, 1200, "$1,200", "$", "recap", is_recap=True, recap_label="RECAP: Starting Balance"))
        self.assertEqual(validate_temporal_fact_ledger(led_recap), [])

        # 3. Tutorial increasing step rollback (Step 1 -> Step 2 -> Step 1) -> FAIL (proves domain-agnostic!)
        led_tut = TemporalFactLedger(time_domain="edited")
        led_tut.add_entity(TemporalEntity("tut_step", "tutorial_step", "Step", monotonic_direction="increasing"))
        led_tut.add_event(TemporalStateEvent("t1", "tut_step", "s1", 0, 50, 1, "Step 1"))
        led_tut.add_event(TemporalStateEvent("t2", "tut_step", "s2", 50, 100, 2, "Step 2", depends_on_event_id="t1"))
        led_tut.add_event(TemporalStateEvent("t3", "tut_step", "s3", 100, 150, 1, "Step 1"))
        errs_tut = validate_temporal_fact_ledger(led_tut)
        self.assertTrue(len(errs_tut) >= 1)

        # 4. Premature state before dependency -> FAIL
        led_pre = TemporalFactLedger(time_domain="edited")
        led_pre.add_entity(TemporalEntity("score", "score", "Score", monotonic_direction="increasing"))
        led_pre.add_event(TemporalStateEvent("p0", "score", "s0", 90, 120, 10))
        led_pre.add_event(TemporalStateEvent("p1", "score", "s1", 30, 60, 50, depends_on_event_id="p0"))
        errs_pre = validate_temporal_fact_ledger(led_pre)
        self.assertTrue(any("premature_state_before_dependency" in e for e in errs_pre))

        # 5. Weekday inversion (Friday -> Tuesday) -> FAIL
        led_day = TemporalFactLedger(time_domain="edited")
        led_day.add_entity(TemporalEntity("timeline_day", "calendar", "Weekday", monotonic_direction="ordered_weekday"))
        led_day.add_event(TemporalStateEvent("d1", "timeline_day", "s1", 0, 60, "Friday"))
        led_day.add_event(TemporalStateEvent("d2", "timeline_day", "s2", 60, 120, "Tuesday"))
        errs_day = validate_temporal_fact_ledger(led_day)
        self.assertTrue(any("weekday" in e or "rollback" in e for e in errs_day))

    def test_d07_d09_word_order_swap_unbiased_pacing_and_fail_closed_rhythm_critic(self) -> None:
        """D07/D09: Swapped days fail preservation; balanced has 0 fixed bonus; missing evidence fails rhythm critic."""
        target = ["rent", "is", "due", "tuesday", "and", "i", "don't", "get", "paid", "until", "friday"]
        swapped = ["rent", "is", "due", "friday", "and", "i", "don't", "get", "paid", "until", "tuesday"]
        sw = validate_semantic_preservation([], target, swapped)
        self.assertFalse(sw["passed"])

        # D09: Pacing candidate scorer has ZERO fixed bonus for `balanced`
        fake_media = SimpleNamespace(duration_s=30.0)
        fake_sg = SimpleNamespace(beats=[])
        base_plan = {
            "eligible": True,
            "removed_total_s": 3.5,
            "cuts_count": 3,
            "decisions": [{"pause_class": "DEAD_AIR", "removed_s": 3.5, "retained_duration_s": 0.2}],
        }
        s_bal = score_pacing_candidate({"plan_id": "pacing_balanced", "variant": "balanced", **base_plan}, fake_media, fake_sg, niche="finance")
        s_tgh = score_pacing_candidate({"plan_id": "pacing_tight", "variant": "tight", **base_plan}, fake_media, fake_sg, niche="finance")
        self.assertEqual(s_bal["score"], s_tgh["score"])

        # D09: Rhythm critic MUST FAIL closed when cuts exist and bed/word_preservation/join_qa are missing!
        fake_map = SimpleNamespace(
            source_duration_s=20.0,
            output_duration_s=17.0,
            removed_total_s=3.0,
            omitted_spans=[SimpleNamespace(pause_class="DEAD_AIR", removed_s=3.0)],
        )
        sb = SimpleNamespace(
            fps_num=30,
            fps_den=1,
            width=1080,
            height=1920,
            niche="finance",
            scenes=[],
        )
        rc = run_rhythm_critic(
            edit_map=fake_map,
            pacing_plan={},
            edited_voice_bed_wav=None,
            word_preservation=None,
            cut_continuity=None,
            join_audio_qa=None,
            storyboard=sb,
            pixel_critic={"severe_findings_count": 0},
        )
        self.assertEqual(rc["objective_rhythm_status"], "FAIL")
        self.assertGreaterEqual(rc["high_severity_count"], 1)


if __name__ == "__main__":
    unittest.main()
