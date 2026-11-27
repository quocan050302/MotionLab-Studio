"""Phase R2-A2 / R2-A8 (ML-01.R5.R2): Unit & Negative Tests for Continuous Timeline, Shared State & Integer-Dollar Lock."""

from __future__ import annotations

import unittest

from motionlab.composition.art_direction import solve_layout_with_protection
from motionlab.composition.continuous_timeline import (
    ElementKeyframe,
    SharedElementTrack,
    build_canonical_numeric_state_trace,
    compile_continuous_timeline,
    evaluate_shared_state,
)
from motionlab.composition.generative_schema import (
    GENERATIVE_SCHEMA_VERSION,
    LayerSpec,
    SceneSpec,
    StoryboardSpec,
    TransitionSpec,
)
from motionlab.composition.primitive_adapters import build_adapted_hyperframes_html
from motionlab.semantic.eval_harness import _make_media, _make_transcript
from motionlab.semantic.story_graph import build_story_graph


def _build_sample_storyboard_and_graph(with_bridges: bool = True, duplicate_ribbon: bool = False):
    media = _make_media(duration_s=12.0, fps=24, tag="cont")
    tr = _make_transcript(
        "Just because you have $1,200 in your bank account doesn't mean you have $1,200 to spend. "
        "Let's say $800 is for bills. You actually have $400 left.",
        duration_s=11.0,
    )
    sg = build_story_graph(tr, None, media)
    s_open = sg.ledger.steps[0].step_id
    s_bill = sg.ledger.steps[1].step_id
    s_rem = sg.ledger.steps[2].step_id

    sc1 = SceneSpec(
        scene_id="sc_01_hook",
        start_frame=0,
        end_frame_exclusive=96,
        role="hook_contrast",
        stage_mode="live_depth_matte",
        viewer_takeaway="$1,200 opening balance",
        persistent_ledger_step_id=s_open,
        layers=[LayerSpec("ly_1", "hf_apple_money_count", "hyperframes", "background_graphics", 10, {"from_cents": 120000, "to_cents": 120000})],
    )
    sc2 = SceneSpec(
        scene_id="sc_02_bills",
        start_frame=96,
        end_frame_exclusive=192,
        role="allocation_step",
        stage_mode="tracked_macro_split",
        viewer_takeaway="-$800 Bills",
        persistent_ledger_step_id=s_bill,
        show_persistent_ribbon=duplicate_ribbon,
        layers=[LayerSpec("ly_2", "hf_mk_progress_stat", "hyperframes", "foreground_graphics", 20, {"amount_cents": 80000, "denom_cents": 120000, "remaining_cents": 40000})],
    )
    sc3 = SceneSpec(
        scene_id="sc_03_payoff",
        start_frame=192,
        end_frame_exclusive=264,
        role="payoff_reveal",
        stage_mode="full_graphic_cutaway",
        viewer_takeaway="$400 left",
        persistent_ledger_step_id=s_rem,
        transition_in=TransitionSpec(kind="line_led" if with_bridges else "none", duration_frames=12 if with_bridges else 0),
        parameters={"transition_out": "paper_fold_out" if with_bridges else "none", "transition_out_frames": 12 if with_bridges else 0},
        layers=[LayerSpec("ly_3", "hf_apple_money_count", "hyperframes", "foreground_graphics", 30, {"from_cents": 120000, "to_cents": 40000})],
    )
    sb = StoryboardSpec(
        schema_version=GENERATIVE_SCHEMA_VERSION,
        mode="AUTO_GENERATIVE",
        source_hash=media.input_hash,
        transcript_hash=tr.transcript_hash,
        story_graph_hash=sg.story_graph_hash,
        niche="finance",
        niche_confidence=0.95,
        core_value_proposition="test",
        width=720,
        height=1280,
        fps_num=24,
        fps_den=1,
        total_frames=288,
        seed=42,
        palette={"ivory": "#F5F0E6", "forest": "#243B32", "espresso": "#40382E", "brass": "#BE9A59", "clay": "#AF745E", "sage": "#84977A"},
        scenes=[sc1, sc2, sc3],
    )
    return sb, sg


class TestR5R2ContinuityAndLedger(unittest.TestCase):
    def test_evaluate_shared_state_deterministic_boundary_continuity(self) -> None:
        trk = SharedElementTrack(
            element_id="persistent_ledger_balance",
            visual_role="canonical_amount_display",
            logical_state_id="ledger_world",
            continuity_group="ledger_world",
            start_frame=0,
            end_frame_exclusive=100,
            color="#243B32",
            keyframes=[
                ElementKeyframe(global_frame=0, x=0.067, y=0.086, w=0.788, h=0.18, opacity=0.0, easing="linear"),
                ElementKeyframe(global_frame=10, x=0.067, y=0.086, w=0.788, h=0.18, opacity=1.0, easing="linear"),
                ElementKeyframe(global_frame=50, x=0.067, y=0.086, w=0.788, h=0.18, opacity=1.0, easing="ease_in_out_cubic"),
            ],
        )
        s5 = evaluate_shared_state(trk, 5)
        self.assertAlmostEqual(s5["opacity"], 0.5, places=3)
        s49 = evaluate_shared_state(trk, 49)
        s50 = evaluate_shared_state(trk, 50)
        self.assertAlmostEqual(s49["x"], s50["x"], places=4)
        self.assertAlmostEqual(s49["y"], s50["y"], places=4)

    def test_continuous_timeline_valid_with_cutaway_bridges_and_single_numeric_owner(self) -> None:
        sb, sg = _build_sample_storyboard_and_graph(with_bridges=True, duplicate_ribbon=False)
        layout, _ = solve_layout_with_protection(sb, None, 720, 1280)
        ctl = compile_continuous_timeline(sb, sg, layout)
        self.assertTrue(ctl["continuous_timeline_valid"], msg=ctl["validation_errors"])
        self.assertTrue(ctl["numeric_state_trace"]["integer_dollar_lock"])
        self.assertTrue(ctl["numeric_state_trace"]["single_numeric_owner_verified"])
        # Verify cutaway exit boundary f264 has CUTAWAY_BRIDGE outgoing handle
        exit_bnd = next(b for b in ctl["boundaries"] if b["boundary_frame"] == 264)
        self.assertEqual(exit_bnd["boundary_class"], "CUTAWAY_BRIDGE")
        self.assertEqual(exit_bnd["bridge_direction"], "outgoing_from_prev")
        self.assertEqual(exit_bnd["handle_frames"], 12)

    def test_negative_unbridged_cutaway_hard_cut_fails_continuous_timeline(self) -> None:
        sb, sg = _build_sample_storyboard_and_graph(with_bridges=False, duplicate_ribbon=False)
        layout, _ = solve_layout_with_protection(sb, None, 720, 1280)
        ctl = compile_continuous_timeline(sb, sg, layout)
        self.assertFalse(ctl["continuous_timeline_valid"])
        self.assertTrue(any("unbridged_cutaway_boundary" in e for e in ctl["validation_errors"]))

    def test_negative_duplicate_numeric_counters_flagged_d03(self) -> None:
        sb, sg = _build_sample_storyboard_and_graph(with_bridges=True, duplicate_ribbon=True)
        trace = build_canonical_numeric_state_trace(sb, sg)
        self.assertFalse(trace["single_numeric_owner_verified"])
        self.assertTrue(any("duplicate_numeric_counters:sc_02_bills" in v for v in trace["violations"]))

    def test_hf_apple_money_count_enforces_integer_dollar_lock_d01_d05(self) -> None:
        html_int = build_adapted_hyperframes_html(
            "hf_apple_money_count", 720, 1280, 24.0, 2.5,
            {"from_cents": 120000, "to_cents": 10000, "currency_symbol": "$"},
        )
        self.assertIn("const ALLOW_FRAC = false;", html_int)
        self.assertIn("Math.round(state.c / 100) * 100", html_int)

        html_frac = build_adapted_hyperframes_html(
            "hf_apple_money_count", 720, 1280, 24.0, 2.5,
            {"from_cents": 5000, "to_cents": 3025, "currency_symbol": "$"},
        )
        self.assertIn("const ALLOW_FRAC = true;", html_frac)

    def test_hf_organic_light_leak_protects_center_face_d08(self) -> None:
        html_ll = build_adapted_hyperframes_html(
            "hf_organic_light_leak", 720, 1280, 24.0, 3.0,
            {"opacity": 0.90},  # intentionally high -> must clamp <= 0.22!
        )
        self.assertIn("mask-image: radial-gradient", html_ll)
        self.assertIn("opacity: 0.220", html_ll)


if __name__ == "__main__":
    unittest.main()
