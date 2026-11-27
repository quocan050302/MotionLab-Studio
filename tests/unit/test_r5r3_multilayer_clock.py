"""Phase A5-A6 (ML-01.R5.R3): Multi-Layer HyperFrames Compiler, Rational FPS Clock & Analytic Motion Physics (D04, D13)."""

from __future__ import annotations

import math
import unittest

from motionlab.composition.continuous_timeline import (
    ElementKeyframe,
    SharedElementTrack,
    build_canonical_numeric_state_trace,
    build_material_motion,
    evaluate_motion_state,
    resolve_contact_event,
    sample_motion_path,
    validate_track_continuity,
)
from motionlab.composition.generative_compiler import (
    compile_scene_hyperframes_layers,
    compile_scene_remotion_props,
    validate_rational_fps_clock,
)
from motionlab.composition.generative_schema import (
    GENERATIVE_SCHEMA_VERSION,
    LayerSpec,
    SceneSpec,
    StoryboardSpec,
)
from motionlab.schemas import SchemaValidationError
from motionlab.semantic.eval_harness import _make_media, _make_transcript
from motionlab.semantic.story_graph import build_story_graph


class TestR5R3MultilayerAndClock(unittest.TestCase):
    def _base_sg(self):
        m = _make_media(10.0, 24, "ml_clock")
        tr = _make_transcript("Start with $500 in your bank account. $300 is for rent. You have $200 left.", 10.0)
        return m, tr, build_story_graph(tr, None, m)

    def test_d04_multi_hf_layers_compiled_in_z_order_without_dropping(self) -> None:
        m, tr, sg = self._base_sg()
        sc = SceneSpec(
            scene_id="sc_01_multi_hf",
            start_frame=0,
            end_frame_exclusive=60,
            role="hook_contrast",
            stage_mode="live_depth_matte",
            viewer_takeaway="2 distinct HF layers",
            persistent_ledger_step_id=sg.ledger.steps[0].step_id,
            layers=[
                LayerSpec("ly_fg", "hf_apple_money_count", "hyperframes", "foreground_graphics", 35, {
                    "from_cents": 50000, "to_cents": 50000, "eyebrow": "BALANCE", "render_mode": "typography_only", "token_count": 0,
                }),
                LayerSpec("ly_bg", "hf_apple_money_count", "hyperframes", "background_graphics", 15, {
                    "from_cents": 50000, "to_cents": 50000, "render_mode": "paper_tokens_only", "token_count": 4,
                }),
            ],
        )
        sb = StoryboardSpec(
            GENERATIVE_SCHEMA_VERSION, "AUTO_GENERATIVE", m.input_hash, tr.transcript_hash, sg.story_graph_hash,
            "finance", 0.95, "v", 720, 1280, 24, 1, 240, 42, {}, [sc],
        )
        overlays = compile_scene_hyperframes_layers(sc, sb, sg)
        self.assertEqual(len(overlays), 2)
        # Sorted strictly by ascending z_order (bg 15 before fg 35!)
        self.assertEqual([ov.layer_id for ov in overlays], ["ly_bg", "ly_fg"])
        self.assertEqual([ov.layer_role for ov in overlays], ["background_graphics", "foreground_graphics"])
        self.assertNotEqual(overlays[0].html_sha256, overlays[1].html_sha256)

        # Single numeric owner check does NOT miscount `paper_tokens_only` as a competing numeric counter!
        trace = build_canonical_numeric_state_trace(sb, sg)
        self.assertTrue(trace["single_numeric_owner_verified"])
        self.assertEqual(trace["violations"], [])

        # Duplicate HF layer_id fails closed!
        sc_dup = SceneSpec(
            scene_id="sc_dup", start_frame=0, end_frame_exclusive=48, role="hook_contrast", stage_mode="live_clean_bridge",
            viewer_takeaway="dup",
            layers=[
                LayerSpec("same_id", "hf_apple_money_count", "hyperframes", "background_graphics", 10, {}),
                LayerSpec("same_id", "hf_lt_mask_reveal", "hyperframes", "foreground_graphics", 30, {}),
            ],
        )
        with self.assertRaises(SchemaValidationError):
            compile_scene_hyperframes_layers(sc_dup, sb, sg)

    def test_d13_rational_fps_clock_exact_and_fail_closed_unsupported(self) -> None:
        m, tr, sg = self._base_sg()
        for num, den, exp_mode in (
            (24, 1, "exact_integer"),
            (25, 1, "exact_integer"),
            (60, 1, "exact_integer"),
            (24000, 1001, "rational_broadcast_ntsc_frame_indexed"),
            (30000, 1001, "rational_broadcast_ntsc_frame_indexed"),
        ):
            clk = validate_rational_fps_clock(num, den)
            self.assertEqual(clk["clock_mode"], exp_mode)
            self.assertEqual(clk["fps_num"], num)
            self.assertEqual(clk["fps_den"], den)

        # Unsupported arbitrary irrational fractional FPS MUST fail closed (no silent rounding!)
        with self.assertRaises(SchemaValidationError):
            validate_rational_fps_clock(1000, 37)
        with self.assertRaises(SchemaValidationError):
            validate_rational_fps_clock(0, 1)

        # Remotion props preserve exact fpsNum/fpsDen/clockMode on NTSC 24000/1001
        sc = SceneSpec("sc_ntsc", 0, 48, "hook_contrast", "live_clean_bridge", "ntsc")
        sb_ntsc = StoryboardSpec(
            GENERATIVE_SCHEMA_VERSION, "AUTO_GENERATIVE", m.input_hash, tr.transcript_hash, sg.story_graph_hash,
            "finance", 0.9, "v", 720, 1280, 24000, 1001, 240, 42, {}, [sc],
        )
        props = compile_scene_remotion_props(
            sc, sb_ntsc, sg, source_file=None, fg_alpha_file=None, prop_alpha_file=None,
            hf_graphic_file=None, hf_layer_role="foreground_graphics",
        )
        self.assertEqual(props["fpsNum"], 24000)
        self.assertEqual(props["fpsDen"], 1001)
        self.assertEqual(props["clockMode"], "rational_broadcast_ntsc_frame_indexed")

    def test_a6_analytic_motion_state_and_track_continuity(self) -> None:
        trk = SharedElementTrack(
            element_id="test_paper", visual_role="token", logical_state_id="ledger",
            continuity_group="ledger_world", start_frame=0, end_frame_exclusive=24, color="#243B32",
            keyframes=[
                ElementKeyframe(0, 0.10, 0.20, 0.30, 0.15, opacity=0.0),
                ElementKeyframe(10, 0.20, 0.12, 0.30, 0.15, opacity=1.0),
                ElementKeyframe(23, 0.20, 0.12, 0.30, 0.15, opacity=1.0),
            ],
        )
        st5 = evaluate_motion_state(trk, 5, fps=24.0, material="linen_paper")
        self.assertEqual(st5["motion_model"], "analytic_physically_inspired")
        self.assertTrue(all(math.isfinite(float(st5[k])) for k in ("x", "y", "vx", "vy", "ax", "ay", "speed", "flex_deg")))
        self.assertGreater(st5["speed"], 0.0)

        ev = resolve_contact_event(trk, fps=24.0)
        self.assertLessEqual(ev["release_frame"], ev["peak_velocity_frame"])
        self.assertLessEqual(ev["peak_velocity_frame"], ev["settle_contact_frame"])

        chk = validate_track_continuity(trk)
        self.assertTrue(chk["valid"], chk["errors"])

        pt = sample_motion_path((0.0, 0.0), (0.5, -0.3), (1.0, 0.0), 0.5)
        self.assertAlmostEqual(pt["x"], 0.875, places=3)  # eased u=0.5 -> t=0.875, linear x when x1=(x0+x2)/2
        self.assertAlmostEqual(pt["y"], -0.065625, places=3)  # curved quadratic Bezier arch
        mat = build_material_motion("linen_paper", 0.5, speed=0.6)
        self.assertEqual(mat["motion_model"], "analytic_physically_inspired")


if __name__ == "__main__":
    unittest.main()
