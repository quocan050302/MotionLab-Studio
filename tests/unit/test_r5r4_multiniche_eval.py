"""Phase A2/D04 (ML-01.R5.R4): Corrected Independent Semantic Oracle & 44-Case Multi-Niche Generalization Tests."""

from __future__ import annotations

import unittest

from motionlab.composition.generative_schema import GENERATIVE_SCHEMA_VERSION, LayerSpec, SceneSpec, StoryboardSpec
from motionlab.config import AnalysisConfig, EditConfig
from motionlab.semantic.eval_harness import (
    _make_media,
    _make_transcript,
    build_independent_eval_fixtures,
    evaluate_storyboard_semantics,
    run_deterministic_reasoning_eval,
)
from motionlab.semantic.generative_director import direct_generative_storyboard
from motionlab.semantic.story_graph import build_story_graph


class TestR5R4MultiNicheAndOracle(unittest.TestCase):
    def test_44_case_multi_niche_deterministic_eval_passes_all_cases(self) -> None:
        """All 44 independent multi-niche reasoning fixtures (including EV37..EV44) pass."""
        rep = run_deterministic_reasoning_eval()
        self.assertEqual(rep["cases_total"], 44)
        self.assertEqual(rep["cases_passed"], 44, f"Failed cases: {rep['failed_cases']}")
        self.assertEqual(rep["accuracy"], 1.0)

    def test_d04_mutation_wrong_niche_with_valid_hash_must_fail_semantic_oracle(self) -> None:
        """P0 Mutation Proof (Fixes D04): A storyboard with valid hash but wrong niche (education/gameplay -> product) MUST FAIL!"""
        fx_map = {f["case_id"]: f for f in build_independent_eval_fixtures()}
        for cid, bad_niche in (("EV24_education_multi_step_no_money", "product"), ("EV32_gameplay_hud_strict_ui_protection", "product")):
            fx = fx_map[cid]
            media = _make_media(duration_s=fx["duration_s"], tag=cid)
            tr = _make_transcript(fx["text"], duration_s=fx["duration_s"])
            sg = build_story_graph(tr, None, media)

            # Mutated storyboard: VALID storyboard_hash, WRONG niche!
            sb_bad = StoryboardSpec(
                GENERATIVE_SCHEMA_VERSION, "AUTO_GENERATIVE", media.input_hash, tr.transcript_hash, sg.story_graph_hash,
                bad_niche, 0.90, "v", 720, 1280, 24, 1, media.duration_frames, 42, {},
                [SceneSpec("sc_01", 0, 48, "generic", "live_clean_bridge", "t", layers=[LayerSpec("l1", "hf_apple_money_count", "hyperframes", "foreground_graphics", 30, {})])],
            )
            sb_bad.storyboard_hash = sb_bad.compute_hash()
            self.assertTrue(bool(sb_bad.storyboard_hash))  # Hash is 100% valid!

            oracle = evaluate_storyboard_semantics(fx, sb_bad, sg)
            self.assertTrue(oracle["schema_hash_valid"])
            self.assertFalse(oracle["niche_semantic_ok"])
            self.assertFalse(oracle["semantic_passed"])
            self.assertGreaterEqual(len(oracle["failure_reasons"]), 1)

    def test_d04_director_reconciles_raw_llm_wrong_niche_on_gameplay_and_education(self) -> None:
        """When small 3B LLM mislabels gameplay/education as 'product', causal reconciliation repairs niche & protects HUD!"""
        cfg = EditConfig(offline=True, composition_mode="generative", allow_remotion=True, allow_hyperframes=True, analysis=AnalysisConfig(use_cache=False), seed=42)
        fx_map = {f["case_id"]: f for f in build_independent_eval_fixtures()}
        # Deliberately feed wrong raw LLM niche="product" on EV32 gameplay!
        fx = fx_map["EV32_gameplay_hud_strict_ui_protection"]
        media = _make_media(duration_s=fx["duration_s"], tag="ev32_mut")
        tr = _make_transcript(fx["text"], duration_s=fx["duration_s"])
        roi = {"has_face_track": False, "has_task_track": True, "keyframes": [{"frame": 12, "task": {"x": 0.1, "y": 0.1, "w": 0.8, "h": 0.8, "conf": 0.9}}]}
        sg = build_story_graph(tr, roi, media)
        sb, _ = direct_generative_storyboard(
            media, sg, roi, config=cfg, matte_available=False, transcript=tr,
            raw_llm_override={"niche": "product", "niche_confidence": 0.8, "core_value_proposition": "bad", "beat_plans": []},
        )
        self.assertEqual(sb.niche, "gameplay")
        self.assertEqual(len(sb.scenes), 0)  # Strict HUD protection!
        oracle = evaluate_storyboard_semantics(fx, sb, sg)
        self.assertTrue(oracle["semantic_passed"], oracle["failure_reasons"])


if __name__ == "__main__":
    unittest.main()
