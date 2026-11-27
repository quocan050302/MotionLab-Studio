"""Phase A1-A4 (ML-01.R5.R3): Unit, Negative & Causal Proof Tests for Causal Global Director (D01, D02, D03, D12, D17)."""

from __future__ import annotations

import unittest

from motionlab.composition.catalog_search import (
    RETRIEVAL_MODE,
    compute_catalog_index_hash,
    search_primitives_hybrid,
)
from motionlab.composition.primitive_registry import load_unified_registry
from motionlab.config import AnalysisConfig, EditConfig
from motionlab.schemas import SchemaValidationError
from motionlab.semantic.eval_harness import _make_media, _make_transcript, run_deterministic_reasoning_eval
from motionlab.semantic.generative_director import direct_generative_storyboard
from motionlab.semantic.model_profiles import compute_director_cache_key, inspect_registered_models
from motionlab.semantic.narrative_brain import (
    NO_ELIGIBLE_CANDIDATE_ID,
    build_evidence_index,
    build_narrative_contract,
    build_state_graph,
    rank_global_candidates,
    rank_scene_candidates,
)
from motionlab.semantic.story_graph import build_story_graph


class TestR5R3CausalDirector(unittest.TestCase):
    def _sample_finance_bundle(self):
        media = _make_media(18.0, 24, "causal_fin")
        text = (
            "Having $1,500 in your checking account doesn't mean you can spend $1,500. "
            "Let's say $900 is for rent, $300 for food and $150 for transit. "
            "You actually have just $150 left."
        )
        tr = _make_transcript(text, 18.0)
        roi = {
            "has_face_track": True,
            "has_task_track": True,
            "summary": {"face_present_ratio": 0.92, "task_present_ratio": 0.65},
            "keyframes": [{"frame": 12, "face": {"x": 0.32, "y": 0.14, "w": 0.30, "h": 0.28, "conf": 0.92}, "task": {"x": 0.30, "y": 0.62, "w": 0.40, "h": 0.28, "conf": 0.72}}],
        }
        sg = build_story_graph(tr, roi, media)
        return media, tr, roi, sg

    def test_d01_causal_rank_reversal_changes_compiled_storyboard_and_hash(self) -> None:
        """P0 Causal Proof: Reversing eligible candidates A vs B MUST alter compiled primitives, layout, sound & storyboard_hash!"""
        media, tr, roi, sg = self._sample_finance_bundle()
        cfg = EditConfig(offline=True, composition_mode="generative", allow_remotion=True, allow_hyperframes=True, analysis=AnalysisConfig(use_cache=False), seed=42)
        stub_llm = {"niche": "finance", "niche_confidence": 0.95, "core_value_proposition": "Clarify spendable cash", "beat_plans": []}
        hook_beat_id = next(b.beat_id for b in sg.beats if b.speech_act == "hook_contrast")

        # Run A: favor depth multi-HF paper tokens
        sb_a, audit_a = direct_generative_storyboard(
            media, sg, roi, config=cfg, matte_available=True, transcript=tr,
            raw_llm_override=stub_llm,
            candidate_score_overrides={f"{hook_beat_id}_depth_multilayer_paper": 50.0, f"{hook_beat_id}_clean_counter": -50.0},
        )
        # Run B: reverse rank to favor clean single-engine Remotion counter
        sb_b, audit_b = direct_generative_storyboard(
            media, sg, roi, config=cfg, matte_available=True, transcript=tr,
            raw_llm_override=stub_llm,
            candidate_score_overrides={f"{hook_beat_id}_depth_multilayer_paper": -50.0, f"{hook_beat_id}_clean_counter": 50.0},
        )

        self.assertTrue(audit_a["brain_audit"]["causal_selection_before_compile"])
        self.assertNotEqual(sb_a.scenes[0].chosen_candidate_id, sb_b.scenes[0].chosen_candidate_id)
        self.assertEqual(sb_a.scenes[0].chosen_candidate_id, f"{hook_beat_id}_depth_multilayer_paper")
        self.assertEqual(sb_b.scenes[0].chosen_candidate_id, f"{hook_beat_id}_clean_counter")
        self.assertEqual(sb_a.scenes[0].stage_mode, "live_depth_matte")
        self.assertEqual(sb_b.scenes[0].stage_mode, "live_clean_bridge")
        self.assertNotEqual([l.primitive_id for l in sb_a.scenes[0].layers], [l.primitive_id for l in sb_b.scenes[0].layers])
        self.assertNotEqual(sb_a.storyboard_hash, sb_b.storyboard_hash)
        self.assertNotEqual(sb_a.selected_graph_hash, sb_b.selected_graph_hash)

    def test_d02_rank_scene_candidates_fail_closed_on_empty_and_all_rejected(self) -> None:
        """P0 Fix D02: Empty candidates must NOT raise IndexError; all-rejected must NOT return ineligible scored[0]!"""
        media, tr, roi, sg = self._sample_finance_bundle()
        ev = build_evidence_index(media, tr, sg, roi_tracks=roi)
        ct = build_narrative_contract(ev, sg)
        nsg = build_state_graph(ev, sg, ct)
        b0, bs0 = sg.beats[0], nsg.beat_states[0]

        chosen_empty, scored_empty = rank_scene_candidates([], b0, bs0, ct)
        self.assertEqual(chosen_empty["candidate_id"], NO_ELIGIBLE_CANDIDATE_ID)
        self.assertFalse(chosen_empty["eligible"])
        self.assertEqual(scored_empty, [])

        all_bad = [
            {"candidate_id": "bad_unbridged", "stage_mode": "full_graphic_cutaway", "remotion_primitive_id": "rem_rc_flip_page", "hyperframes_primitive_id": "none", "transition_in": "none", "transition_out": "none"},
            {"candidate_id": "bad_empty", "stage_mode": "live_clean_bridge", "remotion_primitive_id": "none", "hyperframes_primitive_id": "none", "transition_in": "none", "transition_out": "none"},
        ]
        chosen_bad, scored_bad = rank_scene_candidates(all_bad, b0, bs0, ct)
        self.assertEqual(chosen_bad["candidate_id"], NO_ELIGIBLE_CANDIDATE_ID)
        self.assertFalse(chosen_bad["eligible"])
        self.assertEqual(len(scored_bad), 2)
        self.assertTrue(all(not r["eligible"] for r in scored_bad))

        # Verify bounded replan resolves an all-rejected beat cleanly (or blocks if no verified primitives exist)
        sel, meta = rank_global_candidates(
            [{"beat": b0, "beat_state": bs0, "candidates": all_bad}],
            ct,
            verified_rem_ids={"rem_ru_masked_slide"},
            verified_hf_ids={"hf_mk_progress_stat"},
        )
        self.assertEqual(meta["status"], "OK")
        self.assertGreaterEqual(len(meta["replan_events"]), 1)
        self.assertTrue(sel[0]["chosen"]["eligible"])

        # With empty verified primitive sets, replan cannot succeed -> explicit BLOCKED!
        sel_blk, meta_blk = rank_global_candidates(
            [{"beat": b0, "beat_state": bs0, "candidates": all_bad}],
            ct,
            verified_rem_ids=set(),
            verified_hf_ids=set(),
        )
        self.assertEqual(meta_blk["status"], "BLOCKED_NO_ELIGIBLE_CANDIDATE")
        self.assertEqual(sel_blk, [])

    def test_d03_real_transcript_provenance_vs_approx_fallback(self) -> None:
        """P0 Fix D03: Real Transcript preserves exact provenance; omitted Transcript stays honestly BEAT_LEVEL_APPROX."""
        media, tr, roi, sg = self._sample_finance_bundle()
        cfg = EditConfig(offline=True, composition_mode="generative", allow_remotion=True, allow_hyperframes=True, analysis=AnalysisConfig(use_cache=False), seed=42)
        stub_llm = {"niche": "finance", "niche_confidence": 0.9, "core_value_proposition": "Test", "beat_plans": []}

        _, audit_real = direct_generative_storyboard(media, sg, roi, config=cfg, matte_available=False, transcript=tr, raw_llm_override=stub_llm)
        self.assertTrue(audit_real["brain_audit"]["real_transcript_supplied"])
        self.assertEqual(audit_real["brain_audit"]["timestamp_quality"], "WORD_ALIGNED_APPROX")
        self.assertTrue(audit_real["brain_audit"]["word_timestamps_exact"])

        _, audit_approx = direct_generative_storyboard(media, sg, roi, config=cfg, matte_available=False, transcript=None, raw_llm_override=stub_llm)
        self.assertFalse(audit_approx["brain_audit"]["real_transcript_supplied"])
        self.assertEqual(audit_approx["brain_audit"]["timestamp_quality"], "BEAT_LEVEL_APPROX")
        self.assertFalse(audit_approx["brain_audit"]["word_timestamps_exact"])

    def test_d12_hybrid_ontology_retrieval_anti_overuse_and_catalog_hash(self) -> None:
        reg = load_unified_registry()
        res1 = search_primitives_hybrid(reg, "contradiction negation contrast spend", has_ledger=True, has_matte=True, top_k=5)
        self.assertEqual(res1["retrieval_mode"], RETRIEVAL_MODE)
        self.assertFalse(res1["embedding_model_used"])
        self.assertIn("contradiction", res1["matched_concepts"])
        top_ids = [c["primitive_id"] for c in res1["candidates"]]
        self.assertTrue(any("strikethrough" in pid for pid in top_ids))

        # Anti-overuse decay demotes heavily used primitive
        top0 = res1["candidates"][0]["primitive_id"]
        res2 = search_primitives_hybrid(reg, "contradiction negation contrast spend", has_ledger=True, has_matte=True, top_k=5, used_counts={top0: 4})
        s1 = next(c["score"] for c in res1["candidates"] if c["primitive_id"] == top0)
        s2 = next(c["score"] for c in res2["candidates"] if c["primitive_id"] == top0)
        self.assertLess(s2, s1)
        self.assertEqual(res1["catalog_index_hash"], compute_catalog_index_hash(reg))

    def test_d17_director_cache_key_invalidates_on_any_semantic_input_change(self) -> None:
        base = dict(
            schema_version="3.0.0",
            system_prompt="sys A",
            user_prompt="usr A",
            json_schema={"type": "object", "required": ["niche"]},
            model_id="qwen2_5_3b",
            model_sha256="abc123",
            binary_path=None,
            catalog_hash="cat_1",
            story_graph_hash="sg_1",
            evidence_index_hash="ev_1",
            roi_summary={"face_present_ratio": 0.8},
            seed=42,
            matte_available=True,
        )
        k0 = compute_director_cache_key(**base)["full_hash"]
        for field, mutated in (
            ("user_prompt", "usr B"),
            ("json_schema", {"type": "object", "required": ["niche", "beat_plans"]}),
            ("catalog_hash", "cat_2"),
            ("roi_summary", {"face_present_ratio": 0.2}),
            ("model_sha256", "def456"),
            ("seed", 99),
        ):
            mod = dict(base)
            mod[field] = mutated
            self.assertNotEqual(k0, compute_director_cache_key(**mod)["full_hash"], f"Cache key failed to invalidate on {field}")

    def test_36_case_independent_eval_and_model_registry_honesty(self) -> None:
        ev = run_deterministic_reasoning_eval()
        self.assertGreaterEqual(ev["cases_total"], 36)
        self.assertEqual(ev["cases_failed"], 0, f"Failed: {ev['failed_cases']}")
        reg = inspect_registered_models()
        self.assertIn("host_compute", reg)
        self.assertIn(reg["model_upgrade_status"], ("MODEL_UPGRADE_NOT_RUN", "MODEL_UPGRADE_AVAILABLE"))


if __name__ == "__main__":
    unittest.main()
