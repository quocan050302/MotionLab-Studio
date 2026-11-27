"""Phase R2-A1 / R2-A8 (ML-01.R5.R2): Unit & Negative Tests for Stateful AI Brain, Playbook & 16-Case Independent Eval."""

from __future__ import annotations

import unittest

from motionlab.semantic.eval_harness import run_deterministic_reasoning_eval
from motionlab.semantic.narrative_brain import (
    propose_scene_graphs,
    rank_scene_candidates,
    retrieve_playbook_rules,
)
from motionlab.semantic.story_graph import StoryBeatNode


class TestR5R2AIBrainAndEval(unittest.TestCase):
    def test_independent_16_case_reasoning_eval_harness_passes_all_p0_fixtures(self) -> None:
        report = run_deterministic_reasoning_eval()
        self.assertGreaterEqual(report["cases_total"], 16)
        self.assertEqual(
            report["cases_failed"],
            0,
            msg=f"Independent eval failures: {report['failed_cases']}",
        )
        self.assertGreaterEqual(report["accuracy"], 0.90)

    def test_playbook_returns_approved_defect_derived_rules(self) -> None:
        fin_rules = retrieve_playbook_rules("finance", approved_only=True)
        rule_ids = {r["rule_id"] for r in fin_rules}
        self.assertIn("PB-01", rule_ids)
        self.assertIn("PB-02", rule_ids)
        self.assertIn("PB-03", rule_ids)
        self.assertIn("PB-05", rule_ids)

    def test_candidate_ranker_hard_rejects_role_mismatch_and_unbridged_cutaway(self) -> None:
        beat = StoryBeatNode(
            beat_id="beat_03_allocation_step",
            order=3,
            unit_ids=[],
            word_ids=[18, 19, 20],
            start_s=5.75,
            end_s=8.63,
            start_frame=138,
            end_frame_exclusive=207,
            verbatim_quote="800 is for bills",
            speech_act="allocation_step",
            narrative_role="allocation_bills",
            ledger_step_id="ledger_step_01",
        )
        from motionlab.semantic.narrative_brain import BeatStateSnapshot, NarrativeContract

        bs = BeatStateSnapshot(
            beat_id=beat.beat_id,
            order=3,
            start_frame=138,
            end_frame_exclusive=207,
            speech_act="allocation_step",
            primary_focal_object="cash_pool",
            desired_emotion="tactile_proof",
            pacing_energy="focused_build",
            speaker_presence="macro_hands",
            continuity_group="ledger_world",
            state_before={"ledger_active": True, "available_cents": 120000},
            state_after={"ledger_active": True, "available_cents": 40000},
            state_transition_cause="commit_allocation:Bills:-$800",
            must_not_reveal_before_word_id=18,
            viewer_takeaway="$800 committed to Bills",
            sound_semantic_context={"speech_act": "allocation_step"},
        )
        contract = NarrativeContract(
            schema_version="2.0.0",
            inferred_niche="finance",
            niche_confidence=0.95,
            narrative_pattern="financial_conservation",
            video_purpose="test",
            hook_question="q",
            central_conflict="c",
            proof_mechanism="p",
            payoff_takeaway="t",
            product_value_proposition="v",
            cta_summary="cta",
            tone="editorial",
            allow_fractional_cents=False,
            is_hypothetical_example=True,
            has_authorized_product_asset=False,
        )
        bad_raw = {
            "stage_mode": "full_graphic_cutaway",
            "remotion_primitive_id": "rem_rc_flip_page",  # product-only primitive on allocation beat!
            "hyperframes_primitive_id": "hf_mk_specs_list",
            "transition_in": "none",
        }
        cands = propose_scene_graphs(
            beat,
            bs,
            [],
            llm_proposal=bad_raw,
            matte_available=True,
            is_talking_head=True,
            alloc_index=0,
            verified_rem_ids={"rem_ru_track_matte", "rem_rc_flip_page"},
            verified_hf_ids={"hf_mk_progress_stat", "hf_mk_specs_list"},
        )
        chosen, ranked = rank_scene_candidates(cands, beat, bs, contract)
        self.assertTrue(chosen["eligible"])
        self.assertEqual(chosen["stage_mode"], "tracked_macro_split")
        llm_row = next(r for r in ranked if r["candidate_id"].endswith("_optLLM_raw"))
        self.assertFalse(llm_row["eligible"])
        self.assertIn("role_mismatch_product_primitive_on_non_product_beat", llm_row["hard_rejects"])
        self.assertIn("unbridged_cutaway_boundary" if False else "unbridged_hard_cut_cutaway", llm_row["hard_rejects"])


if __name__ == "__main__":
    unittest.main()
