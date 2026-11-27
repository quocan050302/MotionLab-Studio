"""Phase A5 (ML-01.R5.R4): Edited-Domain Engine Sync, CLI Mode Conflict Guard & D07 Short-Beat Continuous Flow."""

from __future__ import annotations

import unittest

from motionlab.config import AnalysisConfig, EditConfig
from motionlab.semantic.eval_harness import _make_media, _make_transcript
from motionlab.semantic.narrative_brain import (
    build_evidence_index,
    build_narrative_contract,
    build_state_graph,
    propose_scene_graphs,
)
from motionlab.semantic.story_graph import build_story_graph


class TestR5R4EditedEngines(unittest.TestCase):
    def test_cli_pacing_auto_with_audio_preserve_exact_fails_closed(self) -> None:
        """A3.6 Conflict Guard: `--pacing auto` MUST reject incompatible `--audio-mode preserve-exact`!"""
        cfg_bad = EditConfig(offline=True, composition_mode="generative", pacing_mode="auto", audio_mode="preserve-exact")
        with self.assertRaises(ValueError) as ctx:
            cfg_bad.validate()
        self.assertIn("preserve-exact", str(ctx.exception))

        # Valid combinations pass cleanly
        cfg_auto_mix = EditConfig(offline=True, composition_mode="generative", pacing_mode="auto", audio_mode="mix")
        cfg_auto_mix.validate()
        cfg_pres = EditConfig(offline=True, composition_mode="generative", pacing_mode="preserve", audio_mode="preserve-exact")
        cfg_pres.validate()

    def test_d07_short_allocation_beat_prefers_continuous_live_money_world_over_brief_cutaway_flip(self) -> None:
        """D07 Fix: A short ~1.4s allocation beat (like Groceries) prefers continuous live rail over jarring aspect_morph flip."""
        media = _make_media(duration_s=12.0, fps=24, tag="short_alloc")
        tr = _make_transcript(
            "Start with $1,200 in your account. $800 is for bills. $200 is for groceries. $100 is for gas. You have $100 left.",
            duration_s=12.0,
        )
        sg = build_story_graph(tr, None, media)
        ev = build_evidence_index(media, tr, sg)
        ct = build_narrative_contract(ev, sg)
        nsg = build_state_graph(ev, sg, ct)
        state_by_beat = {bs.beat_id: bs for bs in nsg.beat_states}
        alloc_beats = [b for b in sg.beats if b.speech_act == "allocation_step"]
        self.assertGreaterEqual(len(alloc_beats), 2)
        b_short = alloc_beats[1]
        # Force short duration 1.44s (matching Groceries cadence)
        b_short.end_s = round(b_short.start_s + 1.44, 3)
        cands = propose_scene_graphs(
            b_short, state_by_beat[b_short.beat_id], [],
            verified_rem_ids={"rem_ru_track_matte", "rem_viv_aspect_morph"},
            verified_hf_ids={"hf_mk_progress_stat", "hf_data_chart"},
            matte_available=False,
            alloc_index=1,
        )
        by_id = {c["candidate_id"]: c for c in cands}
        rail = next(c for c in cands if c["stage_mode"] == "live_clean_bridge")
        morph = next(c for c in cands if c["stage_mode"] == "aspect_morph_editorial")
        self.assertGreater(rail["visual_clarity_rating"], morph["visual_clarity_rating"])
        self.assertGreater(rail["continuity_rating"], morph["continuity_rating"])

    def test_config_hash_changes_between_preserve_and_auto_pacing_modes(self) -> None:
        """Config hash distinguishes preserve vs auto pacing so caches never cross-contaminate."""
        c1 = EditConfig(offline=True, composition_mode="generative", audio_mode="mix", pacing_mode="preserve")
        c2 = EditConfig(offline=True, composition_mode="generative", audio_mode="mix", pacing_mode="auto")
        self.assertNotEqual(c1.config_hash(), c2.config_hash())


if __name__ == "__main__":
    unittest.main()
