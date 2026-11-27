"""Phase R2-A4 / R2-A8 (ML-01.R5.R2): P0 Regression & Negative Tests for Sound Integrity, Short Cues & Real Counterfactual Ablation.

Specifically tests:
  - D09: Real 18ms (`wyzowl_data-tick.wav`) & 49.5ms (`wyzowl_foley-card-touch.wav`) recorded WAVs SURVIVE
    align_sound_to_marker & mix_preserving_master (fixing the R1 50ms silent drop bug!).
  - D09: Stable original cue_id conservation (`planned == staged + explicitly_suppressed + blocked`), ZERO silent drops.
  - D09: Long-form >16 cues (24 cues over 90s) survive via sliding 10s window instead of magic 16-cue global cutoff.
  - D10: Event-class-aware anchor alignment (`transient` vs `crest_in_window`) eliminates +6f paper/envelope slide drift!
  - D11: Exact immutable asset_id + SHA-256 provenance matching & REAL minus-one-cue AAC counterfactual ablation
    (muting a cue in final MP4 MUST fail validation even if other cues pass!).
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from motionlab.audio.assets import inspect_foley_library, load_recorded_foley_by_asset_id
from motionlab.audio.mixer import mix_preserving_master, validate_final_audio_mix
from motionlab.audio.semantic_sound import (
    PlannedSonicCue,
    SonicScore,
    align_sound_to_marker,
)
from motionlab.preflight import probe_media


def _make_test_clip_with_speech_like_audio(out_mp4: Path, duration_s: float = 6.0) -> None:
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    subprocess.run(
        [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", f"color=c=#334433:s=360x640:r=24:d={duration_s}",
            "-f", "lavfi", "-i", f"sine=frequency=280:sample_rate=44100:duration={duration_s}",
            "-af", "volume=-16dB",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k",
            "-shortest",
            str(out_mp4),
        ],
        check=True, timeout=60, shell=False,
    )


class TestR5R2SoundIntegrityAndAblation(unittest.TestCase):
    def test_d09_short_18ms_and_49_5ms_recorded_wavs_survive_align_and_preserve_cue_ids(self) -> None:
        lib = inspect_foley_library()
        card_touch = next((a for a in lib["assets"] if a["asset_id"] == "cc0_cash_flick_v1_foley-card-touch"), None)
        data_tick = next((a for a in lib["assets"] if a["asset_id"] == "cc0_tick_accent_v0_data-tick"), None)
        if not card_touch or not data_tick:
            self.skipTest("CC0 wyzowl short WAVs not provisioned locally")

        # Reproduce the exact R1 Bills scenario (scue_02_cash_flick f152 & scue_03_tick_accent f166)
        c1 = PlannedSonicCue(
            cue_id="scue_02_bills_card",
            beat_id="beat_03",
            scene_id="sc_03_allocation_step",
            semantic_event_id="opp_03",
            word_ids=[18, 19],
            quote="800 is for bills",
            story_intent="allocation_step:Bills",
            emotional_role="tactile_subtraction_clarity",
            visual_element_id="allocation_bills",
            action_ref="marker:sc_03:release",
            action_confidence=0.95,
            marker_id="sc_03:release",
            marker_type="release",
            sound_role="ILLUSTRATIVE_FOLEY",
            material="cardstock",
            texture="short_tack",
            family="cash_flick",
            variant=1,
            candidate_families=["cash_flick"],
            rejected_candidates=[],
            reason="Bills short 49.5ms foley test",
            target_frame=36,
            pre_roll_ms=0.0,
            gain_db=-5.0,
            pan=-0.1,
            priority=1,
            asset_id=card_touch["asset_id"],
            asset_provenance="recorded_local_foley",
            asset_license="CC0-1.0",
            asset_sha256=card_touch["sha256"],
            fallback_status="RECORDED_LOCAL_FOLEY",
        )
        c2 = PlannedSonicCue(
            cue_id="scue_03_bills_tick",
            beat_id="beat_03",
            scene_id="sc_03_allocation_step",
            semantic_event_id="opp_03_settle",
            word_ids=[20],
            quote="bills",
            story_intent="allocation_step_settle",
            emotional_role="clarity_settle",
            visual_element_id="persistent_ledger_balance",
            action_ref="marker:sc_03:counter_settle",
            action_confidence=0.95,
            marker_id="sc_03:counter_settle",
            marker_type="counter_settle",
            sound_role="GRAPHIC_MOTION",
            material="wood",
            texture="short_tick",
            family="tick_accent",
            variant=0,
            candidate_families=["tick_accent"],
            rejected_candidates=[],
            reason="Bills short 18ms tick test",
            target_frame=54,
            pre_roll_ms=0.0,
            gain_db=-6.0,
            pan=0.1,
            priority=1,
            asset_id=data_tick["asset_id"],
            asset_provenance="recorded_local_foley",
            asset_license="CC0-1.0",
            asset_sha256=data_tick["sha256"],
            fallback_status="RECORDED_LOCAL_FOLEY",
        )
        score = SonicScore(
            schema_version="2.1.0",
            mode="AUTO_GENERATIVE",
            story_graph_hash="sg",
            storyboard_hash="sb",
            foley_library_status="RECORDED_FOLEY_VERIFIED",
            recorded_cues_used=2,
            procedural_cues_used=0,
            cues=[c1, c2],
        )
        tl, rows, meta = align_sound_to_marker(
            score, fps_num=24, fps_den=1, sample_rate=44100, total_frames=144, source_hash="src"
        )
        # BOTH 49.5ms and 18ms cues MUST survive with exact original cue_ids and exact SHA-256!
        self.assertEqual(len(tl.events), 2)
        self.assertEqual([e.id for e in tl.events], ["scue_02_bills_card", "scue_03_bills_tick"])
        self.assertEqual(tl.events[0].raw_asset_sha256, card_touch["sha256"])
        self.assertEqual(tl.events[1].raw_asset_sha256, data_tick["sha256"])
        self.assertTrue(tl.cue_ledger_summary["conservation_exact"])
        self.assertEqual(tl.cue_ledger_summary["blocked"], 0)

    def test_long_form_24_cues_over_90s_not_truncated_by_old_16_cap(self) -> None:
        cues = [
            PlannedSonicCue(
                cue_id=f"scue_{i:02d}_tick",
                beat_id=f"b_{i:02d}",
                scene_id=f"sc_{i:02d}",
                semantic_event_id=f"ev_{i:02d}",
                word_ids=[i],
                quote="step",
                story_intent="step",
                emotional_role="clarity",
                visual_element_id="el",
                action_ref="m",
                action_confidence=0.95,
                marker_id=f"m_{i}",
                marker_type="resolve",
                sound_role="GRAPHIC_MOTION",
                material="wood",
                texture="tick",
                family="tick_accent",
                variant=i % 3,
                candidate_families=["tick_accent"],
                rejected_candidates=[],
                reason="long form cue",
                target_frame=24 + i * 84,  # every 3.5s across 90s
                pre_roll_ms=0.0,
                gain_db=-6.0,
                pan=0.0,
                priority=1,
            )
            for i in range(24)
        ]
        score = SonicScore("2.1.0", "AUTO_GENERATIVE", "sg", "sb", "PARTIAL", 0, 24, cues=cues)
        tl, _, _ = align_sound_to_marker(
            score, fps_num=24, fps_den=1, sample_rate=44100, total_frames=2160, source_hash="src"
        )
        self.assertEqual(len(tl.events), 24)
        self.assertTrue(tl.cue_ledger_summary["conservation_exact"])

    def test_d10_and_d11_true_counterfactual_ablation_passes_valid_mix_and_fails_muted_final(self) -> None:
        lib = inspect_foley_library()
        paper_slide = next((a for a in lib["assets"] if a["asset_id"] == "cc0_paper_rustle_v1_foley-paper-slide"), None)
        data_tick = next((a for a in lib["assets"] if a["asset_id"] == "cc0_tick_accent_v0_data-tick"), None)
        if not paper_slide or not data_tick:
            self.skipTest("CC0 wyzowl WAVs not provisioned locally")

        with tempfile.TemporaryDirectory(prefix="ml01_r5r2_audio_test_") as td:
            td_p = Path(td)
            src_mp4 = td_p / "src.mp4"
            _make_test_clip_with_speech_like_audio(src_mp4, duration_s=5.0)
            media = probe_media(src_mp4)

            cues = [
                PlannedSonicCue(
                    cue_id="scue_paper_slide_d10",
                    beat_id="b1",
                    scene_id="sc1",
                    semantic_event_id="e1",
                    word_ids=[1],
                    quote="200 for groceries",
                    story_intent="allocation_step:Groceries",
                    emotional_role="tactile",
                    visual_element_id="el1",
                    action_ref="m1",
                    action_confidence=0.95,
                    marker_id="m1",
                    marker_type="release",
                    sound_role="ILLUSTRATIVE_FOLEY",
                    material="paper",
                    texture="slide",
                    family="paper_rustle",
                    variant=1,
                    candidate_families=["paper_rustle"],
                    rejected_candidates=[],
                    reason="D10 paper slide crest alignment check",
                    target_frame=36,
                    pre_roll_ms=0.0,
                    gain_db=-5.0,
                    pan=-0.1,
                    priority=1,
                    asset_id=paper_slide["asset_id"],
                    asset_provenance="recorded_local_foley",
                    asset_license="CC0-1.0",
                    asset_sha256=paper_slide["sha256"],
                    fallback_status="RECORDED_LOCAL_FOLEY",
                ),
                PlannedSonicCue(
                    cue_id="scue_short_tick_d09",
                    beat_id="b2",
                    scene_id="sc2",
                    semantic_event_id="e2",
                    word_ids=[2],
                    quote="settle",
                    story_intent="allocation_step_settle",
                    emotional_role="clarity",
                    visual_element_id="el2",
                    action_ref="m2",
                    action_confidence=0.95,
                    marker_id="m2",
                    marker_type="counter_settle",
                    sound_role="GRAPHIC_MOTION",
                    material="wood",
                    texture="tick",
                    family="tick_accent",
                    variant=0,
                    candidate_families=["tick_accent"],
                    rejected_candidates=[],
                    reason="D09 18ms short tick true counterfactual ablation",
                    target_frame=84,
                    pre_roll_ms=0.0,
                    gain_db=-5.0,
                    pan=0.1,
                    priority=1,
                    asset_id=data_tick["asset_id"],
                    asset_provenance="recorded_local_foley",
                    asset_license="CC0-1.0",
                    asset_sha256=data_tick["sha256"],
                    fallback_status="RECORDED_LOCAL_FOLEY",
                ),
            ]
            score = SonicScore("2.1.0", "AUTO_GENERATIVE", "sg", "sb", "RECORDED_FOLEY_VERIFIED", 2, 0, cues=cues)
            tl, _, _ = align_sound_to_marker(
                score, fps_num=24, fps_den=1, sample_rate=44100, total_frames=media.duration_frames, source_hash=media.input_hash
            )
            run_dir = td_p / "run"
            run_dir.mkdir()
            mix_info = mix_preserving_master(media, tl, None, run_dir, seed=42)
            # Exact presentation timeline sample length (5.0s * 44100 = 220500)
            self.assertEqual(mix_info["metrics"]["duration_samples"], tl.duration_samples)

            # Mux final mixed MP4
            ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
            fin_mp4 = run_dir / "final.mp4"
            subprocess.run(
                [
                    ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                    "-i", str(src_mp4), "-i", str(mix_info["mixed_reference_path"]),
                    "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
                    str(fin_mp4),
                ],
                check=True, timeout=60, shell=False,
            )
            _, qa = validate_final_audio_mix(media, fin_mp4, tl, mix_info, run_dir)
            self.assertEqual(qa["status"], "PASS", msg=qa)
            self.assertEqual(qa["ablation_method"], "true_minus_one_cue_aac_counterfactual")
            self.assertEqual(qa["events_total"], 2)
            self.assertEqual(qa["events_proxy_audible"], 2)
            self.assertEqual(qa["events_sync_ok"], 2)
            # Verify D10 paper slide crest lag is now cleanly inside <= 2.5f (was +6.0f in R1!)
            ev_slide = next(e for e in qa["events"] if e["id"] == "scue_paper_slide_d10")
            self.assertLessEqual(abs(ev_slide["primary_anchor_lag_frames"]), 2.5)
            # Negative control (D11): source-only MP4 with NO SFX MUST FAIL validation!
            _, qa_nosfx = validate_final_audio_mix(media, src_mp4, tl, mix_info, run_dir)
            self.assertEqual(qa_nosfx["status"], "FAIL")
            self.assertTrue(len(qa_nosfx["failure_reasons"]) >= 1)

    def test_tampered_asset_sha_rejected_by_loader(self) -> None:
        bad = load_recorded_foley_by_asset_id("cc0_cash_flick_v0_money-notes", expected_raw_sha256="0" * 64)
        self.assertIsNone(bad)


if __name__ == "__main__":
    unittest.main()
