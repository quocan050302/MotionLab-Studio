"""Phase A6 (ML-01.R5.R4): Edited-Domain Audio Retiming, Bit-Exact Original Master & Floor-Free Counterfactual QA."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from motionlab.audio.mixer import build_edited_voice_bed, mix_preserving_master, validate_final_audio_mix
from motionlab.audio.semantic_sound import (
    align_sound_to_marker,
    compile_animation_markers,
    extract_sonic_opportunities,
    plan_sonic_score,
)
from motionlab.composition.edit_time_map import build_edit_time_map, build_edited_mezzanine_video, remap_transcript_to_edited_domain
from motionlab.composition.generative_schema import GENERATIVE_SCHEMA_VERSION, LayerSpec, SceneSpec, StoryboardSpec
from motionlab.preflight import probe_media
from motionlab.schemas import MediaInfo
from motionlab.semantic.eval_harness import _make_transcript
from motionlab.semantic.story_graph import build_story_graph
from tests.unit.test_r5r4_alignment_pacing import _make_speech_gap_clip


class TestR5R4AudioRetime(unittest.TestCase):
    def test_edited_voice_bed_mix_preserves_original_master_and_passes_floor_free_qa_and_fails_tampered(self) -> None:
        """D05/D12 Proof: Edited voice bed mixes on shortened sample clock while keeping original_audio_master.m4a bit-exact!"""
        ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            src_mp4 = _make_speech_gap_clip(td_p / "orig.mp4", speech_blocks=[(0.2, 2.5), (4.6, 7.0)], duration_s=7.5)
            orig_media = probe_media(src_mp4)
            tr_orig = _make_transcript("Start with $500 in your account. $300 is for bills. You have $200 left.", duration_s=7.5)
            # Place words strictly outside our cut [2.8s..4.3s]
            ws = [w for seg in tr_orig.segments for w in seg.words]
            for i, w in enumerate(ws[:6]):
                w.start_s = round(0.25 + i * 0.34, 3)
                w.end_s = round(w.start_s + 0.25, 3)
            for i, w in enumerate(ws[6:]):
                w.start_s = round(4.65 + i * 0.25, 3)
                w.end_s = round(w.start_s + 0.18, 3)

            emap = build_edit_time_map(
                orig_media,
                [{"cut_id": "c_gap", "cut_start_frame": 72, "cut_end_frame_exclusive": 102, "pause_class": "DEAD_AIR"}],  # -1.25s = 30f
                pacing_mode="auto",
            )
            mezz_mp4 = td_p / "edited_mezz.mp4"
            mezz_info = build_edited_mezzanine_video(src_mp4, emap, mezz_mp4, ffmpeg_bin=ffmpeg)
            bed_wav = td_p / "edited_voice_bed.wav"
            bed_info = build_edited_voice_bed(src_mp4, emap, bed_wav, ffmpeg_bin=ffmpeg)
            self.assertTrue(bed_info["sample_exact"])
            self.assertEqual(bed_info["actual_samples"], emap.edited_total_samples)

            edited_tr = remap_transcript_to_edited_domain(tr_orig, emap)
            edited_media = MediaInfo.from_dict({
                **orig_media.to_dict(),
                "source_path": str(mezz_mp4),
                "input_hash": mezz_info["mezzanine_sha256"],
                "nb_frames": emap.output_total_frames,
                "duration_s": emap.output_duration_s,
            })
            sg = build_story_graph(edited_tr, None, edited_media)
            b_alloc = next(b for b in sg.beats if b.speech_act == "allocation_step")
            sc = SceneSpec(
                "sc_01_bills", b_alloc.start_frame, b_alloc.end_frame_exclusive, "allocation_step", "live_clean_bridge", "bills",
                persistent_ledger_step_id=b_alloc.ledger_step_id,
                layers=[LayerSpec("ly_hf", "hf_mk_progress_stat", "hyperframes", "foreground_graphics", 30, {"amount_cents": 30000, "denom_cents": 50000})],
            )
            sb = StoryboardSpec(
                GENERATIVE_SCHEMA_VERSION, "AUTO_GENERATIVE", edited_media.input_hash, edited_tr.transcript_hash, sg.story_graph_hash,
                "finance", 0.95, "v", edited_media.display_width, edited_media.display_height, 24, 1, edited_media.duration_frames, 42, {}, [sc],
            )
            markers = compile_animation_markers(sb, sg, {"sc_01_bills": {"durationInFrames": b_alloc.end_frame_exclusive - b_alloc.start_frame, "globalStartFrame": b_alloc.start_frame}})
            opps = extract_sonic_opportunities(sg, edited_tr, observed_actions=None, storyboard=sb)
            score = plan_sonic_score(sb, opps, sg, markers, niche_style="finance", seed=42, sr=44100)
            tl, _, _ = align_sound_to_marker(
                score, fps_num=24, fps_den=1, sample_rate=44100, total_frames=edited_media.duration_frames,
                source_hash=edited_media.input_hash, audio_mode="mix", sfx_enabled=True, seed=42,
            )
            self.assertGreaterEqual(len(tl.events), 1)
            self.assertEqual(tl.duration_samples, emap.edited_total_samples)

            run_dir = td_p / "run"
            run_dir.mkdir()
            mix_info = mix_preserving_master(
                edited_media, tl, edited_tr, run_dir, seed=42, ffmpeg_bin=ffmpeg,
                original_source_path=src_mp4, edited_voice_bed_wav=bed_wav, edit_map_hash=emap.map_hash,
            )
            self.assertTrue(mix_info["master_exact"])
            self.assertTrue((run_dir / "original_audio_master.m4a").is_file())
            self.assertTrue((run_dir / "evidence" / "no_sfx_edited_reference.m4a").is_file())

            # Genuine edited final MP4 -> PASS with floor-free counterfactual!
            good_final = run_dir / "final_good.mp4"
            subprocess.run(
                [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(mezz_mp4), "-i", mix_info["mixed_reference_path"],
                 "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(good_final)],
                check=True, timeout=45, shell=False,
            )
            _, qa_good = validate_final_audio_mix(edited_media, good_final, tl, mix_info, run_dir, ffmpeg_bin=ffmpeg)
            self.assertEqual(qa_good["status"], "PASS", qa_good["failure_reasons"])
            self.assertTrue(all(not e["counterfactual_details"]["reference_floor_used"] for e in qa_good["events"]))

            # Tampered edited final MP4 (voice bed only, NO SFX) -> MUST FAIL!
            tampered_final = run_dir / "final_nosfx.mp4"
            subprocess.run(
                [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(mezz_mp4), "-i", str(bed_wav),
                 "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(tampered_final)],
                check=True, timeout=45, shell=False,
            )
            _, qa_bad = validate_final_audio_mix(edited_media, tampered_final, tl, mix_info, run_dir, ffmpeg_bin=ffmpeg)
            self.assertEqual(qa_bad["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
