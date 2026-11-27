"""Phase A1–A2 (ML-01.R5.R4): Speech Alignment, Silero VAD, Uncertainty Protection & Causal Pacing Director Tests."""

from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from motionlab.preflight import probe_media
from motionlab.semantic.eval_harness import _make_media, _make_transcript
from motionlab.semantic.pacing_director import choose_pacing_plan, classify_pause_candidates
from motionlab.semantic.speech_perception import align_transcript_words, detect_speech_activity, run_speech_and_action_perception
from motionlab.semantic.story_graph import build_story_graph


def _synth_real_speech_block(text: str, sr: int, max_samples: int, tmp_dir: Path) -> np.ndarray | None:
    """Synthesize genuine offline speech via OS TTS (`say` on macOS) so neural Silero VAD detects real speech."""
    say_bin = shutil.which("say")
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    if not say_bin:
        return None
    aiff_p = tmp_dir / f"tts_{abs(hash(text)) % 100000}.aiff"
    try:
        subprocess.run([say_bin, "-o", str(aiff_p), text], check=True, timeout=20, shell=False)
        proc = subprocess.run(
            [ffmpeg, "-v", "error", "-nostdin", "-i", str(aiff_p), "-ac", "1", "-ar", str(sr), "-f", "f32le", "pipe:1"],
            capture_output=True, check=True, timeout=20, shell=False,
        )
        arr = np.frombuffer(proc.stdout, dtype=np.float32).copy()
        if arr.size == 0:
            return None
        return arr[:max_samples]
    except Exception:
        return None


def _make_speech_gap_clip(out_mp4: Path, *, speech_blocks: list[tuple[float, float]], duration_s: float = 10.0, sr: int = 44100) -> Path:
    """Create a real valid MP4 with speech blocks separated by controlled silent gaps."""
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    n = int(round(duration_s * sr))
    t = np.arange(n, dtype=np.float32) / float(sr)
    sig = np.random.default_rng(42).normal(0.0, 0.0004, n).astype(np.float32)  # gentle room floor -68 dBFS
    phrases = [
        "You start with one thousand dollars in your checking account.",
        "You actually have two hundred dollars left to spend.",
    ]
    for bi, (s0, s1) in enumerate(speech_blocks):
        i0 = max(0, int(round(s0 * sr)))
        i1 = min(n, int(round(s1 * sr)))
        if i1 > i0:
            tts = _synth_real_speech_block(phrases[bi % len(phrases)], sr, i1 - i0, out_mp4.parent)
            if tts is not None and tts.size > 0:
                sig[i0 : i0 + tts.size] += tts * 0.85
            else:
                tt = t[i0:i1]
                env = 0.5 + 0.5 * np.sin(2.0 * math.pi * 4.0 * tt)
                sig[i0:i1] += (0.20 * env * (np.sin(2.0 * math.pi * 220.0 * tt) + 0.6 * np.sin(2.0 * math.pi * 660.0 * tt) + 0.35 * np.sin(2.0 * math.pi * 1100.0 * tt))).astype(np.float32)
    import wave

    wav_p = out_mp4.with_suffix(".wav")
    i16 = np.round(np.clip(sig, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(wav_p), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(i16.tobytes())
    subprocess.run(
        [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", f"color=c=#3a3530:s=360x640:r=24:d={duration_s}",
            "-i", str(wav_p),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
            "-shortest", str(out_mp4),
        ],
        check=True, timeout=45, shell=False,
    )
    return out_mp4


class TestR5R4AlignmentAndPacing(unittest.TestCase):
    def test_a1_alignment_honesty_never_claims_forced_aligned_and_protects_segment_approx(self) -> None:
        """D03 Fix: Whisper/envelope alignment stays honest (forced_aligner_used=False) and segment-only protects whole segment."""
        media = _make_media(duration_s=10.0, tag="seg_approx")
        tr = _make_transcript("Start with $500 in your account. You have $200 left.", duration_s=10.0)
        # Collapse all word starts to segment start to simulate segment-only ASR!
        for seg in tr.segments:
            for w in seg.words:
                w.start_s = seg.start_s
                w.end_s = seg.end_s
        vad_stub = {"noise_floor_dbfs": -55.0, "speech_spans": [{"span_id": "sp_0", "start_s": 0.1, "end_s": 9.5}], "non_speech_spans": [], "uncertain_spans": [], "silent_video": False}
        al = align_transcript_words(tr, vad_stub, media, pcm_16k=np.zeros((0,), dtype=np.float32))
        self.assertFalse(al["forced_aligner_used"])
        self.assertFalse(al["sample_exact_ground_truth"])
        self.assertTrue(al["segment_only_protection"])
        self.assertEqual(al["alignment_method"], "SEGMENT_APPROX")
        self.assertGreaterEqual(al["mean_uncertainty_ms"], 150.0)

    def test_a1_silent_video_never_classified_as_dead_air_or_auto_deleted(self) -> None:
        """Nonverbal / silent-art video must NEVER have its frames deleted as dead air."""
        with tempfile.TemporaryDirectory() as td:
            mp4 = _make_speech_gap_clip(Path(td) / "silent.mp4", speech_blocks=[], duration_s=6.0)
            media = probe_media(mp4)
            tr = _make_transcript("", duration_s=6.0)
            sg = build_story_graph(tr, None, media)
            perc = run_speech_and_action_perception(media, tr, None)
            self.assertTrue(perc["vad_spans"]["silent_video"])
            plan_doc, emap = choose_pacing_plan(media, sg, perc, pacing_mode="auto", niche="silent_art")
            self.assertTrue(emap.is_identity)
            self.assertEqual(emap.output_total_frames, media.duration_frames)
            self.assertEqual(len(emap.omitted_spans), 0)
            self.assertEqual(plan_doc["pause_candidates"][0]["pause_class"], "INTENTIONAL_SILENCE")

    def test_a2_causal_pacing_selection_reversal_changes_edit_time_map_and_output_frames(self) -> None:
        """P0 Causal Proof: Reversing candidate scores (`pacing_identity` vs `pacing_balanced`) MUST causally alter EditTimeMap!"""
        with tempfile.TemporaryDirectory() as td:
            # 2 speech phrases separated by a clean 2.6s dead-air gap (2.2s..4.8s)
            mp4 = _make_speech_gap_clip(Path(td) / "gap.mp4", speech_blocks=[(0.3, 2.2), (4.8, 7.2)], duration_s=8.0)
            media = probe_media(mp4)
            tr = _make_transcript(
                "You start with $1,000 in your account. You actually have $200 left.",
                duration_s=8.0,
                word_gaps={7: 2.6},
            )
            # Align word timestamps directly to our 2 speech blocks
            ws = [w for seg in tr.segments for w in seg.words]
            for i, w in enumerate(ws[:8]):
                w.start_s = round(0.35 + i * 0.22, 3)
                w.end_s = round(w.start_s + 0.18, 3)
            for i, w in enumerate(ws[8:]):
                w.start_s = round(4.90 + i * 0.35, 3)
                w.end_s = round(w.start_s + 0.28, 3)
            sg = build_story_graph(tr, None, media)
            perc = run_speech_and_action_perception(media, tr, None)

            # Run A: favor balanced cutting
            plan_a, emap_a = choose_pacing_plan(
                media, sg, perc, pacing_mode="auto", niche="finance",
                candidate_score_overrides={"pacing_balanced": 60.0, "pacing_identity": -60.0},
            )
            # Run B: reverse scores to favor identity keep
            plan_b, emap_b = choose_pacing_plan(
                media, sg, perc, pacing_mode="auto", niche="finance",
                candidate_score_overrides={"pacing_balanced": -60.0, "pacing_identity": 60.0},
            )
            self.assertEqual(plan_a["chosen_plan_id"], "pacing_balanced")
            self.assertEqual(plan_b["chosen_plan_id"], "pacing_identity")
            self.assertLess(emap_a.output_total_frames, emap_b.output_total_frames)
            self.assertGreater(len(emap_a.omitted_spans), 0)
            self.assertEqual(len(emap_b.omitted_spans), 0)
            self.assertNotEqual(emap_a.map_hash, emap_b.map_hash)

    def test_a2_uncertain_vad_disagreement_abstains_from_cutting(self) -> None:
        """Negative Test: When VAD flags uncertainty in a gap, pacing director MUST classify AMBIGUOUS/REVIEW and NOT cut!"""
        media = _make_media(duration_s=10.0, tag="uncertain_gap")
        tr = _make_transcript("Start with $500 here. End with $200 here.", duration_s=10.0, word_gaps={3: 2.8})
        sg = build_story_graph(tr, None, media)
        ws = [w for seg in tr.segments for w in seg.words]
        gap_t0, gap_t1 = ws[3].end_s, ws[4].start_s
        perc = {
            "speech_alignment": {
                "words": [
                    {"word_id": i, "word": w.word, "source_start_s": w.start_s, "source_end_s": w.end_s,
                     "protected_start_s": max(0.0, w.start_s - 0.06), "protected_end_s": min(10.0, w.end_s + 0.08),
                     "uncertainty_ms": 45.0, "evidence_id": f"w_{i}"}
                    for i, w in enumerate(ws)
                ],
                "segment_only_protection": False,
            },
            "vad_spans": {
                "silent_video": False,
                "speech_spans": [],
                "non_speech_spans": [],
                "uncertain_spans": [{"start_s": gap_t0, "end_s": gap_t1, "reason": "vad_spectral_disagreement"}],
            },
            "prosody_summary": {
                "inter_word_pauses": [{"after_word_id": 3, "before_word_id": 4, "start_s": gap_t0, "end_s": gap_t1, "duration_s": gap_t1 - gap_t0, "phrase_boundary": True, "breath_likelihood": 0.2}]
            },
            "action_evidence": {"windows": []},
        }
        plan_doc, emap = choose_pacing_plan(media, sg, perc, pacing_mode="auto", niche="finance")
        self.assertTrue(emap.is_identity)
        self.assertEqual(len(emap.omitted_spans), 0)
        self.assertTrue(any(c["pause_class"] == "AMBIGUOUS" for c in plan_doc["pause_candidates"]))


if __name__ == "__main__":
    unittest.main()
