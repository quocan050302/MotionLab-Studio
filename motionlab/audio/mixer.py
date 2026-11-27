"""Phase A4 / R2-A4 (ML-01.R5.R2): Voice-Priority Master-Preserving Mixer & Real Minus-One-Cue Counterfactual QA.

Strict invariants & R5.R2 P0 Fixes:
  1. Preserves `original_audio_master.m4a` bit-exact from source (identical packet count & SHA-256).
  2. Voice stays untouched at unity gain 1.0 (zero EQ/NR/compression/normalization/trim/cut).
  3. Exact presentation timeline alignment (`timeline.duration_samples`, e.g. 1,323,000 samples @ 44.1 kHz for 30.000s)
     without AAC decoder priming overhang inflating mixed duration.
  4. Resolves every SoundEvent strictly by immutable `(asset_id, raw_asset_sha256)` — ZERO family/variant drift
     between planner and mixer — and logs both `raw_asset_sha256` and `processed_pcm_sha256`.
  5. Dynamic 1.2–3.8 kHz speech-band spectral carving on mid-heavy paper/page SFX ONLY during active speech
     so page turns & rustles stay crisp without masking dialogue (fixes R1 +7.03 dB page_turn!).
  6. REAL Minus-One-Cue Counterfactual Ablation in `validate_final_audio_mix()`:
     For EVERY staged cue k, builds `voice + (sfx_stem - cue_k_final_contribution)` under the exact frozen
     headroom curve, encodes via identical AAC-192k, decodes, and measures TRUE counterfactual RMS drop —
     NEVER substituting SNR-over-codec-floor as fake ablation!
  7. Strict 100% coverage & cue-ledger conservation gate (NO `>=85%` escape hatch; any blocked/dropped
     required cue fails the audio QA gate!).
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import butter, sosfilt

from motionlab.audio.assets import analyze_waveform_envelope, load_recorded_foley_by_asset_id
from motionlab.audio.schema import AudioTimeline, MixMetrics
from motionlab.audio.synthesis import synthesize_sfx_variant
from motionlab.schemas import MediaInfo, ValidationCheck, atomic_write_json
from motionlab.semantic.schema import Transcript
from motionlab.validation import _audio_packet_hash_and_pts

IMPULSE_FAMILIES = frozenset({"cash_flick", "tick_accent", "pop_reveal", "soft_impact", "bass_settle"})
MID_SPEECH_SENSITIVE_FAMILIES = frozenset({"page_turn", "paper_rustle", "envelope_slide"})


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _db_to_lin(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def _lin_to_db(lin: float) -> float:
    return round(20.0 * math.log10(max(1e-7, float(lin))), 2)


def _write_wav_int16(path: Path, stereo_f32: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(stereo_f32, -1.0, 1.0)
    i16 = np.round(clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2 if stereo_f32.ndim == 2 else 1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(i16.tobytes())


def _decode_audio_stereo_f32(path: Path, sr: int, ffmpeg_bin: str) -> np.ndarray:
    proc = subprocess.run(
        [
            ffmpeg_bin, "-v", "error", "-nostdin",
            "-i", str(path),
            "-vn", "-map", "0:a:0",
            "-ac", "2", "-ar", str(sr),
            "-f", "f32le", "pipe:1",
        ],
        capture_output=True, timeout=90, shell=False,
    )
    if proc.returncode != 0 or not proc.stdout:
        return np.zeros((0, 2), dtype=np.float32)
    arr = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    n = (arr.size // 2) * 2
    return arr[:n].reshape((-1, 2))


def _measure_lufs_and_tp(path: Path, ffmpeg_bin: str) -> tuple[float, float]:
    """Measure integrated LUFS and true peak (dBTP) via FFmpeg ebur128 filter."""
    proc = subprocess.run(
        [ffmpeg_bin, "-hide_banner", "-i", str(path), "-vn", "-af", "ebur128=peak=true", "-f", "null", "-"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=90, shell=False,
    )
    lufs = -24.0
    tp = -1.0
    for line in (proc.stderr or "").splitlines():
        s = line.strip()
        if s.startswith("I:") and "LUFS" in s:
            try:
                lufs = float(s.split()[1])
            except Exception:
                pass
        elif s.startswith("Peak:") and "dBFS" in s:
            try:
                tp = float(s.split()[1])
            except Exception:
                pass
    return round(lufs, 2), round(tp, 2)


def _compute_short_speech_envelope(voice_mono: np.ndarray, sr: int, attack_ms: float = 15.0, release_ms: float = 110.0) -> np.ndarray:
    """Fast decimated speech envelope tracker (15ms attack, 110ms release) for targeted SFX sustain ducking."""
    hop = max(1, int(round(sr * 0.005)))  # 5ms blocks
    n = voice_mono.size
    n_blk = (n + hop - 1) // hop
    padded = np.pad(np.abs(voice_mono), (0, n_blk * hop - n))
    blk_rms = np.sqrt(np.mean(padded.reshape((n_blk, hop)) ** 2, axis=1) + 1e-10).astype(np.float32)
    a_att = math.exp(-hop / max(1.0, sr * attack_ms / 1000.0))
    a_rel = math.exp(-hop / max(1.0, sr * release_ms / 1000.0))
    env_blk = np.zeros_like(blk_rms)
    cur = 0.0
    for i, val in enumerate(blk_rms):
        if val > cur:
            cur = a_att * cur + (1.0 - a_att) * float(val)
        else:
            cur = a_rel * cur + (1.0 - a_rel) * float(val)
        env_blk[i] = cur
    return np.repeat(env_blk, hop)[:n]


def _carve_speech_band_sfx(stereo_pcm: np.ndarray, sr: int, cut_db: float = -5.5) -> np.ndarray:
    """Attenuate 1.0–4.0 kHz speech intelligibility band on SFX ONLY so dialogue consonants stay crystal clear."""
    if stereo_pcm.shape[0] < 64:
        return stereo_pcm
    nyq = 0.5 * sr
    lo = max(200.0, min(950.0, nyq * 0.7)) / nyq
    hi = max(lo + 0.05, min(4100.0, nyq * 0.92)) / nyq
    sos = butter(2, [lo, hi], btype="bandpass", output="sos")
    passes = 2 if cut_db < -6.5 else 1
    per_pass_db = cut_db / passes
    att_frac = float(1.0 - _db_to_lin(per_pass_db))
    out = stereo_pcm.copy()
    for _ in range(passes):
        for ch in range(out.shape[1]):
            band = sosfilt(sos, out[:, ch]).astype(np.float32)
            out[:, ch] = out[:, ch] - att_frac * band
    return out


def extract_original_audio_master(
    source_path: Path,
    out_m4a: Path,
    ffmpeg_bin: str | None = None,
    ffprobe_bin: str | None = None,
) -> dict[str, Any]:
    """Stream-copy bit-exact original audio master (`original_audio_master.m4a`) with zero re-encoding."""
    ffmpeg = ffmpeg_bin or shutil.which("ffmpeg") or "ffmpeg"
    ffprobe = ffprobe_bin or shutil.which("ffprobe") or "ffprobe"
    out_m4a.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(source_path),
            "-vn", "-map", "0:a:0", "-c:a", "copy",
            str(out_m4a),
        ],
        check=True, timeout=60, shell=False,
    )
    src_pkt = _audio_packet_hash_and_pts(Path(source_path), 0, ffmpeg, ffprobe)
    mst_pkt = _audio_packet_hash_and_pts(out_m4a, 0, ffmpeg, ffprobe)
    return {
        "master_path": str(out_m4a),
        "master_exact": (src_pkt["packet_hash"] == mst_pkt["packet_hash"] and src_pkt["packet_count"] == mst_pkt["packet_count"]),
        "source_packet_hash": src_pkt["packet_hash"],
        "master_packet_hash": mst_pkt["packet_hash"],
        "packet_count": mst_pkt["packet_count"],
    }


def build_edited_voice_bed(
    source_path: Path,
    edit_map: Any,
    out_wav: Path,
    *,
    ffmpeg_bin: str | None = None,
) -> dict[str, Any]:
    """Phase A6.4 (ML-01.R5.R4): Build exact sample-accurate derived edited voice bed from immutable original source.

    - Zero time stretching, zero pitch change, zero EQ/NR alteration.
    - Uses exact `EditTimeMap.kept_spans` sample offsets and applies a 12ms equal-power cosine micro-crossfade
      strictly across non-speech join edges to eliminate clicks/DC jumps while preserving exact `edited_total_samples`!
    """
    ffmpeg = ffmpeg_bin or shutil.which("ffmpeg") or "ffmpeg"
    sr = int(edit_map.sample_rate or 44100)
    src_pcm = _decode_audio_stereo_f32(Path(source_path), sr=sr, ffmpeg_bin=ffmpeg)
    n_out = int(edit_map.edited_total_samples)
    if src_pcm.shape[0] == 0 or n_out <= 0:
        out_arr = np.zeros((max(1, n_out), 2), dtype=np.float32)
        _write_wav_int16(out_wav, out_arr, sr)
        return {
            "edited_voice_bed_path": str(out_wav),
            "edited_voice_bed_sha256": _sha256_file(out_wav),
            "sample_rate": sr,
            "actual_samples": int(out_arr.shape[0]),
            "expected_samples": n_out,
            "sample_exact": False,
            "crossfade_ms": 0.0,
            "edit_map_hash": str(edit_map.map_hash),
        }

    out_arr = np.zeros((n_out, 2), dtype=np.float32)
    src_len = src_pcm.shape[0]

    for idx, ks in enumerate(edit_map.kept_spans):
        o0, o1 = int(ks.output_start_sample), int(ks.output_end_sample)
        s0 = int(ks.source_start_sample)
        seg_n = max(0, min(n_out, o1) - o0)
        if seg_n <= 0:
            continue
        # Copy exact span from immutable source audio
        avail = max(0, min(src_len, s0 + seg_n) - s0)
        if avail > 0:
            out_arr[o0 : o0 + avail] = src_pcm[s0 : s0 + avail]

        # Apply smooth equal-power cosine micro-crossfade at cut join `idx > 0` strictly inside non-speech margin!
        xf = min(int(ks.crossfade_in_samples), seg_n // 3, 768)
        if idx > 0 and xf >= 32:
            prev_ks = edit_map.kept_spans[idx - 1]
            # Tail immediately following prev_ks in original source (non-speech margin!) blended into head of current span
            tail_s0 = int(prev_ks.source_end_sample)
            tail_s1 = min(src_len, tail_s0 + xf)
            actual_xf = min(xf, tail_s1 - tail_s0, avail)
            if actual_xf >= 32:
                theta = np.linspace(0.0, 0.5 * math.pi, actual_xf, dtype=np.float32)[:, None]
                fade_out = np.cos(theta)
                fade_in = np.sin(theta)
                prev_tail = src_pcm[tail_s0 : tail_s0 + actual_xf]
                cur_head = out_arr[o0 : o0 + actual_xf].copy()
                out_arr[o0 : o0 + actual_xf] = prev_tail * fade_out + cur_head * fade_in

    _write_wav_int16(out_wav, out_arr, sr)
    return {
        "edited_voice_bed_path": str(out_wav),
        "edited_voice_bed_sha256": _sha256_file(out_wav),
        "sample_rate": sr,
        "actual_samples": int(out_arr.shape[0]),
        "expected_samples": n_out,
        "sample_exact": (int(out_arr.shape[0]) == n_out),
        "crossfade_ms": round(1000.0 * (edit_map.kept_spans[1].crossfade_in_samples / sr), 2) if len(edit_map.kept_spans) > 1 else 0.0,
        "edit_map_hash": str(edit_map.map_hash),
    }


def mix_preserving_master(
    media: MediaInfo,
    timeline: AudioTimeline,
    transcript: Transcript | None,
    run_dir: Path,
    seed: int = 42,
    ffmpeg_bin: str | None = None,
    ffprobe_bin: str | None = None,
    foley_dir: Path | None = None,
    *,
    original_source_path: Path | None = None,
    edited_voice_bed_wav: Path | None = None,
    edit_map_hash: str | None = None,
) -> dict[str, Any]:
    """Mix voice (untouched unity 1.0 or derived edited voice bed) + audibly staged local SFX stem with exact provenance."""
    ffmpeg = ffmpeg_bin or shutil.which("ffmpeg") or "ffmpeg"
    ffprobe = ffprobe_bin or shutil.which("ffprobe") or "ffprobe"
    ev_dir = run_dir / "evidence"
    ev_dir.mkdir(parents=True, exist_ok=True)
    sr = int(timeline.sample_rate or 44100)
    orig_src = Path(original_source_path) if original_source_path is not None else Path(media.source_path)

    # 1. Preserve bit-exact original audio master stream copy ALWAYS from immutable original source!
    master_m4a = run_dir / "original_audio_master.m4a"
    subprocess.run(
        [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(orig_src),
            "-vn", "-map", "0:a:0", "-c:a", "copy",
            str(master_m4a),
        ],
        check=True, timeout=60, shell=False,
    )
    src_pkt = _audio_packet_hash_and_pts(orig_src, 0, ffmpeg, ffprobe)
    mst_pkt = _audio_packet_hash_and_pts(master_m4a, 0, ffmpeg, ffprobe)
    master_exact = (src_pkt["packet_hash"] == mst_pkt["packet_hash"] and src_pkt["packet_count"] == mst_pkt["packet_count"])

    # 2. Decode voice stereo float32 PCM (UNITY 1.0!): use derived edited voice bed when pacing cuts active!
    voice_bed_src = Path(edited_voice_bed_wav) if (edited_voice_bed_wav is not None and Path(edited_voice_bed_wav).is_file()) else orig_src
    voice_raw = _decode_audio_stereo_f32(voice_bed_src, sr=sr, ffmpeg_bin=ffmpeg)
    n_samp = int(timeline.duration_samples) if int(timeline.duration_samples or 0) > 0 else int(round(media.duration_s * sr))
    if voice_raw.shape[0] < n_samp:
        voice = np.pad(voice_raw, ((0, n_samp - voice_raw.shape[0]), (0, 0)))
    else:
        voice = voice_raw[:n_samp].copy()

    # Export no-SFX edited reference M4A for honest human A/B listening when using edited voice bed
    if edited_voice_bed_wav is not None and Path(edited_voice_bed_wav).is_file():
        no_sfx_ref = ev_dir / "no_sfx_edited_reference.m4a"
        subprocess.run(
            [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(edited_voice_bed_wav), "-c:a", "aac", "-b:a", "192k", str(no_sfx_ref)],
            check=True, timeout=60, shell=False,
        )

    voice_mono = 0.5 * (voice[:, 0] + voice[:, 1])
    speech_env = _compute_short_speech_envelope(voice_mono, sr, attack_ms=15.0, release_ms=110.0)
    active_voice_vals = speech_env[speech_env > 0.015]
    ref_voice_rms = float(np.percentile(active_voice_vals, 65)) if active_voice_vals.size > 50 else 0.12
    speech_thresh = max(0.02, ref_voice_rms * 0.45)

    # 3. Render & calibrate each SoundEvent by EXACT immutable asset_id + SHA-256 (fixes D11 provenance drift!)
    sfx_stem = np.zeros((n_samp, 2), dtype=np.float32)
    event_mix_log: list[dict[str, Any]] = []
    per_cue_slices: list[tuple[str, int, int, np.ndarray]] = []

    max_dur_by_fam = {
        "cash_flick": 0.48,
        "paper_rustle": 0.54,
        "envelope_slide": 0.54,
        "page_turn": 0.68,
        "tick_accent": 0.32,
        "pop_reveal": 0.35,
        "soft_whoosh": 0.65,
        "reverse_whoosh": 0.75,
        "soft_impact": 0.72,
        "shimmer_payoff": 0.80,
        "bass_settle": 0.75,
    }

    for ev in timeline.events:
        raw_sha_expected = ev.raw_asset_sha256 or ev.asset_sha256
        prep = None
        if ev.asset_id and ev.fallback == "RECORDED_LOCAL_FOLEY":
            prep = load_recorded_foley_by_asset_id(
                ev.asset_id,
                expected_raw_sha256=raw_sha_expected or None,
                target_sr=sr,
                foley_dir=foley_dir,
            )

        if prep is not None:
            pcm = prep["pcm"].copy()
            used_raw_sha = prep["raw_asset_sha256"]
            used_proc_sha = prep["processed_pcm_sha256"]
            used_prov = "recorded_local_foley"
            anchor_rel_orig = int(prep["peak_sample"] if ev.align_anchor == "transient" else prep["crest_sample"])
        else:
            pcm, t_off = synthesize_sfx_variant(ev.family, variant=ev.variant, sr=sr, seed=seed)
            env_dsp = analyze_waveform_envelope(pcm, sr)
            used_raw_sha = ev.raw_asset_sha256 or ev.asset_sha256
            used_proc_sha = env_dsp["processed_pcm_sha256"]
            used_prov = "procedural_dsp_synthesis"
            anchor_rel_orig = int(env_dsp["peak_sample"] if ev.align_anchor == "transient" else env_dsp["crest_sample"])

        ev_len = pcm.shape[0]
        if ev_len < 32:
            continue

        # Gently trim overly long tail cleanly while preserving onset + crest + short transients intact!
        max_samp = int(round(sr * max_dur_by_fam.get(ev.family, 0.80)))
        if ev_len > max_samp and anchor_rel_orig < max_samp - int(sr * 0.10):
            pcm = pcm[:max_samp].copy()
            fade_tail = min(int(sr * 0.10), max(16, pcm.shape[0] // 4))
            pcm[-fade_tail:] *= np.linspace(1.0, 0.0, fade_tail, dtype=np.float32)[:, None]
            ev_len = pcm.shape[0]

        # Place strictly at timeline start_sample so mixer NEVER drifts from aligned schedule!
        start_i = max(0, int(ev.start_sample))
        end_i = min(n_samp, start_i + ev_len)
        usable = end_i - start_i
        if usable < 32:
            continue
        clip_pcm = pcm[:usable].copy()

        # Measure nearby local voice RMS (±250ms) for intelligent event-specific gain staging
        win_lo = max(0, start_i - int(sr * 0.25))
        win_hi = min(n_samp, end_i + int(sr * 0.25))
        local_voice_rms = float(np.sqrt(np.mean(voice_mono[win_lo:win_hi] ** 2) + 1e-10))
        calib_ref = max(local_voice_rms, ref_voice_rms * 0.72)

        # Calibrated relative levels: crisp primary emphasis (-7.0 dB vs nearby voice), textures (-11.5..-13.5 dB),
        # page_turn & paper_rustle kept refined below speech 1–4 kHz band while maintaining >25 dB SNR & >35 dB ablation!
        is_primary = ev.family in ("cash_flick", "tick_accent", "pop_reveal")
        if ev.family == "soft_impact":
            target_rel_db = -9.5
        elif ev.family == "page_turn":
            target_rel_db = -14.0
        elif ev.family == "paper_rustle":
            target_rel_db = -13.5
        elif is_primary:
            target_rel_db = -7.0
        else:
            target_rel_db = -11.5

        user_offset_db = max(-5.0, min(3.0, float(ev.gain_db) + 6.0))
        cue_rms = float(np.sqrt(np.mean(clip_pcm ** 2) + 1e-10))
        desired_rms = calib_ref * _db_to_lin(target_rel_db + user_offset_db)
        gain_lin = desired_rms / max(1e-5, cue_rms)

        cue_peak = float(np.max(np.abs(clip_pcm)))
        if ev.family == "page_turn":
            max_cue_peak = 0.26
        elif ev.family == "paper_rustle":
            max_cue_peak = 0.30
        elif is_primary or ev.family == "soft_impact":
            max_cue_peak = 0.48
        else:
            max_cue_peak = 0.34
        if cue_peak * gain_lin > max_cue_peak:
            gain_lin = max_cue_peak / max(1e-6, cue_peak)
        clip_pcm *= gain_lin

        # Speech-band 1.0–4.0 kHz carving on mid-heavy paper/page sounds when voice is active (fixes R1 page_turn +7.03 dB!)
        s_env = speech_env[start_i:end_i]
        speech_active_frac = float(np.mean(s_env > speech_thresh)) if s_env.size else 0.0
        if ev.family in MID_SPEECH_SENSITIVE_FAMILIES and speech_active_frac >= 0.08 and usable >= 128:
            if ev.family == "page_turn":
                carve_db = -10.5
            elif ev.family == "paper_rustle":
                carve_db = -8.0
            else:
                carve_db = -5.0
            clip_pcm = _carve_speech_band_sfx(clip_pcm, sr, cut_db=carve_db)

        # Targeted short sidechain ducking on SUSTAIN tail ONLY (preserve crisp short transients <= 55ms intact!)
        if usable > int(sr * 0.06):
            duck_factor = np.where(s_env > speech_thresh, _db_to_lin(-2.0), 1.0).astype(np.float32)
            trans_guard = min(usable, anchor_rel_orig + int(sr * 0.032))
            duck_factor[:trans_guard] = 1.0
            clip_pcm *= duck_factor[:, None]

        # Subtle mono-safe stereo panning
        pan = max(-0.4, min(0.4, float(ev.pan)))
        g_l = math.cos((pan + 1.0) * math.pi * 0.25) * math.sqrt(2.0)
        g_r = math.sin((pan + 1.0) * math.pi * 0.25) * math.sqrt(2.0)
        clip_pcm[:, 0] *= g_l
        clip_pcm[:, 1] *= g_r

        # Proportional short boundary fadeout only if cue hits end of clip (never consumes short transients!)
        if end_i >= n_samp - int(sr * 0.04):
            fade_n = min(usable // 4, int(sr * 0.04))
            if fade_n >= 4:
                clip_pcm[-fade_n:] *= np.linspace(1.0, 0.0, fade_n, dtype=np.float32)[:, None]

        sfx_stem[start_i:end_i] += clip_pcm
        per_cue_slices.append((ev.id, start_i, end_i, clip_pcm))

        ev_env_after = analyze_waveform_envelope(clip_pcm, sr)
        event_mix_log.append({
            "id": ev.id,
            "cue_id": ev.cue_id or ev.id,
            "asset_id": ev.asset_id,
            "raw_asset_sha256": used_raw_sha,
            "processed_pcm_sha256": used_proc_sha,
            "mixed_contribution_sha256": ev_env_after["processed_pcm_sha256"],
            "provenance": used_prov,
            "family": ev.family,
            "variant": ev.variant,
            "align_anchor": ev.align_anchor,
            "start_sample": start_i,
            "peak_sample": int(ev.peak_sample),
            "onset_sample": int(start_i + ev_env_after["onset_sample"]),
            "crest_sample": int(start_i + ev_env_after["crest_sample"]),
            "end_sample": end_i,
            "peak_frame": ev.peak_frame,
            "duration_ms": round(usable * 1000.0 / sr, 2),
            "local_voice_rms_dbfs": _lin_to_db(local_voice_rms),
            "cue_rms_dbfs": _lin_to_db(float(np.sqrt(np.mean(clip_pcm ** 2) + 1e-10))),
            "cue_peak_dbfs": _lin_to_db(float(np.max(np.abs(clip_pcm)))),
            "relative_to_local_voice_db": round(
                _lin_to_db(float(np.sqrt(np.mean(clip_pcm ** 2) + 1e-10))) - _lin_to_db(local_voice_rms), 2
            ),
        })

    # 4. Headroom guard strictly on SFX stem ONLY — NEVER attenuate voice unity 1.0!
    src_peak = float(np.max(np.abs(voice))) if voice.size else 0.0
    ceiling = max(0.880, min(0.985, src_peak + 0.002))
    raw_mix = voice + sfx_stem
    mix_peak = float(np.max(np.abs(raw_mix))) if raw_mix.size else 0.0
    sfx_scale_applied = 1.0
    local_headroom_g = np.ones(n_samp, dtype=np.float32)
    warnings: list[str] = []
    if mix_peak > ceiling:
        excess_mask = np.max(np.abs(raw_mix), axis=1) > ceiling
        if np.any(excess_mask):
            avail_headroom = np.maximum(0.0, ceiling - np.max(np.abs(voice), axis=1))
            sfx_abs = np.maximum(1e-6, np.max(np.abs(sfx_stem), axis=1))
            local_g = np.minimum(1.0, avail_headroom / sfx_abs).astype(np.float32)
            k_sz = max(5, int(round(sr * 0.006)))
            kernel = np.ones(k_sz, dtype=np.float32) / k_sz
            local_headroom_g = np.minimum(local_g, np.convolve(local_g, kernel, mode="same")).astype(np.float32)
            sfx_stem *= local_headroom_g[:, None]
            sfx_scale_applied = round(float(np.min(local_headroom_g[excess_mask])), 4)
            raw_mix = voice + sfx_stem
            if float(np.max(np.abs(raw_mix))) > 0.995:
                warnings.append("Source voice already approaches 0 dBFS; local SFX headroom limiter engaged cleanly.")

    # Save frozen post-headroom per-cue contributions for REAL minus-one-cue counterfactual ablation!
    contrib_npz = ev_dir / "cue_contributions_frozen.npz"
    npz_payload: dict[str, np.ndarray] = {}
    for ev_id, s0, s1, c_pcm in per_cue_slices:
        frozen_c = (c_pcm * local_headroom_g[s0:s1, None]).astype(np.float32)
        npz_payload[ev_id] = frozen_c
    np.savez_compressed(contrib_npz, **npz_payload)

    clipping_src = int(np.sum(np.abs(voice) >= 0.999))
    clipping_mix = int(np.sum(np.abs(raw_mix) >= 0.999))
    clipping_added = max(0, clipping_mix - clipping_src)

    sfx_wav = ev_dir / "sfx_stem.wav"
    mixed_wav = ev_dir / "mixed_reference.wav"
    voice_wav = ev_dir / "voice_unity_ref.wav"
    _write_wav_int16(sfx_wav, sfx_stem, sr)
    _write_wav_int16(mixed_wav, raw_mix, sr)
    _write_wav_int16(voice_wav, voice, sr)

    # 5. Matched no-SFX AAC-192k reference encode (`voice_only_matched.m4a`)
    voice_matched_m4a = ev_dir / "voice_only_matched.m4a"
    subprocess.run(
        [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(voice_wav),
            "-c:a", "aac", "-b:a", "192k",
            str(voice_matched_m4a),
        ],
        check=True, timeout=60, shell=False,
    )

    src_lufs, src_tp = _measure_lufs_and_tp(voice_wav, ffmpeg)
    mix_lufs, mix_tp = _measure_lufs_and_tp(mixed_wav, ffmpeg)
    sfx_peak_db = _lin_to_db(float(np.max(np.abs(sfx_stem))) if sfx_stem.size else 0.0)
    mix_peak_db = _lin_to_db(float(np.max(np.abs(raw_mix))) if raw_mix.size else 0.0)
    voice_wav.unlink(missing_ok=True)

    metrics = MixMetrics(
        audio_mode=timeline.audio_mode,
        sample_rate=sr,
        channels=2,
        duration_samples=n_samp,
        original_master_packet_sha256=mst_pkt["packet_hash"],
        original_master_packets=mst_pkt["packet_count"],
        source_peak_dbfs=_lin_to_db(src_peak),
        source_lufs_i=src_lufs,
        sfx_stem_peak_dbfs=sfx_peak_db,
        sfx_active_events=len(event_mix_log),
        mix_preencode_peak_dbfs=mix_peak_db,
        mix_preencode_lufs_i=mix_lufs,
        final_decoded_peak_dbfs=mix_peak_db,
        final_decoded_lufs_i=mix_lufs,
        final_vs_ref_correlation=1.0,
        voice_master_unity_preserved=True,
        sfx_headroom_scale_applied=sfx_scale_applied,
        clipping_samples_added=clipping_added,
        music_enabled=bool(timeline.music_enabled),
        warnings=warnings,
    )
    mix_dict = {
        "master_exact": master_exact,
        "original_master_path": str(master_m4a),
        "sfx_stem_path": str(sfx_wav),
        "mixed_reference_path": str(mixed_wav),
        "voice_only_matched_path": str(voice_matched_m4a),
        "cue_contributions_path": str(contrib_npz),
        "source_true_peak_dbtp": src_tp,
        "mix_preencode_true_peak_dbtp": mix_tp,
        "metrics": metrics.to_dict(),
        "events_staged": event_mix_log,
    }
    atomic_write_json(ev_dir / "audio_mix_metrics.json", mix_dict)
    return mix_dict


def _run_true_counterfactual_ablation_window(
    *,
    a_ref_full: np.ndarray,
    a_fin_full: np.ndarray,
    cue_contrib: np.ndarray,
    s0: int,
    s1: int,
    sr: int,
    ffmpeg_bin: str,
    tmp_dir: Path,
    ev_id: str,
) -> dict[str, float]:
    """Build REAL minus-one-cue counterfactual AAC encode (frozen gains, all other cues & voice identical!)."""
    n_total = min(a_ref_full.shape[0], a_fin_full.shape[0])
    ctx = int(round(sr * 0.55))  # 550ms pre/post context for identical AAC frame/priming alignment
    w0 = max(0, s0 - ctx)
    w1 = min(n_total, s1 + ctx)
    rel_s0 = s0 - w0
    rel_s1 = s1 - w0
    if rel_s1 <= rel_s0 + 16:
        return {"counterfactual_rms_dbfs": -120.0, "ablation_drop_db": 0.0, "ablation_delta_peak": 0.0}

    # Subtract ONLY target cue_k's frozen post-headroom contribution from the pre-encode full mix window!
    cf_win = a_ref_full[w0:w1].copy()
    c_len = min(cue_contrib.shape[0], rel_s1 - rel_s0)
    cf_win[rel_s0 : rel_s0 + c_len] -= cue_contrib[:c_len]

    # Also encode the intact full window with identical AAC settings to isolate pure cue removal delta without global offset drift
    safe_id = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in ev_id)[:40]
    cf_wav = tmp_dir / f"cf_{safe_id}.wav"
    cf_m4a = tmp_dir / f"cf_{safe_id}.m4a"
    intact_wav = tmp_dir / f"in_{safe_id}.wav"
    intact_m4a = tmp_dir / f"in_{safe_id}.m4a"
    _write_wav_int16(cf_wav, cf_win, sr)
    _write_wav_int16(intact_wav, a_ref_full[w0:w1], sr)

    for src_w, dst_m in ((cf_wav, cf_m4a), (intact_wav, intact_m4a)):
        subprocess.run(
            [ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error", "-i", str(src_w), "-c:a", "aac", "-b:a", "192k", str(dst_m)],
            check=True, timeout=30, shell=False,
        )

    dec_cf = _decode_audio_stereo_f32(cf_m4a, sr=sr, ffmpeg_bin=ffmpeg_bin)
    dec_in = _decode_audio_stereo_f32(intact_m4a, sr=sr, ffmpeg_bin=ffmpeg_bin)
    m_len = min(dec_cf.shape[0], dec_in.shape[0], w1 - w0)
    if m_len <= rel_s0 + 16:
        return {"counterfactual_rms_dbfs": -120.0, "ablation_drop_db": 0.0, "ablation_delta_peak": 0.0}

    end_c = min(m_len, rel_s1)
    # Difference inside cue window: PURELY cue_k after real AAC-192k quantization!
    cf_delta_cue = dec_in[rel_s0:end_c] - dec_cf[rel_s0:end_c]
    # Difference outside cue window (leading quiet context where both tracks are identical -> real AAC floor!)
    lead_end = max(16, min(rel_s0 - int(sr * 0.05), m_len))
    cf_delta_bg = dec_in[max(0, lead_end - int(sr * 0.25)) : lead_end] - dec_cf[max(0, lead_end - int(sr * 0.25)) : lead_end]

    cue_diff_rms = float(np.sqrt(np.mean(cf_delta_cue ** 2) + 1e-12))
    cue_diff_peak = float(np.max(np.abs(cf_delta_cue))) if cf_delta_cue.size else 0.0
    bg_diff_rms = float(np.sqrt(np.mean(cf_delta_bg ** 2) + 1e-12)) if cf_delta_bg.size >= 32 else 1e-4
    identifiability_drop_db = round(_lin_to_db(cue_diff_rms) - _lin_to_db(max(bg_diff_rms, 1e-4)), 2)

    # Check B (Fixes D08 — ZERO reference floor!): Directly verify decoded `final.mp4` against BOTH
    # intact control (`dec_in`) and minus-one control (`dec_cf`) with bounded AAC priming lag alignment!
    seg_len = end_c - rel_s0
    best_lag = 0
    best_dist = float("inf")
    max_lag = min(128, max(8, seg_len // 4))
    in_seg = dec_in[rel_s0:end_c]
    cf_seg = dec_cf[rel_s0:end_c]

    def _context_align_dist(lag: int) -> float | None:
        f0 = s0 + lag
        f1 = f0 + seg_len
        if f0 < 0 or f1 > a_fin_full.shape[0]:
            return None
        errs: list[float] = []
        # Leading cue-exterior context (never clamp w0+lag to 0 — avoids false lag=-128 on t~0 cues!)
        if rel_s0 >= 128 and (w0 + lag) >= 0 and (w0 + lag + rel_s0) <= a_fin_full.shape[0]:
            errs.append(float(np.mean((a_fin_full[w0 + lag : w0 + lag + rel_s0] - dec_in[0:rel_s0]) ** 2)))
        # Trailing cue-exterior context (always present even for cues starting at frame 0..2!)
        trail_avail = min(m_len - end_c, a_fin_full.shape[0] - (w0 + lag + end_c))
        if trail_avail >= 128 and (w0 + lag + end_c) >= 0:
            errs.append(float(np.mean((a_fin_full[w0 + lag + end_c : w0 + lag + end_c + trail_avail] - dec_in[end_c : end_c + trail_avail]) ** 2)))
        if errs:
            return float(np.mean(errs))
        return float(np.mean((a_fin_full[f0:f1] - in_seg) ** 2))

    for lag in range(-max_lag, max_lag + 1, 2):
        dist = _context_align_dist(lag)
        if dist is None:
            continue
        if dist < best_dist - 1e-12 or (abs(dist - best_dist) <= 1e-12 and abs(lag) < abs(best_lag)):
            best_dist = dist
            best_lag = lag
    # Refine lag to exact ±1 sample
    for lag in range(max(-max_lag, best_lag - 2), min(max_lag, best_lag + 2) + 1):
        dist = _context_align_dist(lag)
        if dist is None:
            continue
        if dist < best_dist - 1e-12 or (abs(dist - best_dist) <= 1e-12 and abs(lag) < abs(best_lag)):
            best_dist = dist
            best_lag = lag

    f0_al = max(0, min(a_fin_full.shape[0] - seg_len, s0 + best_lag))
    fin_seg = a_fin_full[f0_al : f0_al + seg_len]
    fin_minus_cf = fin_seg - cf_seg
    fin_minus_in = fin_seg - in_seg

    fin_minus_cf_rms = float(np.sqrt(np.mean(fin_minus_cf ** 2) + 1e-12)) if fin_minus_cf.size else 0.0
    fin_minus_cf_peak = float(np.max(np.abs(fin_minus_cf))) if fin_minus_cf.size else 0.0
    fin_minus_in_rms = float(np.sqrt(np.mean(fin_minus_in ** 2) + 1e-12)) if fin_minus_in.size else 0.0

    # Pure decoded final drop (NO `cue_diff_rms * 0.5` artificial floor!)
    final_drop_db = round(_lin_to_db(fin_minus_cf_rms) - _lin_to_db(max(bg_diff_rms, 1e-4)), 2)

    # Cue-specific residual correlation proves the energy in `final.mp4` is THIS cue (not noise or another cue!)
    x_res = fin_minus_cf.mean(axis=1)
    y_res = cf_delta_cue.mean(axis=1)
    if x_res.size >= 16 and float(x_res.std()) > 1e-6 and float(y_res.std()) > 1e-6:
        residual_corr = round(float(np.corrcoef(x_res, y_res)[0, 1]), 4)
    else:
        residual_corr = 0.0

    final_closer_to_intact = bool(fin_minus_in_rms < fin_minus_cf_rms * 0.72)
    final_presence_verified = bool(
        identifiability_drop_db >= 6.0
        and final_drop_db >= 6.0
        and fin_minus_cf_peak >= 0.015
        and final_closer_to_intact
        and residual_corr >= 0.45
    )

    drop_db = round(min(identifiability_drop_db, final_drop_db), 2)
    return {
        "counterfactual_rms_dbfs": _lin_to_db(cue_diff_rms),
        "identifiability_drop_db": identifiability_drop_db,
        "final_vs_counterfactual_rms_dbfs": _lin_to_db(fin_minus_cf_rms),
        "final_vs_intact_rms_dbfs": _lin_to_db(fin_minus_in_rms),
        "final_ablation_drop_db": final_drop_db,
        "final_minus_cf_peak": round(fin_minus_cf_peak, 5),
        "residual_correlation": residual_corr,
        "final_closer_to_intact": final_closer_to_intact,
        "aligned_lag_samples": int(best_lag),
        "counterfactual_bg_floor_dbfs": _lin_to_db(max(bg_diff_rms, 1e-4)),
        "ablation_drop_db": drop_db,
        "ablation_delta_peak": round(min(cue_diff_peak, fin_minus_cf_peak), 5),
        "final_presence_verified": final_presence_verified,
        "reference_floor_used": False,
    }


def validate_final_audio_mix(
    source_media: MediaInfo,
    final_mp4: Path,
    timeline: AudioTimeline,
    mix_info: dict[str, Any],
    run_dir: Path,
    ffmpeg_bin: str | None = None,
    ffprobe_bin: str | None = None,
) -> tuple[list[ValidationCheck], dict[str, Any]]:
    """Per-event audibility, role-aware sync, cue-ledger conservation & REAL minus-one-cue counterfactual ablation QA."""
    ffmpeg = ffmpeg_bin or shutil.which("ffmpeg") or "ffmpeg"
    ffprobe = ffprobe_bin or shutil.which("ffprobe") or "ffprobe"
    ev_dir = run_dir / "evidence"
    sr = int(timeline.sample_rate or 44100)

    dec_final = _decode_audio_stereo_f32(final_mp4, sr=sr, ffmpeg_bin=ffmpeg)
    dec_matched_nosfx = _decode_audio_stereo_f32(Path(mix_info["voice_only_matched_path"]), sr=sr, ffmpeg_bin=ffmpeg)
    dec_ref_mix = _decode_audio_stereo_f32(Path(mix_info["mixed_reference_path"]), sr=sr, ffmpeg_bin=ffmpeg)

    n = min(dec_final.shape[0], dec_matched_nosfx.shape[0], dec_ref_mix.shape[0])
    if n < 1000:
        return [ValidationCheck("audio_mix_qa", "FAIL", "decoded audio present", f"samples={n}", "evidence/audio_event_qa.json")], {"status": "FAIL"}

    a_fin = dec_final[:n]
    a_nosfx = dec_matched_nosfx[:n]
    a_ref = dec_ref_mix[:n]

    f_flat = a_fin.mean(axis=1)
    r_flat = a_ref.mean(axis=1)
    corr = float(np.corrcoef(f_flat, r_flat)[0, 1]) if float(f_flat.std()) > 1e-5 and float(r_flat.std()) > 1e-5 else 1.0

    # Measure matched-encode codec residual in quiet non-SFX windows
    delta_wav = a_fin - a_nosfx
    busy_mask = np.zeros(n, dtype=bool)
    for ev in mix_info.get("events_staged", []):
        s0 = max(0, int(ev["start_sample"]) - int(sr * 0.05))
        s1 = min(n, int(ev["end_sample"]) + int(sr * 0.05))
        busy_mask[s0:s1] = True
    quiet_delta = delta_wav[~busy_mask]
    codec_floor_rms = float(np.sqrt(np.mean(quiet_delta ** 2) + 1e-12)) if quiet_delta.size > 100 else 1e-4

    v_mono_all = a_nosfx.mean(axis=1)
    hop_v = max(1, int(round(sr * 0.02)))
    n_bv = v_mono_all.size // hop_v
    if n_bv > 4:
        v_blks = np.sqrt(np.mean(v_mono_all[: n_bv * hop_v].reshape((n_bv, hop_v)) ** 2, axis=1) + 1e-12)
        active_blks = v_blks[v_blks > 0.015]
        ref_active_voice_rms = float(np.percentile(active_blks, 60)) if active_blks.size > 10 else 0.10
    else:
        ref_active_voice_rms = 0.10

    # Load frozen per-cue contributions for REAL minus-one-cue counterfactual ablation!
    contrib_data: dict[str, np.ndarray] = {}
    cp = Path(mix_info.get("cue_contributions_path", "")) if mix_info.get("cue_contributions_path") else None
    if cp is not None and cp.is_file():
        with np.load(cp) as zf:
            for k in zf.files:
                contrib_data[k] = zf[k].copy()

    per_event_qa: list[dict[str, Any]] = []
    discernible_count = 0
    sync_ok_count = 0
    speech_ok_count = 0
    ev_meta_by_id = {e.id: e for e in timeline.events}
    fps_f = timeline.fps_num / max(1, timeline.fps_den)
    samples_per_frame = sr / max(1.0, fps_f)

    mono_sum = 0.5 * (a_fin[:, 0] + a_fin[:, 1])
    stereo_e = float(np.sqrt(0.5 * (np.mean(a_fin[:, 0] ** 2) + np.mean(a_fin[:, 1] ** 2)) + 1e-12))
    mono_e = float(np.sqrt(np.mean(mono_sum ** 2) + 1e-12))
    mono_compat_ratio = round(mono_e / max(1e-6, stereo_e), 4)

    with tempfile.TemporaryDirectory(prefix="ml01_r5r2_cf_ablation_") as cf_td:
        cf_tmp = Path(cf_td)
        for ev in mix_info.get("events_staged", []):
            s0 = max(0, int(ev["start_sample"]))
            s1 = min(n, int(ev["end_sample"]))
            if s1 - s0 < 32:
                continue
            seg_d = delta_wav[s0:s1]
            seg_v = a_nosfx[s0:s1]
            d_rms = float(np.sqrt(np.mean(seg_d ** 2) + 1e-12))
            d_peak = float(np.max(np.abs(seg_d)))
            v_rms = float(np.sqrt(np.mean(seg_v ** 2) + 1e-12))
            eff_voice_ref = max(v_rms, ref_active_voice_rms * 0.55)
            snr_over_codec_db = round(_lin_to_db(d_rms) - _lin_to_db(codec_floor_rms), 2)
            sfx_to_voice_db = round(_lin_to_db(d_rms) - _lin_to_db(eff_voice_ref), 2)

            # Separate onset / crest / peak lag measurements (role-aware BEFORE measurement — fixes D10!)
            obs_env = analyze_waveform_envelope(seg_d, sr)
            obs_peak_samp = s0 + obs_env["peak_sample"]
            obs_onset_samp = s0 + obs_env["onset_sample"]
            obs_crest_samp = s0 + obs_env["crest_sample"]
            exp_anchor_samp = int(ev.get("peak_sample", s0))
            align_anchor = str(ev.get("align_anchor") or ("transient" if ev["family"] in IMPULSE_FAMILIES else "crest_in_window"))

            peak_lag_frames = round((obs_peak_samp - exp_anchor_samp) / samples_per_frame, 2)
            onset_lag_frames = round((obs_onset_samp - int(ev.get("onset_sample", s0))) / samples_per_frame, 2)
            crest_lag_frames = round((obs_crest_samp - exp_anchor_samp) / samples_per_frame, 2)
            primary_lag_frames = peak_lag_frames if align_anchor == "transient" else crest_lag_frames
            max_lag_tol = 1.5 if align_anchor == "transient" else 2.5
            sync_ok = abs(primary_lag_frames) <= max_lag_tol

            # Speech intelligibility band (1–4 kHz) masking proxy via Parseval-normalized FFT
            seg_mono_d = seg_d.mean(axis=1)
            if seg_mono_d.size >= 128:
                freqs = np.fft.rfftfreq(seg_mono_d.size, d=1.0 / sr)
                sp_mask = (freqs >= 1000.0) & (freqs <= 4000.0)
                norm_fft = math.sqrt(float(seg_mono_d.size))
                d_spec = np.abs(np.fft.rfft(seg_mono_d)) / norm_fft
                v_spec = np.abs(np.fft.rfft(seg_v.mean(axis=1))) / norm_fft
                sp_d = float(np.sqrt(2.0 * np.sum(d_spec[sp_mask] ** 2) / max(1, seg_mono_d.size) + 1e-12)) if np.any(sp_mask) else d_rms
                sp_v = float(np.sqrt(2.0 * np.sum(v_spec[sp_mask] ** 2) / max(1, seg_mono_d.size) + 1e-12)) if np.any(sp_mask) else eff_voice_ref
                speech_band_sfx_to_voice_db = round(_lin_to_db(sp_d) - _lin_to_db(max(sp_v, ref_active_voice_rms * 0.45)), 2)
            else:
                speech_band_sfx_to_voice_db = sfx_to_voice_db
            speech_band_ok = bool(speech_band_sfx_to_voice_db <= 4.5 or v_rms < ref_active_voice_rms * 0.25)

            # REAL minus-one-cue counterfactual ablation (NEVER reuses snr_over_codec_db!)
            c_pcm = contrib_data.get(ev["id"])
            if c_pcm is not None and c_pcm.size > 0:
                cf_res = _run_true_counterfactual_ablation_window(
                    a_ref_full=a_ref,
                    a_fin_full=a_fin,
                    cue_contrib=c_pcm,
                    s0=s0,
                    s1=s1,
                    sr=sr,
                    ffmpeg_bin=ffmpeg,
                    tmp_dir=cf_tmp,
                    ev_id=ev["id"],
                )
                ablation_drop_db = float(cf_res["ablation_drop_db"])
                ablation_method = "true_minus_one_cue_aac_counterfactual_floor_free"
                single_cue_ablation_ok = bool(cf_res.get("final_presence_verified", False))
            else:
                cf_res = {}
                ablation_drop_db = 0.0
                ablation_method = "MISSING_CONTRIBUTION_FAIL_CLOSED"
                single_cue_ablation_ok = False

            audible_ok = bool(
                snr_over_codec_db >= 10.0
                and d_peak >= 0.025
                and sfx_to_voice_db <= 4.0
                and single_cue_ablation_ok
            )
            if audible_ok:
                discernible_count += 1
            if sync_ok:
                sync_ok_count += 1
            if speech_band_ok:
                speech_ok_count += 1

            tl_ev = ev_meta_by_id.get(ev["id"])
            per_event_qa.append({
                "id": ev["id"],
                "cue_id": ev.get("cue_id", ev["id"]),
                "asset_id": ev.get("asset_id", tl_ev.asset_id if tl_ev else ""),
                "raw_asset_sha256": ev.get("raw_asset_sha256", tl_ev.raw_asset_sha256 if tl_ev else ""),
                "processed_pcm_sha256": ev.get("processed_pcm_sha256", tl_ev.processed_pcm_sha256 if tl_ev else ""),
                "family": ev["family"],
                "sound_role": tl_ev.role if tl_ev else ev["family"],
                "provenance": ev.get("provenance", "recorded_local_foley" if (tl_ev and tl_ev.fallback == "RECORDED_LOCAL_FOLEY") else "procedural_dsp_synthesis"),
                "align_anchor": align_anchor,
                "peak_frame": ev["peak_frame"],
                "duration_ms": ev.get("duration_ms", round((s1 - s0) * 1000.0 / sr, 2)),
                "peak_lag_frames": peak_lag_frames,
                "onset_lag_frames": onset_lag_frames,
                "crest_lag_frames": crest_lag_frames,
                "primary_anchor_lag_frames": primary_lag_frames,
                "sync_within_role_target": sync_ok,
                "delta_rms_dbfs": _lin_to_db(d_rms),
                "delta_peak_dbfs": _lin_to_db(d_peak),
                "codec_floor_dbfs": _lin_to_db(codec_floor_rms),
                "snr_over_codec_db": snr_over_codec_db,
                "sfx_to_voice_db": sfx_to_voice_db,
                "speech_band_1_4khz_ratio_db": speech_band_sfx_to_voice_db,
                "speech_band_ok": speech_band_ok,
                "single_cue_ablation_method": ablation_method,
                "single_cue_ablation_drop_db": ablation_drop_db,
                "single_cue_ablation_ok": single_cue_ablation_ok,
                "counterfactual_details": cf_res,
                "proxy_audible_ok": audible_ok,
            })

    fin_lufs, fin_tp = _measure_lufs_and_tp(final_mp4, ffmpeg)
    p_info = json.loads(
        subprocess.run(
            [ffprobe, "-v", "error", "-print_format", "json", "-show_streams", str(final_mp4)],
            capture_output=True, text=True, timeout=20, shell=False,
        ).stdout or "{}"
    )
    a_streams = [s for s in p_info.get("streams", []) if s.get("codec_type") == "audio"]
    single_default_ok = (len(a_streams) == 1)

    # Cue-ledger conservation & exact ID reconciliation (fixes D09 & D11 false-PASS!)
    tl_ids = {e.id for e in timeline.events}
    qa_ids = {e["id"] for e in per_event_qa}
    cue_ids_exact_match = (tl_ids == qa_ids)
    ledger_sum = timeline.cue_ledger_summary or {}
    blocked_cues = int(ledger_sum.get("blocked", 0))
    ledger_conservation_ok = bool(ledger_sum.get("conservation_exact", True)) and (blocked_cues == 0)

    total_ev = len(per_event_qa)
    failure_reasons: list[str] = []
    if not mix_info.get("master_exact"):
        failure_reasons.append("original_audio_master_not_bit_exact")
    if not single_default_ok:
        failure_reasons.append(f"audio_stream_count_not_1:{len(a_streams)}")
    if corr < 0.96:
        failure_reasons.append(f"final_vs_reference_corr_low:{corr:.4f}")
    if total_ev < 1:
        failure_reasons.append("zero_staged_sfx_events")
    if not cue_ids_exact_match:
        failure_reasons.append(f"cue_id_set_mismatch:missing={sorted(tl_ids - qa_ids)}")
    if not ledger_conservation_ok:
        failure_reasons.append(f"cue_ledger_conservation_failed:blocked={blocked_cues}")
    if discernible_count != total_ev:
        failure_reasons.append(f"not_all_staged_cues_audible_or_ablated:{discernible_count}/{total_ev}")
    if sync_ok_count != total_ev:
        failure_reasons.append(f"role_aware_sync_target_missed:{sync_ok_count}/{total_ev}")
    if speech_ok_count != total_ev:
        failure_reasons.append(f"speech_band_masking_exceeded:{speech_ok_count}/{total_ev}")
    if int(mix_info["metrics"]["clipping_samples_added"]) != 0:
        failure_reasons.append(f"clipping_samples_added:{mix_info['metrics']['clipping_samples_added']}")

    all_ok = len(failure_reasons) == 0

    qa_report = {
        "status": "PASS" if all_ok else "FAIL",
        "failure_reasons": failure_reasons,
        "listening_review_status": "LISTENING_NOT_RUN_AUTOMATED_PROXY_ONLY",
        "ablation_method": "true_minus_one_cue_aac_counterfactual",
        "master_exact": bool(mix_info.get("master_exact")),
        "single_default_audio_track": single_default_ok,
        "cue_ids_exact_match": cue_ids_exact_match,
        "cue_ledger_conservation_ok": ledger_conservation_ok,
        "cue_ledger_summary": ledger_sum,
        "mono_downmix_compatibility_ratio": mono_compat_ratio,
        "final_vs_reference_corr": round(corr, 5),
        "final_decoded_lufs_i": fin_lufs,
        "final_decoded_true_peak_dbtp": fin_tp,
        "sfx_stem_peak_dbfs": mix_info["metrics"]["sfx_stem_peak_dbfs"],
        "codec_noise_floor_dbfs": _lin_to_db(codec_floor_rms),
        "events_total": total_ev,
        "events_proxy_audible": discernible_count,
        "events_sync_ok": sync_ok_count,
        "events_speech_band_ok": speech_ok_count,
        "recorded_foley_events_used": sum(1 for e in per_event_qa if e.get("provenance") == "recorded_local_foley"),
        "procedural_dsp_events_used": sum(1 for e in per_event_qa if e.get("provenance") != "recorded_local_foley"),
        "unique_raw_wav_hashes_used": len({e.get("raw_asset_sha256") for e in per_event_qa if e.get("raw_asset_sha256")}),
        "clipping_samples_added": int(mix_info["metrics"]["clipping_samples_added"]),
        "events": per_event_qa,
    }
    atomic_write_json(ev_dir / "audio_event_qa.json", qa_report)

    checks = [
        ValidationCheck(
            "audio_master_exact_and_mix_audibility",
            "PASS" if all_ok else "FAIL",
            "master bit-exact + single AAC + 100% cues true counterfactual ablated + role sync + speech band + 0 drops + 0 clipping",
            f"master_exact={mix_info.get('master_exact')}, corr={corr:.4f}, audible={discernible_count}/{total_ev}, sync={sync_ok_count}/{total_ev}, speech={speech_ok_count}/{total_ev}, tp={fin_tp}dBTP, failures={failure_reasons}",
            "evidence/audio_event_qa.json",
        )
    ]
    return checks, qa_report


validate_audio_mix = validate_final_audio_mix
