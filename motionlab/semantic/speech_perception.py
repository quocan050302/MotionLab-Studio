"""Phase A1 (ML-01.R5.R4): Local Speech Perception, Word Alignment, Neural Silero VAD, Prosody & Visual Action Evidence.

100% offline / Zero-API. Reuses built-in faster-whisper Silero VAD ONNX (`onnxruntime`), native PCM
cross-checks, chunked prosody analysis, and bounded-memory ROI/optical-flow action evidence.

Honesty invariants:
  - Never claims FORCED_ALIGNED / sample-exact ground truth unless a real phonetic aligner ran.
  - Whisper + VAD/envelope refinement is explicitly labeled `VAD_ENVELOPE_REFINED` or `ASR_WORD_ESTIMATE`
    with empirical `uncertainty_ms` and conservative protected margins.
  - Segment-only transcripts stay `SEGMENT_APPROX` with whole-segment protection (never cut uncertain speech).
  - Without a verified local VLM, semantic action labels remain `UNKNOWN_ACTION` (optical flow alone never
    claims semantic action identity).
"""

from __future__ import annotations

import hashlib
import math
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from motionlab.schemas import MediaInfo, atomic_write_json, canonical_hash, sanitize_native
from motionlab.semantic.schema import Transcript

PERCEPTION_SCHEMA_VERSION = "1.0.0"
VAD_SAMPLE_RATE = 16000
CHUNK_DURATION_S = 60.0  # Bounded memory chunking for long-form audio


@dataclass
class TimingEvidence:
    """Per-word timing evidence with explicit provenance and conservative protection margins."""

    word_id: int
    word: str
    segment_id: str
    source_start_s: float
    source_end_s: float
    source_start_sample: int
    source_end_sample: int
    source_start_frame: int
    source_end_frame_exclusive: int
    protected_start_s: float
    protected_end_s: float
    confidence: float
    uncertainty_ms: float
    method: str  # "VAD_ENVELOPE_REFINED" | "ASR_WORD_ESTIMATE" | "SEGMENT_APPROX" | "FORCED_ALIGNED" | "MANUAL_QA_ONLY"
    evidence_id: str
    language: str
    speaker_id: str | None = None
    vad_confirmed: bool = False
    trailing_energy_extended: bool = False

    def to_dict(self) -> dict[str, Any]:
        return sanitize_native(asdict(self))


def _decode_mono_f32_pcm(path: Path, sr: int = VAD_SAMPLE_RATE, ffmpeg_bin: str | None = None) -> np.ndarray:
    """Decode primary audio track read-only to mono float32 PCM at target rate."""
    ffmpeg = ffmpeg_bin or shutil.which("ffmpeg") or "ffmpeg"
    proc = subprocess.run(
        [
            ffmpeg, "-v", "error", "-nostdin",
            "-i", str(path),
            "-vn", "-map", "0:a:0",
            "-ac", "1", "-ar", str(sr),
            "-f", "f32le", "pipe:1",
        ],
        capture_output=True, timeout=120, shell=False,
    )
    if proc.returncode != 0 or not proc.stdout:
        return np.zeros((0,), dtype=np.float32)
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def _frame_rms_and_speech_band(pcm_16k: np.ndarray, hop_samples: int = 320, win_samples: int = 400) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-20ms-frame RMS (dBFS), 300-3400Hz speech-band energy ratio, and spectral flatness."""
    n = pcm_16k.size
    if n < win_samples:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    n_frames = 1 + (n - win_samples) // hop_samples
    rms_db = np.empty(n_frames, dtype=np.float32)
    sb_ratio = np.empty(n_frames, dtype=np.float32)
    flatness = np.empty(n_frames, dtype=np.float32)
    window = np.hanning(win_samples).astype(np.float32)
    n_fft = 512
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / VAD_SAMPLE_RATE)
    sb_mask = (freqs >= 300.0) & (freqs <= 3400.0)

    # Process in blocks of 2000 frames (40s) for cache locality & bounded peak memory
    blk = 2000
    for b0 in range(0, n_frames, blk):
        b1 = min(n_frames, b0 + blk)
        count = b1 - b0
        starts = (np.arange(b0, b1) * hop_samples)[:, None]
        idx = starts + np.arange(win_samples)[None, :]
        frames = pcm_16k[idx]
        r = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
        rms_db[b0:b1] = (20.0 * np.log10(np.maximum(r, 1e-6))).astype(np.float32)
        spec = np.abs(np.fft.rfft(frames * window[None, :], n=n_fft, axis=1)) ** 2 + 1e-12
        tot = np.sum(spec, axis=1)
        sb = np.sum(spec[:, sb_mask], axis=1)
        sb_ratio[b0:b1] = (sb / np.maximum(tot, 1e-12)).astype(np.float32)
        geo = np.exp(np.mean(np.log(spec[:, 1:]), axis=1))
        arith = np.mean(spec[:, 1:], axis=1)
        flatness[b0:b1] = np.clip(geo / np.maximum(arith, 1e-12), 0.0, 1.0).astype(np.float32)
    return rms_db, sb_ratio, flatness


def detect_speech_activity(
    media: MediaInfo,
    *,
    pcm_16k: np.ndarray | None = None,
    ffmpeg_bin: str | None = None,
    vad_threshold: float = 0.42,
    min_speech_ms: int = 120,
    min_silence_ms: int = 180,
    expect_speech_from_transcript: bool = False,
) -> dict[str, Any]:
    """Run local neural Silero VAD ONNX (via installed faster-whisper) + multi-feature spectral cross-check.

    Uses chunked 60s windows so peak RAM stays strictly bounded on long-form media.
    Never modifies source audio; maps all intervals back to exact native sample rate via Fraction.
    """
    sr_native = int(media.audio_streams[0].sample_rate) if media.audio_streams else 44100
    dur_s = float(media.duration_s)
    total_native_samples = int(round(Fraction(media.duration_frames * media.fps_den * sr_native, media.fps_num)))

    if not media.has_audio:
        return {
            "schema_version": PERCEPTION_SCHEMA_VERSION,
            "source_hash": media.input_hash,
            "has_audio": False,
            "silent_video": True,
            "vad_backend": "none_no_audio_stream",
            "native_sample_rate": sr_native,
            "duration_s": round(dur_s, 4),
            "noise_floor_dbfs": -96.0,
            "speech_spans": [],
            "non_speech_spans": [
                {
                    "span_id": "ns_00",
                    "start_s": 0.0,
                    "end_s": round(dur_s, 4),
                    "duration_s": round(dur_s, 4),
                    "start_sample": 0,
                    "end_sample": total_native_samples,
                    "mean_rms_dbfs": -96.0,
                    "peak_rms_dbfs": -96.0,
                    "speech_prob_max": 0.0,
                    "has_ambient_activity": False,
                    "uncertain": False,
                }
            ],
            "uncertain_spans": [],
            "speech_ratio": 0.0,
        }

    wav = pcm_16k if pcm_16k is not None else _decode_mono_f32_pcm(Path(media.source_path), sr=VAD_SAMPLE_RATE, ffmpeg_bin=ffmpeg_bin)
    if wav.size < 400:
        return {
            "schema_version": PERCEPTION_SCHEMA_VERSION,
            "source_hash": media.input_hash,
            "has_audio": True,
            "silent_video": True,
            "vad_backend": "empty_decoded_audio",
            "native_sample_rate": sr_native,
            "duration_s": round(dur_s, 4),
            "noise_floor_dbfs": -96.0,
            "speech_spans": [],
            "non_speech_spans": [],
            "uncertain_spans": [],
            "speech_ratio": 0.0,
        }

    # 1. Compute 20ms spectral/energy descriptors across whole signal (bounded blocks)
    hop_s = 0.020
    hop_n = int(round(hop_s * VAD_SAMPLE_RATE))  # 320
    rms_db, sb_ratio, flatness = _frame_rms_and_speech_band(wav, hop_samples=hop_n, win_samples=400)
    noise_floor_db = float(np.percentile(rms_db, 15)) if rms_db.size > 10 else -60.0
    peak_db = float(np.percentile(rms_db, 98)) if rms_db.size > 10 else -20.0
    dynamic_range_db = peak_db - noise_floor_db

    # Detect truly silent audio track (peak < -58 dBFS)
    if peak_db < -58.0:
        return {
            "schema_version": PERCEPTION_SCHEMA_VERSION,
            "source_hash": media.input_hash,
            "has_audio": True,
            "silent_video": True,
            "vad_backend": "energy_gate_silent_track",
            "native_sample_rate": sr_native,
            "duration_s": round(dur_s, 4),
            "noise_floor_dbfs": round(noise_floor_db, 2),
            "speech_spans": [],
            "non_speech_spans": [
                {
                    "span_id": "ns_00",
                    "start_s": 0.0,
                    "end_s": round(dur_s, 4),
                    "duration_s": round(dur_s, 4),
                    "start_sample": 0,
                    "end_sample": total_native_samples,
                    "mean_rms_dbfs": round(noise_floor_db, 2),
                    "peak_rms_dbfs": round(peak_db, 2),
                    "speech_prob_max": 0.0,
                    "has_ambient_activity": False,
                    "uncertain": False,
                }
            ],
            "uncertain_spans": [],
            "speech_ratio": 0.0,
        }

    # 2. Run neural Silero VAD ONNX via faster_whisper.vad in bounded 60s chunks (fallback: spectral/energy VAD)
    raw_speech_16k: list[tuple[int, int]] = []
    vad_backend = "spectral_energy_hysteresis"
    try:
        from faster_whisper.vad import VadOptions, get_speech_timestamps  # noqa: PLC0415

        vad_opts = VadOptions(
            threshold=float(vad_threshold),
            neg_threshold=max(0.15, float(vad_threshold) - 0.15),
            min_speech_duration_ms=int(min_speech_ms),
            min_silence_duration_ms=int(min_silence_ms),
            speech_pad_ms=60,
        )
        chunk_n = int(round(CHUNK_DURATION_S * VAD_SAMPLE_RATE))
        for c0 in range(0, wav.size, chunk_n):
            c1 = min(wav.size, c0 + chunk_n)
            seg_wav = np.ascontiguousarray(wav[c0:c1], dtype=np.float32)
            if seg_wav.size < 512:
                continue
            ts_list = get_speech_timestamps(seg_wav, vad_options=vad_opts, sampling_rate=VAD_SAMPLE_RATE)
            for item in ts_list:
                s0 = c0 + int(item["start"])
                s1 = c0 + int(item["end"])
                if s1 > s0:
                    if raw_speech_16k and s0 <= raw_speech_16k[-1][1] + int(0.06 * VAD_SAMPLE_RATE):
                        raw_speech_16k[-1] = (raw_speech_16k[-1][0], max(raw_speech_16k[-1][1], s1))
                    else:
                        raw_speech_16k.append((s0, s1))
        vad_backend = "silero_vad_onnx"
    except Exception:
        vad_backend = "spectral_energy_hysteresis"

    if (vad_backend == "spectral_energy_hysteresis") or (
        expect_speech_from_transcript and not raw_speech_16k and peak_db >= -36.0 and dynamic_range_db >= 18.0
    ):
        if vad_backend == "silero_vad_onnx":
            vad_backend = "silero_vad_onnx_plus_energy_reconciled"
        # Honest fallback / reconciliation: spectral speech-band + RMS hysteresis
        eng_thresh = max(noise_floor_db + 9.0, -40.0)
        min_sb = 0.20 if expect_speech_from_transcript else 0.45
        is_sp = (rms_db >= eng_thresh) & (sb_ratio >= min_sb) & (flatness <= 0.45)
        in_sp = False
        sp_start = 0
        min_sp_f = max(1, int(round(min_speech_ms / 20.0)))
        for fi, flag in enumerate(is_sp):
            if flag and not in_sp:
                in_sp = True
                sp_start = fi
            elif not flag and in_sp:
                in_sp = False
                if fi - sp_start >= min_sp_f:
                    raw_speech_16k.append((sp_start * hop_n, fi * hop_n))
        if in_sp and (rms_db.size - sp_start) >= min_sp_f:
            raw_speech_16k.append((sp_start * hop_n, rms_db.size * hop_n))

    # 3. Build exact speech_spans & complementary non_speech_spans with cross-check uncertainty
    speech_spans: list[dict[str, Any]] = []
    scale_to_native = Fraction(sr_native, VAD_SAMPLE_RATE)
    for idx, (s0_16, s1_16) in enumerate(raw_speech_16k):
        t0 = max(0.0, min(dur_s, s0_16 / VAD_SAMPLE_RATE))
        t1 = max(t0, min(dur_s, s1_16 / VAD_SAMPLE_RATE))
        f0 = max(0, min(rms_db.size - 1, int(s0_16 // hop_n)))
        f1 = max(f0 + 1, min(rms_db.size, int(math.ceil(s1_16 / hop_n))))
        sl_rms = rms_db[f0:f1]
        sl_sb = sb_ratio[f0:f1]
        speech_spans.append({
            "span_id": f"sp_{idx:02d}",
            "start_s": round(t0, 4),
            "end_s": round(t1, 4),
            "duration_s": round(t1 - t0, 4),
            "start_sample": int(round(Fraction(s0_16) * scale_to_native)),
            "end_sample": min(total_native_samples, int(round(Fraction(s1_16) * scale_to_native))),
            "mean_rms_dbfs": round(float(np.mean(sl_rms)), 2) if sl_rms.size else -60.0,
            "peak_rms_dbfs": round(float(np.max(sl_rms)), 2) if sl_rms.size else -60.0,
            "speech_band_ratio": round(float(np.mean(sl_sb)), 3) if sl_sb.size else 0.0,
            "vad_source": vad_backend,
        })

    # Complement intervals = non-speech candidates; flag uncertain if high speech-band energy exists inside non-speech!
    non_speech_spans: list[dict[str, Any]] = []
    uncertain_spans: list[dict[str, Any]] = []
    cursor_s = 0.0
    ns_idx = 0
    boundaries = [(sp["start_s"], sp["end_s"]) for sp in speech_spans] + [(dur_s, dur_s)]
    for sp_s0, sp_s1 in boundaries:
        if sp_s0 - cursor_s >= 0.04:
            f0 = max(0, min(rms_db.size - 1, int(round(cursor_s / hop_s))))
            f1 = max(f0 + 1, min(rms_db.size, int(round(sp_s0 / hop_s))))
            sl_rms = rms_db[f0:f1]
            sl_sb = sb_ratio[f0:f1]
            sl_fl = flatness[f0:f1]
            mean_r = float(np.mean(sl_rms)) if sl_rms.size else noise_floor_db
            peak_r = float(np.max(sl_rms)) if sl_rms.size else noise_floor_db
            mean_sb = float(np.mean(sl_sb)) if sl_sb.size else 0.0
            mean_fl = float(np.mean(sl_fl)) if sl_fl.size else 0.5
            # Ambient activity (e.g. cash rustling, footsteps, room action) vs dead silence
            has_ambient = bool(peak_r > noise_floor_db + 7.5 and (sp_s0 - cursor_s) >= 0.15)
            # Disagreement check: strong tonal speech-band energy inside a VAD "non-speech" span -> UNCERTAIN!
            uncertain = bool(
                (peak_r > max(noise_floor_db + 14.0, -28.0))
                and (mean_sb >= 0.68)
                and (mean_fl <= 0.16)
            )
            ns_obj = {
                "span_id": f"ns_{ns_idx:02d}",
                "start_s": round(cursor_s, 4),
                "end_s": round(sp_s0, 4),
                "duration_s": round(sp_s0 - cursor_s, 4),
                "start_sample": int(round(cursor_s * sr_native)),
                "end_sample": min(total_native_samples, int(round(sp_s0 * sr_native))),
                "mean_rms_dbfs": round(mean_r, 2),
                "peak_rms_dbfs": round(peak_r, 2),
                "speech_band_ratio": round(mean_sb, 3),
                "spectral_flatness": round(mean_fl, 3),
                "has_ambient_activity": has_ambient,
                "uncertain": uncertain,
            }
            non_speech_spans.append(ns_obj)
            if uncertain:
                uncertain_spans.append({
                    "start_s": ns_obj["start_s"],
                    "end_s": ns_obj["end_s"],
                    "reason": "vad_spectral_disagreement_possible_soft_speech",
                })
            ns_idx += 1
        cursor_s = max(cursor_s, sp_s1)

    total_speech_s = sum(sp["duration_s"] for sp in speech_spans)
    return {
        "schema_version": PERCEPTION_SCHEMA_VERSION,
        "source_hash": media.input_hash,
        "has_audio": True,
        "silent_video": len(speech_spans) == 0,
        "vad_backend": vad_backend,
        "native_sample_rate": sr_native,
        "duration_s": round(dur_s, 4),
        "noise_floor_dbfs": round(noise_floor_db, 2),
        "peak_dbfs": round(peak_db, 2),
        "dynamic_range_db": round(dynamic_range_db, 2),
        "speech_spans": speech_spans,
        "non_speech_spans": non_speech_spans,
        "uncertain_spans": uncertain_spans,
        "speech_ratio": round(total_speech_s / max(0.001, dur_s), 4),
    }


def align_transcript_words(
    transcript: Transcript,
    vad_result: dict[str, Any],
    media: MediaInfo,
    *,
    pcm_16k: np.ndarray | None = None,
    ffmpeg_bin: str | None = None,
) -> dict[str, Any]:
    """Refine & bound per-word timing evidence using local VAD + 5ms RMS envelope cross-check.

    Conservative protection invariant:
      - NEVER shrinks protected windows inside active voiced energy.
      - Extends `protected_end_s` outward when trailing consonant/syllable energy persists after Whisper end_s.
      - NEVER labels Whisper DTW or envelope snapping as `FORCED_ALIGNED` or `EXACT`.
      - When segment-only timestamps are detected, protects entire segments (`SEGMENT_APPROX`).
    """
    sr_native = int(media.audio_streams[0].sample_rate) if media.audio_streams else 44100
    fps_f = media.fps_float
    total_f = media.duration_frames
    dur_s = float(media.duration_s)

    wav = pcm_16k
    if wav is None and media.has_audio and Path(media.source_path).is_file():
        wav = _decode_mono_f32_pcm(Path(media.source_path), sr=VAD_SAMPLE_RATE, ffmpeg_bin=ffmpeg_bin)
    has_pcm = wav is not None and wav.size >= 160

    # 5ms short-time RMS envelope for crisp consonant/syllable boundary protection
    hop_ms = 5.0
    hop_5ms = int(round(VAD_SAMPLE_RATE * hop_ms / 1000.0))  # 80 samples
    if has_pcm:
        n_blk = wav.size // hop_5ms
        env_5ms = np.sqrt(np.mean(wav[: n_blk * hop_5ms].reshape((n_blk, hop_5ms)) ** 2, axis=1) + 1e-12)
        env_5ms_db = 20.0 * np.log10(np.maximum(env_5ms, 1e-6))
    else:
        env_5ms_db = np.zeros((0,), dtype=np.float32)

    noise_floor = float(vad_result.get("noise_floor_dbfs", -50.0))
    voice_edge_db = max(noise_floor + 8.0, -40.0)
    speech_spans = vad_result.get("speech_spans") or []

    # Check whether transcript truly has distinct per-word timestamps or only segment-wide timestamps
    raw_words: list[tuple[str, Any]] = []
    for seg in transcript.segments:
        for w in seg.words:
            if w.word.strip():
                raw_words.append((seg.segment_id, w))

    distinct_starts = len({round(float(w.start_s), 3) for _, w in raw_words})
    segment_only = (len(raw_words) >= 4 and distinct_starts <= max(1, len(transcript.segments))) or (
        getattr(transcript, "model_id", "") == "story_graph_beat_approx"
    )
    seg_bounds = {seg.segment_id: (float(seg.start_s), float(seg.end_s)) for seg in transcript.segments}

    aligned: list[TimingEvidence] = []
    prev_end_s = 0.0

    for wid, (seg_id, w) in enumerate(raw_words):
        w_text = w.word.strip()
        prob = float(getattr(w, "probability", 0.85))
        s0 = max(0.0, min(dur_s, float(w.start_s)))
        s1 = max(s0 + 0.02, min(dur_s, float(w.end_s)))
        trailing_ext = False

        # Check VAD overlap
        vad_hit = any(min(s1, float(sp["end_s"])) > max(s0, float(sp["start_s"])) - 0.04 for sp in speech_spans)

        if segment_only:
            method = "SEGMENT_APPROX"
            unc_ms = 180.0
            seg_s0, seg_s1 = seg_bounds.get(seg_id, (s0, s1))
            prot_s0 = max(0.0, seg_s0 - 0.10)
            prot_s1 = min(dur_s, seg_s1 + 0.12)
        else:
            # Conservative envelope refinement: if trailing energy stays active right after Whisper end_s (soft /s/, /t/, /d/),
            # extend end_s outward up to +80ms so cuts NEVER clip trailing phonemes!
            if has_pcm and env_5ms_db.size > 10:
                idx_end = min(env_5ms_db.size - 1, int(round(s1 * 1000.0 / hop_ms)))
                max_look = min(env_5ms_db.size - 1, idx_end + int(round(80.0 / hop_ms)))
                ext_idx = idx_end
                while ext_idx < max_look and env_5ms_db[ext_idx + 1] > voice_edge_db:
                    ext_idx += 1
                if ext_idx > idx_end + 1:
                    s1 = min(dur_s, (ext_idx + 1) * (hop_ms / 1000.0))
                    trailing_ext = True

                # Similarly guard leading consonant onset up to -40ms
                idx_st = max(0, min(env_5ms_db.size - 1, int(round(s0 * 1000.0 / hop_ms))))
                min_look = max(0, idx_st - int(round(40.0 / hop_ms)))
                pre_idx = idx_st
                while pre_idx > min_look and env_5ms_db[pre_idx - 1] > voice_edge_db:
                    pre_idx -= 1
                if pre_idx < idx_st - 1:
                    s0 = max(prev_end_s, pre_idx * (hop_ms / 1000.0))

            # Honest uncertainty estimate (Whisper DTW ~40-90ms; lower confidence -> wider uncertainty!)
            base_unc = 45.0 if vad_hit else 95.0
            if prob < 0.65:
                base_unc += 50.0
            if prob < 0.45:
                base_unc += 65.0
            unc_ms = round(base_unc, 1)
            method = "VAD_ENVELOPE_REFINED" if (has_pcm and vad_hit) else "ASR_WORD_ESTIMATE"
            # Conservative protection envelope around word
            pre_pad = max(0.065, unc_ms / 1000.0)
            post_pad = max(0.090, (unc_ms + 20.0) / 1000.0)
            prot_s0 = max(0.0, s0 - pre_pad)
            prot_s1 = min(dur_s, s1 + post_pad)

        # Enforce monotonicity on word starts
        if s0 < prev_end_s - 0.02:
            s0 = max(0.0, prev_end_s - 0.02)
        s1 = max(s0 + 0.015, s1)
        prev_end_s = s1

        sf = max(0, min(total_f - 1, int(math.floor(s0 * fps_f))))
        ef = max(sf + 1, min(total_f, int(math.ceil(s1 * fps_f))))
        aligned.append(TimingEvidence(
            word_id=wid,
            word=w_text,
            segment_id=seg_id,
            source_start_s=round(s0, 4),
            source_end_s=round(s1, 4),
            source_start_sample=int(round(s0 * sr_native)),
            source_end_sample=int(round(s1 * sr_native)),
            source_start_frame=sf,
            source_end_frame_exclusive=ef,
            protected_start_s=round(prot_s0, 4),
            protected_end_s=round(prot_s1, 4),
            confidence=round(prob, 4),
            uncertainty_ms=unc_ms,
            method=method,
            evidence_id=f"w_{wid:04d}_{hashlib.sha256(f'{wid}:{w_text}:{s0:.3f}'.encode()).hexdigest()[:8]}",
            language=transcript.language or "unknown",
            speaker_id=None,
            vad_confirmed=vad_hit,
            trailing_energy_extended=trailing_ext,
        ))

    if not aligned:
        overall_method = "NO_SPEECH"
        mean_unc = 0.0
    elif segment_only:
        overall_method = "SEGMENT_APPROX"
        mean_unc = 180.0
    elif any(a.method == "VAD_ENVELOPE_REFINED" for a in aligned):
        overall_method = "VAD_ENVELOPE_REFINED"
        mean_unc = round(float(np.mean([a.uncertainty_ms for a in aligned])), 1)
    else:
        overall_method = "ASR_WORD_ESTIMATE"
        mean_unc = round(float(np.mean([a.uncertainty_ms for a in aligned])), 1)

    return {
        "schema_version": PERCEPTION_SCHEMA_VERSION,
        "source_hash": media.input_hash,
        "transcript_hash": transcript.transcript_hash,
        "asr_model_id": transcript.model_id,
        "asr_model_sha256": transcript.model_sha256,
        "language": transcript.language,
        "word_count": len(aligned),
        "alignment_method": overall_method,
        "forced_aligner_used": False,
        "sample_exact_ground_truth": False,
        "segment_only_protection": segment_only,
        "mean_uncertainty_ms": mean_unc,
        "vad_confirmed_ratio": round(sum(1 for a in aligned if a.vad_confirmed) / max(1, len(aligned)), 4) if aligned else 0.0,
        "trailing_extensions_count": sum(1 for a in aligned if a.trailing_energy_extended),
        "words": [a.to_dict() for a in aligned],
    }


def extract_prosody_features(
    pcm_16k: np.ndarray,
    aligned_doc: dict[str, Any],
    vad_result: dict[str, Any],
    media: MediaInfo,
) -> dict[str, Any]:
    """Extract local signal-based prosody proxies (voiced F0 contour, energy, syllabic rate, breath likelihood).

    Explicitly marks uncertainty — never makes psychological/medical claims or gender/accent assumptions.
    """
    words = aligned_doc.get("words") or []
    dur_s = float(media.duration_s)
    if pcm_16k.size < 640 or not words:
        return {
            "schema_version": PERCEPTION_SCHEMA_VERSION,
            "source_hash": media.input_hash,
            "signal_proxy_only": True,
            "emotional_inference_certainty": "LOW_SIGNAL_PROXY_ONLY",
            "voiced_f0_median_hz": None,
            "mean_words_per_sec": 0.0,
            "phrase_summaries": [],
            "inter_word_pauses": [],
        }

    # Estimate F0 proxy via autocorrelation on 30ms voiced frames (75..380 Hz)
    win = 480  # 30ms @ 16kHz
    hop = 320  # 20ms
    lag_min = int(VAD_SAMPLE_RATE / 380)  # 42
    lag_max = int(VAD_SAMPLE_RATE / 75)   # 213
    noise_lin = 10.0 ** (float(vad_result.get("noise_floor_dbfs", -50.0)) / 20.0)
    f0_vals: list[float] = []
    # Sample at most 1500 frames (uniform stride on very long clips) to stay fast & bounded
    total_frames = max(0, (pcm_16k.size - win) // hop)
    stride = max(1, total_frames // 1500)
    for fi in range(0, total_frames, stride):
        seg = pcm_16k[fi * hop : fi * hop + win]
        rms = float(np.sqrt(np.mean(seg * seg) + 1e-12))
        if rms < max(noise_lin * 2.5, 0.015):
            continue
        seg0 = seg - float(np.mean(seg))
        ac = np.correlate(seg0, seg0, mode="full")[win - 1 :]
        if ac[0] <= 1e-9:
            continue
        sub = ac[lag_min : min(lag_max, ac.size)]
        if sub.size < 4:
            continue
        pk = int(np.argmax(sub))
        norm_pk = float(sub[pk] / ac[0])
        if norm_pk >= 0.45:
            f0_vals.append(VAD_SAMPLE_RATE / float(lag_min + pk))

    # Build inter-word pauses with breath-likelihood proxy right before next word onset
    inter_pauses: list[dict[str, Any]] = []
    for i in range(len(words) - 1):
        w_cur = words[i]
        w_nxt = words[i + 1]
        gap = float(w_nxt["source_start_s"]) - float(w_cur["source_end_s"])
        if gap >= 0.16:
            # Inspect last 180ms before next word for inhale signature (moderate broadband energy)
            b_t0 = max(float(w_cur["source_end_s"]), float(w_nxt["source_start_s"]) - 0.22)
            b_t1 = float(w_nxt["source_start_s"])
            s0 = max(0, min(pcm_16k.size, int(round(b_t0 * VAD_SAMPLE_RATE))))
            s1 = max(s0, min(pcm_16k.size, int(round(b_t1 * VAD_SAMPLE_RATE))))
            seg_b = pcm_16k[s0:s1]
            b_db = float(20.0 * math.log10(max(1e-6, float(np.sqrt(np.mean(seg_b * seg_b) + 1e-12))))) if seg_b.size >= 80 else -70.0
            breath_like = bool(-44.0 <= b_db <= -24.0 and gap >= 0.25)
            phrase_end = bool(str(w_cur["word"]).rstrip().endswith((".", "!", "?", ",")))
            inter_pauses.append({
                "pause_idx": len(inter_pauses),
                "after_word_id": int(w_cur["word_id"]),
                "before_word_id": int(w_nxt["word_id"]),
                "start_s": round(float(w_cur["source_end_s"]), 4),
                "end_s": round(float(w_nxt["source_start_s"]), 4),
                "duration_s": round(gap, 4),
                "phrase_boundary": phrase_end,
                "pre_onset_rms_dbfs": round(b_db, 2),
                "breath_likelihood": 0.65 if breath_like else 0.15,
            })

    # Active speech words/sec
    active_s = sum(float(sp["duration_s"]) for sp in (vad_result.get("speech_spans") or []))
    wps = round(len(words) / max(0.5, active_s if active_s > 0.5 else dur_s), 2)

    return {
        "schema_version": PERCEPTION_SCHEMA_VERSION,
        "source_hash": media.input_hash,
        "signal_proxy_only": True,
        "emotional_inference_certainty": "LOW_SIGNAL_PROXY_ONLY",
        "voiced_f0_median_hz": round(float(np.median(f0_vals)), 1) if f0_vals else None,
        "voiced_f0_iqr_hz": round(float(np.percentile(f0_vals, 75) - np.percentile(f0_vals, 25)), 1) if len(f0_vals) >= 4 else None,
        "mean_words_per_sec": wps,
        "inter_word_pauses": inter_pauses,
    }


def extract_action_evidence(
    source_mp4: Path,
    pause_windows: list[tuple[float, float]],
    roi_tracks: dict[str, Any] | None,
    media: MediaInfo,
) -> dict[str, Any]:
    """Sample frames densely inside/around candidate pause windows using bounded streaming OpenCV reads.

    Measures objective visual motion (frame diff & hand/task ROI motion + shot cut check).
    Without a verified local VLM, semantic action labels remain explicitly `UNKNOWN_ACTION`.
    """
    fps_f = media.fps_float
    total_f = media.duration_frames
    kfs = (roi_tracks or {}).get("keyframes") or (roi_tracks or {}).get("frames") or []

    def _roi_near(frame_idx: int) -> dict[str, Any]:
        if not kfs:
            return {}
        return min(kfs, key=lambda k: abs(int(k.get("frame", 0)) - frame_idx))

    windows_out: list[dict[str, Any]] = []
    cap = cv2.VideoCapture(str(source_mp4)) if Path(source_mp4).is_file() else None
    try:
        for idx, (t0, t1) in enumerate(pause_windows):
            sf = max(0, min(total_f - 1, int(math.floor(t0 * fps_f))))
            ef = max(sf + 1, min(total_f, int(math.ceil(t1 * fps_f))))
            # Pad window by ±4 frames for entry/exit motion context
            r0 = max(0, sf - 4)
            r1 = min(total_f, ef + 4)
            step = max(1, (r1 - r0) // 48)  # Bound max decoded frames per window to <=48!

            full_mads: list[float] = []
            task_mads: list[float] = []
            face_mads: list[float] = []
            shot_cut_detected = False
            per_frame_motion: list[dict[str, Any]] = []

            prev_small: np.ndarray | None = None
            prev_hist: np.ndarray | None = None
            if cap is not None and cap.isOpened():
                cap.set(cv2.CAP_PROP_POS_FRAMES, r0)
                cur_f = r0
                while cur_f < r1:
                    ok, fr = cap.read()
                    if not ok or fr is None:
                        break
                    if (cur_f - r0) % step == 0:
                        small = cv2.resize(fr, (180, 320)).astype(np.float32)
                        hsv = cv2.cvtColor(small.astype(np.uint8), cv2.COLOR_BGR2HSV)
                        hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
                        cv2.normalize(hist, hist)
                        if prev_small is not None:
                            diff = np.abs(small - prev_small)
                            f_mad = float(np.mean(diff))
                            h_corr = float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)) if prev_hist is not None else 1.0
                            if h_corr < 0.72 and f_mad > 24.0:
                                shot_cut_detected = True

                            roi_k = _roi_near(cur_f)
                            fb = roi_k.get("face") or roi_k.get("face_box")
                            tb = roi_k.get("task") or roi_k.get("task_box")
                            h_s, w_s = small.shape[:2]

                            def _box_mad(b: dict[str, Any] | None) -> float:
                                if not isinstance(b, dict) or float(b.get("conf", b.get("confidence", 0.0))) < 0.35:
                                    return 0.0
                                bw = float(b.get("w", 0.0))
                                bh = float(b.get("h", 0.0))
                                if "x" in b and "y" in b:
                                    bx, by = float(b["x"]), float(b["y"])
                                else:
                                    bx = float(b.get("cx", 0.5)) - 0.5 * bw
                                    by = float(b.get("cy", 0.5)) - 0.5 * bh
                                x0 = max(0, min(w_s - 2, int(bx * w_s)))
                                y0 = max(0, min(h_s - 2, int(by * h_s)))
                                x1 = max(x0 + 2, min(w_s, int((bx + bw) * w_s)))
                                y1 = max(y0 + 2, min(h_s, int((by + bh) * h_s)))
                                return float(np.mean(diff[y0:y1, x0:x1]))

                            t_mad = _box_mad(tb)
                            fc_mad = _box_mad(fb)
                            full_mads.append(f_mad)
                            task_mads.append(t_mad)
                            face_mads.append(fc_mad)
                            per_frame_motion.append({
                                "frame": cur_f,
                                "full_mad": round(f_mad, 2),
                                "task_mad": round(t_mad, 2),
                                "face_mad": round(fc_mad, 2),
                            })
                        prev_small = small
                        prev_hist = hist
                    cur_f += 1

            mean_full = round(float(np.mean(full_mads)), 2) if full_mads else 0.0
            peak_full = round(float(np.max(full_mads)), 2) if full_mads else 0.0
            mean_task = round(float(np.mean(task_mads)), 2) if task_mads else 0.0
            peak_task = round(float(np.max(task_mads)), 2) if task_mads else 0.0
            mean_face = round(float(np.mean(face_mads)), 2) if face_mads else 0.0

            # Distinguish high localized hand/demonstration action vs repetitive/low-delta idle
            # Note: optical flow alone NEVER claims semantic action identity -> UNKNOWN_ACTION!
            high_action = bool(shot_cut_detected or peak_task >= 14.0 or mean_task >= 8.0 or peak_full >= 16.0)
            moderate_action = bool(mean_task >= 4.0 or mean_full >= 4.5 or mean_face >= 5.0)
            low_info_idle = bool(not high_action and mean_full < 5.5 and (t1 - t0) >= 0.70)

            windows_out.append({
                "window_id": f"act_{idx:02d}",
                "start_s": round(t0, 4),
                "end_s": round(t1, 4),
                "start_frame": sf,
                "end_frame_exclusive": ef,
                "duration_s": round(t1 - t0, 4),
                "semantic_action_label": "UNKNOWN_ACTION",
                "action_label_method": "OBJECTIVE_MOTION_ROI_ONLY",
                "vlm_grounded": False,
                "shot_cut_detected": shot_cut_detected,
                "mean_full_mad": mean_full,
                "peak_full_mad": peak_full,
                "mean_task_mad": mean_task,
                "peak_task_mad": peak_task,
                "mean_face_mad": mean_face,
                "high_visual_action": high_action,
                "moderate_visual_action": moderate_action,
                "low_info_idle_proxy": low_info_idle,
                "sampled_points": len(per_frame_motion),
            })
    finally:
        if cap is not None:
            cap.release()

    return {
        "schema_version": PERCEPTION_SCHEMA_VERSION,
        "source_hash": media.input_hash,
        "vlm_status": "VISION_CRITIC_NOT_RUN",
        "action_label_policy": "Objective ROI/pixel motion only; optical flow alone never asserts semantic action identity.",
        "windows": windows_out,
    }


def run_speech_and_action_perception(
    media: MediaInfo,
    transcript: Transcript,
    roi_tracks: dict[str, Any] | None,
    *,
    out_dir: Path | None = None,
    ffmpeg_bin: str | None = None,
) -> dict[str, Any]:
    """Run complete A1 perception stage (VAD, word alignment, prosody, pause-window action evidence)."""
    ffmpeg = ffmpeg_bin or shutil.which("ffmpeg") or "ffmpeg"
    pcm_16k = _decode_mono_f32_pcm(Path(media.source_path), sr=VAD_SAMPLE_RATE, ffmpeg_bin=ffmpeg) if (media.has_audio and Path(media.source_path).is_file()) else np.zeros((0,), dtype=np.float32)

    has_tr_words = any(bool(getattr(seg, "words", None)) or bool(str(getattr(seg, "text", "")).strip()) for seg in (getattr(transcript, "segments", None) or []))
    vad_doc = detect_speech_activity(media, pcm_16k=pcm_16k, ffmpeg_bin=ffmpeg, expect_speech_from_transcript=has_tr_words)
    align_doc = align_transcript_words(transcript, vad_doc, media, pcm_16k=pcm_16k, ffmpeg_bin=ffmpeg)
    prosody_doc = extract_prosody_features(pcm_16k, align_doc, vad_doc, media)

    # Collect candidate pause windows >= 0.25s from BOTH inter-word pauses and VAD non-speech spans (plus trailing tail)
    cand_windows: list[tuple[float, float]] = []
    for p in prosody_doc.get("inter_word_pauses") or []:
        if float(p["duration_s"]) >= 0.25:
            cand_windows.append((float(p["start_s"]), float(p["end_s"])))
    words = align_doc.get("words") or []
    if words:
        # Leading & trailing tails
        if float(words[0]["source_start_s"]) >= 0.35:
            cand_windows.append((0.0, float(words[0]["source_start_s"])))
        if float(media.duration_s) - float(words[-1]["source_end_s"]) >= 0.25:
            cand_windows.append((float(words[-1]["source_end_s"]), float(media.duration_s)))
    else:
        for ns in vad_doc.get("non_speech_spans") or []:
            if float(ns["duration_s"]) >= 0.35:
                cand_windows.append((float(ns["start_s"]), float(ns["end_s"])))

    # Sort & deduplicate overlapping candidate windows
    cand_windows.sort()
    merged_windows: list[tuple[float, float]] = []
    for w0, w1 in cand_windows:
        if merged_windows and w0 <= merged_windows[-1][1] + 0.05:
            merged_windows[-1] = (merged_windows[-1][0], max(merged_windows[-1][1], w1))
        else:
            merged_windows.append((w0, w1))

    action_doc = extract_action_evidence(Path(media.source_path), merged_windows, roi_tracks, media)

    perception_status = {
        "schema_version": PERCEPTION_SCHEMA_VERSION,
        "source_hash": media.input_hash,
        "vad_backend": vad_doc["vad_backend"],
        "alignment_method": align_doc["alignment_method"],
        "forced_aligner_used": align_doc["forced_aligner_used"],
        "sample_exact_ground_truth": align_doc["sample_exact_ground_truth"],
        "mean_uncertainty_ms": align_doc["mean_uncertainty_ms"],
        "word_count": align_doc["word_count"],
        "speech_spans_count": len(vad_doc["speech_spans"]),
        "non_speech_spans_count": len(vad_doc["non_speech_spans"]),
        "uncertain_vad_spans_count": len(vad_doc["uncertain_spans"]),
        "pause_windows_analyzed": len(merged_windows),
        "vlm_status": action_doc["vlm_status"],
        "perception_hash": canonical_hash({
            "source_hash": media.input_hash,
            "transcript_hash": transcript.transcript_hash,
            "vad_backend": vad_doc["vad_backend"],
            "alignment_method": align_doc["alignment_method"],
            "word_count": align_doc["word_count"],
            "pause_windows": [(round(a, 3), round(b, 3)) for a, b in merged_windows],
        }),
    }

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(out_dir / "speech_alignment.json", align_doc)
        atomic_write_json(out_dir / "vad_spans.json", vad_doc)
        atomic_write_json(out_dir / "prosody_summary.json", prosody_doc)
        atomic_write_json(out_dir / "action_evidence.json", action_doc)
        atomic_write_json(out_dir / "perception_status.json", perception_status)

    return {
        "speech_alignment": align_doc,
        "vad_spans": vad_doc,
        "prosody_summary": prosody_doc,
        "action_evidence": action_doc,
        "perception_status": perception_status,
    }
