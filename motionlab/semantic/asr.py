"""Phase 1 (ML-01.R4): Local Offline ASR Transcription (faster-whisper MIT) + Semantic Unit Segmentation.

Strictly offline: reads SHA-256-verified local weights in `models/faster-whisper-small/`, decodes source audio
read-only via FFmpeg into float32 PCM (never modifying source master), enforces HF_HUB_OFFLINE=1 & local_files_only=True,
and maps word/segment timestamps to video frames and audio sample clocks.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from motionlab.config import MotionLabConfig, REPO_ROOT
from motionlab.schemas import AnalysisResult, MediaInfo, atomic_write_json, atomic_write_text, canonical_hash, ensure_contained_path
from motionlab.semantic.schema import (
    SEMANTIC_SCHEMA_VERSION,
    SemanticUnit,
    Transcript,
    TranscriptSegment,
    WordSpan,
)

MODELS_DIR = REPO_ROOT / "models"
DEFAULT_ASR_MODEL_ID = "faster_whisper_small"
UNCERTAIN_WORD_PROB = 0.55

_NUM_PATTERN = re.compile(r"(?i)(?:[\$€£¥]\s*)?\d{1,3}(?:,\d{3})*(?:\.\d+)?(?:\s*%|\s*k|\s*m)?")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect_asr_model(models_dir: Path | None = None, model_id: str = DEFAULT_ASR_MODEL_ID) -> dict[str, Any]:
    """Verify local faster-whisper runtime and SHA-256-pinned model files with zero network access."""
    mdir = (models_dir or MODELS_DIR).resolve()
    manifest_p = mdir / "manifest.json"
    info: dict[str, Any] = {
        "model_id": model_id,
        "available": False,
        "faster_whisper_available": False,
        "faster_whisper_version": None,
        "model_dir": None,
        "license": "MIT",
        "sha256_expected": None,
        "sha256_actual": None,
        "sha256_verified": False,
        "size_bytes": 0,
        "expected_ram_mb": 650,
        "reason": "",
    }
    try:
        import faster_whisper  # noqa: PLC0415

        info["faster_whisper_available"] = True
        info["faster_whisper_version"] = str(getattr(faster_whisper, "__version__", "unknown"))
    except Exception as exc:
        info["reason"] = f"faster_whisper_unavailable:{exc}"
        return info

    if not manifest_p.is_file():
        info["reason"] = f"models_manifest_missing:{manifest_p}"
        return info

    try:
        manifest = json.loads(manifest_p.read_text(encoding="utf-8"))
        entry = (manifest.get("models") or {}).get(model_id)
        if not isinstance(entry, dict):
            info["reason"] = f"asr_model_id_not_in_manifest:{model_id}"
            return info
        info["license"] = str(entry.get("license", "MIT"))
        info["sha256_expected"] = str(entry.get("sha256", ""))
        info["expected_ram_mb"] = int(entry.get("expected_ram_mb", 650))
        sub_dir = ensure_contained_path(mdir / str(entry.get("directory", "faster-whisper-small")), mdir)
        info["model_dir"] = str(sub_dir)
        files_map: dict[str, str] = dict(entry.get("files") or {"model.bin": info["sha256_expected"]})
        total_bytes = 0
        for fname, exp_sha in files_map.items():
            fp = ensure_contained_path(sub_dir / fname, sub_dir)
            if not fp.is_file():
                info["reason"] = f"asr_model_file_missing:{fp} (run bootstrap --with-semantic)"
                return info
            total_bytes += int(fp.stat().st_size)
            act = _sha256_file(fp)
            if fname == "model.bin":
                info["sha256_actual"] = act
            if exp_sha and act != exp_sha:
                info["reason"] = f"asr_sha256_mismatch:{fname}:expected={exp_sha[:12]},actual={act[:12]}"
                return info
        info["size_bytes"] = total_bytes
        info["sha256_verified"] = True
        info["available"] = True
        info["reason"] = "ok"
        return info
    except Exception as exc:
        info["reason"] = f"asr_manifest_error:{exc}"
        return info


def _decode_readonly_pcm_16k(media_path: Path, ffmpeg_bin: str) -> tuple[np.ndarray, str]:
    """Decode primary audio track read-only to 16 kHz mono float32 PCM via FFmpeg (zero source mutation)."""
    proc = subprocess.run(
        [
            ffmpeg_bin, "-v", "error", "-nostdin",
            "-i", str(media_path),
            "-vn", "-map", "0:a:0",
            "-ac", "1", "-ar", "16000",
            "-f", "f32le", "pipe:1",
        ],
        capture_output=True,
        timeout=90,
        shell=False,
    )
    if proc.returncode != 0 or not proc.stdout:
        return np.zeros(0, dtype=np.float32), ""
    audio_hash = hashlib.sha256(proc.stdout).hexdigest()
    pcm = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    return pcm, audio_hash


def _merge_split_number_tokens(words: list[WordSpan]) -> list[WordSpan]:
    """Recombine Whisper BPE-split currency/comma numbers like ['$1', ',200'] -> ['$1,200'] cleanly."""
    merged: list[WordSpan] = []
    i = 0
    while i < len(words):
        cur = words[i]
        if i + 1 < len(words):
            nxt = words[i + 1]
            if (
                (nxt.word.startswith(",") or nxt.word.startswith("."))
                and any(c.isdigit() for c in cur.word)
                and any(c.isdigit() for c in nxt.word)
            ):
                p = min(cur.probability, nxt.probability)
                merged.append(WordSpan(
                    word=f"{cur.word}{nxt.word}",
                    start_s=cur.start_s,
                    end_s=nxt.end_s,
                    probability=round(p, 4),
                    uncertain=(p < UNCERTAIN_WORD_PROB),
                ))
                i += 2
                continue
            if cur.word in ("$", "€", "£", "¥") and any(c.isdigit() for c in nxt.word):
                p = min(cur.probability, nxt.probability)
                merged.append(WordSpan(
                    word=f"{cur.word}{nxt.word}",
                    start_s=cur.start_s,
                    end_s=nxt.end_s,
                    probability=round(p, 4),
                    uncertain=(p < UNCERTAIN_WORD_PROB),
                ))
                i += 2
                continue
        merged.append(cur)
        i += 1
    return merged


def format_readable_transcript(transcript: Transcript) -> str:
    """Emit timestamped human-auditable transcript text highlighting [UNCERTAIN] words honestly."""
    lines = [
        f"LANGUAGE: {transcript.language} (prob={transcript.language_probability:.3f}) | STATUS: {transcript.status}",
        f"MODEL: {transcript.model_id} (sha256={transcript.model_sha256[:16]}) | WORD_TIMESTAMPS_EXACT: {transcript.word_timestamps_exact}",
        f"PTS_PRIMING_OFFSET_S: {transcript.pts_priming_offset_s:.5f} | PTS_ANOMALIES: {len(transcript.pts_anomalies)}",
        "-" * 76,
    ]
    if not transcript.segments:
        lines.append("[NO SPEECH DETECTED]")
        return "\n".join(lines) + "\n"
    for seg in transcript.segments:
        toks = [f"{w.word}[UNCERTAIN:{w.probability:.2f}]" if w.uncertain else w.word for w in seg.words]
        body = " ".join(toks) if toks else seg.text
        lines.append(f"[{seg.start_s:05.2f}s -> {seg.end_s:05.2f}s] (conf={seg.mean_word_prob:.2f}) {body}")
    return "\n".join(lines) + "\n"


def transcribe_local(
    media: MediaInfo,
    config: MotionLabConfig,
    cache_dir: Path | None = None,
    ffmpeg_bin: str | None = None,
    models_dir: Path | None = None,
) -> Transcript:
    """Run strictly offline local Whisper ASR on source audio, preserving source master untouched."""
    t0 = time.perf_counter()
    ffmpeg = ffmpeg_bin or shutil.which("ffmpeg") or "ffmpeg"
    audio_pts = [s for s in media.pts_summary if s.get("codec_type") == "audio"]
    priming_s = float(audio_pts[0].get("priming_offset_s", 0.0)) if audio_pts else 0.0
    anomalies = list(audio_pts[0].get("anomalies", [])) if audio_pts else []

    if not media.has_audio:
        tr = Transcript(
            schema_version=SEMANTIC_SCHEMA_VERSION,
            source_hash=media.input_hash,
            source_audio_hash="",
            language="none",
            language_probability=0.0,
            duration_s=round(media.duration_s, 4),
            sample_rate_asr=16000,
            model_id="none",
            model_sha256="",
            asr_params={},
            word_timestamps_exact=False,
            pts_priming_offset_s=0.0,
            pts_anomalies=[],
            status="NO_SPEECH",
            full_text="",
            segments=[],
            warnings=["Source media has no audio stream."],
            elapsed_s=round(time.perf_counter() - t0, 3),
        )
        tr.transcript_hash = tr.compute_hash()
        return tr

    asr_info = inspect_asr_model(models_dir=models_dir)
    if not asr_info["available"]:
        tr = Transcript(
            schema_version=SEMANTIC_SCHEMA_VERSION,
            source_hash=media.input_hash,
            source_audio_hash="",
            language="unknown",
            language_probability=0.0,
            duration_s=round(media.duration_s, 4),
            sample_rate_asr=16000,
            model_id=DEFAULT_ASR_MODEL_ID,
            model_sha256="",
            asr_params={},
            word_timestamps_exact=False,
            pts_priming_offset_s=priming_s,
            pts_anomalies=anomalies,
            status="BLOCKED_NO_MODEL",
            full_text="",
            segments=[],
            warnings=[f"Local ASR unavailable: {asr_info['reason']}"],
            elapsed_s=round(time.perf_counter() - t0, 3),
        )
        tr.transcript_hash = tr.compute_hash()
        return tr

    pcm, audio_hash = _decode_readonly_pcm_16k(Path(media.path), ffmpeg)
    if pcm.size < 1600 or float(np.sqrt(np.mean(pcm * pcm))) < 1e-4:
        tr = Transcript(
            schema_version=SEMANTIC_SCHEMA_VERSION,
            source_hash=media.input_hash,
            source_audio_hash=audio_hash,
            language="none",
            language_probability=0.0,
            duration_s=round(media.duration_s, 4),
            sample_rate_asr=16000,
            model_id=DEFAULT_ASR_MODEL_ID,
            model_sha256=str(asr_info["sha256_actual"]),
            asr_params={"beam_size": 5, "vad_filter": True, "compute_type": "int8"},
            word_timestamps_exact=False,
            pts_priming_offset_s=priming_s,
            pts_anomalies=anomalies,
            status="NO_SPEECH",
            full_text="",
            segments=[],
            warnings=["Decoded audio is silent (RMS < 1e-4); zero speech recognized."],
            elapsed_s=round(time.perf_counter() - t0, 3),
        )
        tr.transcript_hash = tr.compute_hash()
        return tr

    asr_params = {
        "beam_size": 5,
        "word_timestamps": True,
        "vad_filter": True,
        "compute_type": "int8",
        "temperature": 0.0,
    }
    cache_key = canonical_hash({
        "audio_hash": audio_hash,
        "model_sha256": asr_info["sha256_actual"],
        "asr_params": asr_params,
        "schema_version": SEMANTIC_SCHEMA_VERSION,
    })
    cache_file = (cache_dir / f"asr_{cache_key}.json") if cache_dir else None
    if config.cache_enabled and cache_file and cache_file.is_file():
        try:
            cached_tr = Transcript.from_dict(json.loads(cache_file.read_text(encoding="utf-8")))
            cached_tr.cache_hit = True
            cached_tr.elapsed_s = round(time.perf_counter() - t0, 3)
            return cached_tr
        except Exception:
            pass

    # Strictly enforce offline env before importing/invoking faster_whisper
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from faster_whisper import WhisperModel  # noqa: PLC0415

    model = WhisperModel(
        str(asr_info["model_dir"]),
        device="cpu",
        compute_type="int8",
        cpu_threads=min(4, os.cpu_count() or 4),
        local_files_only=True,
    )
    raw_segs, info = model.transcribe(
        pcm,
        beam_size=5,
        temperature=0.0,
        word_timestamps=True,
        vad_filter=True,
        condition_on_previous_text=True,
    )

    segments: list[TranscriptSegment] = []
    uncertain_spans: list[dict[str, Any]] = []
    dur_limit = float(media.duration_s)
    for idx, s in enumerate(raw_segs):
        txt = (s.text or "").strip()
        if not txt:
            continue
        seg_start = max(0.0, min(dur_limit, float(s.start)))
        seg_end = max(seg_start, min(dur_limit, float(s.end)))
        raw_words: list[WordSpan] = []
        for w in (s.words or []):
            wt = (w.word or "").strip()
            if not wt:
                continue
            ws = max(0.0, min(dur_limit, float(w.start)))
            we = max(ws, min(dur_limit, float(w.end)))
            wp = max(0.0, min(1.0, float(w.probability)))
            raw_words.append(WordSpan(word=wt, start_s=round(ws, 3), end_s=round(we, 3), probability=round(wp, 4), uncertain=(wp < UNCERTAIN_WORD_PROB)))
        merged_words = _merge_split_number_tokens(raw_words)
        mean_p = float(np.mean([w.probability for w in merged_words])) if merged_words else 0.0
        for mw in merged_words:
            if mw.uncertain:
                uncertain_spans.append({"segment_id": f"seg_{idx:02d}", "word": mw.word, "start_s": mw.start_s, "end_s": mw.end_s, "probability": mw.probability})
        segments.append(TranscriptSegment(
            segment_id=f"seg_{idx:02d}",
            start_s=round(seg_start, 3),
            end_s=round(seg_end, 3),
            text=txt,
            avg_logprob=round(float(getattr(s, "avg_logprob", 0.0)), 4),
            mean_word_prob=round(mean_p, 4),
            words=merged_words,
        ))

    full_text = " ".join(seg.text for seg in segments).strip()
    mean_all = float(np.mean([s.mean_word_prob for s in segments])) if segments else 0.0
    status = "NO_SPEECH" if not segments else ("LOW_CONFIDENCE" if mean_all < 0.50 else "OK")
    warnings: list[str] = []
    if anomalies:
        warnings.append(f"Source audio contains {len(anomalies)} documented PTS priming/concat boundary anomalies; timestamps preserved relative to source PTS.")
    if uncertain_spans:
        warnings.append(f"{len(uncertain_spans)} word(s) fell below confidence threshold {UNCERTAIN_WORD_PROB} and were flagged [UNCERTAIN].")

    tr = Transcript(
        schema_version=SEMANTIC_SCHEMA_VERSION,
        source_hash=media.input_hash,
        source_audio_hash=audio_hash,
        language=str(getattr(info, "language", "unknown")),
        language_probability=round(float(getattr(info, "language_probability", 0.0)), 4),
        duration_s=round(media.duration_s, 4),
        sample_rate_asr=16000,
        model_id=DEFAULT_ASR_MODEL_ID,
        model_sha256=str(asr_info["sha256_actual"]),
        asr_params=asr_params,
        # Honest disclosure: Whisper attention-based word alignment is approximate (~40-120ms), not sample-exact
        word_timestamps_exact=False,
        pts_priming_offset_s=round(priming_s, 6),
        pts_anomalies=anomalies,
        status=status,
        full_text=full_text,
        segments=segments,
        uncertain_spans=uncertain_spans,
        warnings=warnings,
        elapsed_s=round(time.perf_counter() - t0, 3),
        cache_hit=False,
    )
    tr.transcript_hash = tr.compute_hash()
    if config.cache_enabled and cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(cache_file, tr.to_dict())
    return tr


def _extract_spoken_numbers(words: list[WordSpan], text: str) -> list[dict[str, Any]]:
    """Extract numeric/currency tokens strictly from ASR words with timestamp & confidence provenance."""
    found: list[dict[str, Any]] = []
    for w in words:
        for m in _NUM_PATTERN.finditer(w.word):
            found.append({
                "raw": m.group(0),
                "word": w.word,
                "start_s": w.start_s,
                "end_s": w.end_s,
                "confidence": w.probability,
                "uncertain": w.uncertain,
            })
    if not found and text:
        for m in _NUM_PATTERN.finditer(text):
            found.append({"raw": m.group(0), "word": m.group(0), "start_s": 0.0, "end_s": 0.0, "confidence": 0.5, "uncertain": True})
    return found


def build_semantic_units(
    transcript: Transcript,
    analysis: AnalysisResult,
    roi_tracks: dict[str, Any],
    media: MediaInfo,
) -> list[SemanticUnit]:
    """Segment local ASR transcript into meaningful sentence/clause semantic units aligned to frames & samples."""
    if not transcript.segments:
        return []

    fps_frac = media.fps.as_fraction()
    fps_f = float(fps_frac)
    total_f = media.duration_frames
    sr_out = media.audio_streams[0].sample_rate if media.audio_streams else 44100

    # Flatten all words across segments with parent segment_id
    flat_words: list[tuple[str, WordSpan]] = []
    for seg in transcript.segments:
        if seg.words:
            for w in seg.words:
                flat_words.append((seg.segment_id, w))
        else:
            flat_words.append((seg.segment_id, WordSpan(seg.text, seg.start_s, seg.end_s, seg.mean_word_prob, seg.mean_word_prob < UNCERTAIN_WORD_PROB)))

    # Group words into sentence/clause units split on terminal punctuation OR pause >= 0.60s
    grouped: list[list[tuple[str, WordSpan]]] = []
    cur: list[tuple[str, WordSpan]] = []
    for seg_id, w in flat_words:
        if cur:
            prev_w = cur[-1][1]
            gap_s = w.start_s - prev_w.end_s
            prev_ends_sent = prev_w.word.endswith((".", "!", "?"))
            # Split cleanly on sentence boundary or clear phrase pause (keep >=3 words unless terminal sentence)
            if (prev_ends_sent and len(cur) >= 2) or (gap_s >= 0.70 and len(cur) >= 3):
                grouped.append(cur)
                cur = []
        cur.append((seg_id, w))
    if cur:
        grouped.append(cur)

    # Compute local motion energy per frame from analysis motion_windows / action_windows
    motion_by_f: dict[int, float] = {}
    for mw in (analysis.motion_windows or []):
        val = float(getattr(mw, "mean_motion", getattr(mw, "mean_magnitude", getattr(mw, "motion_mean", 0.0))))
        for ff in range(int(mw.start_frame), int(mw.end_frame_exclusive)):
            motion_by_f[ff] = max(motion_by_f.get(ff, 0.0), val)
    for aw in (analysis.action_windows or []):
        val = float(getattr(aw, "motion_mean", 0.0))
        for ff in range(int(aw.start_frame), int(aw.end_frame_exclusive)):
            motion_by_f[ff] = max(motion_by_f.get(ff, 0.0), val)
    action_vecs = roi_tracks.get("action_vectors") or []

    units: list[SemanticUnit] = []
    for u_idx, grp in enumerate(grouped):
        seg_ids = sorted({sid for sid, _ in grp})
        w_list = [w for _, w in grp]
        st_s = max(0.0, float(w_list[0].start_s))
        en_s = min(media.duration_s, max(st_s + 0.25, float(w_list[-1].end_s)))
        st_f = max(0, min(total_f - 1, int(round(st_s * fps_f))))
        en_f = max(st_f + 1, min(total_f, int(round(en_s * fps_f))))
        st_samp = int(round(Fraction(st_f, 1) / fps_frac * sr_out))
        en_samp = int(round(Fraction(en_f, 1) / fps_frac * sr_out))
        quote = " ".join(w.word for w in w_list).strip()
        mean_conf = float(np.mean([w.probability for w in w_list])) if w_list else 0.0
        has_unc = any(w.uncertain for w in w_list)
        nums = _extract_spoken_numbers(w_list, quote)

        # Measure real visual action/motion energy during this unit
        mot_vals = [v for f, v in motion_by_f.items() if st_f <= f < en_f]
        obs_mot = round(float(np.mean(mot_vals)), 4) if mot_vals else 0.0
        ev_refs = list(seg_ids)
        if any(v["start_frame"] < en_f and v["end_frame_exclusive"] > st_f for v in action_vecs):
            ev_refs.append("tracked_action_roi")

        units.append(SemanticUnit(
            unit_id=f"u_{u_idx:02d}",
            start_s=round(st_s, 3),
            end_s=round(en_s, 3),
            start_sample=st_samp,
            end_sample=en_samp,
            start_frame=st_f,
            end_frame_exclusive=en_f,
            transcript_span_ids=seg_ids,
            verbatim_quote=quote,
            mean_confidence=round(mean_conf, 4),
            has_uncertain_words=has_unc,
            spoken_numbers=nums,
            observed_motion_energy=obs_mot,
            evidence_refs=ev_refs,
        ))
    return units
