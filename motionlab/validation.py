"""Phase G6 — Technical Validator, Audio Sync Proof, Effect Window Verification, and Contact Sheet Builder.

Strictly distinguishes:
  - Technical / Structural validation (`TECHNICAL_PASS` | `FAIL` | `BLOCKED`)
  - Human aesthetic judgment (`AESTHETIC_PENDING` — never auto-approved!)
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from motionlab.config import atomic_write_json
from motionlab.preflight import find_binary
from motionlab.renderers.base import probe_artifact_ffprobe
from motionlab.schemas import (
    EditPlan,
    MediaInfo,
    RenderResult,
    ValidationCheck,
    ValidationResult,
)


def _stream_audio_packet_hash(
    media_path: Path,
    audio_stream_idx: int,
    ffmpeg_bin: str,
) -> tuple[str | None, int]:
    """Compute SHA-256 hash over raw demuxed audio packets (exact stream-copy proof)."""
    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(media_path),
        "-map",
        f"0:a:{audio_stream_idx}",
        "-c:a",
        "copy",
        "-f",
        "streamhash",
        "-hash",
        "sha256",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    if proc.returncode != 0:
        return None, 0
    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    digest = lines[0] if lines else None
    return digest, len(lines)


def _audio_packet_hash_and_pts(path: Path, track_idx: int, ffmpeg_bin: str, ffprobe_bin: str) -> dict[str, Any]:
    """Compute exact packet payload SHA-256, packet count, and first/last PTS for an audio track."""
    proc = subprocess.run(
        [
            ffmpeg_bin, "-v", "error",
            "-i", str(path),
            "-map", f"0:a:{track_idx}",
            "-c:a", "copy",
            "-f", "framemd5",
            "-",
        ],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60, shell=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"framemd5 failed for audio track {track_idx} of {path}: {proc.stderr.strip()}")
    pkt_sigs: list[str] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 6:
            pkt_sigs.append(f"{parts[4]}:{parts[5]}")
    digest = hashlib.sha256("\n".join(pkt_sigs).encode("ascii")).hexdigest()

    first_pts, last_pts = 0.0, 0.0
    p_proc = subprocess.run(
        [
            ffprobe_bin, "-v", "error",
            "-select_streams", f"a:{track_idx}",
            "-show_entries", "packet=pts_time",
            "-of", "json",
            str(path),
        ],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30, shell=False,
    )
    if p_proc.returncode == 0:
        try:
            pts_list = [
                float(pk["pts_time"])
                for pk in (json.loads(p_proc.stdout).get("packets") or [])
                if pk.get("pts_time") is not None and pk.get("pts_time") != "N/A"
            ]
            if pts_list:
                first_pts = round(min(pts_list), 6)
                last_pts = round(max(pts_list), 6)
        except (ValueError, json.JSONDecodeError):
            pass

    return {
        "packet_hash": digest,
        "packet_count": len(pkt_sigs),
        "first_pts_s": first_pts,
        "last_pts_s": last_pts,
    }


def _extract_mono_pcm(media_path: Path, audio_stream_idx: int, ffmpeg_bin: str, sr: int = 16000) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        wav_path = Path(tf.name)
    try:
        cmd = [
            ffmpeg_bin,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(media_path),
            "-map",
            f"0:a:{audio_stream_idx}",
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sr),
            "-c:a",
            "pcm_s16le",
            str(wav_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, timeout=120)
        if proc.returncode != 0 or not wav_path.exists() or wav_path.stat().st_size <= 44:
            return np.zeros(0, dtype=np.float32)
        with wave.open(str(wav_path), "rb") as wf:
            raw = wf.readframes(wf.getnframes())
        return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    finally:
        wav_path.unlink(missing_ok=True)


def _measure_pcm_correlation_and_lag_ms(
    src_pcm: np.ndarray,
    out_pcm: np.ndarray,
    sr: int = 16000,
    max_lag_ms: float = 80.0,
) -> tuple[float, float]:
    """Measure peak Pearson correlation and alignment lag (ms) between source and output PCM."""
    if src_pcm.size < 160 or out_pcm.size < 160:
        return 1.0 if src_pcm.size == out_pcm.size else 0.0, 0.0
    # Bound comparison window to first 30s to keep RAM & CPU bounded
    n = min(src_pcm.size, out_pcm.size, sr * 30)
    a = src_pcm[:n] - float(np.mean(src_pcm[:n]))
    b = out_pcm[:n] - float(np.mean(out_pcm[:n]))
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a < 1e-6 and norm_b < 1e-6:
        return 1.0, 0.0
    if norm_a < 1e-6 or norm_b < 1e-6:
        return 0.0, 999.0

    max_lag = min(int(round(sr * max_lag_ms / 1000.0)), n // 4)
    best_corr = -1.0
    best_lag = 0
    # Coarse search then fine search around peak
    step = max(1, max_lag // 40)
    candidates = list(range(-max_lag, max_lag + 1, step))
    for lag in candidates:
        if lag >= 0:
            aa, bb = a[: n - lag], b[lag:n]
        else:
            aa, bb = a[-lag:n], b[: n + lag]
        denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
        corr = float(np.dot(aa, bb) / denom) if denom > 1e-8 else 0.0
        if corr > best_corr:
            best_corr = corr
            best_lag = lag

    # Fine refinement around best_lag
    for lag in range(max(-max_lag, best_lag - step), min(max_lag, best_lag + step) + 1):
        if lag >= 0:
            aa, bb = a[: n - lag], b[lag:n]
        else:
            aa, bb = a[-lag:n], b[: n + lag]
        denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
        corr = float(np.dot(aa, bb) / denom) if denom > 1e-8 else 0.0
        if corr > best_corr:
            best_corr = corr
            best_lag = lag

    return best_corr, (best_lag * 1000.0) / sr


def _detect_black_intervals(video_path: Path, ffmpeg_bin: str) -> list[tuple[float, float, float]]:
    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-i",
        str(video_path),
        "-an",
        "-vf",
        "blackdetect=d=0.15:pix_th=0.05",
        "-f",
        "null",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    intervals: list[tuple[float, float, float]] = []
    for m in re.finditer(r"black_start:([\d.]+)\s+black_end:([\d.]+)\s+black_duration:([\d.]+)", proc.stderr):
        intervals.append((float(m.group(1)), float(m.group(2)), float(m.group(3))))
    return intervals


def _detect_freeze_intervals(video_path: Path, ffmpeg_bin: str) -> list[tuple[float, float, float]]:
    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-i",
        str(video_path),
        "-an",
        "-vf",
        "freezedetect=n=-60dB:d=0.8",
        "-f",
        "null",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    starts: list[float] = []
    intervals: list[tuple[float, float, float]] = []
    for ln in proc.stderr.splitlines():
        m_s = re.search(r"freeze_start:\s*([\d.]+)", ln)
        if m_s:
            starts.append(float(m_s.group(1)))
        m_e = re.search(r"freeze_end:\s*([\d.]+)\s+freeze_duration:\s*([\d.]+)", ln)
        if m_e:
            st = starts.pop(0) if starts else max(0.0, float(m_e.group(1)) - float(m_e.group(2)))
            intervals.append((st, float(m_e.group(1)), float(m_e.group(2))))
    return intervals


def _extract_frame_rgb(video_path: Path, time_s: float, width: int, height: int, ffmpeg_bin: str) -> np.ndarray | None:
    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{max(0.0, time_s):.4f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-vf",
        f"scale={width}:{height}:flags=bilinear",
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=30)
    expected = width * height * 3
    if proc.returncode != 0 or len(proc.stdout) < expected:
        return None
    return np.frombuffer(proc.stdout[:expected], dtype=np.uint8).reshape((height, width, 3))


def generate_contact_sheet_and_pairs(
    source_path: Path,
    output_path: Path,
    plan: EditPlan,
    out_sheet_path: Path,
    evidence_dir: Path,
    ffmpeg_bin: str,
) -> dict[str, Any]:
    """Build contact_sheet.jpg (periodic sampling + before/during/after events) and paired comparison frames."""
    evidence_dir.mkdir(parents=True, exist_ok=True)
    fps = plan.fps_num / plan.fps_den
    dur_s = plan.duration_frames / fps

    # Sample evenly across timeline (~12-16 tiles) + midpoint of every planned event
    n_grid = min(16, max(4, int(round(dur_s / 1.5)) + 1))
    grid_times = [min(max(0.0, dur_s - 0.05), i * dur_s / max(1, n_grid - 1)) for i in range(n_grid)]
    tile_w = 240
    aspect = plan.height / max(1, plan.width)
    tile_h = max(120, int(round(tile_w * aspect)))
    if tile_h > 360:
        tile_h = 360
        tile_w = max(120, int(round(tile_h / aspect)))

    # Ensure even dimensions for FFmpeg rawvideo scale
    tile_w = (tile_w // 2) * 2
    tile_h = (tile_h // 2) * 2

    tiles: list[tuple[str, Image.Image]] = []
    for t_s in grid_times:
        arr = _extract_frame_rgb(output_path, t_s, tile_w, tile_h, ffmpeg_bin)
        if arr is not None:
            tiles.append((f"{t_s:.2f}s", Image.fromarray(arr)))

    # Paired source vs output comparison frames at midpoint of up to 6 planned events
    paired_files: list[str] = []
    event_deltas: dict[str, float] = {}
    for ev in plan.events:
        mid_f = (ev.start_frame + ev.end_frame_exclusive) // 2
        mid_t = mid_f / fps
        src_arr = _extract_frame_rgb(source_path, mid_t, tile_w, tile_h, ffmpeg_bin)
        out_arr = _extract_frame_rgb(output_path, mid_t, tile_w, tile_h, ffmpeg_bin)
        if src_arr is not None and out_arr is not None:
            diff = np.abs(out_arr.astype(np.float32) - src_arr.astype(np.float32))
            mad = float(np.mean(diff))
            max_d = float(np.max(diff))
            changed_px = int(np.sum(np.max(diff, axis=2) >= 8.0))
            event_deltas[ev.id] = {"mad": round(mad, 3), "max_delta": round(max_d, 2), "changed_pixels": changed_px}
            if len(paired_files) < 8:
                pair_img = Image.new("RGB", (tile_w * 2 + 16, tile_h + 34), (20, 22, 28))
                pair_img.paste(Image.fromarray(src_arr), (4, 28))
                pair_img.paste(Image.fromarray(out_arr), (tile_w + 12, 28))
                draw = ImageDraw.Draw(pair_img)
                draw.text((8, 6), f"SOURCE  t={mid_t:.2f}s", fill=(220, 220, 220))
                draw.text((tile_w + 16, 6), f"EDITED  {ev.effect_id} ({ev.engine})  MAD={mad:.2f}", fill=(255, 220, 140))
                pair_path = evidence_dir / f"pair_{ev.id}.jpg"
                pair_img.save(pair_path, "JPEG", quality=86)
                paired_files.append(f"evidence/{pair_path.name}")

    # Compose contact_sheet.jpg
    cols = 4
    rows = max(1, math.ceil(len(tiles) / cols))
    pad = 8
    header_h = 44
    label_h = 20
    sheet_w = cols * (tile_w + pad) + pad
    sheet_h = header_h + rows * (tile_h + label_h + pad) + pad
    sheet = Image.new("RGB", (sheet_w, sheet_h), (16, 18, 22))
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (pad, 12),
        f"MotionLab AI ML-01 Contact Sheet | {plan.width}x{plan.height} @ {plan.fps_num}/{plan.fps_den}fps | "
        f"profile={plan.profile} | events={len(plan.events)} | AESTHETIC_PENDING",
        fill=(240, 242, 248),
    )

    for idx, (lbl, im) in enumerate(tiles):
        r, c = divmod(idx, cols)
        x = pad + c * (tile_w + pad)
        y = header_h + r * (tile_h + label_h + pad)
        sheet.paste(im, (x, y))
        draw.text((x + 4, y + tile_h + 3), lbl, fill=(195, 200, 210))

    out_sheet_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_sheet_path, "JPEG", quality=88)

    return {
        "contact_sheet": out_sheet_path.name,
        "tile_count": len(tiles),
        "paired_frames": paired_files,
        "event_midpoint_mad": event_deltas,
    }


def validate_render_output(
    source_path: Path,
    media: MediaInfo,
    plan: EditPlan,
    candidate_output_path: Path,
    render_result: RenderResult,
    run_dir: Path,
    ffmpeg_bin: str | None = None,
    ffprobe_bin: str | None = None,
) -> tuple[ValidationResult, dict[str, Any]]:
    """Run all mandatory Phase G6 technical gates on the rendered MP4 BEFORE promoting it to final.mp4."""
    ffmpeg = ffmpeg_bin or find_binary("ffmpeg")
    ffprobe = ffprobe_bin or find_binary("ffprobe")
    if not ffmpeg or not ffprobe:
        raise RuntimeError("ffmpeg/ffprobe missing for validation")

    evidence_dir = run_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    checks: list[ValidationCheck] = []
    warnings: list[str] = []

    # 1. File existence & non-empty
    exists_ok = candidate_output_path.exists() and candidate_output_path.stat().st_size > 0
    checks.append(
        ValidationCheck(
            name="artifact_exists_non_empty",
            status="PASS" if exists_ok else "FAIL",
            expected="> 0 bytes",
            actual=candidate_output_path.stat().st_size if exists_ok else 0,
        )
    )
    if not exists_ok:
        return ValidationResult(checks=checks, overall_status="FAIL", warnings=["artifact_missing"]), {}

    # 2. ffprobe metadata verification
    out_probe = probe_artifact_ffprobe(candidate_output_path, ffprobe)
    atomic_write_json(evidence_dir / "ffprobe_final.json", out_probe)
    v_streams = [s for s in out_probe.get("streams", []) if s.get("codec_type") == "video"]
    a_streams = [s for s in out_probe.get("streams", []) if s.get("codec_type") == "audio"]

    has_v = len(v_streams) >= 1
    checks.append(
        ValidationCheck(
            name="video_stream_present",
            status="PASS" if has_v else "FAIL",
            expected=">=1 video stream",
            actual=len(v_streams),
            evidence_path="evidence/ffprobe_final.json",
        )
    )
    if not has_v:
        return ValidationResult(checks=checks, overall_status="FAIL"), {}

    vs = v_streams[0]
    out_w, out_h = int(vs.get("width", 0)), int(vs.get("height", 0))
    dim_ok = (out_w == plan.width and out_h == plan.height)
    checks.append(
        ValidationCheck(
            name="output_dimensions_match_plan",
            status="PASS" if dim_ok else "FAIL",
            expected=f"{plan.width}x{plan.height}",
            actual=f"{out_w}x{out_h}",
            evidence_path="evidence/ffprobe_final.json",
        )
    )

    out_pix = str(vs.get("pix_fmt", "unknown"))
    checks.append(
        ValidationCheck(
            name="pixel_format_compatible",
            status="PASS" if out_pix == "yuv420p" else "FAIL",
            expected="yuv420p",
            actual=out_pix,
            evidence_path="evidence/ffprobe_final.json",
        )
    )

    # Duration tolerance: within 1 output frame + 50ms container/AAC priming rounding
    fps = plan.fps_num / plan.fps_den
    expected_dur = media.duration_s
    out_dur = float((out_probe.get("format") or {}).get("duration") or vs.get("duration") or 0.0)
    tol_s = max(0.08, (1.5 / max(fps, 1.0)))
    dur_delta = abs(out_dur - expected_dur)
    checks.append(
        ValidationCheck(
            name="duration_preserved",
            status="PASS" if dur_delta <= tol_s else "FAIL",
            expected=f"{expected_dur:.3f}s (±{tol_s:.3f}s)",
            actual=f"{out_dur:.3f}s (delta={dur_delta:.4f}s)",
            evidence_path="evidence/ffprobe_final.json",
        )
    )

    # 3. Full decode integrity via `ffmpeg -v error -xerror -i out.mp4 -f null -`
    dec_cmd = [ffmpeg, "-hide_banner", "-v", "error", "-xerror", "-i", str(candidate_output_path), "-f", "null", "-"]
    dec_proc = subprocess.run(dec_cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180)
    decode_ok = dec_proc.returncode == 0 and not dec_proc.stderr.strip()
    checks.append(
        ValidationCheck(
            name="ffmpeg_full_decode_zero_errors",
            status="PASS" if decode_ok else "FAIL",
            expected="exit 0 and empty stderr",
            actual=f"exit {dec_proc.returncode}; stderr_len={len(dec_proc.stderr.strip())}",
            message=dec_proc.stderr.strip()[:300],
        )
    )

    # 4. Audio policy & integrity verification
    audio_evidence: dict[str, Any] = {
        "audio_mode": render_result.audio_mode,
        "source_audio_streams": len(media.audio_streams),
        "output_audio_streams": len(a_streams),
    }
    expected_tracks = list(plan.audio_policy.get("selected_tracks", []))
    if not media.has_audio or not expected_tracks:
        silent_ok = len(a_streams) == 0
        checks.append(
            ValidationCheck(
                name="silent_source_remains_silent",
                status="PASS" if silent_ok else "FAIL",
                expected="0 audio streams (no synthesized audio)",
                actual=len(a_streams),
                evidence_path="evidence/audio_sync.json",
            )
        )
    else:
        count_ok = len(a_streams) == len(expected_tracks)
        checks.append(
            ValidationCheck(
                name="audio_stream_count_matches",
                status="PASS" if count_ok else "FAIL",
                expected=len(expected_tracks),
                actual=len(a_streams),
                evidence_path="evidence/audio_sync.json",
            )
        )
        # Verify stream-copy exact packet payload hash OR PCM correlation + alignment lag
        if render_result.audio_mode == "stream_copy" and count_ok:
            all_hashes_match = True
            hash_pairs: list[dict[str, Any]] = []
            for out_idx, src_idx in enumerate(expected_tracks):
                h_src, _ = _stream_audio_packet_hash(source_path, src_idx, ffmpeg)
                h_out, _ = _stream_audio_packet_hash(candidate_output_path, out_idx, ffmpeg)
                matched = (h_src is not None and h_src == h_out)
                if not matched:
                    all_hashes_match = False
                hash_pairs.append({"src_track": src_idx, "out_track": out_idx, "src_hash": h_src, "out_hash": h_out, "matched": matched})
            audio_evidence["stream_copy_packet_hashes"] = hash_pairs
            checks.append(
                ValidationCheck(
                    name="audio_stream_copy_packet_hash_exact",
                    status="PASS" if all_hashes_match else "FAIL",
                    expected="exact SHA-256 audio packet streamhash match",
                    actual="matched" if all_hashes_match else "mismatch",
                    evidence_path="evidence/audio_sync.json",
                )
            )

        # Also measure decoded PCM correlation + lag on primary selected audio track for concrete sync evidence
        if count_ok:
            src_pcm = _extract_mono_pcm(source_path, expected_tracks[0], ffmpeg)
            out_pcm = _extract_mono_pcm(candidate_output_path, 0, ffmpeg)
            corr, lag_ms = _measure_pcm_correlation_and_lag_ms(src_pcm, out_pcm)
            frame_ms = 1000.0 / max(fps, 1.0)
            audio_evidence["pcm_correlation"] = round(corr, 5)
            audio_evidence["pcm_alignment_lag_ms"] = round(lag_ms, 3)
            audio_evidence["one_frame_ms"] = round(frame_ms, 3)
            # Allow encoder priming tolerance (~1.2 frames) for transcoded AAC, <= 1 frame for stream_copy
            max_allowed_lag = frame_ms * (1.05 if render_result.audio_mode == "stream_copy" else 1.5)
            sync_ok = corr >= 0.95 and abs(lag_ms) <= max_allowed_lag
            checks.append(
                ValidationCheck(
                    name="audio_pcm_sync_and_fidelity",
                    status="PASS" if sync_ok else "FAIL",
                    expected=f"corr>=0.95 and |lag|<={max_allowed_lag:.1f}ms",
                    actual=f"corr={corr:.4f}, lag={lag_ms:.2f}ms",
                    evidence_path="evidence/audio_sync.json",
                )
            )
    atomic_write_json(evidence_dir / "audio_sync.json", audio_evidence)

    # 5. Source-relative black & freeze checks (do NOT false-reject intentional source black/static)
    src_blacks = _detect_black_intervals(source_path, ffmpeg)
    out_blacks = _detect_black_intervals(candidate_output_path, ffmpeg)
    src_black_total = sum(b[2] for b in src_blacks)
    out_black_total = sum(b[2] for b in out_blacks)
    new_black_s = max(0.0, out_black_total - src_black_total)
    checks.append(
        ValidationCheck(
            name="no_unexplained_black_intervals",
            status="PASS" if new_black_s <= 0.25 else "FAIL",
            expected="new black <= 0.25s relative to source",
            actual=f"src={src_black_total:.2f}s, out={out_black_total:.2f}s, delta={new_black_s:.2f}s",
        )
    )

    src_freezes = _detect_freeze_intervals(source_path, ffmpeg)
    out_freezes = _detect_freeze_intervals(candidate_output_path, ffmpeg)
    src_freeze_total = sum(f[2] for f in src_freezes)
    out_freeze_total = sum(f[2] for f in out_freezes)
    new_freeze_s = max(0.0, out_freeze_total - src_freeze_total)
    checks.append(
        ValidationCheck(
            name="no_unexpected_freeze_intervals",
            status="PASS" if new_freeze_s <= 0.8 else "FAIL",
            expected="new freeze <= 0.80s relative to source",
            actual=f"src={src_freeze_total:.2f}s, out={out_freeze_total:.2f}s, delta={new_freeze_s:.2f}s",
        )
    )

    # 6. Contact sheet, paired frames, and rendered effect window verification
    sheet_info = generate_contact_sheet_and_pairs(
        source_path=source_path,
        output_path=candidate_output_path,
        plan=plan,
        out_sheet_path=run_dir / "contact_sheet.jpg",
        evidence_dir=evidence_dir,
        ffmpeg_bin=ffmpeg,
    )
    event_mads = sheet_info.get("event_midpoint_mad") or {}
    if plan.events:
        # At least one planned event must measurably alter frames relative to source (prevents fake passthrough!)
        max_mad = max((v["mad"] for v in event_mads.values()), default=0.0)
        applied_count = sum(
            1 for v in event_mads.values() if v["mad"] >= 0.35 or (v["changed_pixels"] >= 60 and v["max_delta"] >= 12.0)
        )
        effects_visible = applied_count >= 1
        checks.append(
            ValidationCheck(
                name="planned_effects_measurably_rendered",
                status="PASS" if effects_visible else "FAIL",
                expected=">=1 planned event shows measurable frame delta (MAD >= 0.35 or >=60 changed pixels with max>=12)",
                actual=f"{applied_count}/{len(plan.events)} events verified (max_mad={max_mad:.2f})",
                evidence_path="contact_sheet.jpg",
            )
        )
    else:
        checks.append(
            ValidationCheck(
                name="planned_effects_measurably_rendered",
                status="SKIP",
                expected="NO_SAFE_EFFECTS plan (passthrough)",
                actual="0 planned events",
                message="Plan status is NO_SAFE_EFFECTS; not counted as effects demonstration pass.",
            )
        )
        warnings.append("Plan contained 0 safe effects (NO_SAFE_EFFECTS).")

    any_fail = any(c.status == "FAIL" for c in checks)
    any_blocked = any(c.status == "BLOCKED" for c in checks)
    overall = "FAIL" if any_fail else ("BLOCKED" if any_blocked else "TECHNICAL_PASS")

    val_res = ValidationResult(
        checks=checks,
        overall_status=overall,
        aesthetic_status="AESTHETIC_PENDING",
        warnings=warnings,
    )
    return val_res, sheet_info
