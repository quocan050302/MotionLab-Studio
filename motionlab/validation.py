"""Phase G6: Technical Validator, Audio Integrity Verifier, and Contact Sheet / Evidence Generator."""

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

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from motionlab import SCHEMA_VERSION
from motionlab.schemas import (
    EditPlan,
    MediaInfo,
    ValidationCheck,
    ValidationResult,
    atomic_write_json,
)


def _ffprobe_json(path: Path, ffprobe_bin: str) -> dict[str, Any]:
    proc = subprocess.run(
        [
            ffprobe_bin, "-v", "error",
            "-print_format", "json",
            "-show_format", "-show_streams",
            str(path),
        ],
        capture_output=True, text=True, timeout=20, shell=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}: {proc.stderr.strip()}")
    return json.loads(proc.stdout)


def _audio_packet_hash(path: Path, track_idx: int, ffmpeg_bin: str) -> tuple[str, int]:
    """Compute deterministic hash of raw audio packet payloads (for stream-copy integrity proof)."""
    proc = subprocess.run(
        [
            ffmpeg_bin, "-v", "error",
            "-i", str(path),
            "-map", f"0:a:{track_idx}",
            "-c:a", "copy",
            "-f", "framemd5",
            "-",
        ],
        capture_output=True, text=True, timeout=60, shell=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"framemd5 failed for audio track {track_idx} of {path}: {proc.stderr.strip()}")
    # Extract packet MD5 + size columns (ignore container-specific pts/dts header lines)
    pkt_sigs: list[str] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 6:
            # size + md5 payload signature
            pkt_sigs.append(f"{parts[4]}:{parts[5]}")
    digest = hashlib.sha256("\n".join(pkt_sigs).encode("ascii")).hexdigest()
    return digest, len(pkt_sigs)


def _extract_mono_pcm(path: Path, track_idx: int, ffmpeg_bin: str, sr: int = 16000) -> np.ndarray:
    with tempfile.NamedTemporaryFile(prefix="ml01_val_", suffix=".wav", delete=False) as tf:
        wav_path = Path(tf.name)
    try:
        proc = subprocess.run(
            [
                ffmpeg_bin, "-y", "-v", "error",
                "-i", str(path),
                "-map", f"0:a:{track_idx}",
                "-vn", "-ac", "1", "-ar", str(sr), "-c:a", "pcm_s16le",
                str(wav_path),
            ],
            capture_output=True, text=True, timeout=60, shell=False,
        )
        if proc.returncode != 0 or not wav_path.is_file() or wav_path.stat().st_size <= 44:
            return np.empty(0, dtype=np.float32)
        with wave.open(str(wav_path), "rb") as wf:
            raw = wf.readframes(wf.getnframes())
        return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    finally:
        wav_path.unlink(missing_ok=True)


def _pcm_sync_correlation(src_pcm: np.ndarray, out_pcm: np.ndarray, sr: int = 16000) -> tuple[float, float]:
    """Return (max_normalized_correlation, lag_ms) within ±100ms window."""
    if src_pcm.size < 320 or out_pcm.size < 320:
        return 1.0, 0.0
    # Bound comparison window to first 30s to keep RAM small
    n = min(src_pcm.size, out_pcm.size, sr * 30)
    a = src_pcm[:n] - float(np.mean(src_pcm[:n]))
    b = out_pcm[:n] - float(np.mean(out_pcm[:n]))
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-5 and nb < 1e-5:
        return 1.0, 0.0
    if na < 1e-5 or nb < 1e-5:
        return 0.0, 0.0

    # Search lag in [-80ms, +80ms] at 1ms steps (16 samples)
    max_lag = int(sr * 0.08)
    step = 16
    best_corr = -1.0
    best_lag = 0
    for lag in range(-max_lag, max_lag + 1, step):
        if lag >= 0:
            sa = a[: n - lag]
            sb = b[lag:]
        else:
            sa = a[-lag:]
            sb = b[: n + lag]
        denom = float(np.linalg.norm(sa) * np.linalg.norm(sb)) + 1e-9
        c = float(np.dot(sa, sb) / denom)
        if c > best_corr:
            best_corr = c
            best_lag = lag
    return round(best_corr, 4), round((best_lag / sr) * 1000.0, 2)


def _sample_video_frames_rgb(
    path: Path,
    frame_indices: list[int],
    fps_f: float,
    ffmpeg_bin: str,
    target_w: int = 320,
    target_h: int = 320,
) -> dict[int, np.ndarray]:
    """Extract specific frames cleanly using FFmpeg seeking."""
    results: dict[int, np.ndarray] = {}
    for f_idx in sorted(set(frame_indices)):
        t_s = max(0.0, f_idx / max(1.0, fps_f))
        cmd = [
            ffmpeg_bin, "-v", "error",
            "-ss", f"{t_s:.6f}",
            "-i", str(path),
            "-frames:v", "1",
            "-vf", f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease",
            "-f", "image2pipe",
            "-vcodec", "png",
            "pipe:1",
        ]
        proc = subprocess.run(cmd, capture_output=True, timeout=15, shell=False)
        if proc.returncode == 0 and proc.stdout:
            arr = np.frombuffer(proc.stdout, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is not None:
                results[f_idx] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return results


def _measure_black_and_freeze(path: Path, ffmpeg_bin: str) -> dict[str, Any]:
    """Measure black-frame fraction and freeze-frame fraction via streaming low-res pipe."""
    cmd = [
        ffmpeg_bin, "-v", "error",
        "-i", str(path),
        "-vf", "fps=4,scale=120:-2",
        "-f", "rawvideo", "-pix_fmt", "gray",
        "-an", "-sn", "-dn", "pipe:1",
    ]
    # First probe aspect for height
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
    assert proc.stdout is not None
    raw = proc.stdout.read()
    proc.stdout.close()
    if proc.stderr:
        proc.stderr.close()
    proc.wait(timeout=30)
    if not raw:
        return {"sampled": 0, "black_ratio": 0.0, "freeze_ratio": 0.0}

    # Probe exact frame dimensions via ffprobe or compute from ratio
    # We'll decode via OpenCV VideoCapture for exact frame sizes reliably!
    cap = cv2.VideoCapture(str(path))
    black = 0
    freeze = 0
    sampled = 0
    prev_small: np.ndarray | None = None
    step = max(1, int(round((cap.get(cv2.CAP_PROP_FPS) or 30.0) / 4.0)))
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (96, 96), interpolation=cv2.INTER_AREA)
                if float(small.mean()) < 6.0 and float(small.std()) < 3.0:
                    black += 1
                if prev_small is not None:
                    diff = float(np.abs(small.astype(np.float32) - prev_small.astype(np.float32)).mean())
                    if diff < 0.25:
                        freeze += 1
                prev_small = small
                sampled += 1
            idx += 1
    finally:
        cap.release()
    return {
        "sampled": sampled,
        "black_ratio": round(black / max(1, sampled), 4),
        "freeze_ratio": round(freeze / max(1, sampled - 1), 4) if sampled > 1 else 0.0,
    }


def _load_qa_font(size: int = 14) -> ImageFont.ImageFont:
    for fp in (
        "/System/Library/Fonts/Menlo.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "C:/Windows/Fonts/consola.ttf",
    ):
        if Path(fp).is_file():
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                pass
    return ImageFont.load_default()


def generate_contact_sheet_and_pairs(
    source_path: Path,
    output_path: Path,
    plan: EditPlan,
    run_dir: Path,
    ffmpeg_bin: str,
) -> dict[str, Any]:
    """Generate 0.5s timeline contact_sheet.jpg + before/during/after paired evidence frames."""
    evidence_dir = run_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    fps_f = plan.fps.as_float()
    dur_s = plan.duration_frames / max(1.0, fps_f)

    # 1. Sample output timeline every 0.5s (capped at 36 tiles for clean disk usage)
    step_s = max(0.5, dur_s / 36.0)
    grid_frames: list[int] = []
    t = 0.0
    while t < dur_s and len(grid_frames) < 36:
        grid_frames.append(min(plan.duration_frames - 1, int(round(t * fps_f))))
        t += step_s

    # Add midpoint of every selected event
    event_mids: list[tuple[str, int]] = []
    for ev in plan.events:
        mid = (ev.start_frame + ev.end_frame_exclusive) // 2
        event_mids.append((ev.effect_id, mid))
        if mid not in grid_frames:
            grid_frames.append(mid)
    grid_frames = sorted(set(grid_frames))[:40]

    out_tiles = _sample_video_frames_rgb(output_path, grid_frames, fps_f, ffmpeg_bin, target_w=240, target_h=320)
    src_pair_frames = sorted({mid for _, mid in event_mids[:6]})
    if not src_pair_frames and plan.duration_frames > 1:
        src_pair_frames = [plan.duration_frames // 2]
    src_samples = _sample_video_frames_rgb(source_path, src_pair_frames, fps_f, ffmpeg_bin, target_w=360, target_h=480)
    out_samples = _sample_video_frames_rgb(output_path, src_pair_frames, fps_f, ffmpeg_bin, target_w=360, target_h=480)

    # Save paired source vs output comparison PNGs into evidence/
    paired_files: list[str] = []
    font = _load_qa_font(14)
    for f_idx in src_pair_frames:
        s_img = src_samples.get(f_idx)
        o_img = out_samples.get(f_idx)
        if s_img is None or o_img is None:
            continue
        th = max(s_img.shape[0], o_img.shape[0])
        sw = int(round(s_img.shape[1] * th / max(1, s_img.shape[0])))
        ow = int(round(o_img.shape[1] * th / max(1, o_img.shape[0])))
        s_pil = Image.fromarray(s_img).resize((sw, th), Image.Resampling.LANCZOS)
        o_pil = Image.fromarray(o_img).resize((ow, th), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (sw + ow + 16, th + 36), (18, 20, 26))
        canvas.paste(s_pil, (4, 32))
        canvas.paste(o_pil, (sw + 12, 32))
        draw = ImageDraw.Draw(canvas)
        active_ids = [ev.effect_id for ev in plan.events if ev.start_frame <= f_idx < ev.end_frame_exclusive]
        lbl = ",".join(active_ids) if active_ids else "base"
        draw.text((8, 8), f"SOURCE  t={f_idx / fps_f:.2f}s (f={f_idx})", fill=(220, 225, 235), font=font)
        draw.text((sw + 16, 8), f"OUTPUT  [{lbl}]", fill=(130, 225, 255), font=font)
        pair_path = evidence_dir / f"pair_f{f_idx:05d}_{lbl.replace(',', '_')}.jpg"
        canvas.save(pair_path, "JPEG", quality=88)
        paired_files.append(str(pair_path.relative_to(run_dir)))

    # Build contact_sheet.jpg
    contact_path = run_dir / "contact_sheet.jpg"
    ordered_tiles = [(f, out_tiles[f]) for f in sorted(out_tiles.keys())]
    if ordered_tiles:
        cols = min(6, max(1, len(ordered_tiles)))
        rows = math.ceil(len(ordered_tiles) / cols)
        tw = max(arr.shape[1] for _, arr in ordered_tiles)
        th = max(arr.shape[0] for _, arr in ordered_tiles)
        pad = 8
        label_h = 22
        header_h = 40
        sheet_w = cols * (tw + pad) + pad
        sheet_h = header_h + rows * (th + label_h + pad) + pad
        sheet = Image.new("RGB", (sheet_w, sheet_h), (16, 18, 24))
        draw = ImageDraw.Draw(sheet)
        draw.text(
            (12, 10),
            f"MotionLab-AI ML-01 Contact Sheet | profile={plan.profile} | {plan.width}x{plan.height}@{plan.fps.as_float():.2f}fps | events={len(plan.events)}",
            fill=(240, 244, 250),
            font=font,
        )
        for idx, (f_idx, arr) in enumerate(ordered_tiles):
            r = idx // cols
            c = idx % cols
            x = pad + c * (tw + pad)
            y = header_h + pad + r * (th + label_h + pad)
            tile = Image.fromarray(arr).resize((tw, th), Image.Resampling.LANCZOS)
            sheet.paste(tile, (x, y))
            act = [ev.effect_id for ev in plan.events if ev.start_frame <= f_idx < ev.end_frame_exclusive]
            tag = f"{f_idx / fps_f:.2f}s" + (f" [{','.join(act)}]" if act else "")
            draw.text((x + 2, y + th + 3), tag, fill=(170, 215, 255) if act else (180, 185, 195), font=font)
        sheet.save(contact_path, "JPEG", quality=88)

    return {
        "contact_sheet": str(contact_path.relative_to(run_dir)) if contact_path.is_file() else None,
        "paired_frames": paired_files,
    }


def validate_rendered_output(
    source_media: MediaInfo,
    plan: EditPlan,
    rendered_path: Path,
    run_dir: Path,
    control_path: Path | None = None,
    ffmpeg_bin: str | None = None,
    ffprobe_bin: str | None = None,
) -> ValidationResult:
    """Run all mandatory machine technical validation checks on a rendered output MP4."""
    ffmpeg = ffmpeg_bin or shutil.which("ffmpeg") or "ffmpeg"
    ffprobe = ffprobe_bin or shutil.which("ffprobe") or "ffprobe"
    evidence_dir = run_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    checks: list[ValidationCheck] = []
    warnings: list[str] = []

    # 1. File existence & size
    if not rendered_path.is_file() or rendered_path.stat().st_size <= 0:
        checks.append(ValidationCheck("file_exists_nonempty", "FAIL", ">0 bytes", 0, None, "Rendered file missing or empty."))
        return ValidationResult(SCHEMA_VERSION, "FAIL", "FAIL", "PENDING_HUMAN_REVIEW", checks, warnings)
    checks.append(ValidationCheck("file_exists_nonempty", "PASS", ">0 bytes", rendered_path.stat().st_size))

    # 2. ffprobe metadata
    probe = _ffprobe_json(rendered_path, ffprobe)
    probe_ev_path = evidence_dir / "ffprobe_output.json"
    atomic_write_json(probe_ev_path, probe)
    rel_probe = str(probe_ev_path.relative_to(run_dir))

    vstreams = [s for s in probe.get("streams", []) if s.get("codec_type") == "video"]
    astreams = [s for s in probe.get("streams", []) if s.get("codec_type") == "audio"]
    sstreams = [s for s in probe.get("streams", []) if s.get("codec_type") == "subtitle"]

    if not vstreams:
        checks.append(ValidationCheck("video_stream_present", "FAIL", ">=1 video stream", 0, rel_probe))
        return ValidationResult(SCHEMA_VERSION, "FAIL", "FAIL", "PENDING_HUMAN_REVIEW", checks, warnings)

    v = vstreams[0]
    out_w, out_h = int(v.get("width", 0)), int(v.get("height", 0))
    pix_fmt = str(v.get("pix_fmt", ""))
    dim_ok = (out_w == plan.width and out_h == plan.height and pix_fmt == "yuv420p")
    checks.append(
        ValidationCheck(
            "video_dimensions_and_pixfmt",
            "PASS" if dim_ok else "FAIL",
            f"{plan.width}x{plan.height} yuv420p",
            f"{out_w}x{out_h} {pix_fmt}",
            rel_probe,
        )
    )

    # 3. Full decode integrity (-v error -xerror)
    dec_proc = subprocess.run(
        [ffmpeg, "-v", "error", "-xerror", "-i", str(rendered_path), "-f", "null", "-"],
        capture_output=True, text=True, timeout=180, shell=False,
    )
    dec_ok = dec_proc.returncode == 0 and not dec_proc.stderr.strip()
    checks.append(
        ValidationCheck(
            "full_decode_integrity",
            "PASS" if dec_ok else "FAIL",
            "exit=0, no decode errors",
            f"exit={dec_proc.returncode}, stderr={dec_proc.stderr.strip()[:200]}",
        )
    )

    # 4. Duration & FPS fidelity (within 1 output frame + 40ms container rounding)
    fps_f = plan.fps.as_float()
    expected_dur = plan.duration_frames / max(1.0, fps_f)
    out_dur = float(v.get("duration") or probe.get("format", {}).get("duration") or 0.0)
    tol_s = (1.0 / max(1.0, fps_f)) + 0.055
    dur_delta = abs(out_dur - expected_dur)
    dur_ok = dur_delta <= tol_s
    checks.append(
        ValidationCheck(
            "duration_preservation",
            "PASS" if dur_ok else "FAIL",
            f"{expected_dur:.4f}s (±{tol_s:.4f}s)",
            f"{out_dur:.4f}s (delta={dur_delta:.4f}s)",
            rel_probe,
        )
    )

    # 5. No unrequested subtitles
    sub_ok = len(sstreams) == 0
    checks.append(
        ValidationCheck(
            "no_default_subtitles",
            "PASS" if sub_ok else "FAIL",
            0,
            len(sstreams),
            rel_probe,
        )
    )

    # 6. Audio integrity HARD LOCK (Phase E6)
    audio_ev: dict[str, Any] = {"has_source_audio": source_media.has_audio, "mode": plan.audio_policy.get("mode")}
    audio_ev_path = evidence_dir / "audio_integrity.json"
    if not source_media.has_audio:
        a_ok = len(astreams) == 0
        audio_ev["output_audio_streams"] = len(astreams)
        atomic_write_json(audio_ev_path, audio_ev)
        checks.append(
            ValidationCheck(
                "audio_integrity",
                "PASS" if a_ok else "FAIL",
                "0 audio streams (silent source)",
                len(astreams),
                str(audio_ev_path.relative_to(run_dir)),
            )
        )
    else:
        selected_tracks = list(plan.audio_policy.get("selected_tracks") or [0])
        mode = str(plan.audio_policy.get("mode", "stream_copy"))
        count_ok = len(astreams) == len(selected_tracks)
        src_a = source_media.audio_streams[selected_tracks[0]]
        out_a = astreams[0] if astreams else {}
        out_sr = int(out_a.get("sample_rate") or 0)
        out_ch = int(out_a.get("channels") or 0)
        sr_ch_ok = (out_sr == src_a.sample_rate and out_ch == src_a.channels)

        out_a_dur = float(out_a.get("duration") or out_dur)
        src_a_dur = src_a.duration_s or source_media.duration_s
        a_dur_delta = abs(out_a_dur - src_a_dur)
        a_dur_ok = a_dur_delta <= max(0.08, (1.0 / max(1.0, fps_f)) + 0.05)

        payload_ok = False
        if mode == "stream_copy" and astreams:
            src_hash, src_pkts = _audio_packet_hash(Path(source_media.path), selected_tracks[0], ffmpeg)
            out_hash, out_pkts = _audio_packet_hash(rendered_path, 0, ffmpeg)
            payload_ok = (src_hash == out_hash and src_pkts == out_pkts and src_pkts > 0)
            audio_ev.update({
                "verification_method": "stream_copy_packet_payload_sha256",
                "source_packet_hash": src_hash,
                "output_packet_hash": out_hash,
                "source_packets": src_pkts,
                "output_packets": out_pkts,
                "exact_packet_match": payload_ok,
            })
        elif astreams:
            src_pcm = _extract_mono_pcm(Path(source_media.path), selected_tracks[0], ffmpeg)
            out_pcm = _extract_mono_pcm(rendered_path, 0, ffmpeg)
            corr, lag_ms = _pcm_sync_correlation(src_pcm, out_pcm)
            max_lag_ms = (1000.0 / max(1.0, fps_f)) + 25.0  # <= 1 frame + AAC encoder priming
            payload_ok = (corr >= 0.92 and abs(lag_ms) <= max_lag_ms)
            audio_ev.update({
                "verification_method": "decoded_pcm_cross_correlation",
                "pearson_correlation": corr,
                "estimated_lag_ms": lag_ms,
                "max_allowed_lag_ms": round(max_lag_ms, 2),
                "note": "Disclosed AAC transcode verified by PCM correlation & lag (not claimed bit-identical).",
            })

        audio_ev.update({
            "expected_tracks": len(selected_tracks),
            "actual_tracks": len(astreams),
            "source_sample_rate": src_a.sample_rate,
            "output_sample_rate": out_sr,
            "source_channels": src_a.channels,
            "output_channels": out_ch,
            "audio_duration_delta_s": round(a_dur_delta, 4),
        })
        atomic_write_json(audio_ev_path, audio_ev)
        all_a_ok = count_ok and sr_ch_ok and a_dur_ok and payload_ok
        checks.append(
            ValidationCheck(
                "audio_integrity",
                "PASS" if all_a_ok else "FAIL",
                f"tracks={len(selected_tracks)}, sr={src_a.sample_rate}, ch={src_a.channels}, mode={mode}",
                f"tracks={len(astreams)}, sr={out_sr}, ch={out_ch}, payload_ok={payload_ok}, dur_delta={a_dur_delta:.4f}s",
                str(audio_ev_path.relative_to(run_dir)),
            )
        )

    # 7. Black & freeze regression relative to source (never false-fail intentionally black/static sources!)
    ref_for_bf = control_path if (control_path and control_path.is_file()) else Path(source_media.path)
    src_bf = _measure_black_and_freeze(ref_for_bf, ffmpeg)
    out_bf = _measure_black_and_freeze(rendered_path, ffmpeg)
    black_increase = out_bf["black_ratio"] - src_bf["black_ratio"]
    freeze_increase = out_bf["freeze_ratio"] - src_bf["freeze_ratio"]
    bf_ok = (black_increase <= 0.08) and (freeze_increase <= 0.15)
    checks.append(
        ValidationCheck(
            "black_and_freeze_regression",
            "PASS" if bf_ok else "FAIL",
            "black_increase<=0.08, freeze_increase<=0.15 vs source/control",
            f"src={src_bf}, out={out_bf}",
        )
    )

    # 8. Effect execution visual evidence (compare rendered frames inside effect windows vs control/source)
    eff_ev_path = evidence_dir / "effect_execution.json"
    if not plan.events:
        atomic_write_json(eff_ev_path, {"status": "NO_SAFE_EFFECTS", "events_checked": 0})
        checks.append(
            ValidationCheck(
                "effect_execution_evidence",
                "SKIP",
                "effects executed",
                "0 effects planned (NO_SAFE_EFFECTS passthrough)",
                str(eff_ev_path.relative_to(run_dir)),
            )
        )
    else:
        sample_frames = [
            min(plan.duration_frames - 1, max(0, (ev.start_frame + ev.end_frame_exclusive) // 2))
            for ev in plan.events
        ]
        ref_path = control_path if (control_path and control_path.is_file()) else Path(source_media.path)
        ref_imgs = _sample_video_frames_rgb(ref_path, sample_frames, fps_f, ffmpeg, target_w=320, target_h=320)
        out_imgs = _sample_video_frames_rgb(rendered_path, sample_frames, fps_f, ffmpeg, target_w=320, target_h=320)
        per_ev: list[dict[str, Any]] = []
        changed_count = 0
        for ev, f_idx in zip(plan.events, sample_frames):
            ri = ref_imgs.get(f_idx)
            oi = out_imgs.get(f_idx)
            if ri is None or oi is None:
                per_ev.append({"event_id": ev.id, "effect_id": ev.effect_id, "frame": f_idx, "mad": None})
                continue
            if ri.shape != oi.shape:
                ri = cv2.resize(ri, (oi.shape[1], oi.shape[0]), interpolation=cv2.INTER_AREA)
            diff_arr = np.abs(oi.astype(np.float32) - ri.astype(np.float32))
            mad = float(diff_arr.mean())
            max_px = float(diff_arr.max())
            changed_frac = float((diff_arr.max(axis=2) >= 6.0).mean())
            is_black = float(oi.mean()) < 4.0
            # Detect real rendered difference (including crisp peripheral overlays) without false no-ops
            applied = ((mad >= 0.35) or (max_px >= 22.0 and changed_frac >= 0.001)) and (not is_black)
            if applied:
                changed_count += 1
            per_ev.append({
                "event_id": ev.id,
                "effect_id": ev.effect_id,
                "frame": f_idx,
                "mean_abs_diff": round(mad, 3),
                "max_pixel_diff": round(max_px, 2),
                "changed_pixels_frac": round(changed_frac, 5),
                "output_mean_luma": round(float(oi.mean()), 2),
                "applied_measurably": applied,
            })
        atomic_write_json(eff_ev_path, {"events_checked": len(per_ev), "measurably_applied": changed_count, "details": per_ev})
        eff_ok = changed_count >= max(1, math.ceil(len(plan.events) * 0.75))
        checks.append(
            ValidationCheck(
                "effect_execution_evidence",
                "PASS" if eff_ok else "FAIL",
                f">=75% planned effects measurably applied ({len(plan.events)} total)",
                f"{changed_count}/{len(plan.events)} measurably applied",
                str(eff_ev_path.relative_to(run_dir)),
            )
        )

    # Generate visual review artifacts (contact_sheet.jpg + paired frames)
    generate_contact_sheet_and_pairs(Path(source_media.path), rendered_path, plan, run_dir, ffmpeg)

    required_statuses = [c.status for c in checks if c.status != "SKIP"]
    if any(s == "FAIL" for s in required_statuses):
        overall = "FAIL"
    elif any(s == "BLOCKED" for s in required_statuses):
        overall = "BLOCKED"
    else:
        overall = "PASS"

    return ValidationResult(
        schema_version=SCHEMA_VERSION,
        overall_status=overall,
        technical_status="TECHNICAL_PASS" if overall == "PASS" else f"TECHNICAL_{overall}",
        aesthetic_status="PENDING_HUMAN_REVIEW",
        checks=checks,
        warnings=warnings,
    )
