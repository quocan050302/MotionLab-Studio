"""Phase B2-B6: 100% local, offline, deterministic media analysis (zero transcription/API)."""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from motionlab import ANALYZER_VERSION, SCHEMA_VERSION
from motionlab.config import MotionLabConfig
from motionlab.schemas import (
    AnalysisResult,
    AudioEnvelopePoint,
    EnergyPeak,
    MediaInfo,
    MotionWindow,
    PauseWindow,
    ProtectedZone,
    SceneCut,
    SubjectRegion,
)


def _compute_analysis_dims(display_w: int, display_h: int, max_side: int = 320) -> tuple[int, int]:
    """Compute even downscaled analysis dimensions preserving display aspect."""
    if max(display_w, display_h) <= max_side:
        aw = max(32, (display_w // 2) * 2)
        ah = max(32, (display_h // 2) * 2)
        return aw, ah
    if display_w >= display_h:
        aw = max_side
        ah = max(32, int(round(display_h * (max_side / display_w) / 2.0)) * 2)
    else:
        ah = max_side
        aw = max(32, int(round(display_w * (max_side / display_h) / 2.0)) * 2)
    return aw, ah


def _hsv_hist(rgb: np.ndarray) -> np.ndarray:
    """Compute normalized HSV histogram excluding bottom 18% subtitle band."""
    h = rgb.shape[0]
    crop = rgb[: max(1, int(h * 0.82)), :, :]
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256]).astype(np.float32)
    s = float(hist.sum())
    if s > 0:
        hist /= s
    return hist.flatten()


def _detect_letterbox(gray: np.ndarray) -> tuple[float, float, float, float] | None:
    """Return (top_frac, bot_frac, left_frac, right_frac) of dark uniform bars if present."""
    h, w = gray.shape
    row_means = gray.mean(axis=1)
    row_stds = gray.std(axis=1)
    col_means = gray.mean(axis=0)
    col_stds = gray.std(axis=0)

    def _bar_run(means: np.ndarray, stds: np.ndarray, from_end: bool = False) -> int:
        seq_m = means[::-1] if from_end else means
        seq_s = stds[::-1] if from_end else stds
        count = 0
        for m, sd in zip(seq_m, seq_s):
            if m < 14.0 and sd < 6.0:
                count += 1
            else:
                break
        return count

    top = _bar_run(row_means, row_stds, False)
    bot = _bar_run(row_means, row_stds, True)
    left = _bar_run(col_means, col_stds, False)
    right = _bar_run(col_means, col_stds, True)

    top_f, bot_f = top / h, bot / h
    left_f, right_f = left / w, right / w
    # Only flag genuine letterbox/pillarbox (not full black screen)
    if (0.04 <= top_f <= 0.25 and 0.04 <= bot_f <= 0.25) or (0.04 <= left_f <= 0.30 and 0.04 <= right_f <= 0.30):
        return (round(top_f, 4), round(bot_f, 4), round(left_f, 4), round(right_f, 4))
    return None


def _estimate_ui_density(gray: np.ndarray) -> tuple[float, list[tuple[float, float, float, float]]]:
    """Conservative edge/stroke heuristic detecting screen-recording / busy HUD/UI regions."""
    h, w = gray.shape
    edges = cv2.Canny(gray, 70, 160)
    global_edge_density = float((edges > 0).mean())

    # Check top strip (title/toolbar), bottom strip (taskbar/captions), left/right HUD rails
    bands = [
        ("top", 0.0, 0.0, 1.0, 0.14),
        ("bottom", 0.0, 0.82, 1.0, 0.18),
        ("left", 0.0, 0.14, 0.16, 0.68),
        ("right", 0.84, 0.14, 0.16, 0.68),
    ]
    ui_boxes: list[tuple[float, float, float, float]] = []
    # Horizontal gradient richness catches crisp text / UI chrome lines
    sobel_x = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    sobel_y = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    sharp_mask = ((sobel_x > 45.0) | (sobel_y > 45.0)).astype(np.float32)

    for _name, bx, by, bw, bh in bands:
        x0, y0 = int(bx * w), int(by * h)
        x1, y1 = max(x0 + 1, int((bx + bw) * w)), max(y0 + 1, int((by + bh) * h))
        band_edge = float((edges[y0:y1, x0:x1] > 0).mean())
        band_sharp = float(sharp_mask[y0:y1, x0:x1].mean())
        # Crisp UI chrome has high sharp edge density with distinct horizontal/vertical strokes
        if band_edge > 0.12 and band_sharp > 0.18:
            ui_boxes.append((bx, by, bw, bh))
    return global_edge_density, ui_boxes


def _saliency_center(gray: np.ndarray) -> tuple[float, float]:
    """Gradient magnitude weighted center of visual interest in [0, 1]."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    # Mild center prior so blank edges don't dominate
    h, w = mag.shape
    ys = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    xs = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
    total = float(mag.sum())
    if total < 1e-3:
        return 0.5, 0.5
    cx = float((mag * xs).sum() / total)
    cy = float((mag * ys).sum() / total)
    return round(max(0.15, min(0.85, cx)), 4), round(max(0.15, min(0.85, cy)), 4)


def _load_haar_cascades() -> tuple[cv2.CascadeClassifier | None, cv2.CascadeClassifier | None]:
    """Load bundled local OpenCV Haar face detectors (zero network calls)."""
    try:
        base = Path(cv2.data.haarcascades)
        c1_path = base / "haarcascade_frontalface_default.xml"
        c2_path = base / "haarcascade_frontalface_alt2.xml"
        c1 = cv2.CascadeClassifier(str(c1_path)) if c1_path.is_file() else None
        c2 = cv2.CascadeClassifier(str(c2_path)) if c2_path.is_file() else None
        if c1 is not None and c1.empty():
            c1 = None
        if c2 is not None and c2.empty():
            c2 = None
        return c1, c2
    except Exception:
        return None, None


def _box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 1e-9 else 0.0


def _analyze_video_stream(
    media: MediaInfo,
    config: MotionLabConfig,
    ffmpeg_bin: str,
) -> dict[str, Any]:
    """Stream downscaled RGB frames via FFmpeg pipe to detect cuts, motion, subjects, and safe zones."""
    sample_fps = min(config.analysis_fps, max(1.0, media.fps.as_float()))
    aw, ah = _compute_analysis_dims(media.display_width, media.display_height, max_side=320)
    frame_bytes = aw * ah * 3

    # Note: FFmpeg autorotates display-matrix sources automatically unless -noautorotate is given.
    vf = f"fps={sample_fps:.4f},scale={aw}:{ah}:flags=bilinear"
    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel", "error",
        "-i", media.path,
        "-vf", vf,
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-an", "-sn", "-dn",
        "pipe:1",
    ]

    primary_cascade, confirm_cascade = _load_haar_cascades() if config.enable_face_tracking else (None, None)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
    )
    assert proc.stdout is not None

    # Streaming state — keep only tiny rolling metadata (never full RGB frame list!)
    sample_frames_idx: list[int] = []
    hist_ring: list[np.ndarray] = []  # keep only last 6 histograms for flash rejection
    luma_series: list[float] = []
    hist_diffs: list[float] = []
    luma_diffs: list[float] = []
    motion_scores: list[float] = []
    motion_classes: list[str] = []
    saliency_pts: list[tuple[float, float]] = []
    raw_face_samples: list[tuple[int, list[tuple[float, float, float, float, float]]]] = []
    flash_samples: list[int] = []
    ui_band_hits: dict[tuple[float, float, float, float], int] = {}
    edge_densities: list[float] = []
    letterbox_hits = 0
    letterbox_acc = [0.0, 0.0, 0.0, 0.0]

    prev_gray: np.ndarray | None = None
    prev_hist: np.ndarray | None = None
    prev_luma: float | None = None
    s_idx = 0

    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if not buf or len(buf) < frame_bytes:
                break
            rgb = np.frombuffer(buf, dtype=np.uint8).reshape((ah, aw, 3))
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

            src_frame = min(
                max(0, media.duration_frames - 1),
                int(round((s_idx / sample_fps) * media.fps.as_float())),
            )
            sample_frames_idx.append(src_frame)

            hist = _hsv_hist(rgb)
            mean_luma = float(gray.mean()) / 255.0
            luma_series.append(mean_luma)

            if prev_hist is not None and prev_gray is not None and prev_luma is not None:
                # Bhattacharyya distance in [0, 1]
                hdiff = float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
                if not math.isfinite(hdiff):
                    hdiff = 0.0
                # Mask bottom 18% subtitle area for luma MAD
                cut_h = max(1, int(ah * 0.82))
                ldiff = float(np.abs(gray[:cut_h].astype(np.float32) - prev_gray[:cut_h].astype(np.float32)).mean()) / 255.0
                hist_diffs.append(hdiff)
                luma_diffs.append(ldiff)

                # Abrupt global brightness spike heuristic (camera flash / existing flash transition)
                if abs(mean_luma - prev_luma) > 0.28 and hdiff < 0.35:
                    flash_samples.append(s_idx)

                # Optical flow on further downscaled 160px grayscale for speed & determinism
                flow_w = max(40, aw // 2)
                flow_h = max(40, ah // 2)
                g0 = cv2.resize(prev_gray, (flow_w, flow_h), interpolation=cv2.INTER_AREA)
                g1 = cv2.resize(gray, (flow_w, flow_h), interpolation=cv2.INTER_AREA)
                flow = cv2.calcOpticalFlowFarneback(
                    g0, g1, None,
                    pyr_scale=0.5, levels=2, winsize=11, iterations=2, poly_n=5, poly_sigma=1.1, flags=0,
                )
                mag = np.hypot(flow[..., 0], flow[..., 1])
                mean_mag = float(mag.mean())
                norm_motion = min(1.0, mean_mag / max(1.0, min(flow_w, flow_h) * 0.06))
                motion_scores.append(norm_motion)

                # Camera vs subject motion classification ONLY when coherent
                if norm_motion < 0.04:
                    mclass = "static"
                else:
                    fx_mean = float(flow[..., 0].mean())
                    fy_mean = float(flow[..., 1].mean())
                    global_vec = math.hypot(fx_mean, fy_mean)
                    coherence = global_vec / (mean_mag + 1e-6)
                    active_frac = float((mag > max(0.8, mean_mag * 1.5)).mean())
                    if coherence > 0.68 and active_frac > 0.35:
                        mclass = "camera"
                    elif coherence < 0.38 and 0.04 <= active_frac <= 0.45:
                        mclass = "subject"
                    else:
                        mclass = "unknown"
                motion_classes.append(mclass)
            else:
                hist_diffs.append(0.0)
                luma_diffs.append(0.0)
                motion_scores.append(0.0)
                motion_classes.append("static")

            # Keep rolling histogram history (bounded size <= 1024; for very long videos still tiny 128 floats each)
            hist_ring.append(hist)

            # Saliency & UI/letterbox sampling (every 2nd sample for speed)
            if s_idx % 2 == 0:
                saliency_pts.append(_saliency_center(gray))
                ed, boxes = _estimate_ui_density(gray)
                edge_densities.append(ed)
                for b in boxes:
                    ui_band_hits[b] = ui_band_hits.get(b, 0) + 1
                lb = _detect_letterbox(gray)
                if lb is not None:
                    letterbox_hits += 1
                    for j in range(4):
                        letterbox_acc[j] += lb[j]
            else:
                saliency_pts.append(saliency_pts[-1] if saliency_pts else (0.5, 0.5))

            # Optional local OpenCV Haar face tracking (every 2nd sample, ~3 fps)
            if primary_cascade is not None and (s_idx % 2 == 0):
                eq_gray = cv2.equalizeHist(gray)
                min_face = max(20, int(min(aw, ah) * 0.11))
                dets = primary_cascade.detectMultiScale(
                    eq_gray, scaleFactor=1.15, minNeighbors=5, minSize=(min_face, min_face)
                )
                sample_faces: list[tuple[float, float, float, float, float]] = []
                if len(dets) > 0:
                    alt_dets = (
                        confirm_cascade.detectMultiScale(
                            eq_gray, scaleFactor=1.15, minNeighbors=4, minSize=(min_face, min_face)
                        )
                        if confirm_cascade is not None
                        else ()
                    )
                    alt_norm = [(x / aw, y / ah, w / aw, h / ah) for (x, y, w, h) in alt_dets]
                    for (x, y, w, h) in dets:
                        box = (x / aw, y / ah, w / aw, h / ah)
                        confirmed = any(_box_iou(box, ab) > 0.35 for ab in alt_norm)
                        conf = 0.85 if confirmed else 0.55
                        sample_faces.append((box[0], box[1], box[2], box[3], conf))
                raw_face_samples.append((src_frame, sample_faces))

            prev_gray = gray
            prev_hist = hist
            prev_luma = mean_luma
            s_idx += 1
    finally:
        proc.stdout.close()
        stderr_out = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        if proc.stderr:
            proc.stderr.close()
        ret = proc.wait(timeout=15)
        if ret != 0 and s_idx == 0:
            raise RuntimeError(f"FFmpeg analysis decode failed (exit {ret}): {stderr_out[:400]}")

    n_samples = len(sample_frames_idx)
    if n_samples == 0:
        return {
            "cuts": [],
            "motion_windows": [],
            "subject_regions": [],
            "protected_zones": [],
            "quality_flags": ["empty_video_samples"],
            "summary": {"sampled_frames": 0, "mean_motion": 0.0, "ui_edge_density": 0.0, "has_letterbox": False},
        }

    # --- B2: Scene Cut Detection with Adaptive Threshold & Flash Rejection ---
    cuts: list[SceneCut] = []
    min_shot_frames = max(6, int(round(0.60 * media.fps.as_float())))
    if n_samples >= 3:
        hd_arr = np.array(hist_diffs[1:], dtype=np.float32)
        ld_arr = np.array(luma_diffs[1:], dtype=np.float32)
        med_hd = float(np.median(hd_arr))
        mad_hd = float(np.median(np.abs(hd_arr - med_hd))) + 1e-4
        hist_thresh = max(0.32, min(0.75, med_hd + 4.5 * mad_hd))
        luma_thresh = max(0.11, float(np.median(ld_arr)) + 4.0 * (float(np.median(np.abs(ld_arr - np.median(ld_arr)))) + 1e-4))

        last_cut_frame = -min_shot_frames
        for i in range(1, n_samples):
            hd = hist_diffs[i]
            ld = luma_diffs[i]
            if (hd >= hist_thresh and ld >= 0.07) or (hd >= 0.48 and ld >= 0.05) or (ld >= max(0.24, luma_thresh) and hd >= 0.22):
                # Flash rejection: if frame i+1 or i+2 returns close to frame i-1 histogram, it's a flash!
                is_flash = False
                for lookahead in (1, 2):
                    if i + lookahead < n_samples:
                        ret_diff = float(cv2.compareHist(hist_ring[i - 1], hist_ring[i + lookahead], cv2.HISTCMP_BHATTACHARYYA))
                        if ret_diff < min(0.22, hd * 0.55):
                            is_flash = True
                            flash_samples.append(i)
                            break
                if is_flash:
                    continue
                cut_frame = sample_frames_idx[i]
                if cut_frame - last_cut_frame < min_shot_frames:
                    continue
                if cut_frame <= 3 or cut_frame >= media.duration_frames - 4:
                    continue
                conf = min(0.99, round(0.55 + 0.65 * hd + 0.45 * ld, 3))
                cuts.append(
                    SceneCut(
                        frame=cut_frame,
                        confidence=conf,
                        boundary_type="detected",
                        hist_diff=round(hd, 4),
                        luma_diff=round(ld, 4),
                    )
                )
                last_cut_frame = cut_frame

    # --- B3: Motion Windows (~1.0s windows aligned to integer frames) ---
    motion_windows: list[MotionWindow] = []
    win_samples = max(2, int(round(sample_fps * 1.0)))
    for start_i in range(0, n_samples, win_samples):
        end_i = min(n_samples, start_i + win_samples)
        sf = sample_frames_idx[start_i]
        ef = sample_frames_idx[end_i] if end_i < n_samples else media.duration_frames
        if ef <= sf:
            ef = min(media.duration_frames, sf + 1)
        m_slice = motion_scores[start_i:end_i]
        c_slice = motion_classes[start_i:end_i]
        s_slice = saliency_pts[start_i:end_i]

        mean_m = float(np.mean(m_slice))
        std_m = float(np.std(m_slice))
        stability = max(0.0, min(1.0, 1.0 - (mean_m * 0.7 + std_m * 0.6)))
        near_static = mean_m < 0.085 and stability > 0.80

        # Majority class with unknown fallback
        counts: dict[str, int] = {}
        for mc in c_slice:
            counts[mc] = counts.get(mc, 0) + 1
        dom_class = max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
        cx = float(np.mean([p[0] for p in s_slice]))
        cy = float(np.mean([p[1] for p in s_slice]))

        motion_windows.append(
            MotionWindow(
                start_frame=sf,
                end_frame_exclusive=ef,
                motion_score=round(mean_m, 4),
                stability=round(stability, 4),
                near_static=near_static,
                motion_class=dom_class,
                saliency_center_x=round(cx, 4),
                saliency_center_y=round(cy, 4),
            )
        )

    # --- B5: Temporal Subject Tracking & Smoothing ---
    subject_regions: list[SubjectRegion] = []
    protected_zones: list[ProtectedZone] = []

    active_box: tuple[float, float, float, float] | None = None
    active_conf = 0.0
    seg_start_f: int | None = None
    seg_last_f: int = 0
    seg_hits = 0
    seg_multi = False
    seg_max_count = 0

    def _flush_subject() -> None:
        nonlocal active_box, active_conf, seg_start_f, seg_hits, seg_multi, seg_max_count
        if active_box is not None and seg_start_f is not None and seg_hits >= 2:
            ef = min(media.duration_frames, max(seg_start_f + 1, seg_last_f + int(round(media.fps.as_float() * 0.5))))
            bx, by, bw, bh = active_box
            # Penalize confidence if multiple subjects make single framing ambiguous
            final_conf = round(active_conf * (0.65 if seg_multi else 1.0), 3)
            sr = SubjectRegion(
                start_frame=seg_start_f,
                end_frame_exclusive=ef,
                x=round(bx, 4),
                y=round(by, 4),
                w=round(bw, 4),
                h=round(bh, 4),
                confidence=final_conf,
                subject_count=max(1, seg_max_count),
                ambiguous=seg_multi,
                detector="opencv_haar_frontal",
            )
            subject_regions.append(sr)
            # Protect face + head margin
            px = max(0.0, bx - bw * 0.20)
            py = max(0.0, by - bh * 0.25)
            pw = min(1.0 - px, bw * 1.40)
            ph = min(1.0 - py, bh * 1.50)
            protected_zones.append(
                ProtectedZone(
                    start_frame=seg_start_f,
                    end_frame_exclusive=ef,
                    x=round(px, 4),
                    y=round(py, 4),
                    w=round(pw, 4),
                    h=round(ph, 4),
                    reason="face",
                    confidence=final_conf,
                )
            )
        active_box = None
        active_conf = 0.0
        seg_start_f = None
        seg_hits = 0
        seg_multi = False
        seg_max_count = 0

    for src_f, faces in raw_face_samples:
        if not faces:
            if active_box is not None and (src_f - seg_last_f) > int(round(media.fps.as_float() * 0.8)):
                _flush_subject()
            continue
        # Pick largest face by area as primary; record ambiguity if >=2 comparable faces
        faces_sorted = sorted(faces, key=lambda item: item[2] * item[3], reverse=True)
        best = faces_sorted[0]
        box = (best[0], best[1], best[2], best[3])
        is_multi = len(faces_sorted) >= 2 and (faces_sorted[1][2] * faces_sorted[1][3]) >= 0.4 * (best[2] * best[3])

        if active_box is None:
            active_box = box
            active_conf = best[4]
            seg_start_f = src_f
            seg_last_f = src_f
            seg_hits = 1
            seg_multi = is_multi
            seg_max_count = len(faces_sorted)
        elif _box_iou(active_box, box) > 0.20 or abs((active_box[0] + active_box[2] / 2) - (box[0] + box[2] / 2)) < 0.18:
            # EMA smooth box to avoid jitter
            alpha = 0.35
            active_box = (
                active_box[0] * (1 - alpha) + box[0] * alpha,
                active_box[1] * (1 - alpha) + box[1] * alpha,
                active_box[2] * (1 - alpha) + box[2] * alpha,
                active_box[3] * (1 - alpha) + box[3] * alpha,
            )
            active_conf = min(0.95, active_conf * 0.7 + best[4] * 0.3 + 0.02)
            seg_last_f = src_f
            seg_hits += 1
            seg_multi = seg_multi or is_multi
            seg_max_count = max(seg_max_count, len(faces_sorted))
        else:
            _flush_subject()
            active_box = box
            active_conf = best[4]
            seg_start_f = src_f
            seg_last_f = src_f
            seg_hits = 1
            seg_multi = is_multi
            seg_max_count = len(faces_sorted)
    _flush_subject()

    # --- B6: Existing Effects / Flashes / Letterbox / Busy UI Protected Zones ---
    quality_flags: list[str] = []
    if flash_samples:
        quality_flags.append(f"existing_flash_events:{len(set(flash_samples))}")
        for fi in sorted(set(flash_samples)):
            sf = max(0, sample_frames_idx[fi] - int(round(media.fps.as_float() * 0.35)))
            ef = min(media.duration_frames, sample_frames_idx[fi] + int(round(media.fps.as_float() * 0.45)))
            protected_zones.append(
                ProtectedZone(
                    start_frame=sf,
                    end_frame_exclusive=ef,
                    x=0.0, y=0.0, w=1.0, h=1.0,
                    reason="existing_effect",
                    confidence=0.80,
                )
            )

    # High-motion bursts protected from heavy zoom stacking
    high_motion_count = sum(1 for m in motion_windows if m.motion_score > 0.55)
    if high_motion_count > max(1, len(motion_windows) // 2):
        quality_flags.append("high_existing_motion")

    n_ui_checks = max(1, len(edge_densities))
    mean_edge_density = float(np.mean(edge_densities)) if edge_densities else 0.0
    for box, hits in sorted(ui_band_hits.items()):
        if hits / n_ui_checks >= 0.45:
            protected_zones.append(
                ProtectedZone(
                    start_frame=0,
                    end_frame_exclusive=media.duration_frames,
                    x=box[0], y=box[1], w=box[2], h=box[3],
                    reason="ui_text",
                    confidence=round(min(0.95, hits / n_ui_checks), 3),
                )
            )
    if any(z.reason == "ui_text" for z in protected_zones):
        quality_flags.append("busy_ui_or_text_detected")

    has_letterbox = (letterbox_hits / n_ui_checks) >= 0.50
    if has_letterbox:
        quality_flags.append("letterbox_or_pillarbox_detected")
        t_f, b_f, l_f, r_f = [v / max(1, letterbox_hits) for v in letterbox_acc]
        if t_f >= 0.04:
            protected_zones.append(ProtectedZone(0, media.duration_frames, 0.0, 0.0, 1.0, round(t_f, 4), "letterbox", 0.9))
        if b_f >= 0.04:
            protected_zones.append(ProtectedZone(0, media.duration_frames, 0.0, round(1.0 - b_f, 4), 1.0, round(b_f, 4), "letterbox", 0.9))

    if media.duration_frames <= max(2, int(round(media.fps.as_float() * 0.5))):
        quality_flags.append("very_short_media")
    if len(subject_regions) == 0:
        quality_flags.append("no_confident_face_subject")
    elif any(s.ambiguous for s in subject_regions):
        quality_flags.append("multi_subject_ambiguous")

    return {
        "cuts": cuts,
        "motion_windows": motion_windows,
        "subject_regions": subject_regions,
        "protected_zones": protected_zones,
        "quality_flags": quality_flags,
        "summary": {
            "sampled_frames": n_samples,
            "sample_fps": round(sample_fps, 3),
            "mean_motion": round(float(np.mean(motion_scores)) if motion_scores else 0.0, 4),
            "static_window_ratio": round(
                sum(1 for m in motion_windows if m.near_static) / max(1, len(motion_windows)), 4
            ),
            "ui_edge_density": round(mean_edge_density, 4),
            "has_letterbox": has_letterbox,
            "face_coverage_frames": sum(s.end_frame_exclusive - s.start_frame for s in subject_regions),
            "cut_count": len(cuts),
        },
    }


def _analyze_audio_stream(
    media: MediaInfo,
    config: MotionLabConfig,
    ffmpeg_bin: str,
) -> dict[str, Any]:
    """Phase B4: Bounded-chunk 16kHz mono PCM extraction + RMS/peak/STFT onset/silence analysis.

    Never performs STT or network calls. Never replaces master audio.
    """
    if not media.has_audio:
        return {
            "audio_envelope": [],
            "energy_peaks": [],
            "pauses": [],
            "quality_flags": ["no_audio_stream"],
            "summary": {"has_audio": False, "clipped_frames": 0, "noise_floor_db": -96.0},
        }

    track_idx = config.audio_track if config.audio_track is not None else 0
    if track_idx >= len(media.audio_streams):
        raise ValueError(
            f"Requested --audio-track {track_idx}, but source only has {len(media.audio_streams)} audio stream(s)."
        )

    sr = 16000
    fps_f = media.fps.as_float()
    samples_per_frame = max(1, int(round(sr / fps_f)))
    total_frames = media.duration_frames

    with tempfile.NamedTemporaryFile(prefix="ml01_audio_", suffix=".wav", delete=False) as tmp_f:
        wav_path = Path(tmp_f.name)

    try:
        cmd = [
            ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error",
            "-i", media.path,
            "-map", f"0:a:{track_idx}",
            "-vn", "-ac", "1", "-ar", str(sr), "-c:a", "pcm_s16le",
            str(wav_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120, shell=False)
        if proc.returncode != 0 or not wav_path.is_file() or wav_path.stat().st_size <= 44:
            return {
                "audio_envelope": [],
                "energy_peaks": [],
                "pauses": [],
                "quality_flags": ["audio_extraction_empty"],
                "summary": {"has_audio": False, "clipped_frames": 0, "noise_floor_db": -96.0},
            }

        # Stream WAV in bounded chunks (64k samples at a time) -> per-frame RMS, peak, STFT flux
        frame_rms = np.zeros(total_frames, dtype=np.float32)
        frame_peak = np.zeros(total_frames, dtype=np.float32)
        frame_flux = np.zeros(total_frames, dtype=np.float32)

        # Pure NumPy STFT spectral flux (512-sample Hann window)
        fft_n = 512
        hann = np.hanning(fft_n).astype(np.float32)
        prev_mag: np.ndarray | None = None

        sample_cursor = 0
        leftover = np.empty(0, dtype=np.float32)

        with wave.open(str(wav_path), "rb") as wf:
            chunk_frames = 65536
            while True:
                raw = wf.readframes(chunk_frames)
                if not raw:
                    break
                pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                buf = np.concatenate([leftover, pcm]) if leftover.size else pcm

                n_complete = buf.size // samples_per_frame
                for k in range(n_complete):
                    f_idx = sample_cursor // samples_per_frame
                    if f_idx >= total_frames:
                        break
                    seg = buf[k * samples_per_frame : (k + 1) * samples_per_frame]
                    rms = float(np.sqrt(np.mean(seg * seg)))
                    pk = float(np.max(np.abs(seg)))
                    frame_rms[f_idx] = rms
                    frame_peak[f_idx] = pk

                    # Spectral flux on frame window
                    if seg.size >= fft_n:
                        spec = np.abs(np.fft.rfft(seg[:fft_n] * hann))
                    else:
                        padded = np.zeros(fft_n, dtype=np.float32)
                        padded[: seg.size] = seg
                        spec = np.abs(np.fft.rfft(padded * hann))
                    if prev_mag is not None:
                        diff = np.maximum(0.0, spec - prev_mag)
                        frame_flux[f_idx] = float(np.mean(diff))
                    prev_mag = spec
                    sample_cursor += samples_per_frame

                leftover = buf[n_complete * samples_per_frame :]
                if sample_cursor // samples_per_frame >= total_frames:
                    break
    finally:
        wav_path.unlink(missing_ok=True)

    # Robust percentile normalization & noise floor
    eps = 1e-6
    p20 = float(np.percentile(frame_rms, 20))
    p95 = float(np.percentile(frame_rms, 95))
    max_rms = float(np.max(frame_rms))
    noise_floor_db = round(20.0 * math.log10(max(eps, p20)), 2)

    q_flags: list[str] = []
    if max_rms < 1e-3:
        q_flags.append("silent_audio_track")

    denom = max(1e-4, p95 - p20)
    norm_rms = np.clip((frame_rms - p20) / denom, 0.0, 1.0)

    clipped_frames = int(np.sum(frame_peak >= 0.985))
    if clipped_frames > max(3, int(total_frames * 0.01)):
        q_flags.append(f"audio_clipping_detected:{clipped_frames}")

    envelope: list[AudioEnvelopePoint] = [
        AudioEnvelopePoint(
            frame=i,
            rms=round(float(frame_rms[i]), 5),
            peak=round(float(frame_peak[i]), 5),
            normalized_rms=round(float(norm_rms[i]), 4),
            clipped=bool(frame_peak[i] >= 0.985),
        )
        for i in range(total_frames)
    ]

    # Onsets / energy peaks with robust adaptive threshold and minimum distance (0.55s)
    energy_peaks: list[EnergyPeak] = []
    if max_rms >= 1e-3 and total_frames >= 6:
        flux_p50 = float(np.median(frame_flux))
        flux_p90 = float(np.percentile(frame_flux, 90))
        flux_thresh = max(0.05, flux_p50 + 0.65 * (flux_p90 - flux_p50))
        min_dist = max(4, int(round(0.55 * fps_f)))
        last_peak = -min_dist
        for i in range(2, total_frames - 2):
            if norm_rms[i] < 0.45:
                continue
            local_max = frame_flux[i] >= frame_flux[i - 1] and frame_flux[i] >= frame_flux[i + 1]
            rms_rise = norm_rms[i] - float(np.mean(norm_rms[max(0, i - 4) : i]))
            if (local_max and frame_flux[i] >= flux_thresh) or (rms_rise >= 0.35 and norm_rms[i] >= 0.60):
                if i - last_peak >= min_dist:
                    strength = min(1.0, round(0.5 * float(norm_rms[i]) + 0.5 * min(1.0, float(frame_flux[i]) / max(1e-4, flux_p90 * 1.5)), 4))
                    energy_peaks.append(
                        EnergyPeak(
                            frame=i,
                            strength=strength,
                            onset_flux=round(float(frame_flux[i]), 4),
                            rms_norm=round(float(norm_rms[i]), 4),
                        )
                    )
                    last_peak = i

    # Pause / low-energy windows (>= 0.40s below noise_floor + ~5dB; NEVER used to cut!)
    pauses: list[PauseWindow] = []
    silence_thresh = max(0.004, p20 * 1.8)
    min_pause_frames = max(4, int(round(0.40 * fps_f)))
    in_pause = False
    p_start = 0
    for i in range(total_frames):
        quiet = float(frame_rms[i]) <= silence_thresh
        if quiet and not in_pause:
            in_pause = True
            p_start = i
        elif not quiet and in_pause:
            if i - p_start >= min_pause_frames:
                mean_r = float(np.mean(frame_rms[p_start:i]))
                pauses.append(PauseWindow(p_start, i, round(20.0 * math.log10(max(eps, mean_r)), 2)))
            in_pause = False
    if in_pause and total_frames - p_start >= min_pause_frames:
        mean_r = float(np.mean(frame_rms[p_start:total_frames]))
        pauses.append(PauseWindow(p_start, total_frames, round(20.0 * math.log10(max(eps, mean_r)), 2)))

    return {
        "audio_envelope": envelope,
        "energy_peaks": energy_peaks,
        "pauses": pauses,
        "quality_flags": q_flags,
        "summary": {
            "has_audio": max_rms >= 1e-3,
            "audio_track_analyzed": track_idx,
            "clipped_frames": clipped_frames,
            "noise_floor_db": noise_floor_db,
            "p95_rms": round(p95, 5),
            "energy_peak_count": len(energy_peaks),
            "pause_count": len(pauses),
        },
    }


def analyze_media(
    media: MediaInfo,
    config: MotionLabConfig,
    ffmpeg_bin: str | None = None,
    analysis_workers: int = 2,
) -> AnalysisResult:
    """Run local video + audio signal analysis with bounded parallel execution."""
    ffmpeg = ffmpeg_bin or shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg executable not found on PATH.")

    if analysis_workers >= 2 and media.has_audio:
        with ThreadPoolExecutor(max_workers=2) as pool:
            vid_fut = pool.submit(_analyze_video_stream, media, config, ffmpeg)
            aud_fut = pool.submit(_analyze_audio_stream, media, config, ffmpeg)
            vid_res = vid_fut.result()
            aud_res = aud_fut.result()
    else:
        vid_res = _analyze_video_stream(media, config, ffmpeg)
        aud_res = _analyze_audio_stream(media, config, ffmpeg)

    capabilities: list[str] = ["video_basic"]
    if aud_res["summary"].get("has_audio"):
        capabilities.append("audio_envelope")
    if vid_res["cuts"]:
        capabilities.append("scene_cuts")
    if any(s.confidence >= 0.60 and not s.ambiguous for s in vid_res["subject_regions"]):
        capabilities.append("confident_subject")
    if not media.is_hdr or config.allow_hdr_tonemap:
        capabilities.append("sdr_color")

    quality_flags = sorted(set(vid_res["quality_flags"] + aud_res["quality_flags"]))
    summary_stats = {
        **vid_res["summary"],
        **aud_res["summary"],
        "display_width": media.display_width,
        "display_height": media.display_height,
        "aspect_ratio": round(media.display_width / max(1, media.display_height), 4),
        "duration_s": round(media.duration_s, 4),
    }

    return AnalysisResult(
        schema_version=SCHEMA_VERSION,
        analyzer_version=ANALYZER_VERSION,
        source_hash=media.input_hash,
        timebase=media.timebase,
        fps=media.fps,
        duration_frames=media.duration_frames,
        cuts=vid_res["cuts"],
        motion_windows=vid_res["motion_windows"],
        audio_envelope=aud_res["audio_envelope"],
        energy_peaks=aud_res["energy_peaks"],
        pauses=aud_res["pauses"],
        subject_regions=vid_res["subject_regions"],
        protected_zones=vid_res["protected_zones"],
        quality_flags=quality_flags,
        capabilities=sorted(set(capabilities)),
        summary_stats=summary_stats,
    )
