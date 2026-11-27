"""Phase B2–B6 — Local Zero-API Media Analyzer.

Extracts deterministic signal-level cues from video and audio without any STT,
LLM, cloud API, or full-video RAM buffering:
  - B2: detect_scene_cuts() via streaming low-res HSV histogram + luma diff with flash/subtitle guards
  - B3: measure_motion() via downscaled grayscale optical flow / frame diff + saliency proxy
  - B4: analyze_audio() via bounded chunked mono PCM extraction -> RMS/peak envelope, spectral-flux onsets, pauses
  - B5: track_subject() via local OpenCV Haar frontal face detector (Apache-2.0, bundled in opencv-python-headless)
  - B6: estimate_existing_effects() & safe zones (luminance spikes, letterbox, busy UI edges, face protection)
"""

from __future__ import annotations

import asyncio
import math
import os
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from motionlab import ANALYZER_VERSION
from motionlab.cache import LocalAnalysisCache
from motionlab.config import AnalysisConfig, EditConfig
from motionlab.events import build_candidates
from motionlab.preflight import find_binary
from motionlab.resource import ResourceBudget, run_bounded_async
from motionlab.schemas import (
    AnalysisResult,
    AudioEnvelopePoint,
    EnergyPeak,
    MediaInfo,
    MotionWindow,
    PauseWindow,
    ProtectedInterval,
    SceneCut,
    SubjectRegion,
)


def _dependency_signature() -> str:
    return f"cv2={cv2.__version__};np={np.__version__}"


def _open_ffmpeg_raw_rgb_stream(
    video_path: Path,
    sample_fps: float,
    target_width: int,
    target_height: int,
    ffmpeg_bin: str,
    ffmpeg_threads: int = 2,
) -> subprocess.Popen[bytes]:
    """Stream downscaled RGB24 frames via FFmpeg without loading the full video into RAM."""
    vf = f"fps={sample_fps:.4f},scale={target_width}:{target_height}:flags=bilinear"
    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-threads",
        str(max(1, ffmpeg_threads)),
        "-i",
        str(video_path),
        "-an",
        "-sn",
        "-dn",
        "-vf",
        vf,
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _compute_analysis_dimensions(media: MediaInfo, target_w: int) -> tuple[int, int]:
    w = max(64, (target_w // 2) * 2)
    aspect = media.display_height / max(1, media.display_width)
    h = max(64, (int(round(w * aspect)) // 2) * 2)
    return w, h


def _analyze_video_stream(
    media: MediaInfo,
    cfg: AnalysisConfig,
    ffmpeg_bin: str,
    ffmpeg_threads: int = 2,
) -> dict[str, Any]:
    """Single streaming pass over downscaled video frames for cuts, motion, saliency, existing effects, and subjects."""
    w, h = _compute_analysis_dimensions(media, cfg.downscale_width)
    sample_fps = min(cfg.sample_fps, max(1.0, media.fps_float))
    frame_bytes = w * h * 3

    # Optional bundled local OpenCV Haar face detector (no downloads!)
    haar_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    face_cascade = None
    has_face_detector = False
    if cfg.enable_subject_tracking and haar_path.exists():
        cascade = cv2.CascadeClassifier(str(haar_path))
        if not cascade.empty():
            face_cascade = cascade
            has_face_detector = True

    subject_stride = max(1, int(round(sample_fps / max(0.5, cfg.subject_sample_fps))))

    proc = _open_ffmpeg_raw_rgb_stream(
        Path(media.source_path),
        sample_fps=sample_fps,
        target_width=w,
        target_height=h,
        ffmpeg_bin=ffmpeg_bin,
        ffmpeg_threads=ffmpeg_threads,
    )

    hist_diffs: list[float] = []
    luma_means: list[float] = []
    luma_diffs: list[float] = []
    motion_scores: list[float] = []
    cam_motion_flags: list[str] = []
    saliency_xs: list[float] = []
    saliency_ys: list[float] = []
    edge_densities: list[float] = []
    sample_frames: list[int] = []
    border_edge_ratios: list[float] = []
    letterbox_votes = 0
    raw_subjects: list[SubjectRegion] = []

    prev_gray: np.ndarray | None = None
    prev_hist: np.ndarray | None = None
    prev_luma: float | None = None

    idx = 0
    try:
        assert proc.stdout is not None
        while True:
            buf = proc.stdout.read(frame_bytes)
            if not buf or len(buf) < frame_bytes:
                break
            rgb = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 3))
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

            # Restrict cut histogram to central 80% vertically to reduce subtitle false triggers
            y0_sub = int(h * 0.10)
            y1_sub = int(h * 0.85)
            hsv_core = hsv[y0_sub:y1_sub, :, :]
            hist = cv2.calcHist([hsv_core], [0, 1], None, [24, 16], [0, 180, 0, 256])
            cv2.normalize(hist, hist, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)

            mean_luma = float(np.mean(gray)) / 255.0
            luma_means.append(mean_luma)

            # Letterbox / pillarbox detection (top/bottom or left/right uniform black strips)
            strip_h = max(4, int(h * 0.08))
            if float(np.mean(gray[:strip_h, :])) < 10.0 and float(np.mean(gray[-strip_h:, :])) < 10.0:
                letterbox_votes += 1

            # Edge density (useful for UI/screen-recording/text detection)
            edges = cv2.Canny(gray, 80, 160)
            edge_density = float(np.mean(edges > 0))
            edge_densities.append(edge_density)

            # Peripheral/HUD edge ratio (top 15%, bottom 15%, sides 12%)
            periph_mask = np.zeros_like(edges, dtype=bool)
            periph_mask[: max(2, int(h * 0.15)), :] = True
            periph_mask[int(h * 0.85) :, :] = True
            periph_mask[:, : max(2, int(w * 0.12))] = True
            periph_mask[:, int(w * 0.88) :] = True
            border_edge_ratios.append(float(np.mean(edges[periph_mask] > 0)))

            # Map sample index to exact source frame index
            src_time = idx / sample_fps
            src_frame = min(media.nb_frames - 1, max(0, media.seconds_to_frame(src_time)))
            sample_frames.append(src_frame)

            if prev_gray is not None and prev_hist is not None and prev_luma is not None:
                # B2: Bhattacharyya distance in [0, 1]
                hdiff = float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
                ldiff = abs(mean_luma - prev_luma)
                hist_diffs.append(hdiff)
                luma_diffs.append(ldiff)

                # B3: Dense optical flow (Farneback) on small downscaled gray frame
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray,
                    gray,
                    None,
                    pyr_scale=0.5,
                    levels=2,
                    winsize=9,
                    iterations=2,
                    poly_n=5,
                    poly_sigma=1.1,
                    flags=0,
                )
                mag, _ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                norm_mag = mag / max(1.0, float(min(w, h)) * 0.08)
                m_score = float(np.clip(np.mean(norm_mag), 0.0, 1.0))
                motion_scores.append(m_score)

                # Classify camera vs subject motion ONLY when confident
                fx_med = float(np.median(flow[..., 0]))
                fy_med = float(np.median(flow[..., 1]))
                global_vec = math.hypot(fx_med, fy_med)
                residual = float(np.mean(np.hypot(flow[..., 0] - fx_med, flow[..., 1] - fy_med)))
                if m_score < 0.06:
                    cam_motion_flags.append("unknown")
                elif global_vec > 1.2 and global_vec > residual * 1.45:
                    cam_motion_flags.append("camera")
                elif residual > global_vec * 1.65 and float(np.percentile(mag, 90)) > 1.0:
                    cam_motion_flags.append("subject")
                else:
                    cam_motion_flags.append("unknown")

                # Saliency proxy: blend edge energy + motion magnitude
                sal = (edges.astype(np.float32) / 255.0) * 0.6 + np.clip(norm_mag, 0.0, 1.0) * 0.4
                sal_sum = float(np.sum(sal))
                if sal_sum > 1e-4:
                    ys, xs = np.indices((h, w), dtype=np.float32)
                    sx = float(np.sum(sal * xs) / sal_sum) / w
                    sy = float(np.sum(sal * ys) / sal_sum) / h
                else:
                    sx, sy = 0.5, 0.5
                saliency_xs.append(sx)
                saliency_ys.append(sy)
            else:
                hist_diffs.append(0.0)
                luma_diffs.append(0.0)
                motion_scores.append(0.0)
                cam_motion_flags.append("unknown")
                saliency_xs.append(0.5)
                saliency_ys.append(0.5)

            # B5: Optional local Haar face detection (strided)
            if face_cascade is not None and (idx % subject_stride == 0):
                # Upscale slightly if very small so Haar works reliably on portraits
                det_scale = 2.0 if w < 220 else 1.0
                det_gray = (
                    cv2.resize(gray, (int(w * det_scale), int(h * det_scale)), interpolation=cv2.INTER_LINEAR)
                    if det_scale != 1.0
                    else gray
                )
                det_gray = cv2.equalizeHist(det_gray)
                min_face = max(20, int(min(det_gray.shape) * 0.10))
                rects, reject_levels, level_weights = face_cascade.detectMultiScale3(
                    det_gray,
                    scaleFactor=1.12,
                    minNeighbors=5,
                    minSize=(min_face, min_face),
                    outputRejectLevels=True,
                )
                if len(rects) > 0:
                    dh, dw = det_gray.shape[:2]
                    # Pick highest-weight / largest face; mark multi_subject if >= 2 strong detections
                    scored = []
                    for ri, (rx, ry, rw, rh) in enumerate(rects):
                        wt = float(level_weights[ri]) if len(level_weights) > ri else 1.5
                        conf = float(np.clip(1.0 / (1.0 + math.exp(-(wt - 2.2) * 0.8)), 0.1, 0.98))
                        scored.append((conf * (rw * rh), conf, rx / dw, ry / dh, rw / dw, rh / dh))
                    scored.sort(key=lambda t: t[0], reverse=True)
                    strong_count = sum(1 for s in scored if s[1] >= 0.45)
                    _, conf, nx, ny, nw, nh = scored[0]
                    if conf >= 0.35:
                        raw_subjects.append(
                            SubjectRegion(
                                frame=src_frame,
                                x=nx,
                                y=ny,
                                w=nw,
                                h=nh,
                                confidence=conf,
                                multi_subject=(strong_count >= 2),
                            )
                        )

            prev_gray = gray
            prev_hist = hist
            prev_luma = mean_luma
            idx += 1
    finally:
        if proc.stdout:
            proc.stdout.close()
        stderr_out = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        if proc.stderr:
            proc.stderr.close()
        proc.wait(timeout=30)
        if proc.returncode != 0 and idx == 0:
            raise RuntimeError(f"FFmpeg video decode failed during analysis: {stderr_out.strip()}")

    n_samples = len(sample_frames)
    if n_samples == 0:
        return {
            "cuts": [],
            "motion_windows": [],
            "subject_regions": [],
            "protected_intervals": [],
            "safe_zones": {"has_face_detector": has_face_detector, "face_box": None, "letterbox": False},
            "quality_flags": ["empty_video_samples"],
            "signal_summary": {},
            "has_face_detector": has_face_detector,
        }

    # --- B2: Adaptive scene cut detection with flash & minimum-shot-interval protection ---
    cuts: list[SceneCut] = []
    protected_intervals: list[ProtectedInterval] = []
    hd_arr = np.array(hist_diffs, dtype=np.float64)
    ld_arr = np.array(luma_diffs, dtype=np.float64)
    med_hd = float(np.median(hd_arr[1:])) if n_samples > 1 else 0.0
    mad_hd = float(np.median(np.abs(hd_arr[1:] - med_hd))) if n_samples > 1 else 0.01
    robust_sigma = max(0.025, mad_hd * 1.4826)
    adaptive_thresh = max(cfg.cut_min_hist_diff, med_hd + cfg.cut_threshold_sigma * robust_sigma)

    min_cut_frames = max(1, int(round(cfg.min_shot_interval_s * media.fps_float)))
    last_cut_frame = -min_cut_frames

    for i in range(1, n_samples):
        hd = float(hd_arr[i])
        ld = float(ld_arr[i])
        if hd < adaptive_thresh:
            continue

        # Flash guard: if luma spikes abruptly AND returns close to previous level within 1-2 samples -> flash, not scene cut
        is_flash = False
        if ld > 0.25 and i + 2 < n_samples:
            luma_before = luma_means[i - 1]
            luma_after = luma_means[min(n_samples - 1, i + 2)]
            if abs(luma_after - luma_before) < 0.09 and abs(luma_means[i] - luma_before) > 0.22:
                is_flash = True

        src_frame = sample_frames[i]
        if is_flash:
            protected_intervals.append(
                ProtectedInterval(
                    start_frame=max(0, src_frame - int(media.fps_float * 0.25)),
                    end_frame_exclusive=min(media.nb_frames, src_frame + int(media.fps_float * 0.45)),
                    reason="abrupt_luminance_flash",
                    severity=min(1.0, ld * 2.0),
                )
            )
            continue

        # Existing gradual transition / rapid-edit guard: multiple high diffs clustered closely
        existing_transition = bool(
            (i >= 2 and hd_arr[i - 1] > adaptive_thresh * 0.65)
            or (i + 1 < n_samples and hd_arr[i + 1] > adaptive_thresh * 0.65)
        )

        if src_frame - last_cut_frame < min_cut_frames:
            protected_intervals.append(
                ProtectedInterval(
                    start_frame=max(0, last_cut_frame),
                    end_frame_exclusive=min(media.nb_frames, src_frame + int(media.fps_float * 0.4)),
                    reason="rapid_cut_cluster",
                    severity=0.8,
                )
            )
            continue

        conf = float(np.clip((hd - adaptive_thresh * 0.7) / max(0.2, 1.0 - adaptive_thresh * 0.7), 0.35, 0.99))
        cuts.append(
            SceneCut(
                frame=src_frame,
                time_s=media.frame_to_seconds(src_frame),
                confidence=conf,
                boundary_type="detected",
                hist_diff=hd,
                luma_diff=ld,
                protected_existing_transition=existing_transition,
            )
        )
        last_cut_frame = src_frame
        if existing_transition:
            protected_intervals.append(
                ProtectedInterval(
                    start_frame=max(0, src_frame - int(media.fps_float * 0.3)),
                    end_frame_exclusive=min(media.nb_frames, src_frame + int(media.fps_float * 0.5)),
                    reason="existing_gradual_transition",
                    severity=0.85,
                )
            )

    # --- B3: Aggregate motion windows (1.0s contiguous windows aligned to integer frames) ---
    motion_windows: list[MotionWindow] = []
    win_samples = max(2, int(round(sample_fps * 1.0)))
    for start_i in range(0, n_samples, win_samples):
        end_i = min(n_samples, start_i + win_samples)
        sf = sample_frames[start_i]
        ef = sample_frames[end_i] if end_i < n_samples else media.nb_frames
        if ef <= sf:
            continue
        m_slice = motion_scores[start_i:end_i]
        m_mean = float(np.mean(m_slice))
        m_std = float(np.std(m_slice))
        stability = float(np.clip(1.0 - (m_mean * 0.7 + m_std * 0.9), 0.0, 1.0))
        if m_mean < 0.08:
            m_cls = "near_static"
        elif m_mean < 0.30:
            m_cls = "moderate"
        else:
            m_cls = "high"

        flags = cam_motion_flags[start_i:end_i]
        cam_cnt = flags.count("camera")
        subj_cnt = flags.count("subject")
        if cam_cnt >= max(2, len(flags) // 2) and cam_cnt > subj_cnt:
            src_cls = "camera"
        elif subj_cnt >= max(2, len(flags) // 2) and subj_cnt > cam_cnt:
            src_cls = "subject"
        else:
            src_cls = "unknown"

        motion_windows.append(
            MotionWindow(
                start_frame=sf,
                end_frame_exclusive=ef,
                motion_score=m_mean,
                stability_score=stability,
                motion_class=m_cls,
                source_class=src_cls,
                saliency_center_x=float(np.mean(saliency_xs[start_i:end_i])),
                saliency_center_y=float(np.mean(saliency_ys[start_i:end_i])),
                edge_density=float(np.mean(edge_densities[start_i:end_i])),
            )
        )

        # Protect intervals with very aggressive existing motion/shake
        if m_mean >= 0.48:
            protected_intervals.append(
                ProtectedInterval(
                    start_frame=sf,
                    end_frame_exclusive=ef,
                    reason="high_existing_motion",
                    severity=min(1.0, m_mean),
                )
            )

    # --- B5: Temporal smoothing on subject regions (exponential moving average) ---
    smoothed_subjects: list[SubjectRegion] = []
    if raw_subjects:
        alpha = 0.35
        sx, sy, sw, sh = raw_subjects[0].x, raw_subjects[0].y, raw_subjects[0].w, raw_subjects[0].h
        for sr in raw_subjects:
            sx = alpha * sr.x + (1.0 - alpha) * sx
            sy = alpha * sr.y + (1.0 - alpha) * sy
            sw = alpha * sr.w + (1.0 - alpha) * sw
            sh = alpha * sr.h + (1.0 - alpha) * sh
            smoothed_subjects.append(
                SubjectRegion(
                    frame=sr.frame,
                    x=float(np.clip(sx, 0.0, 1.0 - sw)),
                    y=float(np.clip(sy, 0.0, 1.0 - sh)),
                    w=float(np.clip(sw, 0.02, 1.0)),
                    h=float(np.clip(sh, 0.02, 1.0)),
                    confidence=sr.confidence,
                    multi_subject=sr.multi_subject,
                    detector=sr.detector,
                )
            )

    # --- B6: Safe zones & signal summary ---
    letterbox = (letterbox_votes / max(1, n_samples)) > 0.65
    mean_edge = float(np.mean(edge_densities))
    mean_border_edge = float(np.mean(border_edge_ratios))
    mean_motion = float(np.mean(motion_scores[1:])) if n_samples > 1 else 0.0
    static_ratio = float(np.mean([1.0 if m < 0.08 else 0.0 for m in motion_scores[1:]])) if n_samples > 1 else 1.0

    # Aggregate subject coverage & confidence
    expected_subj_samples = max(1, (n_samples + subject_stride - 1) // subject_stride)
    subject_coverage = len(smoothed_subjects) / expected_subj_samples
    mean_subj_conf = float(np.mean([s.confidence for s in smoothed_subjects])) if smoothed_subjects else 0.0
    multi_subject_ratio = (
        float(np.mean([1.0 if s.multi_subject else 0.0 for s in smoothed_subjects]))
        if smoothed_subjects
        else 0.0
    )

    face_box = None
    if smoothed_subjects and subject_coverage >= 0.25 and mean_subj_conf >= 0.45:
        face_box = {
            "x": round(float(np.median([s.x for s in smoothed_subjects])), 4),
            "y": round(float(np.median([s.y for s in smoothed_subjects])), 4),
            "w": round(float(np.median([s.w for s in smoothed_subjects])), 4),
            "h": round(float(np.median([s.h for s in smoothed_subjects])), 4),
            "cx": round(float(np.median([s.center_x for s in smoothed_subjects])), 4),
            "cy": round(float(np.median([s.center_y for s in smoothed_subjects])), 4),
            "coverage": round(subject_coverage, 4),
            "confidence": round(mean_subj_conf, 4),
            "multi_subject": multi_subject_ratio >= 0.3,
        }

    # Heuristic UI/HUD edge protection when borders carry dense crisp edges (screen recordings/games)
    busy_ui_edges = mean_border_edge >= 0.14 or (mean_edge >= 0.16 and static_ratio >= 0.65)
    quality_flags: list[str] = []
    if letterbox:
        quality_flags.append("letterbox_detected")
    if mean_motion >= 0.35:
        quality_flags.append("high_motion_footage")
    if static_ratio >= 0.78:
        quality_flags.append("mostly_static_footage")
    if busy_ui_edges:
        quality_flags.append("dense_ui_or_text_edges")
    if len(protected_intervals) >= max(3, int(media.duration_s / 3.0)):
        quality_flags.append("heavily_edited_or_flashing_source")
    if multi_subject_ratio >= 0.3:
        quality_flags.append("multi_subject_ambiguity")

    safe_zones = {
        "has_face_detector": has_face_detector,
        "face_box": face_box,
        "letterbox": letterbox,
        "busy_ui_edges": busy_ui_edges,
        "peripheral_overlay_margin": 0.05,
        "safe_anchor_x": face_box["cx"] if (face_box and not face_box["multi_subject"]) else 0.5,
        "safe_anchor_y": face_box["cy"] if (face_box and not face_box["multi_subject"]) else 0.5,
    }

    signal_summary = {
        "n_video_samples": n_samples,
        "sample_fps": round(sample_fps, 4),
        "cut_count": len(cuts),
        "cuts_per_minute": round(len(cuts) / max(media.duration_s / 60.0, 1e-3), 3),
        "mean_motion": round(mean_motion, 4),
        "static_ratio": round(static_ratio, 4),
        "mean_edge_density": round(mean_edge, 4),
        "mean_border_edge_density": round(mean_border_edge, 4),
        "subject_coverage": round(subject_coverage, 4),
        "mean_subject_confidence": round(mean_subj_conf, 4),
        "multi_subject_ratio": round(multi_subject_ratio, 4),
        "mean_luma": round(float(np.mean(luma_means)), 4),
    }

    return {
        "cuts": cuts,
        "motion_windows": motion_windows,
        "subject_regions": smoothed_subjects,
        "protected_intervals": protected_intervals,
        "safe_zones": safe_zones,
        "quality_flags": quality_flags,
        "signal_summary": signal_summary,
        "has_face_detector": has_face_detector,
    }


def _analyze_audio_stream(
    media: MediaInfo,
    cfg: AnalysisConfig,
    ffmpeg_bin: str,
    audio_track: int | None,
    work_dir: Path,
) -> dict[str, Any]:
    """Phase B4: Bounded-chunk mono PCM audio analysis (RMS, peaks, spectral-flux onsets, pauses, clipping).

    Never alters or replaces original audio master. Treats acoustic energy purely as signal cues, NOT speech meaning.
    """
    if not media.has_audio:
        return {
            "audio_envelope": [],
            "energy_peaks": [],
            "pauses": [],
            "quality_flags": ["silent_source_no_audio_stream"],
            "audio_summary": {"has_audio": False},
        }

    selected_idx = audio_track if audio_track is not None else 0
    if selected_idx < 0 or selected_idx >= len(media.audio_streams):
        raise ValueError(
            f"Requested --audio-track {selected_idx} out of range (source has {len(media.audio_streams)} audio stream(s))"
        )

    work_dir.mkdir(parents=True, exist_ok=True)
    tmp_wav = work_dir / f"_analysis_audio_t{selected_idx}.wav"
    try:
        cmd = [
            ffmpeg_bin,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(media.source_path),
            "-map",
            f"0:a:{selected_idx}",
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(cfg.audio_sample_rate),
            "-c:a",
            "pcm_s16le",
            str(tmp_wav),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if proc.returncode != 0 or not tmp_wav.exists() or tmp_wav.stat().st_size <= 44:
            return {
                "audio_envelope": [],
                "energy_peaks": [],
                "pauses": [],
                "quality_flags": ["audio_decode_empty"],
                "audio_summary": {"has_audio": True, "decoded": False},
            }

        hop_samples = max(160, int(round(cfg.audio_sample_rate * (cfg.audio_hop_ms / 1000.0))))
        raw_rms: list[float] = []
        raw_peak: list[float] = []
        raw_flux: list[float] = []
        clipped_samples = 0
        total_samples = 0
        prev_mag_spec: np.ndarray | None = None

        # Read in bounded chunks (never allocate huge RAM for long recordings)
        chunk_hops = 200
        read_frames = hop_samples * chunk_hops
        window_fn = np.hanning(hop_samples).astype(np.float32)

        with wave.open(str(tmp_wav), "rb") as wf:
            while True:
                data = wf.readframes(read_frames)
                if not data:
                    break
                pcm_i16 = np.frombuffer(data, dtype=np.int16)
                if pcm_i16.size == 0:
                    break
                total_samples += int(pcm_i16.size)
                clipped_samples += int(np.sum(np.abs(pcm_i16) >= 32760))
                pcm = pcm_i16.astype(np.float32) / 32768.0

                n_hops = pcm.size // hop_samples
                for h_i in range(n_hops):
                    seg = pcm[h_i * hop_samples : (h_i + 1) * hop_samples]
                    rms = float(np.sqrt(np.mean(seg * seg)))
                    pk = float(np.max(np.abs(seg)))
                    raw_rms.append(rms)
                    raw_peak.append(pk)

                    # Lightweight half-wave rectified spectral flux for acoustic onset cues
                    spec = np.abs(np.fft.rfft(seg * window_fn))
                    if prev_mag_spec is not None and prev_mag_spec.shape == spec.shape:
                        diff = spec - prev_mag_spec
                        flux = float(np.mean(np.maximum(diff, 0.0)))
                    else:
                        flux = 0.0
                    raw_flux.append(flux)
                    prev_mag_spec = spec

        if not raw_rms:
            return {
                "audio_envelope": [],
                "energy_peaks": [],
                "pauses": [],
                "quality_flags": ["audio_silent"],
                "audio_summary": {"has_audio": True, "decoded": True, "silent": True},
            }

        rms_arr = np.array(raw_rms, dtype=np.float64)
        peak_arr = np.array(raw_peak, dtype=np.float64)
        flux_arr = np.array(raw_flux, dtype=np.float64)

        # Robust percentile normalization (noise floor = p10, ceiling = p95)
        p10 = float(np.percentile(rms_arr, 10))
        p95 = float(np.percentile(rms_arr, 95))
        p_max = float(np.max(rms_arr))
        denom = max(1e-5, p95 - p10)
        is_effectively_silent = p_max < 1e-4

        quality_flags: list[str] = []
        if is_effectively_silent:
            quality_flags.append("audio_effectively_silent")
        if total_samples > 0 and (clipped_samples / total_samples) > 0.002:
            quality_flags.append("audio_clipping_detected")

        hop_s = cfg.audio_hop_ms / 1000.0
        envelope: list[AudioEnvelopePoint] = []
        for i in range(len(raw_rms)):
            t_s = i * hop_s
            frame_idx = min(media.nb_frames - 1, max(0, media.seconds_to_frame(t_s)))
            norm_rms = 0.0 if is_effectively_silent else float(np.clip((rms_arr[i] - p10) / denom, 0.0, 1.0))
            db_val = 20.0 * math.log10(max(rms_arr[i], 1e-6))
            envelope.append(
                AudioEnvelopePoint(
                    frame=frame_idx,
                    time_s=t_s,
                    rms_norm=norm_rms,
                    peak_norm=float(np.clip(peak_arr[i], 0.0, 1.0)),
                    db=db_val,
                )
            )

        # Acoustic onset / energy peaks with minimum distance (>= 1.2s) so we never trigger on every syllable
        energy_peaks: list[EnergyPeak] = []
        if not is_effectively_silent and len(envelope) >= 5:
            flux_p50 = float(np.median(flux_arr))
            flux_p95 = float(np.percentile(flux_arr, 95))
            flux_scale = max(1e-6, flux_p95 - flux_p50)
            norm_flux = np.clip((flux_arr - flux_p50) / flux_scale, 0.0, 1.0)
            norm_rms_arr = np.array([e.rms_norm for e in envelope], dtype=np.float64)
            combined = 0.55 * norm_rms_arr + 0.45 * norm_flux

            min_dist_hops = max(1, int(round(1.2 / hop_s)))
            threshold = max(0.55, float(np.percentile(combined, 80)))
            last_peak_i = -min_dist_hops

            for i in range(1, len(combined) - 1):
                val = float(combined[i])
                if val < threshold:
                    continue
                if val < combined[i - 1] or val < combined[i + 1]:
                    continue
                if i - last_peak_i < min_dist_hops:
                    continue
                t_s = i * hop_s
                frame_idx = min(media.nb_frames - 1, max(0, media.seconds_to_frame(t_s)))
                energy_peaks.append(
                    EnergyPeak(
                        frame=frame_idx,
                        time_s=t_s,
                        strength=float(norm_rms_arr[i]),
                        onset_flux=float(norm_flux[i]),
                        confidence=float(np.clip(val, 0.0, 1.0)),
                    )
                )
                last_peak_i = i

        # Pause / low-energy windows (>= 0.45s below noise-floor margin). Informational ONLY — never cut audio!
        pauses: list[PauseWindow] = []
        silence_thresh = max(p10 * 1.6, 0.006)
        in_pause = False
        pause_start_i = 0
        min_pause_hops = max(2, int(round(0.45 / hop_s)))
        for i, r_val in enumerate(rms_arr):
            if r_val <= silence_thresh:
                if not in_pause:
                    in_pause = True
                    pause_start_i = i
            else:
                if in_pause:
                    if (i - pause_start_i) >= min_pause_hops:
                        sf = min(media.nb_frames - 1, media.seconds_to_frame(pause_start_i * hop_s))
                        ef = min(media.nb_frames, media.seconds_to_frame(i * hop_s))
                        if ef > sf:
                            pauses.append(
                                PauseWindow(
                                    start_frame=sf,
                                    end_frame_exclusive=ef,
                                    duration_s=(i - pause_start_i) * hop_s,
                                    mean_db=float(20.0 * math.log10(max(float(np.mean(rms_arr[pause_start_i:i])), 1e-6))),
                                )
                            )
                    in_pause = False
        if in_pause and (len(rms_arr) - pause_start_i) >= min_pause_hops:
            sf = min(media.nb_frames - 1, media.seconds_to_frame(pause_start_i * hop_s))
            ef = media.nb_frames
            if ef > sf:
                pauses.append(
                    PauseWindow(
                        start_frame=sf,
                        end_frame_exclusive=ef,
                        duration_s=(len(rms_arr) - pause_start_i) * hop_s,
                        mean_db=float(20.0 * math.log10(max(float(np.mean(rms_arr[pause_start_i:])), 1e-6))),
                    )
                )

        audio_summary = {
            "has_audio": True,
            "decoded": True,
            "selected_audio_track": selected_idx,
            "silent": is_effectively_silent,
            "peak_dbfs": round(20.0 * math.log10(max(float(np.max(peak_arr)), 1e-6)), 2),
            "rms_p10_db": round(20.0 * math.log10(max(p10, 1e-6)), 2),
            "rms_p95_db": round(20.0 * math.log10(max(p95, 1e-6)), 2),
            "energy_peak_count": len(energy_peaks),
            "pause_count": len(pauses),
            "clipped_ratio": round(clipped_samples / max(1, total_samples), 6),
        }

        return {
            "audio_envelope": envelope,
            "energy_peaks": energy_peaks,
            "pauses": pauses,
            "quality_flags": quality_flags,
            "audio_summary": audio_summary,
        }
    finally:
        tmp_wav.unlink(missing_ok=True)


async def analyze_media_async(
    media: MediaInfo,
    edit_config: EditConfig,
    work_dir: Path,
    budget: ResourceBudget | None = None,
    ffmpeg_bin: str | None = None,
) -> tuple[AnalysisResult, bool]:
    """Run local video + audio analysis concurrently under ResourceBudget, using LocalAnalysisCache if enabled.

    Returns (AnalysisResult, cache_hit).
    """
    cfg = edit_config.analysis
    cfg.validate()
    ffmpeg = ffmpeg_bin or find_binary("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found on PATH")

    cache: LocalAnalysisCache | None = None
    cache_key: str | None = None
    if cfg.use_cache:
        cache_dir = edit_config.cache_dir or (work_dir.parent / ".cache")
        cache = LocalAnalysisCache(cache_dir)
        cache_key = LocalAnalysisCache.build_key(
            source_hash=media.input_hash,
            analysis_config_hash=cfg.canonical_hash(),
            audio_track=edit_config.audio_track,
            dependency_signature=_dependency_signature(),
            analyzer_version=ANALYZER_VERSION,
        )
        cached = cache.get(cache_key)
        if cached is not None:
            return AnalysisResult.from_dict(cached), True

    workers = budget.snapshot.analysis_workers if (budget and budget.snapshot) else 2
    ff_threads = budget.snapshot.ffmpeg_threads if (budget and budget.snapshot) else 2

    tasks = [
        ("video", lambda: _analyze_video_stream(media, cfg, ffmpeg, ffmpeg_threads=ff_threads)),
        ("audio", lambda: _analyze_audio_stream(media, cfg, ffmpeg, edit_config.audio_track, work_dir)),
    ]
    results = await run_bounded_async(tasks, max_concurrency=min(workers, 2))
    v_res = results["video"]
    a_res = results["audio"]

    capabilities: list[str] = ["video_stream"]
    if media.has_audio and not a_res["audio_summary"].get("silent", False):
        capabilities.append("audio")
    if v_res["cuts"]:
        capabilities.append("scene_cut")
    face_box = v_res["safe_zones"].get("face_box")
    if face_box and face_box.get("confidence", 0.0) >= 0.45 and not face_box.get("multi_subject", False):
        capabilities.append("confident_subject")
    if not v_res["safe_zones"].get("busy_ui_edges", False):
        capabilities.append("clean_edges")

    quality_flags = sorted(set(v_res["quality_flags"] + a_res["quality_flags"] + media.warnings))
    signal_summary = {
        **v_res["signal_summary"],
        "audio": a_res["audio_summary"],
    }

    analysis = AnalysisResult(
        source_hash=media.input_hash,
        config_hash=cfg.canonical_hash(),
        analyzer_version=ANALYZER_VERSION,
        timebase=f"{media.time_base_num}/{media.time_base_den}",
        fps_num=media.fps_num,
        fps_den=media.fps_den,
        duration_frames=media.nb_frames,
        cuts=v_res["cuts"],
        motion_windows=v_res["motion_windows"],
        audio_envelope=a_res["audio_envelope"],
        energy_peaks=a_res["energy_peaks"],
        pauses=a_res["pauses"],
        subject_regions=v_res["subject_regions"],
        protected_intervals=v_res["protected_intervals"],
        safe_zones=v_res["safe_zones"],
        quality_flags=quality_flags,
        capabilities=sorted(set(capabilities)),
        signal_summary=signal_summary,
        candidates=[],
    )
    analysis.candidates = build_candidates(analysis, media, max_per_minute=cfg.max_candidates_per_minute)

    if cache is not None and cache_key is not None:
        cache.put(cache_key, analysis.to_dict())

    if budget:
        budget.sample_metrics()
    return analysis, False


def analyze_media(
    media: MediaInfo,
    edit_config: EditConfig,
    work_dir: Path,
    budget: ResourceBudget | None = None,
    ffmpeg_bin: str | None = None,
) -> tuple[AnalysisResult, bool]:
    """Synchronous convenience wrapper around `analyze_media_async`."""
    return asyncio.run(
        analyze_media_async(
            media=media,
            edit_config=edit_config,
            work_dir=work_dir,
            budget=budget,
            ffmpeg_bin=ffmpeg_bin,
        )
    )
