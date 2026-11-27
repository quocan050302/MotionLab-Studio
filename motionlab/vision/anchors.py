"""Canonical Normalized Anchor Tracks, Matte Visual Debug Sheets & Vision Pipeline Coordinator."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from motionlab.config import MotionLabConfig
from motionlab.resource import ResourceBudget
from motionlab.schemas import AnalysisResult, MediaInfo, atomic_write_json
from motionlab.vision.segmentation import inspect_vision_model, segment_subject_sequence
from motionlab.vision.tracking import track_action_rois


def _sample_track_at(keyframes: list[dict[str, Any]], frame: int, key: str) -> dict[str, float]:
    if not keyframes:
        return {"cx": 0.5, "cy": 0.5, "w": 0.3, "h": 0.25, "conf": 0.0}
    if frame <= keyframes[0]["frame"]:
        return dict(keyframes[0][key])
    if frame >= keyframes[-1]["frame"]:
        return dict(keyframes[-1][key])
    # Binary search surrounding keyframes for exact linear interpolation
    lo, hi = 0, len(keyframes) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if keyframes[mid]["frame"] <= frame:
            lo = mid
        else:
            hi = mid
    a, b = keyframes[lo], keyframes[hi]
    span = max(1, b["frame"] - a["frame"])
    t = (frame - a["frame"]) / span
    fa, fb = a[key], b[key]
    return {
        "cx": round(fa["cx"] + (fb["cx"] - fa["cx"]) * t, 4),
        "cy": round(fa["cy"] + (fb["cy"] - fa["cy"]) * t, 4),
        "w": round(fa["w"] + (fb["w"] - fa["w"]) * t, 4),
        "h": round(fa["h"] + (fb["h"] - fa["h"]) * t, 4),
        "conf": round(fa["conf"] + (fb["conf"] - fa["conf"]) * t, 4),
    }


def build_anchor_tracks(
    media: MediaInfo,
    analysis: AnalysisResult,
    roi_tracks: dict[str, Any],
    matte_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute canonical normalized anchor tracks, safe negative-space slots, and protected bounds."""
    kfs = roi_tracks.get("keyframes") or []
    summary = roi_tracks.get("summary") or {}
    med_face = summary.get("median_face_center") or [0.5, 0.34]
    med_task = summary.get("median_task_center") or [0.5, 0.68]

    # Compute face bounding envelope so peripheral detail apertures NEVER collide with speaker face/hairline
    face_boxes = [k["face"] for k in kfs if k.get("face", {}).get("w", 0.0) > 0.01]
    if face_boxes:
        face_top = float(np.median([f["cy"] - f["h"] * 0.5 for f in face_boxes]))
    else:
        face_top = max(0.24, float(med_face[1]) - 0.10)

    # Choose open peripheral headroom slot guaranteed clear of speaker face & hairline!
    open_side = "left" if med_face[0] >= 0.51 else "right"
    slot_h = 0.165
    # Place capsule strictly above face_top with breathing margin so it NEVER touches hair/forehead
    safe_cy = round(max(0.105, min(0.125, face_top - slot_h * 0.52)), 4)
    detail_slot = {
        "side": open_side,
        "cx": 0.25 if open_side == "left" else 0.75,
        "cy": safe_cy,
        "w": 0.41,
        "h": slot_h,
    }
    alt_detail_slot = {
        "side": "right" if open_side == "left" else "left",
        "cx": 0.75 if open_side == "left" else 0.25,
        "cy": safe_cy,
        "w": 0.41,
        "h": slot_h,
    }
    behind_stage = {
        "cx": round(float(med_face[0]), 4),
        "cy": round(max(0.28, min(0.46, float(med_face[1]) + 0.05)), 4),
        "radius_w": 0.44,
        "radius_h": 0.28,
    }

    return {
        "source_hash": media.input_hash,
        "width": media.display_width,
        "height": media.display_height,
        "fps": media.fps.to_ffmpeg(),
        "duration_frames": media.duration_frames,
        "has_face_track": bool(roi_tracks.get("has_face_track")),
        "has_task_track": bool(roi_tracks.get("has_task_track")),
        "matte_available": bool(matte_result and matte_result.get("available") and matte_result.get("status") == "READY"),
        "tabletop_guard_y": roi_tracks.get("tabletop_guard_y"),
        "palette": analysis.dominant_palette or {
            "tone": "warm_editorial_bright",
            "primary_hex": "#C8A26A",
            "secondary_hex": "#4A3F35",
            "tertiary_hex": "#F4EBDD",
        },
        "detail_aperture_slot": detail_slot,
        "alt_detail_slot": alt_detail_slot,
        "behind_stage_anchor": behind_stage,
        "keyframes": kfs,
        "action_vectors": roi_tracks.get("action_vectors") or [],
    }


def query_span_anchors(anchors: dict[str, Any], start_frame: int, end_frame_exclusive: int) -> dict[str, Any]:
    """Extract compact normalized trajectory points and best action vector for a specific scene span."""
    kfs = anchors.get("keyframes") or []
    span_len = max(1, end_frame_exclusive - start_frame)
    # Sample 9 evenly spaced points across the span for crisp JS/GSAP spline interpolation
    pts_face: list[dict[str, float]] = []
    pts_task: list[dict[str, float]] = []
    for k in range(9):
        f = min(end_frame_exclusive - 1, start_frame + int(round((k / 8.0) * max(1, span_len - 1))))
        fc = _sample_track_at(kfs, f, "face")
        tk = _sample_track_at(kfs, f, "task")
        pts_face.append({"t": round(k / 8.0, 4), **fc})
        pts_task.append({"t": round(k / 8.0, 4), **tk})

    # Find highest-confidence overlapping action vector
    vecs = [
        v for v in (anchors.get("action_vectors") or [])
        if v["start_frame"] < end_frame_exclusive and v["end_frame_exclusive"] > start_frame
    ]
    if vecs:
        best_v = max(vecs, key=lambda v: (v["confidence"], v["mean_energy"]))
    else:
        f0, f1 = pts_task[1], pts_task[-2]
        best_v = {
            "origin_x": f0["cx"], "origin_y": f0["cy"],
            "dest_x": f1["cx"] if abs(f1["cx"] - f0["cx"]) > 0.04 else min(0.82, max(0.18, f0["cx"] + 0.14)),
            "dest_y": f1["cy"],
            "confidence": round((f0["conf"] + f1["conf"]) * 0.5, 3),
        }

    return {
        "start_frame": start_frame,
        "end_frame_exclusive": end_frame_exclusive,
        "face_mid": pts_face[4],
        "task_mid": pts_task[4],
        "face_path": pts_face,
        "task_path": pts_task,
        "action_vector": best_v,
        "detail_slot": anchors.get("detail_aperture_slot") or {"cx": 0.27, "cy": 0.19, "w": 0.42, "h": 0.26, "side": "left"},
        "alt_detail_slot": anchors.get("alt_detail_slot") or {"cx": 0.72, "cy": 0.21, "w": 0.40, "h": 0.24, "side": "right"},
        "behind_stage": anchors.get("behind_stage_anchor") or {"cx": 0.5, "cy": 0.38, "radius_w": 0.44, "radius_h": 0.28},
    }


def generate_matte_debug_sheet(
    source_path: Path,
    matte_result: dict[str, Any],
    anchors: dict[str, Any],
    out_jpg: Path,
) -> str | None:
    """Create a compact visual QA sheet showing source+ROIs, raw MODNet alpha, and occlusion composite."""
    spans = matte_result.get("spans") or []
    if not spans:
        return None
    out_jpg.parent.mkdir(parents=True, exist_ok=True)
    src_cap = cv2.VideoCapture(str(source_path))
    rows: list[np.ndarray] = []
    try:
        for sp in spans[:4]:
            mp = Path(sp["matte_mp4"])
            rp = Path(sp.get("raw_matte_mp4", sp["matte_mp4"]))
            if not mp.is_file():
                continue
            mid_local = max(0, sp["frame_count"] // 2)
            mid_global = sp["start_frame"] + mid_local

            src_cap.set(cv2.CAP_PROP_POS_FRAMES, mid_global)
            ok_s, bgr = src_cap.read()
            m_cap = cv2.VideoCapture(str(mp))
            m_cap.set(cv2.CAP_PROP_POS_FRAMES, mid_local)
            ok_m, m_bgr = m_cap.read()
            m_cap.release()
            r_cap = cv2.VideoCapture(str(rp))
            r_cap.set(cv2.CAP_PROP_POS_FRAMES, mid_local)
            ok_r, r_bgr = r_cap.read()
            r_cap.release()
            if not (ok_s and ok_m and ok_r):
                continue

            h, w = bgr.shape[:2]
            src_vis = bgr.copy()
            kfs = anchors.get("keyframes") or []
            fc = _sample_track_at(kfs, mid_global, "face")
            tk = _sample_track_at(kfs, mid_global, "task")
            # Draw green face ROI & amber task ROI
            if fc["w"] > 0.01:
                x0, y0 = int((fc["cx"] - fc["w"] * 0.5) * w), int((fc["cy"] - fc["h"] * 0.5) * h)
                x1, y1 = int((fc["cx"] + fc["w"] * 0.5) * w), int((fc["cy"] + fc["h"] * 0.5) * h)
                cv2.rectangle(src_vis, (x0, y0), (x1, y1), (80, 220, 120), 3)
            if tk["conf"] > 0.1:
                x0, y0 = int((tk["cx"] - tk["w"] * 0.5) * w), int((tk["cy"] - tk["h"] * 0.5) * h)
                x1, y1 = int((tk["cx"] + tk["w"] * 0.5) * w), int((tk["cy"] + tk["h"] * 0.5) * h)
                cv2.rectangle(src_vis, (x0, y0), (x1, y1), (60, 180, 245), 3)

            # Composite occlusion test: warm antique-gold arch strictly BEHIND composite matte!
            alpha = cv2.cvtColor(m_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            bg = bgr.copy()
            bg = cv2.GaussianBlur(bg, (25, 25), 8.0)
            cv2.circle(bg, (int(w * fc["cx"]), int(h * (fc["cy"] + 0.06))), int(min(w, h) * 0.42), (106, 162, 200), -1)
            a3 = alpha[..., None]
            comp = np.clip(bgr.astype(np.float32) * a3 + bg.astype(np.float32) * (1.0 - a3), 0, 255).astype(np.uint8)

            tw = 240
            th = max(120, int(round(h * (tw / max(1, w)))))
            lbl = f"f{mid_global} | cov={sp['mean_coverage']:.2f} iou={sp['mean_temporal_iou']:.3f}"
            p1 = cv2.resize(src_vis, (tw, th))
            p2 = cv2.resize(r_bgr, (tw, th))
            p3 = cv2.resize(m_bgr, (tw, th))
            p4 = cv2.resize(comp, (tw, th))
            cv2.putText(p1, lbl, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            rows.append(np.hstack([p1, p2, p3, p4]))
    finally:
        src_cap.release()

    if not rows:
        return None
    sheet = np.vstack(rows)
    cv2.imwrite(str(out_jpg), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return str(out_jpg)


def run_vision_pipeline(
    media: MediaInfo,
    analysis: AnalysisResult,
    config: MotionLabConfig,
    run_dir: Path,
    budget: ResourceBudget | None = None,
    matte_spans: list[tuple[int, int]] | None = None,
    cache_dir: Path | None = None,
    ffmpeg_bin: str | None = None,
) -> dict[str, Any]:
    """Coordinate Phase 1 ROI tracking + optional MODNet segmentation + anchor tracks + evidence JSONs."""
    ev_dir = run_dir / "evidence"
    ev_dir.mkdir(parents=True, exist_ok=True)

    roi_tracks = track_action_rois(media, analysis, config, ffmpeg_bin=ffmpeg_bin)
    model_info = inspect_vision_model()
    matte_res: dict[str, Any] = {
        "available": False,
        "status": "SKIPPED_NO_SPANS",
        "model": model_info,
        "spans": [],
        "quality_summary": {},
    }
    if matte_spans:
        matte_res = segment_subject_sequence(
            media=media,
            spans=matte_spans,
            workdir=run_dir / "temp",
            cache_dir=cache_dir,
            preserve_tabletop_y=roi_tracks.get("tabletop_guard_y"),
            keyframe_stride=2,
            budget=budget,
            ffmpeg_bin=ffmpeg_bin,
        )

    anchors = build_anchor_tracks(media, analysis, roi_tracks, matte_res)
    debug_sheet = generate_matte_debug_sheet(
        Path(media.path), matte_res, anchors, ev_dir / "matte_debug_sheet.jpg"
    )

    vision_manifest = {
        "schema_version": "1.0.0",
        "source_hash": media.input_hash,
        "model": model_info,
        "matte_status": matte_res.get("status"),
        "matte_available": bool(anchors["matte_available"]),
        "matte_cache_hit": bool(matte_res.get("cache_hit", False)),
        "matte_elapsed_s": float(matte_res.get("elapsed_s", 0.0)),
        "roi_summary": roi_tracks.get("summary", {}),
        "tabletop_guard_y": roi_tracks.get("tabletop_guard_y"),
        "debug_sheet": str(Path(debug_sheet).relative_to(run_dir)) if debug_sheet else None,
        "matte_spans": [
            {
                "span_index": s["span_index"],
                "start_frame": s["start_frame"],
                "end_frame_exclusive": s["end_frame_exclusive"],
                "frame_count": s["frame_count"],
                "matte_sha256": s["matte_sha256"],
                "raw_matte_sha256": s["raw_matte_sha256"],
                "quality_ok": s["quality_ok"],
            }
            for s in (matte_res.get("spans") or [])
        ],
    }

    atomic_write_json(ev_dir / "vision_manifest.json", vision_manifest)
    atomic_write_json(ev_dir / "roi_tracks.json", roi_tracks)
    atomic_write_json(ev_dir / "anchors.json", anchors)
    atomic_write_json(ev_dir / "mask_quality.json", {
        "status": matte_res.get("status"),
        "quality_summary": matte_res.get("quality_summary", {}),
        "spans": matte_res.get("spans", []),
    })

    return {
        "vision_manifest": vision_manifest,
        "roi_tracks": roi_tracks,
        "anchors": anchors,
        "matte_result": matte_res,
    }
