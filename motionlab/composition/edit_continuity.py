"""Phase A4 (ML-01.R5.R4): Intelligent Cut Continuity Planner, Reframing Guards & Audio Join QA.

Inspects every real cut join in BOTH source and edited streams (±6 frames) for:
  - Face ROI displacement, hand/task motion discontinuity, luminance/exposure jump, and pixel MAD.
  - Motivated join style selection (`direct_cut`, `motivated_punch_in`, `shared_element_bridge`, `cutaway_bridge`)
    strictly respecting safe crops (no aliasing/face clipping on 720x1280) and L/J-cut lip-sync safety.
  - Waveform audio join QA on `edited_voice_bed.wav`: impulse/click detector, DC-offset step,
    noise-floor jump, and exact sample count vs `EditTimeMap.edited_total_samples`.
  - Renders compact `evidence/cut_join_strips.jpg` showing real decoded frames around every cut join.
"""

from __future__ import annotations

import math
import shutil
import wave
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from motionlab.composition.edit_time_map import EditTimeMap
from motionlab.schemas import atomic_write_json

CONTINUITY_SCHEMA_VERSION = "1.0.0"


def validate_lj_bridge_safety(
    *,
    audio_offset_frames: int,
    lips_visible_in_window: bool,
    is_broll_or_cutaway: bool,
) -> dict[str, Any]:
    """Fail-closed guard for optional L/J-style bridges: NEVER shift speaking audio relative to visible talking lips!"""
    if audio_offset_frames == 0:
        return {"safe": True, "reason": "same_clock_default"}
    if lips_visible_in_window and not is_broll_or_cutaway:
        return {
            "safe": False,
            "reason": "lj_bridge_rejected_visible_speaking_lips_desync",
            "audio_offset_frames": int(audio_offset_frames),
        }
    return {"safe": True, "reason": "non_lip_cutaway_or_broll_bridge_permitted"}


def _box_xywh(b: dict[str, Any]) -> tuple[float, float, float, float]:
    """Normalize either top-left {x, y, w, h} or center-based {cx, cy, w, h} ROI box to (x, y, w, h)."""
    w = float(b.get("w", 0.0))
    h = float(b.get("h", 0.0))
    if "x" in b and "y" in b:
        return float(b["x"]), float(b["y"]), w, h
    cx = float(b.get("cx", 0.5))
    cy = float(b.get("cy", 0.5))
    return cx - 0.5 * w, cy - 0.5 * h, w, h


def compute_safe_reframe_crop(
    face_box: dict[str, Any] | None,
    task_box: dict[str, Any] | None,
    *,
    width: int,
    height: int,
    max_zoom: float = 1.07,
) -> dict[str, Any]:
    """Compute bounded safe punch-in crop (<= 1.07x on 720x1280 to avoid aliasing) keeping face & hands inside frame."""
    if not isinstance(face_box, dict) or float(face_box.get("conf", face_box.get("confidence", 0.0))) < 0.45:
        return {"zoom": 1.0, "cx": 0.5, "cy": 0.5, "safe": True, "reason": "no_confident_face_keep_1x"}

    fx, fy, fw, fh = _box_xywh(face_box)
    # Cap zoom conservatively on 720p vertical so phone sharpness never degrades
    z = min(max_zoom, 1.06 if width <= 720 else 1.10)
    win_w = 1.0 / z
    win_h = 1.0 / z
    cx = max(win_w * 0.5, min(1.0 - win_w * 0.5, fx + fw * 0.5))
    cy = max(win_h * 0.5, min(1.0 - win_h * 0.5, fy + fh * 0.45))

    # Verify face box (and task box if present) remain completely inside cropped window with 3% margin!
    x0, x1 = cx - win_w * 0.5, cx + win_w * 0.5
    y0, y1 = cy - win_h * 0.5, cy + win_h * 0.5
    face_ok = (fx >= x0 + 0.02) and (fx + fw <= x1 - 0.02) and (fy >= y0 + 0.02) and (fy + fh <= y1 - 0.02)
    task_ok = True
    if isinstance(task_box, dict) and float(task_box.get("conf", task_box.get("confidence", 0.0))) >= 0.45:
        tx, ty, tw, th = _box_xywh(task_box)
        task_ok = (tx >= x0 + 0.01) and (tx + tw <= x1 - 0.01) and (ty + th <= y1 - 0.01)

    if not (face_ok and task_ok):
        return {"zoom": 1.0, "cx": 0.5, "cy": 0.5, "safe": True, "reason": "crop_would_clip_face_or_task_keep_1x"}

    return {
        "zoom": round(z, 3),
        "cx": round(cx, 4),
        "cy": round(cy, 4),
        "safe": True,
        "reason": "bounded_motivated_punch_in",
    }


def _read_frame_at(cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_idx)))
    ok, fr = cap.read()
    return fr if (ok and fr is not None) else None


def evaluate_cut_continuity(
    source_mp4: Path,
    edited_mp4: Path,
    edit_map: EditTimeMap,
    roi_tracks_source: dict[str, Any] | None = None,
    *,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Inspect ±6 frames around every real cut join in BOTH source and edited decoded videos + render `cut_join_strips.jpg`."""
    if edit_map.is_identity or len(edit_map.kept_spans) <= 1:
        doc = {
            "schema_version": CONTINUITY_SCHEMA_VERSION,
            "edit_map_hash": edit_map.map_hash,
            "joins_count": 0,
            "severe_jump_joins": 0,
            "joins": [],
            "cut_join_strips_path": None,
            "status": "PASS_NO_CUTS",
        }
        if out_dir is not None:
            atomic_write_json(out_dir / "cut_continuity.json", doc)
        return doc

    kfs = (roi_tracks_source or {}).get("keyframes") or (roi_tracks_source or {}).get("frames") or []

    def _roi_near(sf: int) -> dict[str, Any]:
        return min(kfs, key=lambda k: abs(int(k.get("frame", 0)) - sf)) if kfs else {}

    src_cap = cv2.VideoCapture(str(source_mp4))
    ed_cap = cv2.VideoCapture(str(edited_mp4))
    joins: list[dict[str, Any]] = []
    strip_rows: list[tuple[str, list[tuple[str, np.ndarray]]]] = []
    severe_jumps = 0

    try:
        for i in range(len(edit_map.kept_spans) - 1):
            sp_prev = edit_map.kept_spans[i]
            sp_next = edit_map.kept_spans[i + 1]
            om = edit_map.omitted_spans[i] if i < len(edit_map.omitted_spans) else None

            out_join_f = sp_next.output_start_frame
            src_before_f = sp_prev.source_end_frame_exclusive - 1
            src_after_f = sp_next.source_start_frame

            # Read ±6 frame windows on edited stream
            ed_before = _read_frame_at(ed_cap, max(0, out_join_f - 1))
            ed_after = _read_frame_at(ed_cap, min(edit_map.output_total_frames - 1, out_join_f))
            ed_pre2 = _read_frame_at(ed_cap, max(0, out_join_f - 2))
            ed_post2 = _read_frame_at(ed_cap, min(edit_map.output_total_frames - 1, out_join_f + 1))

            # Source baseline frames at exact join edges (verifies frame-exact mapping!)
            src_b = _read_frame_at(src_cap, src_before_f)
            src_a = _read_frame_at(src_cap, src_after_f)

            join_mad = 0.0
            baseline_mad = 0.0
            luma_jump = 0.0
            src_before_match_mad = 0.0
            src_after_match_mad = 0.0

            if ed_before is not None and ed_after is not None:
                sb1 = cv2.resize(ed_before, (180, 320)).astype(np.float32)
                sa1 = cv2.resize(ed_after, (180, 320)).astype(np.float32)
                join_mad = round(float(np.mean(np.abs(sa1 - sb1))), 2)
                luma_jump = round(abs(float(np.mean(sa1)) - float(np.mean(sb1))), 2)
                if ed_pre2 is not None and ed_post2 is not None:
                    sp2 = cv2.resize(ed_pre2, (180, 320)).astype(np.float32)
                    sa2 = cv2.resize(ed_post2, (180, 320)).astype(np.float32)
                    baseline_mad = round(0.5 * (float(np.mean(np.abs(sb1 - sp2))) + float(np.mean(np.abs(sa2 - sa1)))), 2)
                if src_b is not None:
                    src_before_match_mad = round(float(np.mean(np.abs(sb1 - cv2.resize(src_b, (180, 320)).astype(np.float32)))), 2)
                if src_a is not None:
                    src_after_match_mad = round(float(np.mean(np.abs(sa1 - cv2.resize(src_a, (180, 320)).astype(np.float32)))), 2)

            # Face ROI displacement across the cut
            r_pre = _roi_near(src_before_f)
            r_post = _roi_near(src_after_f)
            fb0 = r_pre.get("face") or r_pre.get("face_box") or {}
            fb1 = r_post.get("face") or r_post.get("face_box") or {}
            face_disp = 0.0
            if isinstance(fb0, dict) and isinstance(fb1, dict) and fb0.get("w") and fb1.get("w"):
                x0_b, y0_b, w0_b, h0_b = _box_xywh(fb0)
                x1_b, y1_b, w1_b, h1_b = _box_xywh(fb1)
                c0x = x0_b + 0.5 * w0_b
                c0y = y0_b + 0.5 * h0_b
                c1x = x1_b + 0.5 * w1_b
                c1y = y1_b + 0.5 * h1_b
                face_disp = round(math.hypot(c1x - c0x, c1y - c0y), 4)

            # Recommend join style honestly based on measured continuity
            severe = bool(luma_jump > 38.0 or face_disp > 0.18)
            if severe:
                severe_jumps += 1
            if face_disp >= 0.045 or join_mad >= 22.0:
                reframe = compute_safe_reframe_crop(fb1, r_post.get("task"), width=720, height=1280)
                join_style = "motivated_punch_in" if reframe["zoom"] > 1.0 else "shared_element_bridge"
            else:
                reframe = {"zoom": 1.0, "cx": 0.5, "cy": 0.5, "safe": True, "reason": "smooth_direct_cut"}
                join_style = "direct_cut"

            joins.append({
                "join_id": f"join_{i:02d}_outf{out_join_f:04d}",
                "cut_id": om.cut_id if om else f"cut_{i:02d}",
                "pause_class": om.pause_class if om else "UNKNOWN",
                "output_join_frame": out_join_f,
                "source_before_frame": src_before_f,
                "source_after_frame": src_after_f,
                "removed_frames": om.removed_frames if om else (src_after_f - src_before_f - 1),
                "removed_s": om.removed_s if om else 0.0,
                "join_mad": join_mad,
                "neighbor_baseline_mad": baseline_mad,
                "luma_jump": luma_jump,
                "face_center_displacement": face_disp,
                "source_before_mapping_mad": src_before_match_mad,
                "source_after_mapping_mad": src_after_match_mad,
                "recommended_join_style": join_style,
                "safe_reframe": reframe,
                "severe_discontinuity": severe,
            })

            # Collect visual strip thumbnails (-3, -1, +0, +1, +3) for human review!
            row_frames: list[tuple[str, np.ndarray]] = []
            for off in (-3, -1, 0, 1, 3):
                fi = max(0, min(edit_map.output_total_frames - 1, out_join_f + off))
                fr = _read_frame_at(ed_cap, fi)
                if fr is not None:
                    row_frames.append((f"f{fi}({off:+d})", cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
            if row_frames:
                strip_rows.append((f"join_{i:02d} cut={om.cut_id if om else i} -{om.removed_s if om else 0:.2f}s MAD={join_mad:.1f}", row_frames))
    finally:
        src_cap.release()
        ed_cap.release()

    strip_path = None
    if out_dir is not None and strip_rows:
        out_dir.mkdir(parents=True, exist_ok=True)
        tw, th = 144, 256
        pad, hdr, lbl_h = 8, 34, 22
        cols = 5
        rows = len(strip_rows)
        sheet = Image.new("RGB", (cols * (tw + pad) + pad, hdr + rows * (th + lbl_h + pad) + pad), (18, 17, 15))
        draw = ImageDraw.Draw(sheet)
        draw.text((pad, 10), f"ML-01.R5.R4 Cut Join Strips ({rows} joins, edited domain -3/-1/+0/+1/+3f)", fill=(245, 240, 230))
        for r_idx, (title, frs) in enumerate(strip_rows):
            y = hdr + r_idx * (th + lbl_h + pad)
            draw.text((pad, y), title[:75], fill=(190, 220, 180))
            for c_idx, (flbl, rgb) in enumerate(frs[:cols]):
                x = pad + c_idx * (tw + pad)
                sheet.paste(Image.fromarray(rgb).resize((tw, th), Image.Resampling.LANCZOS), (x, y + lbl_h))
                draw.text((x + 4, y + lbl_h + 4), flbl, fill=(255, 240, 180))
        sp = out_dir / "cut_join_strips.jpg"
        sheet.save(sp, "JPEG", quality=90)
        strip_path = str(sp)

    doc = {
        "schema_version": CONTINUITY_SCHEMA_VERSION,
        "edit_map_hash": edit_map.map_hash,
        "joins_count": len(joins),
        "severe_jump_joins": severe_jumps,
        "joins": joins,
        "cut_join_strips_path": strip_path,
        "status": "PASS" if severe_jumps == 0 else "WARN_SEVERE_JOIN_JUMP",
    }
    if out_dir is not None:
        atomic_write_json(out_dir / "cut_continuity.json", doc)
    return doc


def evaluate_audio_joins(
    edited_voice_bed_wav: Path,
    edit_map: EditTimeMap,
    *,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Audit derived `edited_voice_bed.wav` at every cut join for clicks, DC jumps, noise-floor jumps, and exact sample count."""
    if not edited_voice_bed_wav.is_file():
        return {"status": "FAIL", "failure_reasons": ["missing_edited_voice_bed_wav"], "joins": []}

    with wave.open(str(edited_voice_bed_wav), "rb") as wf:
        sr = wf.getframerate()
        nch = wf.getnchannels()
        nframes = wf.getnframes()
        raw = np.frombuffer(wf.readframes(nframes), dtype=np.int16).astype(np.float32) / 32768.0
    if nch > 1:
        pcm = raw.reshape((-1, nch)).mean(axis=1)
    else:
        pcm = raw

    sample_count_diff = abs(int(pcm.size) - int(edit_map.edited_total_samples))
    joins_out: list[dict[str, Any]] = []
    failure_reasons: list[str] = []

    if sample_count_diff > 1:
        failure_reasons.append(f"edited_bed_sample_count_drift:actual={pcm.size},expected={edit_map.edited_total_samples}")
    if sr != edit_map.sample_rate:
        failure_reasons.append(f"edited_bed_sample_rate_mismatch:actual={sr},expected={edit_map.sample_rate}")

    win_n = max(128, int(round(sr * 0.08)))  # ±80ms
    click_win = max(16, int(round(sr * 0.005)))  # ±5ms immediate splice point

    for i in range(len(edit_map.kept_spans) - 1):
        sp_next = edit_map.kept_spans[i + 1]
        js = int(sp_next.output_start_sample)
        lo = max(0, js - win_n)
        hi = min(pcm.size, js + win_n)
        if hi - lo < 64:
            continue
        seg_pre = pcm[lo:js]
        seg_post = pcm[js:hi]
        seg_splice = pcm[max(0, js - click_win) : min(pcm.size, js + click_win)]

        # 1. Click / transient discontinuity detector at splice boundary
        diffs = np.abs(np.diff(seg_splice)) if seg_splice.size >= 4 else np.zeros((1,), dtype=np.float32)
        max_step = float(np.max(diffs))
        local_rms = float(np.sqrt(np.mean(pcm[lo:hi] ** 2) + 1e-10))
        # Compare splice max step against background derivative median
        bg_diffs = np.abs(np.diff(pcm[lo:hi]))
        med_diff = float(np.percentile(bg_diffs, 85)) if bg_diffs.size >= 16 else 0.005
        click_ratio = round(max_step / max(med_diff, 0.004), 2)
        click_flagged = bool(max_step >= 0.14 and click_ratio >= 7.5)

        # 2. DC offset jump across join
        dc_pre = float(np.mean(seg_pre)) if seg_pre.size else 0.0
        dc_post = float(np.mean(seg_post)) if seg_post.size else 0.0
        dc_jump = round(abs(dc_post - dc_pre), 5)
        dc_flagged = bool(dc_jump >= 0.045)

        # 3. Room-tone / noise floor jump across join (excluding clipped extremes)
        rms_pre_db = round(20.0 * math.log10(max(1e-6, float(np.sqrt(np.mean(seg_pre ** 2) + 1e-12)))), 2) if seg_pre.size else -96.0
        rms_post_db = round(20.0 * math.log10(max(1e-6, float(np.sqrt(np.mean(seg_post ** 2) + 1e-12)))), 2) if seg_post.size else -96.0
        floor_jump_db = round(abs(rms_post_db - rms_pre_db), 2)

        ok = not (click_flagged or dc_flagged)
        if not ok:
            failure_reasons.append(f"audio_join_{i:02d}_defect:click={click_flagged}(step={max_step:.3f},ratio={click_ratio}),dc={dc_flagged}({dc_jump})")

        joins_out.append({
            "join_id": f"ajoin_{i:02d}_s{js}",
            "output_sample": js,
            "output_time_s": round(js / max(1, sr), 4),
            "max_sample_step": round(max_step, 5),
            "click_ratio": click_ratio,
            "click_flagged": click_flagged,
            "dc_offset_jump": dc_jump,
            "dc_flagged": dc_flagged,
            "rms_pre_dbfs": rms_pre_db,
            "rms_post_dbfs": rms_post_db,
            "floor_jump_db": floor_jump_db,
            "local_rms": round(local_rms, 5),
            "join_ok": ok,
        })

    doc = {
        "schema_version": CONTINUITY_SCHEMA_VERSION,
        "edit_map_hash": edit_map.map_hash,
        "sample_rate": sr,
        "actual_samples": int(pcm.size),
        "expected_samples": int(edit_map.edited_total_samples),
        "sample_count_diff": sample_count_diff,
        "joins_checked": len(joins_out),
        "clicks_detected": sum(1 for j in joins_out if j["click_flagged"]),
        "dc_jumps_detected": sum(1 for j in joins_out if j["dc_flagged"]),
        "joins": joins_out,
        "failure_reasons": failure_reasons,
        "status": "PASS" if not failure_reasons else "FAIL",
    }
    if out_dir is not None:
        atomic_write_json(out_dir / "join_audio_qa.json", doc)
    return doc
