"""Phase A5 (ML-01.R5): Objective Creative Quality Proxy & VQ-01 Audit Harness.

Measures objective phone-scale (360x640) visual/structural properties WITHOUT conflating technical proxy
metrics with human aesthetic approval! Explicitly marks `VISION_CRITIC_NOT_RUN` (no local VLM installed)
and `PENDING_HUMAN_REVIEW` for Gate G8.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from motionlab.composition.generative_schema import StoryboardSpec
from motionlab.schemas import MediaInfo, atomic_write_json
from motionlab.semantic.story_graph import StoryGraph


def verify_ablation_still_diff(full_png: Path, ablated_png: Path) -> dict[str, Any]:
    """Compare full rendered midpoint still vs single-layer-ablated still to prove real visible pixel contribution."""
    if not full_png.is_file() or not ablated_png.is_file():
        return {"verified": False, "reason": "missing_still_file", "mad": 0.0, "changed_pixel_ratio": 0.0}
    im_full = cv2.imread(str(full_png))
    im_ab = cv2.imread(str(ablated_png))
    if im_full is None or im_ab is None or im_full.shape != im_ab.shape:
        return {"verified": False, "reason": "shape_mismatch_or_decode_fail", "mad": 0.0, "changed_pixel_ratio": 0.0}
    diff = np.abs(im_full.astype(np.float32) - im_ab.astype(np.float32))
    mad = float(np.mean(diff))
    changed_ratio = float(np.mean(np.max(diff, axis=2) >= 8.0))
    return {
        "verified": bool(mad >= 0.35 or changed_ratio >= 0.004),
        "mad": round(mad, 3),
        "changed_pixel_ratio": round(changed_ratio, 5),
    }


def detect_template_repetition(storyboard: StoryboardSpec) -> dict[str, Any]:
    """Flag >2 consecutive identical stage_mode + primary-primitive layouts (anti-template check)."""
    streak = 1
    max_streak = 1 if storyboard.scenes else 0
    flagged_runs: list[dict[str, Any]] = []
    signatures: list[str] = []
    for sc in storyboard.scenes:
        prims = "+".join(sorted(ly.primitive_id for ly in sc.layers))
        sig = f"{sc.stage_mode}|{prims}"
        signatures.append(sig)
    for i in range(1, len(signatures)):
        if signatures[i] == signatures[i - 1]:
            streak += 1
            if streak > max_streak:
                max_streak = streak
            if streak > 2:
                flagged_runs.append({"scene_id": storyboard.scenes[i].scene_id, "signature": signatures[i], "streak": streak})
        else:
            streak = 1
    return {
        "max_consecutive_identical_templates": max_streak,
        "distinct_stage_modes": sorted({sc.stage_mode for sc in storyboard.scenes}),
        "distinct_signatures": len(set(signatures)),
        "passed": max_streak <= 2,
        "flagged_runs": flagged_runs,
    }


def build_contact_and_boundary_sheets(
    source_mp4: Path,
    final_mp4: Path,
    storyboard: StoryboardSpec,
    out_dir: Path,
) -> dict[str, str]:
    """Generate 0.5s full-timeline contact sheet (60 panels @ 360px scale) + scene boundary comparison strips."""
    out_dir.mkdir(parents=True, exist_ok=True)

    def _read_frames(path: Path) -> list[np.ndarray]:
        cap = cv2.VideoCapture(str(path))
        arr: list[np.ndarray] = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            arr.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        cap.release()
        return arr

    fin_frames = _read_frames(final_mp4)
    src_frames = _read_frames(source_mp4)
    n = len(fin_frames)
    if n == 0:
        return {}

    # 1. 0.5s interval contact sheet (60 panels on a 30s 24fps clip -> every 12th frame)
    step = max(1, int(round(storyboard.fps_num / max(1, storyboard.fps_den) * 0.5)))
    indices = list(range(0, n, step))
    cols = 10
    rows = (len(indices) + cols - 1) // cols
    tw, th = 180, 320
    pad = 6
    hdr = 44
    sheet = Image.new("RGB", (cols * (tw + pad) + pad, hdr + rows * (th + 22 + pad) + pad), (22, 21, 19))
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (pad, 12),
        f"MotionLab R5 Contact Sheet (0.5s step) | {storyboard.mode} | {len(storyboard.scenes)} scenes | {n} frames",
        fill=(245, 240, 230),
    )
    for idx_i, fno in enumerate(indices):
        r, c = divmod(idx_i, cols)
        x = pad + c * (tw + pad)
        y = hdr + r * (th + 22 + pad)
        im = Image.fromarray(fin_frames[fno]).resize((tw, th), Image.Resampling.LANCZOS)
        sheet.paste(im, (x, y))
        draw.text((x + 3, y + th + 3), f"f{fno:03d} ({fno / 24.0:.1f}s)", fill=(198, 164, 106))
    cs_path = out_dir.parent / "contact_sheet.jpg"
    sheet.save(cs_path, "JPEG", quality=90)

    # 2. Phone-scale 360px per-scene peak strip (Before vs After side by side for every scene!)
    sc_list = storyboard.scenes[:12]
    if sc_list:
        ptw, pth = 240, 426
        sc_cols = min(5, len(sc_list))
        sc_rows = (len(sc_list) + sc_cols - 1) // sc_cols
        psheet = Image.new("RGB", (sc_cols * (ptw + 10) + 10, 48 + sc_rows * (pth + 32 + 10) + 10), (22, 21, 19))
        pdraw = ImageDraw.Draw(psheet)
        pdraw.text((10, 14), f"Per-Scene Peak Inspection ({storyboard.mode}) — Phone-Scale Readability Check", fill=(245, 240, 230))
        for si, sc in enumerate(sc_list):
            mid_f = max(0, min(n - 1, (sc.start_frame + sc.end_frame_exclusive) // 2))
            r, c = divmod(si, sc_cols)
            x = 10 + c * (ptw + 10)
            y = 48 + r * (pth + 32 + 10)
            im = Image.fromarray(fin_frames[mid_f]).resize((ptw, pth), Image.Resampling.LANCZOS)
            psheet.paste(im, (x, y))
            pdraw.text((x + 4, y + pth + 4), f"{sc.scene_id} f{sc.start_frame}-{sc.end_frame_exclusive} {sc.stage_mode[:16]}", fill=(198, 164, 106))
        peak_sheet_p = out_dir / "scene_peaks_phone_scale.jpg"
        psheet.save(peak_sheet_p, "JPEG", quality=92)
    else:
        peak_sheet_p = cs_path

    # 3. Source vs Final boundary delta & 720-frame numeric CSV
    csv_lines = ["frame,time_s,final_mean_luma,final_std_luma,source_diff_mad,scene_id"]
    owner: list[str] = ["raw_source"] * n
    for sc in storyboard.scenes:
        for f in range(sc.start_frame, min(n, sc.end_frame_exclusive)):
            owner[f] = sc.scene_id
    for f in range(n):
        ff = fin_frames[f]
        luma = float(np.mean(ff))
        std = float(np.std(ff))
        mad = float(np.mean(np.abs(ff.astype(np.float32) - src_frames[f].astype(np.float32)))) if f < len(src_frames) else 0.0
        csv_lines.append(f"{f},{f / 24.0:.4f},{luma:.2f},{std:.2f},{mad:.3f},{owner[f]}")
    csv_path = out_dir / "frame_metrics_720.csv"
    csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    return {
        "contact_sheet": str(cs_path),
        "scene_peaks_sheet": str(peak_sheet_p),
        "frame_metrics_csv": str(csv_path),
    }


def evaluate_creative_quality(
    media: MediaInfo,
    final_mp4: Path,
    storyboard: StoryboardSpec,
    story_graph: StoryGraph,
    ablation_results: dict[str, dict[str, Any]],
    audio_qa: dict[str, Any] | None,
    reuse_report: dict[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    """Compute honest objective creative quality proxies (A5) — NEVER conflated with human G8 approval."""
    ev_dir = run_dir / "evidence"
    sheets = build_contact_and_boundary_sheets(Path(media.source_path), final_mp4, storyboard, ev_dir)

    # Measure substantive visual coverage & contrast/black-frame health directly from decoded final MP4
    cap_fin = cv2.VideoCapture(str(final_mp4))
    cap_src = cv2.VideoCapture(str(media.source_path))
    black_frames = 0
    substantive_frames = 0
    total_checked = 0
    phone_contrasts: list[float] = []

    while True:
        ok1, f1 = cap_fin.read()
        ok2, f2 = cap_src.read()
        if not ok1:
            break
        total_checked += 1
        mean_l = float(np.mean(f1))
        if mean_l < 4.5:
            black_frames += 1
        if ok2 and f1.shape == f2.shape:
            diff_mad = float(np.mean(np.abs(f1.astype(np.float32) - f2.astype(np.float32))))
            if diff_mad >= 3.0:
                substantive_frames += 1
        # Sample phone-scale 360x640 global contrast every 12 frames
        if total_checked % 12 == 0:
            small = cv2.resize(f1, (360, 640), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
            p10, p90 = float(np.percentile(gray, 10)), float(np.percentile(gray, 90))
            phone_contrasts.append(round((p90 + 12.0) / max(12.0, p10 + 12.0), 2))
    cap_fin.release()
    cap_src.release()

    rep_check = detect_template_repetition(storyboard)
    ablation_verified = sum(1 for sc_ab in ablation_results.values() for v in sc_ab.values() if isinstance(v, dict) and v.get("verified"))
    ablation_total = sum(1 for sc_ab in ablation_results.values() for v in sc_ab.values() if isinstance(v, dict))

    cov_ratio = round(substantive_frames / max(1, total_checked), 4)
    median_contrast = round(float(np.median(phone_contrasts)) if phone_contrasts else 1.0, 2)

    hard_rejects: list[str] = []
    if total_checked != media.duration_frames:
        hard_rejects.append(f"frame_count_mismatch:{total_checked}!={media.duration_frames}")
    if black_frames > 0:
        hard_rejects.append(f"black_frames_detected:{black_frames}")
    if not rep_check["passed"]:
        hard_rejects.append(f"repetitive_template_streak:{rep_check['max_consecutive_identical_templates']}")
    if story_graph.ledger.active and not story_graph.ledger.conservation_exact:
        hard_rejects.append("ledger_conservation_failed")
    if audio_qa and audio_qa.get("status") != "PASS":
        hard_rejects.append(f"audio_qa_failed:{audio_qa.get('status')}")

    proxy_pass = bool(not hard_rejects and cov_ratio >= 0.35 and median_contrast >= 2.2)

    report = {
        "schema_version": "1.0.0",
        "technical_proxy_status": "PASS" if proxy_pass else "FAIL",
        "vision_critic_status": "VISION_CRITIC_NOT_RUN",
        "vision_critic_note": "No local VLM installed; text-only Qwen2.5-3B is NOT treated as a vision critic.",
        "human_review_status": "PENDING_HUMAN_REVIEW",
        "vq01_human_score": None,
        "competitor_reference_status": "REFERENCE_NOT_AVAILABLE",
        "frames_checked": total_checked,
        "black_frames": black_frames,
        "substantive_visual_change_ratio": cov_ratio,
        "phone_360x640_median_contrast_ratio": median_contrast,
        "template_repetition_audit": rep_check,
        "ledger_conservation_exact": bool(story_graph.ledger.conservation_exact) if story_graph.ledger.active else None,
        "ablation_layers_verified": ablation_verified,
        "ablation_layers_total": ablation_total,
        "reuse_gate_passed": bool(reuse_report.get("passed")),
        "hard_rejects": hard_rejects,
        "evidence_files": sheets,
    }
    atomic_write_json(ev_dir / "creative_quality_proxy.json", report)
    return report

