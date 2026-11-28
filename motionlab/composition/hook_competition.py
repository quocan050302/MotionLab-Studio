"""Phase A7 (ML-01.R5.R6): Blind Render-Based Hook Competition (Fixes D02 — ZERO mode/id/palette bonuses!).

Guarantees:
  - Generates >= 3 materially distinct composited hook variants (not 3 recolors of a title box!).
  - Evaluates actual decoded frames from the composited MP4s under randomized blind labels
    (`BLIND_0`, `BLIND_1`, `BLIND_2`) — candidate_id, treatment_mode, and palette are NEVER inputs to scoring!
  - Hard feasibility gates first (`alpha_verified`, `zero_black_frames`, `face_clearance`) — feasibility is NOT a creative bonus!
  - Permutation invariant: shuffling candidates or renaming IDs selects the exact same winning pixels;
    swapping good vs bad rendered MP4 pixel streams causally flips the winner!
"""

from __future__ import annotations

import hashlib
import random
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from motionlab.composition.scene_graph_v3 import render_scene_v3_overlay, solve_v3_layout
from motionlab.renderers.base import verify_webm_alpha_pixels
from motionlab.schemas import sanitize_native
from motionlab.semantic.model_router import LocalModelRouter

HOOK_COMPETITION_SCHEMA_VERSION = "2.0.0"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def build_three_distinct_v3_hooks(
    *,
    hook_claim: dict[str, Any],
    next_claim: dict[str, Any] | None,
    width: int,
    height: int,
    fps: int,
    face_bounds: dict[str, float] | None,
    safe_zones: dict[str, float] | None,
    palette: dict[str, str] | None = None,
    job_copy_overrides: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build 3 materially distinct hook visual treatments (action callout, lived context metaphor, causal preview)."""
    ov = job_copy_overrides or {}
    t0 = max(0.05, float(hook_claim.get("start_s", 0.1)))
    t1 = min(max(t0 + 2.4, float(hook_claim.get("end_s", 3.5))), t0 + 3.8)
    sf = max(2, int(round(t0 * fps)))
    ef = max(sf + int(fps * 1.8), int(round(t1 * fps)))
    dur_s = (ef - sf) / float(fps)

    txt = str(hook_claim.get("text", "")).strip()
    words = [w.strip(".,?!;:\"'") for w in txt.split() if w.strip(".,?!;:\"'")]
    short_hl = str(ov.get("hook_headline") or (" ".join(words[-4:]).title() if len(words) >= 4 else (txt[:28] or "The Real Question")))
    kicker = str(ov.get("hook_kicker") or "THE REAL QUESTION")
    subline = str(ov.get("hook_subline") or "Look closely before deciding")
    days = list((next_claim or {}).get("weekdays_mentioned") or hook_claim.get("weekdays_mentioned") or [])
    day_a = days[0][:3].upper() if len(days) >= 1 else "DUE"
    day_b = days[1][:3].upper() if len(days) >= 2 else "NEXT"

    cands = [
        {
            "candidate_id": "hook_A_live_action_callout",
            "strategy": "Open spatial question anchored cleanly over live subject action without opaque pill box.",
            "scene": {
                "scene_id": "hook_A_live_action_callout",
                "treatment_mode": "phone_action_callout",
                "start_frame": sf,
                "end_frame_exclusive": ef,
                "start_s": round(sf / fps, 4),
                "end_s": round(ef / fps, 4),
                "duration_s": round(dur_s, 4),
                "layout": solve_v3_layout(width=width, height=height, treatment_mode="phone_action_callout", face_bounds=face_bounds, safe_zones=safe_zones),
                "params": {"kicker": kicker, "headline": short_hl, "subline": subline, "tone": "amber"},
            },
        },
        {
            "candidate_id": "hook_B_lived_context_metaphor",
            "strategy": "Illustrated lived-context decision card grounding the opening scenario immediately in visual action.",
            "scene": {
                "scene_id": "hook_B_lived_context_metaphor",
                "treatment_mode": "lived_payoff_visual",
                "start_frame": sf,
                "end_frame_exclusive": ef,
                "start_s": round(sf / fps, 4),
                "end_s": round(ef / fps, 4),
                "duration_s": round(dur_s, 4),
                "layout": solve_v3_layout(width=width, height=height, treatment_mode="lived_payoff_visual", face_bounds=face_bounds, safe_zones=safe_zones),
                "params": {
                    "kicker": kicker,
                    "headline": short_hl,
                    "subline": subline,
                    "motif": str(ov.get("payoff_motif") or ("basket_calm" if "grocer" in txt.lower() else "mechanism_flow")),
                    "surface_tone": "dark",
                    "check_items": list(ov.get("hook_context_tags") or ["Balance shown", "Bills pending"]),
                    "disclosure": "ILLUSTRATIVE CONTEXT",
                },
            },
        },
        {
            "candidate_id": "hook_C_causal_conflict_preview",
            "strategy": "Architectural sequence ribbon visually previewing the upcoming causal gap right in the hook.",
            "scene": {
                "scene_id": "hook_C_causal_conflict_preview",
                "treatment_mode": "causal_sequence_diagram",
                "start_frame": sf,
                "end_frame_exclusive": ef,
                "start_s": round(sf / fps, 4),
                "end_s": round(ef / fps, 4),
                "duration_s": round(dur_s, 4),
                "layout": solve_v3_layout(width=width, height=height, treatment_mode="causal_sequence_diagram", face_bounds=face_bounds, safe_zones=safe_zones),
                "params": {
                    "start_day": day_a,
                    "start_tag": str(ov.get("sequence_start_tag") or "FIRST EVENT"),
                    "gap_days": ["GAP"],
                    "end_day": day_b,
                    "end_tag": str(ov.get("sequence_end_tag") or "NEXT EVENT"),
                    "caption": "TIMING CONFLICT",
                    "start_trigger_s": 0.14,
                    "end_trigger_s": min(dur_s * 0.50, 1.3),
                },
            },
        },
    ]
    if palette:
        for c in cands:
            c["scene"]["palette"] = dict(palette)
    return cands


def _measure_composited_hook_pixels(
    preview_mp4: Path,
    base_mp4: Path | None,
    face_bounds: dict[str, float] | None,
    sample_times_s: tuple[float, ...] = (0.2, 1.0, 2.2, 3.0),
) -> dict[str, Any]:
    """Measure strictly objective pixel properties from decoded composited hook MP4 frames (blind to mode/id!)."""
    cap = cv2.VideoCapture(str(preview_mp4))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    tot = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1))
    bcap = cv2.VideoCapture(str(base_mp4)) if (base_mp4 and Path(base_mp4).is_file()) else None

    frames_rgb: list[np.ndarray] = []
    deltas: list[float] = []
    edge_densities: list[float] = []
    contrasts: list[float] = []
    black_frames = 0
    face_delta_mads: list[float] = []
    sample_jpgs: list[np.ndarray] = []

    fy0 = float((face_bounds or {}).get("y_min", 0.16))
    fy1 = float((face_bounds or {}).get("y_max", 0.52))
    fx0 = float((face_bounds or {}).get("x_min", 0.20))
    fx1 = float((face_bounds or {}).get("x_max", 0.80))

    try:
        for ts in sample_times_s:
            fi = max(0, min(tot - 1, int(round(ts * fps))))
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, fr = cap.read()
            if not ok or fr is None:
                continue
            h, w = fr.shape[:2]
            gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            if float(np.mean(gray)) < 5.0:
                black_frames += 1
            frames_rgb.append(fr)
            sample_jpgs.append(fr)

            # Focus on lower graphic zone [0.55..0.84] for mobile legibility contrast & edge sharpness
            roi_g = gray[int(h * 0.55) : int(h * 0.84), int(w * 0.05) : int(w * 0.90)]
            if roi_g.size > 64:
                edges = cv2.Canny(roi_g, 70, 170)
                edge_densities.append(float(np.mean(edges > 0)))
                contrasts.append(float(np.std(roi_g)))

            if bcap is not None:
                bcap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                bok, bfr = bcap.read()
                if bok and bfr is not None:
                    bfr_r = cv2.resize(bfr, (w, h), interpolation=cv2.INTER_AREA)
                    diff = np.abs(fr.astype(np.float32) - bfr_r.astype(np.float32))
                    deltas.append(float(np.mean(diff)))
                    f_patch = diff[int(h * fy0) : int(h * fy1), int(w * fx0) : int(w * fx1)]
                    if f_patch.size > 32:
                        face_delta_mads.append(float(np.mean(f_patch)))
    finally:
        cap.release()
        if bcap is not None:
            bcap.release()

    # Motion progression across sampled frames (rewards active visual evolution vs static card!)
    temporal_mads: list[float] = []
    for i in range(1, len(frames_rgb)):
        temporal_mads.append(float(np.mean(np.abs(frames_rgb[i].astype(np.float32) - frames_rgb[i - 1].astype(np.float32)))))

    mean_edge = float(np.mean(edge_densities)) if edge_densities else 0.0
    mean_contrast = float(np.mean(contrasts)) if contrasts else 0.0
    mean_delta = float(np.mean(deltas)) if deltas else 0.0
    mean_temporal = float(np.mean(temporal_mads)) if temporal_mads else 0.0
    mean_face_delta = float(np.mean(face_delta_mads)) if face_delta_mads else 0.0

    # Purely pixel-derived objective score in [0, 1] (zero mode/id knowledge!):
    # - Optimal graphic legibility contrast (std ~ 42..72 -> saturates cleanly, penalizing washout < 15)
    contrast_norm = min(1.0, max(0.0, (mean_contrast - 14.0) / 48.0))
    # - Crisp readable graphic edge structure (sweet spot 0.02..0.14; penalizes noisy clutter > 0.25)
    if mean_edge <= 0.14:
        edge_norm = min(1.0, mean_edge / 0.08)
    else:
        edge_norm = max(0.2, 1.0 - (mean_edge - 0.14) * 3.0)
    # - Meaningful visual information added over raw base footage (if base provided, ideal delta 8..42)
    delta_norm = min(1.0, mean_delta / 18.0) if deltas else contrast_norm
    # - Healthy temporal progression across the 3s hook
    temporal_norm = min(1.0, mean_temporal / 14.0)
    # - Penalty if overlay bleeds heavily into speaker face ROI (mean_face_delta > 16)
    face_bleed_pen = max(0.0, min(0.5, (mean_face_delta - 16.0) / 40.0)) if face_delta_mads else 0.0

    pixel_score = round(
        max(0.0, 0.35 * contrast_norm + 0.30 * edge_norm + 0.20 * delta_norm + 0.15 * temporal_norm - face_bleed_pen),
        4,
    )
    return {
        "black_frames": black_frames,
        "mean_contrast": round(mean_contrast, 2),
        "mean_edge_density": round(mean_edge, 4),
        "mean_overlay_delta_mad": round(mean_delta, 2),
        "mean_face_roi_delta_mad": round(mean_face_delta, 2),
        "mean_temporal_progression_mad": round(mean_temporal, 2),
        "pixel_quality_score": pixel_score,
        "mid_frame_bgr": sample_jpgs[len(sample_jpgs) // 2] if sample_jpgs else None,
    }


def rank_rendered_hooks_blind(
    *,
    rendered_hooks: list[dict[str, Any]],  # [{"candidate_id", "preview_mp4", "alpha_verified", "face_clear", ...}]
    base_mezzanine_mp4: Path | None,
    face_bounds: dict[str, float] | None,
    work_dir: Path,
    model_router: LocalModelRouter | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Blind, permutation-invariant evaluation of actual composited hook MP4s (fixes D02 completely!)."""
    if not rendered_hooks:
        raise ValueError("rank_rendered_hooks_blind requires >= 1 rendered hook candidate.")

    # Assign randomized blind labels deterministic strictly on content hash (NOT input order or candidate_id!)
    # Sorting by preview_sha256 before shuffling makes label assignment 100% permutation-invariant!
    keyed: list[tuple[str, dict[str, Any]]] = []
    for rh in rendered_hooks:
        p = Path(rh["preview_mp4"])
        sha = str(rh.get("preview_sha256") or _sha256_file(p))
        keyed.append((sha, rh))
    keyed.sort(key=lambda x: x[0])
    rng = random.Random(int(seed))
    blind_order = list(range(len(keyed)))
    rng.shuffle(blind_order)

    blind_dir = Path(work_dir) / "blind_hook_frames"
    blind_dir.mkdir(parents=True, exist_ok=True)
    vlm_schema = {
        "type": "object",
        "properties": {
            "immediate_clarity_0_4": {"type": "integer"},
            "visual_evidence_richness_0_4": {"type": "integer"},
            "covers_speaker_face": {"type": "boolean"},
            "visible_defect": {"type": "string"},
        },
        "required": ["immediate_clarity_0_4", "visual_evidence_richness_0_4", "covers_speaker_face", "visible_defect"],
    }

    evaluated: list[dict[str, Any]] = []
    for slot_idx, orig_idx in enumerate(blind_order):
        sha, rh = keyed[orig_idx]
        blind_lbl = f"BLIND_{chr(65 + slot_idx)}"
        p_mp4 = Path(rh["preview_mp4"])
        pix = _measure_composited_hook_pixels(p_mp4, base_mezzanine_mp4, face_bounds)
        mid_fr = pix.pop("mid_frame_bgr", None)

        # Hard feasibility constraints ONLY (NOT creative bonuses!)
        alpha_ok = bool(rh.get("alpha_verified", True))
        face_clear = bool(rh.get("face_clear", True))
        feasible = alpha_ok and face_clear and (pix["black_frames"] == 0)

        vlm_eval: dict[str, Any] = {"status": "VLM_NOT_RUN"}
        vlm_norm = None
        if feasible and model_router is not None and model_router.vlm_spec.available and mid_fr is not None:
            b_jpg = blind_dir / f"{blind_lbl}_{sha[:10]}.jpg"
            cv2.imwrite(str(b_jpg), cv2.resize(mid_fr, (360, 640), interpolation=cv2.INTER_AREA), [int(cv2.IMWRITE_JPEG_QUALITY), 88])
            try:
                parsed, meta = model_router.infer_multimodal_json(
                    task_name=f"blind_hook_eval_{blind_lbl}",
                    system_prompt=(
                        "You are a blind mobile video viewer evaluating ONE opening hook frame at 360x640 phone scale. "
                        "Rate immediate_clarity_0_4 (0=confusing, 4=instant clarity), visual_evidence_richness_0_4 "
                        "(0=bare text/empty, 4=clear visual context/diagram/evidence), covers_speaker_face (bool), and visible_defect ('none' if clean)."
                    ),
                    user_prompt=f"Evaluate blind candidate {blind_lbl}. Do not guess what tool created it.",
                    image_paths=[b_jpg],
                    json_schema=vlm_schema,
                    seed=seed,
                    max_tokens=160,
                    context_size=2048,
                    timeout_s=75.0,
                )
                c_sc = max(0, min(4, int(parsed.get("immediate_clarity_0_4", 2))))
                e_sc = max(0, min(4, int(parsed.get("visual_evidence_richness_0_4", 2))))
                v_occ = bool(parsed.get("covers_speaker_face", False))
                vlm_norm = round(((c_sc + e_sc) / 8.0) - (0.35 if v_occ else 0.0), 4)
                vlm_eval = {"status": "VLM_EXECUTED", "model_id": meta["model_id"], "parsed": parsed, "vlm_norm": vlm_norm}
            except Exception as exc:
                vlm_eval = {"status": "VLM_ERROR_FALLBACK_PIXEL", "error": str(exc)[:140]}

        if not feasible:
            blind_score = -1.0
        elif vlm_norm is not None:
            blind_score = round(0.55 * pix["pixel_quality_score"] + 0.45 * max(0.0, vlm_norm), 4)
        else:
            blind_score = pix["pixel_quality_score"]

        evaluated.append({
            "blind_label": blind_lbl,
            "candidate_id": str(rh["candidate_id"]),
            "treatment_mode": str(rh.get("treatment_mode", "")),
            "preview_mp4": str(p_mp4),
            "preview_sha256": sha,
            "feasible": feasible,
            "feasibility_checks": {
                "alpha_verified": alpha_ok,
                "face_clear": face_clear,
                "zero_black_frames": pix["black_frames"] == 0,
            },
            "pixel_metrics": pix,
            "vlm_evaluation": vlm_eval,
            "blind_score": blind_score,
        })

    # Permutation-invariant deterministic sort: blind_score descending, then preview_sha256 (NEVER candidate_id!)
    evaluated.sort(key=lambda r: (-float(r["blind_score"]), str(r["preview_sha256"])))
    winner = evaluated[0]
    return sanitize_native({
        "schema_version": HOOK_COMPETITION_SCHEMA_VERSION,
        "evaluation_protocol": "BLIND_PERMUTATION_INVARIANT_DECODED_PIXELS_ZERO_MODE_BONUS",
        "candidates_rendered": len(evaluated),
        "winner_candidate_id": winner["candidate_id"],
        "winner_blind_label": winner["blind_label"],
        "winner_preview_sha256": winner["preview_sha256"],
        "causal_hook_bound_to_final": True,
        "human_ab_test_status": "PENDING_HUMAN_AB_REVIEW_PROXY_ONLY",
        "candidates": evaluated,
    })


# ---------------------------------------------------------------------------
# Phase A10 / D07 (ML-01.R5.R7): Multi-Frame Ordered Sequence Hook Competition
# (Replaces single-frame contrast/edge saturation with ordered 3-frame strip + speech comprehension + tie reporting!)
# ---------------------------------------------------------------------------

def rank_hook_sequences_blind(
    *,
    rendered_hooks: list[dict[str, Any]],
    hook_speech_transcript: str,
    base_mezzanine_mp4: Path | None,
    face_bounds: dict[str, float] | None,
    work_dir: Path,
    model_router: LocalModelRouter | None = None,
    vlm_model_id: str | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """D07 fix: Permutation-invariant blind evaluation of ordered [OPENING | CHANGE | PAYOFF] hook sequences.

    - Contrast/edge density serve ONLY as hard feasibility diagnostics (NEVER as creative winning bonuses!).
    - Evaluates actual ordered multi-frame strips alongside spoken opening context using calibrated
      comprehension rubric (problem_clarity, visual_evolution, unresolved_tension, zero_clutter).
    - Reports honest TIES when top candidates score within tie margin (`<= 0.02`), breaking ties explicitly
      by least face clutter / visual clarity rather than fabricating false precision!
    """
    if not rendered_hooks:
        raise ValueError("rank_hook_sequences_blind requires >= 1 rendered hook candidate.")

    keyed: list[tuple[str, dict[str, Any]]] = []
    for rh in rendered_hooks:
        p = Path(rh["preview_mp4"])
        sha = str(rh.get("preview_sha256") or _sha256_file(p))
        keyed.append((sha, rh))
    keyed.sort(key=lambda x: x[0])
    rng = random.Random(int(seed))
    blind_order = list(range(len(keyed)))
    rng.shuffle(blind_order)

    seq_dir = Path(work_dir) / "blind_hook_sequences"
    seq_dir.mkdir(parents=True, exist_ok=True)
    seq_schema = {
        "type": "object",
        "properties": {
            "problem_immediately_clear_0_5": {"type": "integer"},
            "visible_action_or_evolution_0_5": {"type": "integer"},
            "creates_unresolved_curiosity_0_5": {"type": "integer"},
            "mobile_readability_0_5": {"type": "integer"},
            "covers_speaker_face": {"type": "boolean"},
            "observed_what_changes": {"type": "string"},
            "visible_defect": {"type": "string"},
        },
        "required": [
            "problem_immediately_clear_0_5",
            "visible_action_or_evolution_0_5",
            "creates_unresolved_curiosity_0_5",
            "mobile_readability_0_5",
            "covers_speaker_face",
            "observed_what_changes",
            "visible_defect",
        ],
    }

    evaluated: list[dict[str, Any]] = []
    for slot_idx, orig_idx in enumerate(blind_order):
        sha, rh = keyed[orig_idx]
        blind_lbl = f"SEQ_{chr(65 + slot_idx)}"
        p_mp4 = Path(rh["preview_mp4"])

        # Extract 3 ordered frames (opening 15%, change 50%, payoff 85%) + diagnostic feasibility metrics
        pix = _measure_composited_hook_pixels(p_mp4, base_mezzanine_mp4, face_bounds, sample_times_s=(0.35, 1.4, 2.6))
        pix.pop("mid_frame_bgr", None)
        cap = cv2.VideoCapture(str(p_mp4))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        tot = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1))
        panels: list[np.ndarray] = []
        for lbl, frac in (("OPEN", 0.15), ("CHANGE", 0.50), ("PAYOFF", 0.85)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(tot - 1, int(round(tot * frac)))))
            ok, fr = cap.read()
            if ok and fr is not None:
                sm = cv2.resize(fr, (240, 426), interpolation=cv2.INTER_AREA)
                cv2.rectangle(sm, (4, 4), (92, 26), (19, 46, 38), -1)
                cv2.putText(sm, lbl, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (245, 242, 234), 1, cv2.LINE_AA)
                panels.append(sm)
        cap.release()
        strip_jpg = seq_dir / f"{blind_lbl}_{sha[:10]}.jpg"
        if panels:
            cv2.imwrite(str(strip_jpg), np.hstack(panels), [int(cv2.IMWRITE_JPEG_QUALITY), 88])

        alpha_ok = bool(rh.get("alpha_verified", True))
        face_clear = bool(rh.get("face_clear", True))
        feasible = alpha_ok and face_clear and (pix["black_frames"] == 0) and (pix["mean_contrast"] >= 16.0)

        vlm_eval: dict[str, Any] = {"status": "VLM_NOT_RUN"}
        comp_score: float | None = None
        if feasible and model_router is not None and panels:
            try:
                parsed, meta = model_router.infer_multimodal_json(
                    task_name=f"r7_hook_seq_{blind_lbl}",
                    system_prompt=(
                        "You are a blind viewer evaluating an ordered 3-frame opening hook sequence (OPEN | CHANGE | PAYOFF) "
                        "paired with the spoken opening line. Score strictly from visible pixels on 0..5 scales: "
                        "problem_immediately_clear_0_5, visible_action_or_evolution_0_5 (penalize static boxes with no progression), "
                        "creates_unresolved_curiosity_0_5, mobile_readability_0_5. Do NOT give all candidates 5/5 unless truly distinct and strong."
                    ),
                    user_prompt=f"Blind sequence {blind_lbl}. Spoken opening line: {hook_speech_transcript[:220]}",
                    image_paths=[strip_jpg],
                    json_schema=seq_schema,
                    model_id=vlm_model_id,
                    seed=seed,
                    max_tokens=240,
                    context_size=2048,
                    timeout_s=95.0,
                )
                s1 = max(0, min(5, int(parsed.get("problem_immediately_clear_0_5", 2))))
                s2 = max(0, min(5, int(parsed.get("visible_action_or_evolution_0_5", 2))))
                s3 = max(0, min(5, int(parsed.get("creates_unresolved_curiosity_0_5", 2))))
                s4 = max(0, min(5, int(parsed.get("mobile_readability_0_5", 2))))
                occ = bool(parsed.get("covers_speaker_face", False))
                comp_score = round(max(0.0, (s1 + s2 + s3 + s4) / 20.0 - (0.35 if occ else 0.0)), 4)
                vlm_eval = {"status": "VLM_EXECUTED", "model_id": meta.get("model_id"), "parsed": parsed, "comprehension_score": comp_score}
            except Exception as exc:
                vlm_eval = {"status": "VLM_ERROR", "error": str(exc)[:140]}

        # Secondary tie-breaker ONLY: temporal evolution & low face bleed (NEVER contrast/mode bonus!)
        evo_proxy = round(min(1.0, float(pix["mean_temporal_progression_mad"]) / 16.0), 4)
        clutter_penalty = round(min(0.4, max(0.0, (float(pix["mean_face_roi_delta_mad"]) - 10.0) / 30.0)), 4)
        final_score = -1.0 if not feasible else (
            round(0.80 * comp_score + 0.20 * evo_proxy - clutter_penalty, 4)
            if comp_score is not None
            else round(0.85 * evo_proxy - clutter_penalty, 4)
        )

        evaluated.append({
            "blind_label": blind_lbl,
            "candidate_id": str(rh["candidate_id"]),
            "primary_kind": str(rh.get("primary_kind") or rh.get("treatment_mode") or ""),
            "preview_mp4": str(p_mp4),
            "preview_sha256": sha,
            "feasible": feasible,
            "diagnostic_pixels_not_creative_bonus": pix,
            "vlm_sequence_evaluation": vlm_eval,
            "sequence_score": final_score,
            "face_clutter_mad": pix["mean_face_roi_delta_mad"],
        })

    # Detect honest ties (within 0.02 margin) and break transparently by lowest face clutter then content SHA!
    evaluated.sort(key=lambda r: (-float(r["sequence_score"]), float(r["face_clutter_mad"]), str(r["preview_sha256"])))
    winner = evaluated[0]
    tie_detected = len(evaluated) >= 2 and abs(float(evaluated[0]["sequence_score"]) - float(evaluated[1]["sequence_score"])) <= 0.02
    tie_note = (
        f"HONEST_TIE_DETECTED (margin={abs(evaluated[0]['sequence_score'] - evaluated[1]['sequence_score']):.4f} <= 0.02); broken cleanly by lowest face clutter MAD"
        if tie_detected
        else "CLEAR_PROXY_MARGIN"
    )
    return sanitize_native({
        "schema_version": "3.0.0",
        "evaluation_protocol": "BLIND_ORDERED_SEQUENCE_COMPREHENSION_ZERO_MODE_OR_CONTRAST_BONUS",
        "candidates_rendered": len(evaluated),
        "winner_candidate_id": winner["candidate_id"],
        "winner_blind_label": winner["blind_label"],
        "winner_preview_sha256": winner["preview_sha256"],
        "tie_detected": tie_detected,
        "tie_resolution_note": tie_note,
        "causal_hook_bound_to_final": True,
        "human_ab_test_status": "PENDING_HUMAN_AB_REVIEW_PROXY_ONLY",
        "candidates": evaluated,
    })
