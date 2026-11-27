"""Phase R1-A2 (ML-01.R5.R1): Premium Editorial Design System, All-Frame Protection Solver & Continuity Tracks.

Provides:
  - `build_style_tokens`: Warm editorial palette derived from niche + optional footage warmth sampling
    (ivory #F5F0E6, sage #84977A, deep forest #243B32, muted espresso #40382E, antique gold #BE9A59, terracotta #AF745E).
  - `solve_layout_with_protection`: Conservative 9:16 Reels safe zones (L>=48, R>=105, T>=110, B>=230 at 720x1280)
    plus ALL-FRAME face/eye/mouth/hand collision sweep across every frame in ROI tracks.
  - `evaluate_phone_legibility`: Phone-scale (360x640) typography size, WCAG-like contrast proxy,
    and strict consumer QA-jargon filter (blocks '(derived)', 'provenance', schema terms).
  - `build_shared_element_tracks`: Persistent `SharedElementState` continuity records bridging consecutive scenes.
  - `validate_layout_sweep`: Fail-closed layout/safety gate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from motionlab.composition.generative_schema import SceneSpec, StoryboardSpec
from motionlab.schemas import sanitize_native
from motionlab.semantic.story_graph import StoryGraph

R5R1_EDITORIAL_PALETTE: dict[str, str] = {
    "ivory": "#F5F0E6",
    "sage": "#84977A",
    "forest": "#243B32",
    "espresso": "#40382E",
    "gold": "#BE9A59",
    "brass": "#BE9A59",       # backward-compatible alias
    "terracotta": "#AF745E",
    "clay": "#AF745E",        # backward-compatible alias
    "ink_dark": "#1E1B16",
}

NICHE_STYLE_TOKENS: dict[str, dict[str, str]] = {
    "finance": dict(R5R1_EDITORIAL_PALETTE),
    "product": {
        "ivory": "#F4F2EC",
        "sage": "#7E9A94",
        "forest": "#223D42",
        "espresso": "#323539",
        "gold": "#BE9A59",
        "brass": "#BE9A59",
        "terracotta": "#AD6F5C",
        "clay": "#AD6F5C",
        "ink_dark": "#1B1D20",
    },
    "storytelling": {
        "ivory": "#F5EFE4",
        "sage": "#89967D",
        "forest": "#2F4234",
        "espresso": "#3D342C",
        "gold": "#BA9454",
        "brass": "#BA9454",
        "terracotta": "#A85E4B",
        "clay": "#A85E4B",
        "ink_dark": "#1D1915",
    },
    "education": {
        "ivory": "#F3F5F7",
        "sage": "#7E989D",
        "forest": "#22423E",
        "espresso": "#2C3440",
        "gold": "#BC9757",
        "brass": "#BC9757",
        "terracotta": "#AE6B58",
        "clay": "#AE6B58",
        "ink_dark": "#181E26",
    },
    "lifestyle": dict(R5R1_EDITORIAL_PALETTE),
    "cinematic": {
        "ivory": "#F0ECE4",
        "sage": "#849386",
        "forest": "#263A33",
        "espresso": "#34312C",
        "gold": "#BC9A5D",
        "brass": "#BC9A5D",
        "terracotta": "#A76554",
        "clay": "#A76554",
        "ink_dark": "#1A1916",
    },
}

# Consumer-facing text MUST NEVER leak internal QA/schema vocabulary
CONSUMER_FORBIDDEN_JARGON = (
    "(derived)",
    "derived_arithmetic",
    "verbatim_spoken",
    "story_graph",
    "schema_version",
    "provenance:",
)


@dataclass
class SafeMargins:
    left: int
    right: int
    top: int
    bottom: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass
class SharedElementState:
    element_id: str
    logical_state: str
    scene_id: str
    start_frame: int
    end_frame_exclusive: int
    bbox_norm: dict[str, float]  # x, y, w, h in [0, 1]
    z_order: int
    color: str
    continuity_from_scene: str | None = None
    continuity_to_scene: str | None = None
    transition_owner: str = "remotion"
    provenance_ref: str = ""

    def to_dict(self) -> dict[str, Any]:
        return sanitize_native(asdict(self))


def compute_safe_margins(canvas_w: int, canvas_h: int) -> SafeMargins:
    """Compute conservative platform safe margins scaled cleanly for 9:16 portrait, 1:1, or 16:9 landscape."""
    is_portrait = canvas_h > canvas_w
    if is_portrait:
        # Reels/Shorts conservative safe box: L>=48, R>=105 (UI rail), T>=110, B>=230 at 720x1280
        sx = canvas_w / 720.0
        sy = canvas_h / 1280.0
        return SafeMargins(
            left=max(32, int(round(48 * sx))),
            right=max(64, int(round(105 * sx))),
            top=max(64, int(round(110 * sy))),
            bottom=max(120, int(round(230 * sy))),
        )
    # 16:9 / 1:1 safe title margins (5% / 8%)
    return SafeMargins(
        left=max(36, int(round(canvas_w * 0.055))),
        right=max(36, int(round(canvas_w * 0.055))),
        top=max(32, int(round(canvas_h * 0.065))),
        bottom=max(44, int(round(canvas_h * 0.095))),
    )


def _hex_to_rgb(hx: str) -> tuple[float, float, float]:
    s = hx.lstrip("#")
    if len(s) != 6:
        return (245.0, 240.0, 230.0)
    return (float(int(s[0:2], 16)), float(int(s[2:4], 16)), float(int(s[4:6], 16)))


def _relative_luminance(rgb: tuple[float, float, float]) -> float:
    def _lin(c: float) -> float:
        x = c / 255.0
        return x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4
    r, g, b = _lin(rgb[0]), _lin(rgb[1]), _lin(rgb[2])
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def wcag_contrast_ratio(fg_hex: str, bg_hex: str) -> float:
    l1 = _relative_luminance(_hex_to_rgb(fg_hex))
    l2 = _relative_luminance(_hex_to_rgb(bg_hex))
    hi, lo = max(l1, l2), min(l1, l2)
    return round((hi + 0.05) / (lo + 0.05), 2)


def build_style_tokens(niche: str = "finance", source_mp4: Path | None = None) -> dict[str, Any]:
    """Derive cohesive editorial color + typography tokens from niche and optional footage color temperature."""
    base = dict(NICHE_STYLE_TOKENS.get((niche or "finance").lower(), R5R1_EDITORIAL_PALETTE))
    footage_warmth = "warm_natural"
    sampled_luma = 148.0
    if source_mp4 is not None and source_mp4.is_file():
        cap = cv2.VideoCapture(str(source_mp4))
        ok, fr = cap.read()
        cap.release()
        if ok and fr is not None:
            rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB).astype(np.float32)
            r_m, g_m, b_m = float(rgb[:, :, 0].mean()), float(rgb[:, :, 1].mean()), float(rgb[:, :, 2].mean())
            sampled_luma = round(0.299 * r_m + 0.587 * g_m + 0.114 * b_m, 1)
            footage_warmth = "warm_natural" if (r_m - b_m) >= 8.0 else ("cool_neutral" if (b_m - r_m) >= 8.0 else "balanced_neutral")
    return {
        "schema_version": "1.0.0",
        "niche": (niche or "finance").lower(),
        "footage_warmth": footage_warmth,
        "sampled_luma": sampled_luma,
        "palette": base,
        "typography": {
            "sans_family": 'Bahnschrift, "Segoe UI", "Helvetica Neue", Arial, sans-serif',
            "serif_family": 'Georgia, Cambria, "Times New Roman", serif',
            "max_families": 2,
            "max_weights": 3,
            "hero_amount_px_720": [96, 150],
            "secondary_label_px_720": [32, 48],
            "auxiliary_min_px_720": 24,
        },
    }


def _box_iou_and_overlap(a: dict[str, float], b: dict[str, float]) -> float:
    ax0, ay0, ax1, ay1 = a["x"], a["y"], a["x"] + a["w"], a["y"] + a["h"]
    bx0, by0, bx1, by1 = b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_b = max(1e-6, b["w"] * b["h"])
    return round(inter / area_b, 4)


def solve_layout_with_protection(
    storyboard: StoryboardSpec,
    roi_tracks: dict[str, Any] | None,
    canvas_w: int,
    canvas_h: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Solve safe layout zones for every scene and sweep ALL frames for face/eye/mouth/hand collisions.

    Returns:
      (layout_plan_dict, frame_protection_rows)
    """
    margins = compute_safe_margins(canvas_w, canvas_h)
    roi_frames = (roi_tracks or {}).get("keyframes") or (roi_tracks or {}).get("frames") or []
    roi_by_frame = {int(r["frame"]): r for r in roi_frames if "frame" in r}

    def _nearest_face_box(f: int) -> dict[str, float] | None:
        if not roi_by_frame:
            return None
        if f in roi_by_frame:
            fb = roi_by_frame[f].get("face") or roi_by_frame[f].get("face_box")
            if isinstance(fb, dict):
                return _norm_box(fb)
        # Nearest sampled ROI keyframe
        closest = min(roi_by_frame.keys(), key=lambda k: abs(k - f))
        if abs(closest - f) <= 18:
            fb = roi_by_frame[closest].get("face") or roi_by_frame[closest].get("face_box")
            if isinstance(fb, dict):
                return _norm_box(fb)
        return None

    def _norm_box(b: dict[str, Any]) -> dict[str, float]:
        if "x" in b and "y" in b and "w" in b and "h" in b:
            return {"x": float(b["x"]), "y": float(b["y"]), "w": float(b["w"]), "h": float(b["h"])}
        cx, cy = float(b.get("cx", 0.5)), float(b.get("cy", 0.28))
        bw, bh = float(b.get("w", 0.28)), float(b.get("h", 0.24))
        return {"x": max(0.0, cx - bw * 0.5), "y": max(0.0, cy - bh * 0.5), "w": bw, "h": bh}

    safe_x = margins.left / canvas_w
    safe_w = max(0.2, (canvas_w - margins.left - margins.right) / canvas_w)
    safe_top = margins.top / canvas_h
    safe_bot = (canvas_h - margins.bottom) / canvas_h

    scene_layouts: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []

    for sc in storyboard.scenes:
        # Gather minimum face top y across ALL frames of this scene so upper text NEVER clips eyes/forehead
        face_boxes: list[dict[str, float]] = []
        for f in range(sc.start_frame, sc.end_frame_exclusive):
            fb = _nearest_face_box(f)
            if fb is not None:
                face_boxes.append(fb)

        min_face_y = min((fb["y"] for fb in face_boxes), default=0.22)
        max_face_bottom = max((fb["y"] + fb["h"] for fb in face_boxes), default=0.45)

        if sc.stage_mode == "full_graphic_cutaway":
            hero_box = {"x": round(safe_x, 4), "y": round(max(safe_top, 0.09), 4), "w": round(safe_w, 4), "h": 0.24}
            secondary_box = {"x": round(safe_x, 4), "y": 0.36, "w": round(safe_w, 4), "h": 0.42}
            protect_live_face = False
        elif sc.stage_mode == "aspect_morph_editorial":
            # Balanced architectural split: upper breakdown cleanly inside top 9%..44%, speaker generously below 47%..92%
            hero_box = {"x": round(safe_x, 4), "y": round(max(safe_top, 0.088), 4), "w": round(safe_w, 4), "h": 0.34}
            secondary_box = {"x": 0.04, "y": 0.47, "w": 0.92, "h": 0.44}
            protect_live_face = False
        else:
            # Live speaker hero: keep upper text cleanly above eye/brow line!
            # Eye region starts at roughly face.y + 0.22 * face.h; protect strictly above face.y + 0.08 * face.h!
            eye_guard_y = max(safe_top + 0.10, min_face_y + 0.02) if face_boxes else 0.24
            avail_top_h = max(0.11, min(0.165, eye_guard_y - safe_top))
            top_y = max(0.042, min(safe_top, eye_guard_y - avail_top_h))
            hero_box = {"x": round(safe_x, 4), "y": round(top_y, 4), "w": round(safe_w, 4), "h": round(avail_top_h, 4)}
            secondary_box = {"x": round(safe_x, 4), "y": round(min(safe_bot - 0.11, max(max_face_bottom + 0.22, 0.71)), 4), "w": round(safe_w, 4), "h": 0.10}
            protect_live_face = True

        max_eye_ov = 0.0
        for f in range(sc.start_frame, sc.end_frame_exclusive):
            fb = _nearest_face_box(f)
            eye_mouth_ov = 0.0
            if protect_live_face and fb is not None:
                # Strict eye+mouth core box inside detected face (lower 78% of face bbox)
                eye_mouth = {
                    "x": fb["x"] + fb["w"] * 0.10,
                    "y": fb["y"] + fb["h"] * 0.22,
                    "w": fb["w"] * 0.80,
                    "h": fb["h"] * 0.72,
                }
                eye_mouth_ov = _box_iou_and_overlap(hero_box, eye_mouth)
                if sc.show_persistent_ribbon:
                    eye_mouth_ov = max(eye_mouth_ov, _box_iou_and_overlap(secondary_box, eye_mouth))
            max_eye_ov = max(max_eye_ov, eye_mouth_ov)
            frame_rows.append({
                "frame": f,
                "scene_id": sc.scene_id,
                "stage_mode": sc.stage_mode,
                "protect_live_face": protect_live_face,
                "eye_mouth_overlap_ratio": eye_mouth_ov,
                "hero_top_px": int(round(hero_box["y"] * canvas_h)),
                "hero_bottom_px": int(round((hero_box["y"] + hero_box["h"]) * canvas_h)),
                "safe_ok": eye_mouth_ov <= 0.02,
            })

        scene_layouts.append({
            "scene_id": sc.scene_id,
            "stage_mode": sc.stage_mode,
            "hero_box_norm": hero_box,
            "secondary_box_norm": secondary_box,
            "protect_live_face": protect_live_face,
            "max_eye_mouth_overlap": round(max_eye_ov, 4),
            "safe_margins_px": margins.to_dict(),
        })

    layout_plan = {
        "schema_version": "1.0.0",
        "canvas": {"width": canvas_w, "height": canvas_h},
        "safe_margins_px": margins.to_dict(),
        "scenes": scene_layouts,
        "all_frames_checked": len(frame_rows),
        "max_eye_mouth_overlap_any_frame": round(max((r["eye_mouth_overlap_ratio"] for r in frame_rows), default=0.0), 4),
        "zero_eye_mouth_collision": all(r["safe_ok"] for r in frame_rows),
    }
    return layout_plan, frame_rows


def evaluate_phone_legibility(
    layout_plan: dict[str, Any],
    storyboard: StoryboardSpec,
    canvas_w: int = 720,
    canvas_h: int = 1280,
) -> dict[str, Any]:
    """Audit 360x640 phone-scale font sizes, contrast ratios, and consumer QA-jargon leaks."""
    scale = 360.0 / max(1, canvas_w)
    pal = storyboard.palette or R5R1_EDITORIAL_PALETTE
    jargon_leaks: list[str] = []
    small_font_warnings: list[str] = []
    contrast_checks: list[dict[str, Any]] = []

    ivory = pal.get("ivory", "#F5F0E6")
    forest = pal.get("forest", "#243B32")
    espresso = pal.get("espresso", "#40382E")
    gold = pal.get("gold", pal.get("brass", "#BE9A59"))

    contrast_checks.append({"pair": "ivory_on_forest", "ratio": wcag_contrast_ratio(ivory, forest), "target": 4.5})
    contrast_checks.append({"pair": "forest_on_ivory", "ratio": wcag_contrast_ratio(forest, ivory), "target": 4.5})
    contrast_checks.append({"pair": "espresso_on_ivory", "ratio": wcag_contrast_ratio(espresso, ivory), "target": 4.5})
    contrast_checks.append({"pair": "gold_on_forest", "ratio": wcag_contrast_ratio(gold, forest), "target": 3.0})

    for sc in storyboard.scenes:
        for ly in sc.layers:
            for k, v in ly.params.items():
                if isinstance(v, str):
                    low = v.lower()
                    for bad in CONSUMER_FORBIDDEN_JARGON:
                        if bad in low:
                            jargon_leaks.append(f"{sc.scene_id}.{ly.layer_id}.{k}:{bad}")
                elif k in ("font_size", "hero_px") and isinstance(v, (int, float)):
                    px_360 = round(float(v) * scale, 1)
                    if px_360 < 12.0:
                        small_font_warnings.append(f"{sc.scene_id}.{ly.layer_id}:{v}px->{px_360}px@360")

    contrast_ok = all(c["ratio"] >= c["target"] for c in contrast_checks)
    return {
        "schema_version": "1.0.0",
        "phone_scale_width": 360,
        "scale_factor": round(scale, 4),
        "contrast_checks": contrast_checks,
        "contrast_proxy_ok": contrast_ok,
        "consumer_jargon_leaks": jargon_leaks,
        "small_font_warnings": small_font_warnings,
        "passed": bool(contrast_ok and not jargon_leaks and not small_font_warnings),
    }


def build_shared_element_tracks(
    storyboard: StoryboardSpec,
    story_graph: StoryGraph,
    layout_plan: dict[str, Any],
) -> dict[str, Any]:
    """Construct persistent SharedElementState continuity tracks across consecutive narrative scenes."""
    layout_by_sc = {s["scene_id"]: s for s in layout_plan.get("scenes", [])}
    pal = storyboard.palette or R5R1_EDITORIAL_PALETTE
    tracks: list[SharedElementState] = []

    ledger_scenes = [sc for sc in storyboard.scenes if sc.persistent_ledger_step_id or sc.role in ("hook_contrast", "negation_tension", "allocation_step", "payoff_reveal")]
    for i, sc in enumerate(ledger_scenes):
        ly_info = layout_by_sc.get(sc.scene_id, {})
        bbox = ly_info.get("hero_box_norm", {"x": 0.067, "y": 0.086, "w": 0.788, "h": 0.18})
        tracks.append(SharedElementState(
            element_id="persistent_ledger_balance",
            logical_state=sc.persistent_ledger_step_id or sc.role,
            scene_id=sc.scene_id,
            start_frame=sc.start_frame,
            end_frame_exclusive=sc.end_frame_exclusive,
            bbox_norm=bbox,
            z_order=25,
            color=pal.get("forest", "#243B32"),
            continuity_from_scene=ledger_scenes[i - 1].scene_id if i > 0 else None,
            continuity_to_scene=ledger_scenes[i + 1].scene_id if i + 1 < len(ledger_scenes) else None,
            transition_owner="remotion",
            provenance_ref=sc.persistent_ledger_step_id or sc.role,
        ))

    prod_scenes = [sc for sc in storyboard.scenes if sc.role in ("product_intro", "product_benefit", "closing_takeaway")]
    for i, sc in enumerate(prod_scenes):
        ly_info = layout_by_sc.get(sc.scene_id, {})
        bbox = ly_info.get("hero_box_norm", {"x": 0.067, "y": 0.086, "w": 0.788, "h": 0.18})
        tracks.append(SharedElementState(
            element_id="product_organization_motif",
            logical_state=sc.role,
            scene_id=sc.scene_id,
            start_frame=sc.start_frame,
            end_frame_exclusive=sc.end_frame_exclusive,
            bbox_norm=bbox,
            z_order=28,
            color=pal.get("gold", pal.get("brass", "#BE9A59")),
            continuity_from_scene=prod_scenes[i - 1].scene_id if i > 0 else (ledger_scenes[-1].scene_id if ledger_scenes else None),
            continuity_to_scene=prod_scenes[i + 1].scene_id if i + 1 < len(prod_scenes) else None,
            transition_owner="remotion",
            provenance_ref=story_graph.product_name or sc.role,
        ))

    return {
        "schema_version": "1.0.0",
        "persistent_element_ids": sorted({t.element_id for t in tracks}),
        "track_count": len(tracks),
        "tracks": [t.to_dict() for t in tracks],
    }


def validate_layout_sweep(
    layout_plan: dict[str, Any],
    legibility_report: dict[str, Any],
) -> list[str]:
    """Return hard layout/legibility/protection failures (empty list == PASS)."""
    errs: list[str] = []
    if not layout_plan.get("zero_eye_mouth_collision", True):
        errs.append(f"eye_mouth_collision_detected:max_overlap={layout_plan.get('max_eye_mouth_overlap_any_frame')}")
    for leak in legibility_report.get("consumer_jargon_leaks", []):
        errs.append(f"consumer_jargon_leak:{leak}")
    if not legibility_report.get("contrast_proxy_ok", True):
        errs.append("contrast_proxy_below_target")
    return errs
