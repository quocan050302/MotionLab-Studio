"""Phase R2 (Section 05): Hierarchical Motion Choreography, Dual-Anchor Reframing, and Adaptive Density.

100% deterministic, rule + media-signal driven (ZERO LLM, ZERO hardcoded video/filename logic).
Coordinates main motion + supporting editorial overlays/color + genuine quiet holds across
both multi-cut and continuous single-shot videos.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from typing import Any

from motionlab.config import MotionLabConfig
from motionlab.policies import ProfilePolicy
from motionlab.registry import EffectRegistry
from motionlab.schemas import (
    AnalysisResult,
    EffectEvent,
    EffectSpec,
    ProtectedZone,
    SubjectRegion,
    sanitize_native,
)


@dataclass
class BeatSpec:
    beat_id: str
    start_frame: int
    end_frame_exclusive: int
    beat_role: str  # hook | activity_focus | emphasis | visual_reset | quiet_hold | closing_hold | cut_bridge
    anchor_x: float
    anchor_y: float
    protected_union: dict[str, float]
    confidence: float
    evidence_refs: list[str]
    allow_overlay: bool = True
    allow_color: bool = True
    main_event_id: str | None = None
    support_event_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return sanitize_native(asdict(self))


def _overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
    return a0 < b1 and b0 < a1


def _box_union(boxes: list[tuple[float, float, float, float]]) -> dict[str, float]:
    if not boxes:
        return {"x": 0.15, "y": 0.15, "w": 0.70, "h": 0.70}
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes)
    y1 = max(b[1] + b[3] for b in boxes)
    return {
        "x": round(max(0.0, x0), 4),
        "y": round(max(0.0, y0), 4),
        "w": round(min(1.0 - max(0.0, x0), max(0.05, x1 - x0)), 4),
        "h": round(min(1.0 - max(0.0, y0), max(0.05, y1 - y0)), 4),
    }


def ensure_protected_visibility(
    anchor_x: float,
    anchor_y: float,
    zoom: float,
    protected_union: dict[str, float],
) -> tuple[float, float, float]:
    """Clamp anchor and zoom so BOTH face and active_task_region stay safely inside the visible viewport."""
    if zoom <= 1.001:
        return round(anchor_x, 4), round(anchor_y, 4), 1.0

    px0 = protected_union.get("x", 0.15)
    py0 = protected_union.get("y", 0.15)
    pw = protected_union.get("w", 0.70)
    ph = protected_union.get("h", 0.70)
    px1 = min(1.0, px0 + pw)
    py1 = min(1.0, py0 + ph)

    # Visible span at zoom z is 1/z. Ensure protected union fits with >=3% breathing margin.
    needed_w = min(0.96, pw + 0.06)
    needed_h = min(0.96, ph + 0.06)
    max_safe_zoom = min(1.0 / max(0.25, needed_w), 1.0 / max(0.25, needed_h), 1.12)
    safe_zoom = max(1.02, min(zoom, max(1.03, max_safe_zoom)))

    half_vw = 0.5 / safe_zoom
    half_vh = 0.5 / safe_zoom
    # Viewport center must keep [px0, px1] inside [cx - half_vw, cx + half_vw]
    min_cx = max(half_vw, px1 - half_vw)
    max_cx = min(1.0 - half_vw, px0 + half_vw)
    min_cy = max(half_vh, py1 - half_vh)
    max_cy = min(1.0 - half_vh, py0 + half_vh)

    if min_cx <= max_cx:
        cx = max(min_cx, min(max_cx, anchor_x))
    else:
        cx = (px0 + px1) * 0.5
    if min_cy <= max_cy:
        cy = max(min_cy, min(max_cy, anchor_y))
    else:
        cy = (py0 + py1) * 0.5

    return round(max(0.22, min(0.78, cx)), 4), round(max(0.22, min(0.78, cy)), 4), round(safe_zoom, 4)


def optimize_safe_anchor(
    start_frame: int,
    end_frame_exclusive: int,
    analysis: AnalysisResult,
    bias_to_task: bool = False,
) -> tuple[float, float, dict[str, float]]:
    """Compute smooth dual-anchor framing preserving face + foreground active task region."""
    faces = [
        (s.x, s.y, s.w, s.h)
        for s in analysis.subject_regions
        if _overlaps(start_frame, end_frame_exclusive, s.start_frame, s.end_frame_exclusive)
        and s.confidence >= 0.55
        and not s.ambiguous
    ]
    tasks = [
        (z.x, z.y, z.w, z.h)
        for z in analysis.protected_zones
        if z.reason == "active_task_region"
        and _overlaps(start_frame, end_frame_exclusive, z.start_frame, z.end_frame_exclusive)
    ]
    union = _box_union(faces + tasks)

    if faces and tasks:
        fx = sum(b[0] + b[2] * 0.5 for b in faces) / len(faces)
        fy = sum(b[1] + b[3] * 0.45 for b in faces) / len(faces)
        tx = sum(b[0] + b[2] * 0.5 for b in tasks) / len(tasks)
        ty = sum(b[1] + b[3] * 0.5 for b in tasks) / len(tasks)
        w_task = 0.58 if bias_to_task else 0.32
        ax = fx * (1.0 - w_task) + tx * w_task
        ay = fy * (1.0 - w_task) + ty * w_task
        return round(ax, 4), round(ay, 4), union

    if faces:
        fx = sum(b[0] + b[2] * 0.5 for b in faces) / len(faces)
        fy = sum(b[1] + b[3] * 0.42 for b in faces) / len(faces)
        return round(fx, 4), round(fy, 4), union

    if tasks:
        tx = sum(b[0] + b[2] * 0.5 for b in tasks) / len(tasks)
        ty = sum(b[1] + b[3] * 0.5 for b in tasks) / len(tasks)
        return round(tx, 4), round(ty, 4), union

    return 0.5, 0.46, union


def _find_open_corner_for_overlay(protected_union: dict[str, float]) -> tuple[float, float]:
    """Choose an open peripheral corner strictly outside protected face & active task regions."""
    px0 = protected_union.get("x", 0.2)
    py0 = protected_union.get("y", 0.2)
    px1 = px0 + protected_union.get("w", 0.6)
    py1 = py0 + protected_union.get("h", 0.6)
    # Candidate safe corner centers: top-right, top-left, bottom-right, bottom-left
    corners = [(0.86, 0.09), (0.14, 0.09), (0.86, 0.91), (0.14, 0.91)]
    for cx, cy in corners:
        inside_prot = (px0 - 0.06 <= cx <= px1 + 0.06) and (py0 - 0.06 <= cy <= py1 + 0.06)
        if not inside_prot:
            return cx, cy
    return (0.86, 0.08)


def _best_waveform_position(protected_zones: list[ProtectedZone], sf: int, ef: int) -> str | None:
    """Return 'bottom' or 'top' if safe margin is clear of UI/face/active-task collisions."""
    bot_blocked = any(
        z.reason in ("ui_text", "letterbox", "face")
        and _overlaps(sf, ef, z.start_frame, z.end_frame_exclusive)
        and (z.y + z.h) >= 0.88
        for z in protected_zones
    )
    if not bot_blocked:
        return "bottom"
    top_blocked = any(
        z.reason in ("ui_text", "letterbox", "face")
        and _overlaps(sf, ef, z.start_frame, z.end_frame_exclusive)
        and z.y <= 0.14
        for z in protected_zones
    )
    if not top_blocked:
        return "top"
    return None


def build_beats(
    analysis: AnalysisResult,
    policy: ProfilePolicy,
    config: MotionLabConfig,
) -> list[BeatSpec]:
    """Partition timeline into hierarchical editorial beats using acoustic pauses, motion peaks, cuts, and dwell rhythm."""
    fps_f = analysis.fps.as_float()
    total_f = analysis.duration_frames
    dur_s = total_f / max(1.0, fps_f)
    if dur_s < 1.2:
        return []

    beats: list[BeatSpec] = []
    # Identify genuine quiet breathing windows (acoustic pauses >= 0.55s) so choreography respects natural silence
    quiet_spans: list[tuple[int, int, str]] = []
    for idx, p in enumerate(analysis.pauses):
        p_dur = (p.end_frame_exclusive - p.start_frame) / fps_f
        if p_dur >= 0.55 and p.start_frame > int(fps_f * 3.0) and p.end_frame_exclusive < total_f - int(fps_f * 2.5):
            # Avoid clustering two quiet holds closer than 6.0s apart
            if quiet_spans and (p.start_frame - quiet_spans[-1][1]) < int(fps_f * 6.0):
                continue
            quiet_spans.append((p.start_frame, p.end_frame_exclusive, f"pause_{idx:03d}"))

    # Guarantee at least 1 natural quiet breathing window on >= 15s clips even when speech has no long silence
    if not quiet_spans and dur_s >= 15.0:
        mid_q0 = int(round(total_f * 0.56))
        mid_q1 = min(total_f - int(round(fps_f * 3.2)), mid_q0 + int(round(fps_f * 1.4)))
        if mid_q1 - mid_q0 >= int(round(fps_f * 0.9)):
            quiet_spans.append((mid_q0, mid_q1, "mid_editorial_breather"))

    # For very short clips (< 5s), build 1-2 clean beats
    if dur_s < 5.0:
        sf, ef = 0, max(4, total_f - max(2, int(round(fps_f * 0.2))))
        ax, ay, union = optimize_safe_anchor(sf, ef, analysis)
        beats.append(BeatSpec("beat_000_hook", sf, ef, "hook", ax, ay, union, 0.78, ["short_clip_opening"]))
        return beats

    # Compute adaptive beat duration from motion_density & duration
    density_mult = {"low": 1.35, "adaptive": 1.0, "rich": 0.80}.get(config.motion_density, 1.0)
    if policy.protect_edges_strict:
        density_mult *= 1.45

    hook_end = min(total_f - int(fps_f * 2.0), int(round(fps_f * 2.4)))
    closing_start = max(hook_end + int(fps_f * 1.5), total_f - int(round(fps_f * 2.8)))

    # 1. Opening Hook Beat (clean full-bleed entry)
    ax, ay, union = optimize_safe_anchor(0, hook_end, analysis, bias_to_task=False)
    beats.append(
        BeatSpec("beat_000_hook", 0, hook_end, "hook", ax, ay, union, 0.88, ["opening_hook_beat"], allow_overlay=True, allow_color=False)
    )

    # 2. Middle Narrative / Action Beats (respecting cuts, motion peaks, and quiet holds)
    base_beat_f = int(round(fps_f * 2.8 * density_mult))
    peak_frames = [aw.start_frame for aw in analysis.action_windows if aw.window_type == "action_peak"]
    cut_frames = [c.frame for c in analysis.cuts]
    used_quiet: set[str] = set()

    cursor = hook_end + int(round(fps_f * 0.25))
    b_idx = 1
    role_cycle = ["activity_focus", "emphasis", "visual_reset", "activity_focus", "emphasis", "visual_reset"]
    cycle_pos = 0

    while cursor + int(round(fps_f * 1.2)) < closing_start:
        # Check if an upcoming quiet breathing pause starts right here (within 1.1s of cursor)
        matched_quiet = None
        for qs, qe, qref in quiet_spans:
            if qref in used_quiet:
                continue
            if (qs - cursor) <= int(round(fps_f * 1.1)) and qe > cursor:
                q_start = max(cursor, qs)
                q_end = min(closing_start - int(round(fps_f * 0.2)), max(qe, q_start + int(round(fps_f * 1.1))))
                matched_quiet = (q_start, q_end, qref)
                break

        if matched_quiet is not None:
            qsf, qef, qref = matched_quiet
            used_quiet.add(qref)
            if qef - qsf >= int(round(fps_f * 0.65)):
                ax, ay, union = optimize_safe_anchor(qsf, qef, analysis, bias_to_task=False)
                beats.append(
                    BeatSpec(
                        f"beat_{b_idx:03d}_quiet", qsf, qef, "quiet_hold", ax, ay, union,
                        0.88, [qref], allow_overlay=False, allow_color=False,
                    )
                )
                b_idx += 1
                cursor = qef + int(round(fps_f * 0.15))
                continue

        # Standard active beat aligned near local action peaks, clamped cleanly before cuts or upcoming quiet pauses
        end_f = min(closing_start - int(round(fps_f * 0.2)), cursor + base_beat_f)
        for qs, _qe, qref in quiet_spans:
            if qref not in used_quiet and cursor + int(round(fps_f * 1.1)) < qs < end_f:
                end_f = max(cursor + int(round(fps_f * 1.1)), qs - int(round(fps_f * 0.12)))
                break
        # Snap end cleanly before an upcoming hard cut so camera transforms don't cross cuts
        for cf in cut_frames:
            if cursor + int(round(fps_f * 1.2)) < cf < end_f:
                end_f = max(cursor + int(round(fps_f * 1.1)), cf - int(round(fps_f * 0.25)))
                break
        if end_f - cursor < int(round(fps_f * 1.1)):
            cursor = end_f + int(round(fps_f * 0.3))
            continue

        # Check if this window holds a motion peak or active task ROI
        has_peak = any(cursor <= pf < end_f for pf in peak_frames)
        has_task = any(
            z.reason == "active_task_region" and _overlaps(cursor, end_f, z.start_frame, z.end_frame_exclusive)
            for z in analysis.protected_zones
        )
        role = role_cycle[cycle_pos % len(role_cycle)]
        if has_task and role == "emphasis" and (cycle_pos % 2 == 0):
            role = "activity_focus"
        cycle_pos += 1

        ax, ay, union = optimize_safe_anchor(cursor, end_f, analysis, bias_to_task=(role == "activity_focus"))
        ev_refs = [f"dwell_{cursor}_{end_f}"]
        if has_peak:
            ev_refs.append("local_motion_peak")
        if has_task:
            ev_refs.append("active_task_region")
        beats.append(
            BeatSpec(
                f"beat_{b_idx:03d}_{role}",
                cursor, end_f, role, ax, ay, union,
                0.84 if (has_peak or has_task) else 0.72,
                ev_refs,
                allow_overlay=True,
                allow_color=(role in ("emphasis", "visual_reset")),
            )
        )
        b_idx += 1
        # Small calm breathing gap between beats so camera settles smoothly at 1.0 before next beat
        cursor = end_f + int(round(fps_f * 0.35))

    # 3. Calm Closing Beat (stable ending preserving final frames and speech)
    if total_f - closing_start >= int(round(fps_f * 1.4)):
        ax, ay, union = optimize_safe_anchor(closing_start, total_f, analysis, bias_to_task=False)
        beats.append(
            BeatSpec(
                f"beat_{b_idx:03d}_closing",
                closing_start, total_f - 1, "closing_hold", ax, ay, union,
                0.86, ["closing_calm_release"], allow_overlay=False, allow_color=True,
            )
        )

    return beats


def compose_layer_tracks(
    beats: list[BeatSpec],
    registry: EffectRegistry,
    policy: ProfilePolicy,
    analysis: AnalysisResult,
    config: MotionLabConfig,
    engine_caps: dict[str, Any],
    rng: random.Random,
) -> tuple[list[EffectEvent], list[dict[str, Any]]]:
    """Map hierarchical beats -> compatible multi-layer EffectEvents (camera/mask + HyperFrames overlay + color)."""
    fps_f = analysis.fps.as_float()
    total_f = analysis.duration_frames
    caps = set(analysis.capabilities)
    palette = analysis.dominant_palette or {}
    pal_primary = str(palette.get("primary_hex", "#F4EBDD"))
    pal_secondary = str(palette.get("secondary_hex", "#D8BE94"))
    pal_tertiary = str(palette.get("tertiary_hex", "#B5C4B1"))
    palette_cycle = [pal_primary, pal_secondary, pal_tertiary]

    chosen: list[EffectEvent] = []
    decisions: list[dict[str, Any]] = []
    ev_seq = 0
    used_main_history: list[str] = []
    used_overlay_history: list[str] = []
    used_color_history: list[str] = []

    def _can_use_spec(spec: EffectSpec, sf: int, ef: int) -> tuple[bool, str, str | None]:
        if policy.name not in spec.supported_profiles:
            return False, f"profile_disallowed:{policy.name}", None
        if not set(spec.required_features).issubset(caps):
            return False, "missing_capability", None
        if spec.safety_class == "subject_aware" and not policy.allow_subject_zoom:
            return False, "policy_disallows_subject_zoom", None
        if spec.family == "layout" and spec.layer == "mask" and not policy.allow_layout_masks:
            return False, "policy_disallows_layout_masks", None
        if spec.id == "L05" and not (policy.allow_split_layout and config.allow_split_layout):
            return False, "split_opt_in_required", None
        if spec.safety_class == "audio_reactive" and not policy.allow_audio_reactive:
            return False, "policy_disallows_audio_reactive", None
        dur_s = (ef - sf) / fps_f
        if dur_s < spec.min_duration_s * 0.85:
            return False, "beat_too_short", None
        eng = registry.resolve_engine(spec, engine_caps, preferred_engine=config.preferred_engine)
        if eng is None:
            return False, "no_available_engine", None
        # Check existing-effect flash protection
        for z in analysis.protected_zones:
            if z.reason == "existing_effect" and _overlaps(sf, ef, z.start_frame, z.end_frame_exclusive):
                if spec.layer in ("transform", "mask", "transition"):
                    return False, "existing_effect_protected", None
        # Check high local motion before adding spatial zoom
        m_hits = [mw.motion_score for mw in analysis.motion_windows if _overlaps(sf, ef, mw.start_frame, mw.end_frame_exclusive)]
        if m_hits and (sum(m_hits) / len(m_hits)) > 0.50 and spec.layer in ("transform", "mask"):
            return False, "high_motion_skip_spatial", None
        return True, "ok", eng

    def _make_event(
        spec: EffectSpec,
        sf: int,
        ef: int,
        engine: str,
        beat: BeatSpec,
        accent_hex: str,
        extra_reasons: list[str],
    ) -> EffectEvent:
        nonlocal ev_seq
        # Clamp duration strictly to spec max_duration_s
        max_f = max(4, int(round(spec.max_duration_s * fps_f)))
        ef_clamped = min(total_f, sf + min(ef - sf, max_f))
        if ef_clamped <= sf + 2:
            ef_clamped = min(total_f, sf + 4)

        params: dict[str, Any] = {k: b.default for k, b in spec.parameter_schema.items()}
        style_scale = {"minimal": 0.78, "editorial": 0.90, "dynamic": 1.0}.get(config.style, 0.90)
        if "intensity" in params:
            params["intensity"] = round(min(spec.max_intensity, float(params["intensity"]) * policy.intensity_scale * style_scale), 4)

        # Dual-anchor safe visibility clamping so face + hands NEVER clip out of frame!
        for z_key in ("max_zoom", "start_zoom", "target_zoom", "overscan_zoom"):
            if z_key in params:
                sx, sy, sz = ensure_protected_visibility(beat.anchor_x, beat.anchor_y, float(params[z_key]), beat.protected_union)
                b_z = spec.parameter_schema[z_key]
                params[z_key] = round(max(b_z.min_val or 1.0, min(b_z.max_val or 1.12, sz)), 4)
                if "anchor_x" in params:
                    bx = spec.parameter_schema["anchor_x"]
                    params["anchor_x"] = round(max(bx.min_val or 0.0, min(bx.max_val or 1.0, sx)), 4)
                if "anchor_y" in params:
                    by = spec.parameter_schema["anchor_y"]
                    params["anchor_y"] = round(max(by.min_val or 0.0, min(by.max_val or 1.0, sy)), 4)

        if "anchor_x" in params and not any(k in params for k in ("max_zoom", "start_zoom", "target_zoom", "overscan_zoom")):
            bx = spec.parameter_schema["anchor_x"]
            params["anchor_x"] = round(max(bx.min_val or 0.0, min(bx.max_val or 1.0, beat.anchor_x)), 4)
        if "anchor_y" in params and not any(k in params for k in ("max_zoom", "start_zoom", "target_zoom", "overscan_zoom")):
            by = spec.parameter_schema["anchor_y"]
            params["anchor_y"] = round(max(by.min_val or 0.0, min(by.max_val or 1.0, beat.anchor_y)), 4)

        if "accent_hex" in params and accent_hex in (spec.parameter_schema["accent_hex"].choices or ()):
            params["accent_hex"] = accent_hex

        if spec.id == "G02":
            cx, cy = _find_open_corner_for_overlay(beat.protected_union)
            params["cx"] = round(max(0.12, min(0.88, cx)), 4)
            params["cy"] = round(max(0.08, min(0.88, cy)), 4)
        if spec.id == "G05":
            wf_pos = _best_waveform_position(analysis.protected_zones, sf, ef_clamped)
            params["position"] = wf_pos or "bottom"
        if "direction" in params and spec.parameter_schema["direction"].choices:
            choices = spec.parameter_schema["direction"].choices
            params["direction"] = choices[(ev_seq + rng.randrange(len(choices))) % len(choices)]

        validated = spec.validate_params(params)
        ev = EffectEvent(
            id=f"ev_{ev_seq:03d}_{spec.id}",
            effect_id=spec.id,
            effect_version=spec.version,
            start_frame=int(sf),
            end_frame_exclusive=int(ef_clamped),
            layer=spec.layer,
            region=dict(beat.protected_union),
            parameters=validated,
            easing="spring_damped" if spec.id in ("C03", "C08") else "ease_in_out_cubic",
            priority=10 if spec.family == "transition" else 5,
            engine=engine,
            reason_codes=[f"family:{spec.family}", f"beat:{beat.beat_role}"] + extra_reasons,
            evidence_refs=list(beat.evidence_refs),
            fallback_id=spec.fallback_id,
            uses_neighbor_frames=spec.uses_neighbor_frames,
            beat_id=beat.beat_id,
            intended_family=spec.family,
        )
        ev_seq += 1
        return ev

    # 1. First schedule real detected scene cut transitions (bounded & duration-preserving!)
    cut_candidates = ["T01", "T04", "T03", "T06", "T05", "T07", "T02", "T08"]
    max_trans = max(0, int(round(policy.max_transitions_per_15s * max(0.5, (total_f / fps_f) / 15.0))))
    trans_used = 0
    for c_idx, cut in enumerate(analysis.cuts):
        if trans_used >= max_trans:
            break
        margin_f = max(4, int(round(0.35 * fps_f)))
        if cut.frame < margin_f or cut.frame > total_f - margin_f:
            continue
        prev_cut = analysis.cuts[c_idx - 1].frame if c_idx > 0 else 0
        next_cut = analysis.cuts[c_idx + 1].frame if c_idx + 1 < len(analysis.cuts) else total_f
        if (cut.frame - prev_cut) / fps_f < 0.70 or (next_cut - cut.frame) / fps_f < 0.70:
            continue
        half = max(3, int(round(0.22 * fps_f)))
        sf, ef = max(1, cut.frame - half), min(total_f - 1, cut.frame + half)
        for tid in cut_candidates:
            if not registry.has(tid):
                continue
            spec = registry.get(tid)
            ok, _why, eng = _can_use_spec(spec, sf, ef)
            if ok and eng:
                # Rotate transition IDs for diversity
                if trans_used > 0 and tid == cut_candidates[0]:
                    continue
                dummy_beat = BeatSpec(f"beat_cut_{c_idx:02d}", sf, ef, "cut_bridge", 0.5, 0.5, {"x": 0.15, "y": 0.15, "w": 0.7, "h": 0.7}, cut.confidence, [f"cut_{c_idx:03d}"])
                ev = _make_event(spec, sf, ef, eng, dummy_beat, pal_primary, ["detected_scene_cut"])
                chosen.append(ev)
                trans_used += 1
                decisions.append({"beat_id": dummy_beat.beat_id, "status": "SELECTED", "effect_id": tid, "engine": eng, "frame_range": [sf, ef]})
                break

    # 2. Schedule hierarchical editorial beats (main motion + compatible supporting overlay/color + quiet holds)
    # Role -> ordered pool of refined editorial primitives (rotated to avoid repetition!)
    # On continuous single-shot talking-head footage (0 scene cuts), avoid mid-sentence aperture-shrinking pillbox masks (L01/C10)
    # and favor seamless lighting/focus masks (L02, L03, L04) + smooth camera phrasing.
    single_shot_speaker = (policy.name == "talking-head" and len(analysis.cuts) == 0)
    role_main_pools: dict[str, list[str]] = {
        "hook": ["C01", "C02", "V02"],
        "activity_focus": ["C09", "C04", "L03", "C06", "L02", "C05", "C01"],
        "emphasis": (
            ["C03", "L02", "C08", "L04", "C05", "L03"]
            if single_shot_speaker
            else ["C03", "L01", "C08", "L04", "C05", "L10", "C10"]
        ),
        "visual_reset": ["C02", "C05", "L03", "C09", "C06"],
        "closing_hold": ["C02", "C01", "V04"],
    }
    overlay_pool = ["G01", "G05", "L08", "G02", "G03", "G06"]
    color_pool = ["V02", "V01", "V04", "V03", "V05"]

    overlay_count = 0
    max_overlays = max(2, int(round((total_f / fps_f) / 6.5))) if not policy.protect_edges_strict else 1
    if config.motion_density == "low":
        max_overlays = max(1, max_overlays - 1)
    color_count = 0
    max_colors = max(1, int(round((total_f / fps_f) / 9.0)))

    for b_num, beat in enumerate(beats):
        sf, ef = beat.start_frame, beat.end_frame_exclusive
        if beat.beat_role == "quiet_hold":
            decisions.append({
                "beat_id": beat.beat_id,
                "status": "QUIET_HOLD_PRESERVED",
                "frame_range": [sf, ef],
                "reason": "intentional_breathing_room_no_overlay_spam",
            })
            continue

        # Skip beats overlapping a scheduled cut transition
        if any(_overlaps(sf, ef, ev.start_frame, ev.end_frame_exclusive) and ev.layer == "transition" for ev in chosen):
            decisions.append({"beat_id": beat.beat_id, "status": "SKIPPED", "reason": "yields_to_cut_transition"})
            continue

        # Pick main motion for beat (avoid repeating last 2 main effects!)
        main_pool = role_main_pools.get(beat.beat_role, ["C01", "C05"])
        if policy.protect_edges_strict:
            main_pool = ["V02", "V04", "V01"]  # Screen-recording / gameplay safe: zero crop/zoom!

        main_ev: EffectEvent | None = None
        # Prefer effects not recently used
        ordered_pool = sorted(
            [eid for eid in main_pool if registry.has(eid)],
            key=lambda eid: (
                2 if eid in used_main_history[-2:] else (1 if eid in used_main_history else 0),
                main_pool.index(eid),
            ),
        )
        for eid in ordered_pool:
            spec = registry.get(eid)
            ok, why, eng = _can_use_spec(spec, sf, ef)
            if not ok or eng is None:
                continue
            # Ensure no incompatible overlap with already-chosen events
            conflict = any(
                _overlaps(sf, ef, prev.start_frame, prev.end_frame_exclusive)
                and (spec.layer == prev.layer or spec.layer in registry.get(prev.effect_id).incompatibilities or prev.layer in spec.incompatibilities)
                for prev in chosen
            )
            if conflict:
                continue
            accent = palette_cycle[b_num % len(palette_cycle)]
            main_ev = _make_event(spec, sf, ef, eng, beat, accent, ["choreography_main"])
            chosen.append(main_ev)
            used_main_history.append(eid)
            beat.main_event_id = main_ev.id
            decisions.append({
                "beat_id": beat.beat_id,
                "status": "SELECTED_MAIN",
                "event_id": main_ev.id,
                "effect_id": eid,
                "engine": eng,
                "frame_range": [main_ev.start_frame, main_ev.end_frame_exclusive],
            })
            break

        # Optional supporting Layer 2: Peripheral editorial graphic overlay (routed to HyperFrames when available!)
        # Multi-layer moments on selected beats (not every beat, keeping clean rhythm & breathing space!)
        should_add_overlay = (
            beat.allow_overlay
            and overlay_count < max_overlays
            and (beat.beat_role in ("hook", "activity_focus", "emphasis") or (policy.protect_edges_strict and overlay_count == 0))
            and (b_num == 0 or (b_num % 2 == 1))
        )
        if should_add_overlay:
            ov_ordered = sorted(
                [eid for eid in overlay_pool if registry.has(eid)],
                key=lambda eid: (2 if eid in used_overlay_history[-2:] else (1 if eid in used_overlay_history else 0), overlay_pool.index(eid)),
            )
            # Stagger overlay entrance slightly inside beat for anticipation/settle layering
            ov_sf = min(ef - max(8, int(round(fps_f * 0.9))), sf + int(round(fps_f * 0.25)))
            ov_ef = min(ef, ov_sf + int(round(fps_f * 2.1)))
            if ov_ef - ov_sf >= max(8, int(round(fps_f * 0.9))):
                for oid in ov_ordered:
                    if oid == "G05" and _best_waveform_position(analysis.protected_zones, ov_sf, ov_ef) is None:
                        continue
                    spec = registry.get(oid)
                    ok, _why, eng = _can_use_spec(spec, ov_sf, ov_ef)
                    if not ok or eng is None:
                        continue
                    # Enforce overlay cooldown (>= 2.2s gap between overlays so frame stays crisp, never cluttered)
                    ov_gap = int(round(fps_f * 2.0))
                    if any(
                        prev.layer == "overlay" and max(ov_sf - prev.end_frame_exclusive, prev.start_frame - ov_ef) < ov_gap
                        for prev in chosen
                    ):
                        continue
                    accent = palette_cycle[(overlay_count + 1) % len(palette_cycle)]
                    ov_ev = _make_event(spec, ov_sf, ov_ef, eng, beat, accent, ["choreography_multilayer_overlay"])
                    chosen.append(ov_ev)
                    used_overlay_history.append(oid)
                    beat.support_event_ids.append(ov_ev.id)
                    overlay_count += 1
                    decisions.append({
                        "beat_id": beat.beat_id,
                        "status": "SELECTED_SUPPORT_OVERLAY",
                        "event_id": ov_ev.id,
                        "effect_id": oid,
                        "engine": eng,
                        "frame_range": [ov_ev.start_frame, ov_ev.end_frame_exclusive],
                    })
                    break

        # Optional supporting Layer 3: Subtle editorial color/optical polish (non-overlapping with other color events)
        if beat.allow_color and color_count < max_colors and (main_ev is None or main_ev.layer != "color"):
            col_ordered = sorted(
                [eid for eid in color_pool if registry.has(eid)],
                key=lambda eid: (1 if eid in used_color_history else 0, color_pool.index(eid)),
            )
            for cid in col_ordered:
                spec = registry.get(cid)
                ok, _why, eng = _can_use_spec(spec, sf, ef)
                if not ok or eng is None:
                    continue
                if any(prev.layer == "color" and _overlaps(sf, ef, prev.start_frame, prev.end_frame_exclusive) for prev in chosen):
                    continue
                col_ev = _make_event(spec, sf, ef, eng, beat, pal_primary, ["choreography_color_polish"])
                chosen.append(col_ev)
                used_color_history.append(cid)
                beat.support_event_ids.append(col_ev.id)
                color_count += 1
                decisions.append({
                    "beat_id": beat.beat_id,
                    "status": "SELECTED_SUPPORT_COLOR",
                    "event_id": col_ev.id,
                    "effect_id": cid,
                    "engine": eng,
                    "frame_range": [col_ev.start_frame, col_ev.end_frame_exclusive],
                })
                break

    chosen.sort(key=lambda e: (e.start_frame, e.layer, e.id))
    return chosen, decisions


def evaluate_candidate_sequence(
    beats: list[BeatSpec],
    events: list[EffectEvent],
    registry: EffectRegistry,
) -> dict[str, Any]:
    """Summarize choreography sequence diversity, multi-layer moments, and quiet-hold pacing."""
    families = sorted({registry.get(e.effect_id).family for e in events})
    unique_effects = sorted({e.effect_id for e in events})
    engines = sorted({e.engine for e in events})
    quiet_beats = sum(1 for b in beats if b.beat_role == "quiet_hold")
    active_beats = sum(1 for b in beats if b.main_event_id is not None)
    multilayer_beats = sum(1 for b in beats if b.main_event_id is not None and len(b.support_event_ids) >= 1)
    visual_resets = sum(1 for b in beats if b.beat_role in ("visual_reset", "closing_hold") and b.main_event_id is not None)
    return {
        "total_beats": len(beats),
        "active_motion_beats": active_beats,
        "quiet_hold_beats": quiet_beats,
        "multilayer_beats": multilayer_beats,
        "visual_reset_beats": visual_resets,
        "total_events": len(events),
        "distinct_effects": len(unique_effects),
        "effect_ids": unique_effects,
        "distinct_families": len(families),
        "families": families,
        "planned_engines": engines,
    }


def plan_choreography(
    analysis: AnalysisResult,
    policy: ProfilePolicy,
    config: MotionLabConfig,
    engine_caps: dict[str, Any],
    registry: EffectRegistry,
    rng: random.Random,
) -> tuple[list[EffectEvent], dict[str, Any]]:
    """Run full R2 hierarchical motion choreography -> (events, choreography_report)."""
    beats = build_beats(analysis, policy, config)
    events, decisions = compose_layer_tracks(beats, registry, policy, analysis, config, engine_caps, rng)
    metrics = evaluate_candidate_sequence(beats, events, registry)
    report = {
        "style": config.style,
        "motion_density": config.motion_density,
        "dominant_palette": analysis.dominant_palette,
        "metrics": metrics,
        "beats": [b.to_dict() for b in beats],
        "decisions": decisions,
    }
    return events, report
