"""Phase D3-D6: Deterministic rule-based automatic effect planner, layout resolver, and plan validator."""

from __future__ import annotations

import math
import random
from typing import Any

from motionlab import PLANNER_VERSION, SCHEMA_VERSION
from motionlab.config import MotionLabConfig
from motionlab.events import build_candidates
from motionlab.policies import ProfilePolicy, select_profile
from motionlab.registry import EffectRegistry, load_effects
from motionlab.schemas import (
    AnalysisResult,
    CandidateEvent,
    EditPlan,
    EffectEvent,
    EffectSpec,
    MediaInfo,
    ProtectedZone,
    SchemaValidationError,
    SubjectRegion,
    ensure_finite,
)

STRONG_LAYERS = {"transform", "mask", "transition"}


def resolve_layout(
    media: MediaInfo,
    analysis: AnalysisResult,
    config: MotionLabConfig,
) -> tuple[int, int, str, list[str]]:
    """Phase D5: Resolve output dimensions and framing policy without blind cropping."""
    warnings: list[str] = []
    src_w = media.display_width
    src_h = media.display_height

    if config.aspect == "source":
        if config.quality == "preview":
            # Scale proportionally to <= 720p short-edge / 960p portrait equivalent
            if src_h >= src_w and src_h > 960:
                scale = 960.0 / src_h
                out_w = max(2, int(round(src_w * scale / 2.0)) * 2)
                out_h = 960
            elif src_w > src_h and src_w > 1280:
                scale = 1280.0 / src_w
                out_w = 1280
                out_h = max(2, int(round(src_h * scale / 2.0)) * 2)
            else:
                out_w = src_w + (src_w % 2)
                out_h = src_h + (src_h % 2)
        else:
            out_w = src_w + (src_w % 2)
            out_h = src_h + (src_h % 2)
        if out_w != src_w or out_h != src_h:
            warnings.append(f"Adjusted odd source dimensions {src_w}x{src_h} -> {out_w}x{out_h} for yuv420p alignment.")
        return out_w, out_h, "preserve", warnings

    # Explicit opt-in --aspect 9:16
    target_w, target_h = (540, 960) if config.quality == "preview" else (1080, 1920)
    src_ratio = src_w / max(1, src_h)
    target_ratio = target_w / target_h

    if abs(src_ratio - target_ratio) / target_ratio <= 0.03:
        return target_w, target_h, "preserve", warnings

    # Source is NOT 9:16: default fit/pad (no blind crop!) unless stable single-subject tracking permits smart crop
    face_cov = float(analysis.summary_stats.get("face_coverage_frames", 0)) / max(1, analysis.duration_frames)
    has_ui = "busy_ui_or_text_detected" in analysis.quality_flags
    has_multi = "multi_subject_ambiguous" in analysis.quality_flags

    if face_cov >= 0.65 and not has_ui and not has_multi and config.profile in ("talking-head", "auto"):
        warnings.append("Using subject-tracked smart_crop_9x16 based on >=65% confident single-subject tracking.")
        return target_w, target_h, "smart_crop_9x16", warnings

    policy = "pad_blur_9x16" if config.pad_background == "blur" else "pad_solid_9x16"
    warnings.append(
        f"Source aspect {src_w}x{src_h} differs from 9:16; using {policy} fit/pad to avoid blind content loss."
    )
    return target_w, target_h, policy, warnings


def _overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
    return a0 < b1 and b0 < a1


def _subject_at_window(
    regions: list[SubjectRegion], start_f: int, end_f: int
) -> SubjectRegion | None:
    best: SubjectRegion | None = None
    best_overlap = 0
    for r in regions:
        ov = min(end_f, r.end_frame_exclusive) - max(start_f, r.start_frame)
        if ov > best_overlap and r.confidence >= 0.60 and not r.ambiguous:
            best_overlap = ov
            best = r
    return best


def _window_motion_stats(analysis: AnalysisResult, start_f: int, end_f: int) -> tuple[float, float, float, float]:
    """Return (mean_motion, mean_stability, anchor_x, anchor_y) for frame window."""
    hits = [mw for mw in analysis.motion_windows if _overlaps(start_f, end_f, mw.start_frame, mw.end_frame_exclusive)]
    if not hits:
        return 0.1, 0.85, 0.5, 0.48
    m = sum(h.motion_score for h in hits) / len(hits)
    st = sum(h.stability for h in hits) / len(hits)
    cx = sum(h.saliency_center_x for h in hits) / len(hits)
    cy = sum(h.saliency_center_y for h in hits) / len(hits)
    return m, st, cx, cy


def _has_existing_effect_protection(zones: list[ProtectedZone], start_f: int, end_f: int) -> bool:
    return any(
        z.reason == "existing_effect" and _overlaps(start_f, end_f, z.start_frame, z.end_frame_exclusive)
        for z in zones
    )


def _ui_collides_margin(zones: list[ProtectedZone], start_f: int, end_f: int, position: str = "bottom") -> bool:
    for z in zones:
        if z.reason not in ("ui_text", "letterbox"):
            continue
        if not _overlaps(start_f, end_f, z.start_frame, z.end_frame_exclusive):
            continue
        if position == "bottom" and (z.y + z.h) >= 0.85:
            return True
        if position == "top" and z.y <= 0.15:
            return True
    return False


def _shot_bounds_around(frame: int, analysis: AnalysisResult) -> tuple[int, int]:
    """Find [shot_start, shot_end_exclusive) containing `frame`."""
    cut_frames = sorted(c.frame for c in analysis.cuts)
    lo = 0
    hi = analysis.duration_frames
    for cf in cut_frames:
        if cf <= frame:
            lo = cf
        elif cf > frame:
            hi = cf
            break
    return lo, hi


def _is_candidate_compatible_with_spec(
    cand: CandidateEvent,
    spec: EffectSpec,
    policy: ProfilePolicy,
    analysis: AnalysisResult,
    config: MotionLabConfig,
) -> tuple[bool, str]:
    """Hard eligibility gate for (candidate, spec)."""
    if policy.name not in spec.supported_profiles:
        return False, f"profile_disallowed:{policy.name}"

    if not set(spec.required_features).issubset(set(analysis.capabilities)):
        missing = sorted(set(spec.required_features) - set(analysis.capabilities))
        return False, f"missing_capability:{','.join(missing)}"

    if spec.family == "transition" and cand.event_type != "scene_boundary":
        return False, "transition_requires_scene_boundary"
    if spec.family != "transition" and cand.event_type == "scene_boundary":
        return False, "non_transition_on_cut_candidate"

    if spec.safety_class == "subject_aware" and not policy.allow_subject_zoom:
        return False, "policy_disallows_subject_zoom"
    if spec.family == "layout" and not policy.allow_layout_masks:
        return False, "policy_disallows_layout_masks"
    if spec.id == "L05" and not (policy.allow_split_layout and config.allow_split_layout):
        return False, "split_layout_requires_explicit_opt_in"
    if spec.safety_class == "audio_reactive" and not policy.allow_audio_reactive:
        return False, "policy_disallows_audio_reactive"
    if spec.safety_class == "audio_reactive" and cand.event_type not in ("audio_onset", "energy_rise", "safe_hold"):
        return False, "audio_reactive_needs_audio_or_hold"

    if _has_existing_effect_protection(analysis.protected_zones, cand.start_frame, cand.end_frame_exclusive):
        if spec.layer in STRONG_LAYERS:
            return False, "protected_existing_effect_interval"

    fps_f = analysis.fps.as_float()
    if spec.uses_neighbor_frames:
        # Must not be near source start/end, and both adjacent shots must be >= 0.7s
        margin_f = max(4, int(round(0.35 * fps_f)))
        if cand.frame < margin_f or cand.frame > analysis.duration_frames - margin_f:
            return False, "transition_too_close_to_source_edge"
        prev_cuts = [c.frame for c in analysis.cuts if c.frame < cand.frame]
        next_cuts = [c.frame for c in analysis.cuts if c.frame > cand.frame]
        prev_len = (cand.frame - (prev_cuts[-1] if prev_cuts else 0)) / fps_f
        next_len = ((next_cuts[0] if next_cuts else analysis.duration_frames) - cand.frame) / fps_f
        if prev_len < 0.70 or next_len < 0.70:
            return False, f"adjacent_shot_too_short({prev_len:.2f}s,{next_len:.2f}s)"

    mean_motion, _stab, _, _ = _window_motion_stats(analysis, cand.start_frame, cand.end_frame_exclusive)
    if mean_motion > 0.55 and spec.layer in ("transform", "mask"):
        return False, f"high_motion_skip_spatial({mean_motion:.2f})"

    if spec.safety_class == "subject_aware":
        subj = _subject_at_window(analysis.subject_regions, cand.start_frame, cand.end_frame_exclusive)
        if subj is None:
            return False, "no_confident_subject_in_window"

    if spec.id == "G05" and _ui_collides_margin(analysis.protected_zones, cand.start_frame, cand.end_frame_exclusive, "bottom"):
        return False, "bottom_margin_ui_collision"

    return True, "ok"


def score_candidate(
    cand: CandidateEvent,
    spec: EffectSpec,
    policy: ProfilePolicy,
    analysis: AnalysisResult,
    chosen_events: list[EffectEvent],
) -> tuple[float, dict[str, float], list[str]]:
    """Phase D3: Compute interpretable multi-term score for (candidate, effect_spec)."""
    mean_motion, stability, _, _ = _window_motion_stats(analysis, cand.start_frame, cand.end_frame_exclusive)

    confidence_term = 0.45 * cand.confidence

    # Visual suitability by event_type & family alignment
    suit = 0.25
    reasons: list[str] = [cand.event_type]
    if cand.event_type == "scene_boundary" and spec.family == "transition":
        suit = 0.55
        reasons.append("aligned_boundary_transition")
    elif cand.event_type == "sustained_static" and spec.family in ("camera", "layout"):
        suit = 0.50
        reasons.append("static_window_gentle_motion")
    elif cand.event_type in ("audio_onset", "energy_rise") and spec.family in ("graphics", "camera"):
        suit = 0.48
        reasons.append("acoustic_cue_accent")
    elif cand.event_type == "safe_hold" and spec.family in ("color", "camera", "graphics"):
        suit = 0.40
        reasons.append("paced_hold_polish")
    elif cand.event_type == "motion_change" and spec.family in ("color", "graphics"):
        suit = 0.38
        reasons.append("motion_shift_accent")

    if spec.family in policy.preferred_families:
        pref_rank = policy.preferred_families.index(spec.family)
        suit += max(0.05, 0.20 - 0.04 * pref_rank)

    # Novelty & family diversity bonus (encourages tasteful variation across families)
    used_ids = {ev.effect_id for ev in chosen_events}
    used_families = {ev.reason_codes[0] if ev.reason_codes else "" for ev in chosen_events}
    novelty_term = 0.25 if spec.id not in used_ids else -0.35
    if f"family:{spec.family}" not in used_families:
        novelty_term += 0.22

    # Rhythm bonus for steady calm shots
    rhythm_term = 0.15 * stability

    # Penalties: crowding, cost, existing motion, repetition
    fps_f = analysis.fps.as_float()
    nearby_count = sum(
        1 for ev in chosen_events
        if abs(ev.start_frame - cand.start_frame) < int(round(3.5 * fps_f))
    )
    crowding_pen = 0.22 * nearby_count
    cost_pen = 0.08 * spec.estimated_cost
    motion_pen = 0.35 * mean_motion if spec.layer in ("transform", "mask") else 0.05 * mean_motion
    rep_pen = 0.30 * sum(1 for ev in chosen_events if ev.effect_id == spec.id)

    total = confidence_term + suit + novelty_term + rhythm_term - crowding_pen - cost_pen - motion_pen - rep_pen
    breakdown = {
        "confidence": round(confidence_term, 4),
        "visual_suitability": round(suit, 4),
        "novelty": round(novelty_term, 4),
        "rhythm": round(rhythm_term, 4),
        "crowding_penalty": round(-crowding_pen, 4),
        "cost_penalty": round(-cost_pen, 4),
        "motion_penalty": round(-motion_pen, 4),
        "repetition_penalty": round(-rep_pen, 4),
        "total": round(total, 4),
    }
    return total, breakdown, [f"family:{spec.family}"] + reasons


def _compute_event_window(
    cand: CandidateEvent,
    spec: EffectSpec,
    analysis: AnalysisResult,
) -> tuple[int, int] | None:
    """Compute valid integer [start_frame, end_frame_exclusive) respecting shot boundaries and bounds."""
    fps_f = analysis.fps.as_float()
    total_f = analysis.duration_frames
    min_f = max(4, int(round(spec.min_duration_s * fps_f)))
    max_f = max(min_f, int(round(spec.max_duration_s * fps_f)))

    if spec.family == "transition":
        half = max(3, min(max_f // 2, int(round(0.22 * fps_f))))
        sf = max(1, cand.frame - half)
        ef = min(total_f - 1, cand.frame + half)
        if ef - sf < 4:
            return None
        return sf, ef

    shot_lo, shot_hi = _shot_bounds_around(cand.frame, analysis)
    # Keep small safety buffer from hard cuts
    buf = max(2, int(round(0.10 * fps_f)))
    safe_lo = shot_lo + (buf if shot_lo > 0 else 0)
    safe_hi = shot_hi - (buf if shot_hi < total_f else 0)
    if safe_hi - safe_lo < min_f:
        return None

    target_dur = int(round(min(spec.max_duration_s, max(spec.min_duration_s, 2.2)) * fps_f))
    if spec.family == "color":
        target_dur = int(round(min(spec.max_duration_s, max(spec.min_duration_s, 3.6)) * fps_f))
    elif spec.family == "layout":
        target_dur = int(round(min(spec.max_duration_s, max(spec.min_duration_s, 1.8)) * fps_f))
    elif spec.family == "graphics":
        target_dur = int(round(min(spec.max_duration_s, max(spec.min_duration_s, 2.2)) * fps_f))

    # Anticipation 150ms before acoustic cues
    if cand.event_type in ("audio_onset", "energy_rise"):
        sf = max(safe_lo, cand.frame - int(round(0.15 * fps_f)))
    else:
        sf = max(safe_lo, cand.start_frame)

    ef = min(safe_hi, sf + target_dur)
    if ef - sf < min_f:
        sf = max(safe_lo, ef - min_f)
    if ef - sf < min_f or sf < 0 or ef > total_f or ef <= sf:
        return None
    if ef - sf > max_f:
        ef = sf + max_f
    return sf, ef


def _build_safe_params(
    spec: EffectSpec,
    policy: ProfilePolicy,
    analysis: AnalysisResult,
    start_f: int,
    end_f: int,
    rng: random.Random,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Populate deterministic, bounded, safe parameters & anchor region for an event."""
    params: dict[str, Any] = {k: b.default for k, b in spec.parameter_schema.items()}
    if "intensity" in params:
        params["intensity"] = round(min(spec.max_intensity, float(params["intensity"]) * policy.intensity_scale), 4)

    subj = _subject_at_window(analysis.subject_regions, start_f, end_f)
    _m, _st, sal_x, sal_y = _window_motion_stats(analysis, start_f, end_f)
    if subj is not None:
        ax = max(0.25, min(0.75, subj.x + subj.w * 0.5))
        ay = max(0.22, min(0.70, subj.y + subj.h * 0.42))
        region = {"x": round(subj.x, 4), "y": round(subj.y, 4), "w": round(subj.w, 4), "h": round(subj.h, 4)}
    else:
        ax = max(0.30, min(0.70, sal_x))
        ay = max(0.30, min(0.70, sal_y))
        region = {"x": 0.15, "y": 0.15, "w": 0.70, "h": 0.70}

    if "anchor_x" in params:
        b = spec.parameter_schema["anchor_x"]
        params["anchor_x"] = round(max(b.min_val or 0.0, min(b.max_val or 1.0, ax)), 4)
    if "anchor_y" in params:
        b = spec.parameter_schema["anchor_y"]
        params["anchor_y"] = round(max(b.min_val or 0.0, min(b.max_val or 1.0, ay)), 4)

    if "direction" in params and spec.parameter_schema["direction"].choices:
        choices = spec.parameter_schema["direction"].choices
        params["direction"] = choices[rng.randrange(len(choices))]

    validated = spec.validate_params(params)
    return validated, region


def _violates_spacing_or_overlap(
    spec: EffectSpec,
    start_f: int,
    end_f: int,
    chosen: list[EffectEvent],
    registry: EffectRegistry,
    policy: ProfilePolicy,
    fps_f: float,
) -> tuple[bool, str]:
    """Phase D4: Enforce concurrency compatibility, cooldowns, and pacing rules."""
    min_strong_gap = int(round(policy.min_strong_spacing_s * fps_f))
    same_eff_gap = int(round(policy.same_effect_cooldown_s * fps_f))
    same_fam_gap = int(round(policy.same_family_cooldown_s * fps_f))

    for ev in chosen:
        ev_spec = registry.get(ev.effect_id)
        overlap = _overlaps(start_f, end_f, ev.start_frame, ev.end_frame_exclusive)
        if overlap:
            if spec.layer in ev_spec.incompatibilities or ev_spec.layer in spec.incompatibilities:
                return True, f"layer_conflict:{spec.layer}_vs_{ev_spec.layer}"
            if spec.layer in STRONG_LAYERS and ev_spec.layer in STRONG_LAYERS:
                return True, "concurrent_strong_transform_forbidden"
            if spec.layer == ev_spec.layer:
                return True, f"same_layer_overlap:{spec.layer}"

        # Cooldown distance between interval edges
        gap = max(start_f - ev.end_frame_exclusive, ev.start_frame - end_f)
        if ev.effect_id == spec.id and gap < same_eff_gap:
            return True, f"same_effect_cooldown:{spec.id}"
        if ev_spec.family == spec.family and spec.family != "transition" and gap < same_fam_gap:
            return True, f"same_family_cooldown:{spec.family}"
        if spec.layer in STRONG_LAYERS and ev_spec.layer in STRONG_LAYERS and spec.family != "transition":
            if gap < min_strong_gap:
                return True, "min_strong_spacing_violation"
        # Prevent consecutive inset cards (no repetitive card layout)
        if spec.id in ("L06", "L07") and ev.effect_id in ("L06", "L07"):
            if gap < int(round(8.0 * fps_f)):
                return True, "no_repetitive_inset_card_layout"

    return False, "ok"


def validate_plan(plan: EditPlan, media: MediaInfo, registry: EffectRegistry) -> list[str]:
    """Phase D6: Validate every EditPlan invariant strictly before serialization or rendering."""
    ensure_finite(plan.to_dict())
    errors: list[str] = []

    if plan.duration_frames <= 0:
        errors.append("Plan duration_frames must be positive.")
    if plan.width <= 0 or plan.height <= 0 or plan.width % 2 != 0 or plan.height % 2 != 0:
        errors.append(f"Plan output dimensions must be positive even integers, got {plan.width}x{plan.height}.")
    if plan.audio_policy.get("retime_allowed", False):
        errors.append("Audio retime is strictly forbidden in ML-01.")
    if plan.audio_policy.get("default_subtitles", False) or plan.audio_policy.get("default_music", False):
        errors.append("Default subtitles or music injection is forbidden in ML-01.")

    seen_ids: set[str] = set()
    for ev in plan.events:
        if ev.id in seen_ids:
            errors.append(f"Duplicate event id: {ev.id}")
        seen_ids.add(ev.id)
        if not registry.has(ev.effect_id):
            errors.append(f"Unknown effect_id in plan: {ev.effect_id}")
            continue
        spec = registry.get(ev.effect_id)
        try:
            spec.validate_params(ev.parameters)
        except SchemaValidationError as exc:
            errors.append(str(exc))
        if ev.start_frame < 0 or ev.end_frame_exclusive > plan.duration_frames or ev.end_frame_exclusive <= ev.start_frame:
            errors.append(
                f"Invalid frame bounds on {ev.id} ({ev.effect_id}): [{ev.start_frame}, {ev.end_frame_exclusive}) "
                f"vs duration {plan.duration_frames}"
            )
        if ev.fallback_id is not None and not registry.has(ev.fallback_id):
            errors.append(f"Unknown fallback_id {ev.fallback_id} on {ev.id}")

    # Verify pairwise layer compatibility
    for i in range(len(plan.events)):
        for j in range(i + 1, len(plan.events)):
            a, b = plan.events[i], plan.events[j]
            if _overlaps(a.start_frame, a.end_frame_exclusive, b.start_frame, b.end_frame_exclusive):
                sa, sb = registry.get(a.effect_id), registry.get(b.effect_id)
                if sa.layer in sb.incompatibilities or sb.layer in sa.incompatibilities:
                    errors.append(f"Incompatible overlapping events {a.id} ({a.effect_id}) and {b.id} ({b.effect_id})")

    if errors:
        raise SchemaValidationError("EditPlan validation failed:\n- " + "\n- ".join(errors))
    return []


def build_edit_plan(
    media: MediaInfo,
    analysis: AnalysisResult,
    config: MotionLabConfig,
    engine_caps: dict[str, Any],
    registry: EffectRegistry | None = None,
) -> tuple[EditPlan, dict[str, Any]]:
    """Run deterministic rule-based planner -> (EditPlan, plan_explain dict)."""
    config.validate()
    reg = registry or load_effects()
    rng = random.Random(config.seed)

    policy, profile_reason = select_profile(analysis, config.profile)
    out_w, out_h, aspect_policy, layout_warnings = resolve_layout(media, analysis, config)
    candidates = build_candidates(analysis)

    warnings: list[str] = list(media.warnings) + list(layout_warnings)
    if media.is_hdr and not config.allow_hdr_tonemap:
        warnings.append("HDR input without --allow-hdr-tonemap: SDR color grading effects disabled to prevent silent tone-map errors.")

    fps_f = media.fps.as_float()
    scale_15s = max(0.5, media.duration_s / 15.0)
    max_accents = max(1, int(round(policy.max_accents_per_15s * scale_15s)))
    max_transitions = max(0, int(round(policy.max_transitions_per_15s * scale_15s)))

    # Very short media (< 1.2s) gets conservative cap
    if media.duration_s < 1.2:
        max_accents = 1
        max_transitions = 0
        warnings.append("Short clip (<1.2s): capping effect density conservatively.")

    specs = reg.all_specs()
    chosen: list[EffectEvent] = []
    explain_decisions: list[dict[str, Any]] = []
    chosen_accents = 0
    chosen_transitions = 0

    # Process scene_boundary candidates first (bounded by max_transitions), then remaining cues
    ordered_cands = sorted(
        candidates,
        key=lambda c: (0 if c.event_type == "scene_boundary" else 1, -c.confidence, c.frame, c.id),
    )

    event_seq = 0
    for cand in ordered_cands:
        is_cut = cand.event_type == "scene_boundary"
        if is_cut and chosen_transitions >= max_transitions:
            explain_decisions.append({
                "candidate_id": cand.id,
                "status": "SKIPPED",
                "reason": f"transition_budget_reached({max_transitions})",
            })
            continue
        if not is_cut and chosen_accents >= max_accents:
            explain_decisions.append({
                "candidate_id": cand.id,
                "status": "SKIPPED",
                "reason": f"accent_budget_reached({max_accents})",
            })
            continue

        scored_specs: list[tuple[float, str, EffectSpec, dict[str, float], list[str], tuple[int, int], str]] = []
        rejections: list[dict[str, str]] = []

        for spec in specs:
            ok, why = _is_candidate_compatible_with_spec(cand, spec, policy, analysis, config)
            if not ok:
                rejections.append({"effect_id": spec.id, "reason": why})
                continue

            engine = reg.resolve_engine(spec, engine_caps, preferred_engine=config.preferred_engine)
            if engine is None:
                rejections.append({"effect_id": spec.id, "reason": "no_available_licensed_engine"})
                continue

            win = _compute_event_window(cand, spec, analysis)
            if win is None:
                rejections.append({"effect_id": spec.id, "reason": "window_out_of_shot_bounds"})
                continue

            sf, ef = win
            blocked, block_why = _violates_spacing_or_overlap(spec, sf, ef, chosen, reg, policy, fps_f)
            if blocked:
                rejections.append({"effect_id": spec.id, "reason": block_why})
                continue

            sc, breakdown, reasons = score_candidate(cand, spec, policy, analysis, chosen)
            # Tiny deterministic seeded jitter (<0.015) purely to break exact ties reproducibly
            tie_break = rng.uniform(0.0, 0.012)
            scored_specs.append((sc + tie_break, spec.id, spec, breakdown, reasons, (sf, ef), engine))

        if not scored_specs:
            explain_decisions.append({
                "candidate_id": cand.id,
                "frame": cand.frame,
                "event_type": cand.event_type,
                "status": "REJECTED",
                "rejections_sample": rejections[:8],
            })
            continue

        scored_specs.sort(key=lambda item: (-item[0], item[1]))
        best_score, _, best_spec, best_breakdown, best_reasons, (sf, ef), best_engine = scored_specs[0]

        if best_score < 0.25:
            explain_decisions.append({
                "candidate_id": cand.id,
                "status": "REJECTED_LOW_SCORE",
                "top_effect": best_spec.id,
                "score": round(best_score, 4),
            })
            continue

        params, region = _build_safe_params(best_spec, policy, analysis, sf, ef, rng)
        easing_name = "spring_damped" if best_spec.id in ("C03", "C08") else "ease_in_out_cubic"

        ev = EffectEvent(
            id=f"ev_{event_seq:03d}_{best_spec.id}",
            effect_id=best_spec.id,
            effect_version=best_spec.version,
            start_frame=sf,
            end_frame_exclusive=ef,
            layer=best_spec.layer,
            region=region,
            parameters=params,
            easing=easing_name,
            priority=10 if best_spec.family == "transition" else 5,
            engine=best_engine,
            reason_codes=best_reasons,
            evidence_refs=[cand.id],
            fallback_id=best_spec.fallback_id,
            uses_neighbor_frames=best_spec.uses_neighbor_frames,
        )
        chosen.append(ev)
        event_seq += 1
        if is_cut:
            chosen_transitions += 1
        else:
            chosen_accents += 1

        explain_decisions.append({
            "candidate_id": cand.id,
            "frame": cand.frame,
            "event_type": cand.event_type,
            "status": "SELECTED",
            "selected_event_id": ev.id,
            "effect_id": best_spec.id,
            "effect_name": best_spec.name,
            "engine": best_engine,
            "frame_range": [sf, ef],
            "score_breakdown": best_breakdown,
            "alternatives_considered": len(scored_specs),
        })

    chosen.sort(key=lambda e: (e.start_frame, e.layer, e.id))

    # Audio hard-lock policy
    selected_tracks = (
        [config.audio_track]
        if config.audio_track is not None and config.audio_track < len(media.audio_streams)
        else [a.audio_track_index for a in media.audio_streams]
    )
    copy_ok = all(media.audio_streams[idx].mp4_copy_compatible for idx in selected_tracks) if selected_tracks else True
    audio_policy = {
        "has_source_audio": media.has_audio,
        "preserve_master": True,
        "retime_allowed": False,
        "filters_applied": [],
        "selected_tracks": selected_tracks,
        "mode": "none" if not media.has_audio else ("stream_copy" if copy_ok else "disclosed_aac_transcode"),
        "default_subtitles": False,
        "default_music": False,
        "default_sfx": False,
    }

    plan_status = "READY" if chosen else "NO_SAFE_EFFECTS"
    if not chosen:
        warnings.append(
            "NO_SAFE_EFFECTS: No effect met safety, capability, and shot-duration constraints. "
            "Output will preserve source media cleanly without synthetic filler."
        )

    plan = EditPlan(
        schema_version=SCHEMA_VERSION,
        planner_version=PLANNER_VERSION,
        source_hash=media.input_hash,
        config_hash=config.config_hash(),
        seed=config.seed,
        profile=policy.name,
        profile_selection_reason=profile_reason,
        width=out_w,
        height=out_h,
        fps=media.fps,
        duration_frames=media.duration_frames,
        aspect_policy=aspect_policy,
        audio_policy=audio_policy,
        engine_capabilities={
            k: {"available": bool(v.get("available")), "license": v.get("license", "")}
            for k, v in engine_caps.items()
            if isinstance(v, dict)
        },
        events=chosen,
        status=plan_status,
        warnings=warnings,
    )
    validate_plan(plan, media, reg)
    plan.plan_hash = plan.compute_plan_hash()

    plan_explain = {
        "schema_version": SCHEMA_VERSION,
        "planner_version": PLANNER_VERSION,
        "plan_hash": plan.plan_hash,
        "profile_policy": policy.to_dict(),
        "profile_selection_reason": profile_reason,
        "candidate_count": len(candidates),
        "selected_event_count": len(chosen),
        "selected_families": sorted({reg.get(e.effect_id).family for e in chosen}),
        "budgets": {
            "max_accents": max_accents,
            "max_transitions": max_transitions,
            "used_accents": chosen_accents,
            "used_transitions": chosen_transitions,
        },
        "decisions": explain_decisions,
    }
    return plan, plan_explain
