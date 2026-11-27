"""Phase D2–D6 — Rule-based Automatic Effect Planner with Deterministic Explainability.

Produces:
  - EditPlan (`edit_plan.json`)
  - Detailed explanation log (`plan_explain.json`) with per-term scores, reason codes,
    rejections, and fallback decisions.

Guarantees:
  - Same input + config + seed + versions -> identical plan hash.
  - Hard audio lock (no retiming, no cuts, no fades, no loudnorm, no voice generation).
  - Enforces profile budgets, cooldowns, overlap rules, holds between moves, and safe zones.
  - Emits explicit `NO_SAFE_EFFECTS` status if no effects can safely apply (never forces unsafe effects).
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any

from motionlab import PLANNER_VERSION, SCHEMA_VERSION
from motionlab.config import EditConfig
from motionlab.policies import ProfilePolicy, select_profile
from motionlab.registry import EffectRegistry, load_effects
from motionlab.schemas import (
    AnalysisResult,
    CandidateEvent,
    EditPlan,
    EffectEvent,
    EffectSpec,
    MediaInfo,
)

# Candidate type -> naturally suitable effect IDs
CANDIDATE_EFFECT_AFFINITY: dict[str, tuple[str, ...]] = {
    "scene_boundary": (
        "dissolve_short",
        "soft_mask_wipe",
        "blur_through_cut",
        "directional_wipe",
        "dark_dip",
        "edge_slide",
        "zoom_through_cut",
        "match_scale_bridge",
    ),
    "sustained_static": (
        "subtle_push_in",
        "lateral_drift",
        "reframed_focus_hold",
        "eased_pull_out",
        "vertical_drift",
        "rounded_mask_reveal",
        "full_to_inset",
        "inset_to_full",
        "diagonal_mask_sweep",
        "neutral_contrast",
        "warm_subtle",
        "edge_vignette",
        "depth_shadow_push",
        "geometric_window",
    ),
    "safe_hold": (
        "neutral_contrast",
        "warm_subtle",
        "subtle_push_in",
        "reframed_focus_hold",
        "eased_pull_out",
        "soft_edge_vignette_reveal",
        "edge_vignette",
        "edge_lines",
        "foreground_frame_accent",
        "waveform_accent",
        "film_grain_subtle",
        "soft_glow",
        "ambient_particles",
    ),
    "audio_onset": (
        "subject_punch_in",
        "settle_bounce",
        "waveform_accent",
        "pulse_border",
        "ring_accent",
        "edge_lines",
        "light_sweep",
        "directional_shapes",
        "reframed_focus_hold",
    ),
    "energy_rise": (
        "waveform_accent",
        "pulse_border",
        "subtle_push_in",
        "warm_subtle",
        "edge_lines",
        "ring_accent",
        "light_sweep",
        "subject_punch_in",
        "rounded_mask_reveal",
    ),
    "motion_change": (
        "reframed_focus_hold",
        "subject_follow_pan",
        "eased_pull_out",
        "lateral_drift",
        "neutral_contrast",
        "soft_edge_vignette_reveal",
        "edge_lines",
    ),
}


@dataclass
class ScoredOption:
    candidate: CandidateEvent
    spec: EffectSpec
    engine: str
    score: float
    terms: dict[str, float]
    reasons: list[str]
    start_frame: int
    end_frame_exclusive: int
    region: dict[str, float]


def resolve_target_dimensions(
    media: MediaInfo,
    aspect: str,
    quality: str,
    aspect_bg_mode: str,
    analysis: AnalysisResult,
) -> tuple[int, int, dict[str, Any]]:
    """Phase D5: Resolve safe output dimensions & layout policy without blind cropping."""
    src_w = media.display_width
    src_h = media.display_height
    src_aspect = src_w / max(1, src_h)

    if aspect == "source":
        w, h = src_w, src_h
        if quality == "preview":
            max_side = max(w, h)
            if max_side > 960:
                scale = 960.0 / max_side
                w = int(round(w * scale))
                h = int(round(h * scale))
        # Enforce positive even dimensions for yuv420p H.264
        w = max(64, (w // 2) * 2)
        h = max(64, (h // 2) * 2)
        layout = {
            "mode": "preserve_source",
            "source_dimensions": [src_w, src_h],
            "target_dimensions": [w, h],
            "crop_applied": False,
            "pad_applied": False,
            "background_fill": "none",
        }
        return w, h, layout

    if aspect == "9:16":
        w, h = (540, 960) if quality == "preview" else (1080, 1920)
    elif aspect == "16:9":
        w, h = (960, 540) if quality == "preview" else (1920, 1080)
    elif aspect == "1:1":
        w, h = (720, 720) if quality == "preview" else (1080, 1080)
    else:
        raise ValueError(f"Unsupported aspect '{aspect}'")

    target_aspect = w / h
    aspect_diff = abs(src_aspect - target_aspect) / max(target_aspect, 1e-6)

    if aspect_diff <= 0.03:
        # Already matching target aspect (e.g. 720x1280 -> 1080x1920 9:16)
        layout = {
            "mode": "aspect_match_scale",
            "source_dimensions": [src_w, src_h],
            "target_dimensions": [w, h],
            "crop_applied": False,
            "pad_applied": False,
            "background_fill": "none",
        }
        return w, h, layout

    # Different aspect ratio: never crop blindly!
    face_box = analysis.safe_zones.get("face_box")
    busy_ui = bool(analysis.safe_zones.get("busy_ui_edges", False))
    can_smart_crop = (
        aspect_bg_mode == "smart_crop"
        and not busy_ui
        and face_box is not None
        and not face_box.get("multi_subject", False)
        and float(face_box.get("confidence", 0.0)) >= 0.55
    )

    if can_smart_crop:
        layout = {
            "mode": "smart_crop_tracked_subject",
            "source_dimensions": [src_w, src_h],
            "target_dimensions": [w, h],
            "crop_applied": True,
            "pad_applied": False,
            "crop_anchor_x": float(face_box["cx"]),
            "crop_anchor_y": float(face_box["cy"]),
            "background_fill": "none",
        }
    else:
        # Default safe fit/pad over blurred or solid same-media background (preserves 100% of source content)
        bg = "solid" if aspect_bg_mode == "solid" else "blur_same_media"
        layout = {
            "mode": "fit_pad_safe",
            "source_dimensions": [src_w, src_h],
            "target_dimensions": [w, h],
            "crop_applied": False,
            "pad_applied": True,
            "background_fill": bg,
        }
    return w, h, layout


def _pick_engine_for_spec(
    spec: EffectSpec,
    engine_capabilities: dict[str, Any],
    preferred_engine: str,
) -> tuple[str | None, str | None]:
    """Select an eligible available engine for `spec`, honoring license/capability flags."""
    available = {
        name
        for name, info in engine_capabilities.items()
        if isinstance(info, dict) and info.get("available") is True
    }
    if preferred_engine != "auto":
        if preferred_engine in spec.supported_engines and preferred_engine in available:
            return preferred_engine, None
        # Fallback if primary preferred doesn't support this specific effect
    for eng in spec.supported_engines:
        if eng in available:
            reason = None if eng == spec.engine else f"engine_fallback:{spec.engine}->{eng}"
            return eng, reason
    return None, f"no_available_engine_in:{','.join(spec.supported_engines)}"


def _overlaps_protected(start_f: int, end_f: int, analysis: AnalysisResult) -> str | None:
    for pi in analysis.protected_intervals:
        if start_f < pi.end_frame_exclusive and end_f > pi.start_frame:
            return pi.reason
    return None


def _window_motion_at(frame: int, analysis: AnalysisResult) -> float:
    for mw in analysis.motion_windows:
        if mw.start_frame <= frame < mw.end_frame_exclusive:
            return mw.motion_score
    return float(analysis.signal_summary.get("mean_motion", 0.0))


def _subject_anchor_at(frame: int, analysis: AnalysisResult) -> tuple[float, float, float]:
    """Return (cx, cy, confidence) near `frame`."""
    if not analysis.subject_regions:
        sz = analysis.safe_zones
        return float(sz.get("safe_anchor_x", 0.5)), float(sz.get("safe_anchor_y", 0.5)), 0.0
    best = min(analysis.subject_regions, key=lambda s: abs(s.frame - frame))
    return best.center_x, best.center_y, best.confidence


def _choose_peripheral_region(analysis: AnalysisResult) -> dict[str, float]:
    """Place abstract accents strictly away from detected face/subject center."""
    face = analysis.safe_zones.get("face_box")
    if not face:
        return {"cx": 0.5, "cy": 0.88, "w": 0.90, "h": 0.14}
    # Put peripheral accents in opposite corner/edge from face
    fx = float(face["cx"])
    fy = float(face["cy"])
    accent_x = 0.84 if fx < 0.55 else 0.16
    accent_y = 0.86 if fy < 0.65 else 0.14
    return {"cx": accent_x, "cy": accent_y, "w": 0.24, "h": 0.18}


def _compute_event_window(
    cand: CandidateEvent,
    spec: EffectSpec,
    fps: float,
    duration_frames: int,
) -> tuple[int, int] | None:
    """Compute exact integer frame interval [start, end_exclusive) with anticipation where appropriate."""
    min_f = max(2, int(round(spec.min_duration_s * fps)))
    max_f = max(min_f, int(round(spec.max_duration_s * fps)))

    if spec.family == "transition":
        half = max(2, min(int(round(0.24 * fps)), max_f // 2))
        sf = cand.frame - half
        ef = cand.frame + half
        # Never touch video ends
        if sf < 2 or ef > duration_frames - 2 or (ef - sf) < 4:
            return None
        return sf, ef

    # Anticipation (100–220ms) before acoustic onset / energy rise cues
    anticipation = 0
    if cand.event_type in ("audio_onset", "energy_rise"):
        anticipation = int(round(0.15 * fps))

    sf = max(0, cand.frame - anticipation)
    natural_len = max(min_f, cand.end_frame_exclusive - cand.frame)
    target_len = min(max_f, max(min_f, int(round(natural_len * 1.15))))

    # Keep camera & grade durations tasteful (not cramped, not covering whole video excessively)
    if spec.family == "camera":
        target_len = min(max_f, max(min_f, int(round(2.4 * fps))))
    elif spec.family == "color_optical":
        target_len = min(max_f, max(min_f, int(round(3.6 * fps))))
    elif spec.family == "motion_graphics":
        target_len = min(max_f, max(min_f, int(round(2.0 * fps))))
    elif spec.family == "mask_layout":
        target_len = min(max_f, max(min_f, int(round(1.8 * fps))))

    ef = min(duration_frames, sf + target_len)
    if ef - sf < min_f:
        sf = max(0, ef - min_f)
    if ef - sf < max(2, int(round(min_f * 0.75))) or ef <= sf:
        return None
    return sf, ef


def build_edit_plan(
    media: MediaInfo,
    analysis: AnalysisResult,
    config: EditConfig,
    engine_capabilities: dict[str, Any],
    registry: EffectRegistry | None = None,
) -> tuple[EditPlan, dict[str, Any]]:
    """Phase D2–D6: Score candidates, allocate effects under strict policies, validate plan & explain."""
    config.validate()
    reg = registry or load_effects()
    profile, profile_reasons = select_profile(analysis, media, requested=config.profile)

    fps = media.fps_float
    dur_frames = analysis.duration_frames
    dur_s = media.duration_s

    target_w, target_h, layout_plan = resolve_target_dimensions(
        media=media,
        aspect=config.aspect,
        quality=config.quality,
        aspect_bg_mode=config.aspect_bg_mode,
        analysis=analysis,
    )

    # Audio policy hard lock
    selected_tracks = (
        [config.audio_track]
        if (config.audio_track is not None and media.has_audio)
        else [a.audio_index for a in media.audio_streams]
    )
    all_copy_ok = all(
        media.audio_streams[i].mp4_copy_compatible
        for i in selected_tracks
        if i < len(media.audio_streams)
    )
    audio_policy = {
        "hard_lock": True,
        "has_audio": media.has_audio,
        "selected_tracks": selected_tracks,
        "mode": "none" if not media.has_audio else ("stream_copy" if all_copy_ok else "transcode_compat"),
        "retime_allowed": False,
        "filters_allowed": False,
        "loudnorm_allowed": False,
        "fades_allowed": False,
        "default_subtitles": False,
        "default_music_or_sfx": False,
    }

    warnings: list[str] = list(media.warnings)
    if media.color_metadata.is_hdr and not config.allow_hdr_tonemap:
        warnings.append("HDR source without --allow-hdr-tonemap: color effects restricted")

    # Deterministic PRNG seeded from config.seed + source_hash + profile
    seed_material = f"{config.seed}:{media.input_hash}:{profile.name}:{PLANNER_VERSION}"
    seed_int = int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:12], 16)
    rng = random.Random(seed_int)

    # Very short clip (< 0.8s) or heavily flashing/protected source -> conservative budget reduction
    scale_factor = max(0.25, dur_s / 15.0) * config.density_scale
    if "heavily_edited_or_flashing_source" in analysis.quality_flags:
        scale_factor *= 0.5
        warnings.append("reduced_density_due_to_heavily_edited_source")
    if "high_motion_footage" in analysis.quality_flags:
        scale_factor *= 0.7
        warnings.append("reduced_motion_effects_due_to_high_motion_footage")

    max_main_events = int(round(profile.max_events_per_15s * scale_factor))
    max_transitions = int(round(profile.max_transitions_per_15s * max(1.0, dur_s / 15.0) * config.density_scale))
    if dur_s >= 1.5 and config.density_scale > 0.0:
        max_main_events = max(1, max_main_events)
    if dur_s < 0.8 or config.density_scale == 0.0:
        max_main_events = 0
        max_transitions = 0

    # Ensure we always have at least gentle safe_hold candidates if the clip is valid (>= 1.2s) and quiet
    candidates = list(analysis.candidates)
    if not candidates and dur_frames >= max(10, int(round(fps * 1.2))):
        candidates.append(
            CandidateEvent(
                id="cand_fallback_hold_001",
                frame=max(1, int(round(fps * 0.2))),
                end_frame_exclusive=max(2, dur_frames - int(round(fps * 0.2))),
                event_type="safe_hold",
                confidence=0.65,
                reason="fallback_clean_window",
                metrics={"motion_score": float(analysis.signal_summary.get("mean_motion", 0.0))},
            )
        )

    rejections: list[dict[str, Any]] = []
    scored_options: list[ScoredOption] = []
    caps_set = set(analysis.capabilities)
    flags_set = set(analysis.quality_flags)

    # Score every (candidate, eligible_effect) pair deterministically
    for cand in candidates:
        affinity_ids = CANDIDATE_EFFECT_AFFINITY.get(cand.event_type, ())
        for eid in affinity_ids:
            spec = reg.get(eid)

            # Hard rejects
            if profile.name not in spec.supported_profiles:
                rejections.append({"candidate": cand.id, "effect": eid, "reason": "profile_unsupported"})
                continue
            if eid in profile.blocked_effects:
                rejections.append({"candidate": cand.id, "effect": eid, "reason": "profile_blocked"})
                continue
            if spec.safety_class == "explicit_only":
                rejections.append({"candidate": cand.id, "effect": eid, "reason": "explicit_only_safety_class"})
                continue
            if spec.family == "mask_layout" and not profile.allow_layout_masks:
                rejections.append({"candidate": cand.id, "effect": eid, "reason": "profile_disallows_layout_masks"})
                continue
            if spec.safety_class == "subject_sensitive" and not profile.allow_face_transforms and "confident_subject" not in caps_set:
                rejections.append({"candidate": cand.id, "effect": eid, "reason": "no_confident_subject"})
                continue
            missing_feats = [f for f in spec.required_features if f not in caps_set]
            if missing_feats:
                rejections.append({"candidate": cand.id, "effect": eid, "reason": f"missing_features:{','.join(missing_feats)}"})
                continue
            hit_exclusions = [ex for ex in spec.excluded_cases if ex in flags_set or ex in caps_set]
            if hit_exclusions:
                rejections.append({"candidate": cand.id, "effect": eid, "reason": f"excluded_case:{','.join(hit_exclusions)}"})
                continue
            if media.color_metadata.is_hdr and not config.allow_hdr_tonemap and spec.family == "color_optical":
                rejections.append({"candidate": cand.id, "effect": eid, "reason": "hdr_grade_blocked_without_opt_in"})
                continue

            engine, eng_note = _pick_engine_for_spec(spec, engine_capabilities, config.preferred_engine)
            if engine is None:
                rejections.append({"candidate": cand.id, "effect": eid, "reason": eng_note or "no_engine"})
                continue

            win = _compute_event_window(cand, spec, fps, dur_frames)
            if win is None:
                rejections.append({"candidate": cand.id, "effect": eid, "reason": "shot_or_clip_too_short"})
                continue
            sf, ef = win

            prot = _overlaps_protected(sf, ef, analysis)
            if prot is not None and (spec.is_strong_transform or spec.family == "transition" or spec.id in ("light_sweep", "pulse_border")):
                rejections.append({"candidate": cand.id, "effect": eid, "reason": f"protected_interval:{prot}"})
                continue

            local_motion = _window_motion_at(cand.frame, analysis)
            if local_motion >= 0.35 and spec.family == "camera":
                rejections.append({"candidate": cand.id, "effect": eid, "reason": "high_local_motion_camera_guard"})
                continue

            # Compute per-term score
            pref_rank = (
                1.0 - (profile.preferred_families.index(spec.family) / max(1, len(profile.preferred_families)))
                if spec.family in profile.preferred_families
                else 0.25
            )
            term_conf = cand.confidence * 0.35
            term_suit = pref_rank * 0.30 + (0.08 if spec.is_core else 0.03)
            term_rhythm = 0.14 if cand.event_type in ("scene_boundary", "audio_onset", "sustained_static") else 0.08
            term_cost_pen = -0.05 * spec.estimated_cost
            term_motion_pen = -0.25 * local_motion if spec.family == "camera" else 0.0
            # Tiny deterministic seeded tie-breaker for controlled variety across seeds
            term_seed = rng.uniform(0.0, 0.06)

            total_score = term_conf + term_suit + term_rhythm + term_cost_pen + term_motion_pen + term_seed
            reasons = [cand.reason, f"profile:{profile.name}", f"family:{spec.family}"]
            if eng_note:
                reasons.append(eng_note)

            if spec.family == "motion_graphics":
                region = _choose_peripheral_region(analysis)
            else:
                cx, cy, _ = _subject_anchor_at(cand.frame, analysis)
                region = {"cx": cx, "cy": cy, "w": 1.0, "h": 1.0}

            scored_options.append(
                ScoredOption(
                    candidate=cand,
                    spec=spec,
                    engine=engine,
                    score=total_score,
                    terms={
                        "confidence": round(term_conf, 4),
                        "suitability": round(term_suit, 4),
                        "rhythm": round(term_rhythm, 4),
                        "cost_penalty": round(term_cost_pen, 4),
                        "motion_penalty": round(term_motion_pen, 4),
                        "seeded_tiebreak": round(term_seed, 4),
                    },
                    reasons=reasons,
                    start_frame=sf,
                    end_frame_exclusive=ef,
                    region=region,
                )
            )

    # Phase D4: Allocate events enforcing budgets, cooldowns, family diversity, and overlap rules
    selected_events: list[EffectEvent] = []
    explain_selected: list[dict[str, Any]] = []
    used_counts: dict[str, int] = {}
    used_families: dict[str, int] = {}
    last_effect_end: dict[str, int] = {}
    last_family_end: dict[str, int] = {}
    last_strong_end: int = -100000

    repeat_cd_f = int(round(profile.repeat_effect_cooldown_s * fps))
    family_cd_f = int(round(profile.same_family_cooldown_s * fps))
    strong_cd_f = int(round(profile.strong_accent_cooldown_s * fps))

    # First pass: transitions (counted separately from main effects budget)
    trans_options = [o for o in scored_options if o.spec.family == "transition"]
    trans_options.sort(key=lambda o: (-o.score, o.start_frame, o.spec.code, o.candidate.id))
    trans_selected = 0

    # Second pass: main effects (camera, mask_layout, color_optical, motion_graphics)
    main_options = [o for o in scored_options if o.spec.family != "transition"]

    def _can_place(opt: ScoredOption) -> tuple[bool, str]:
        sf, ef = opt.start_frame, opt.end_frame_exclusive
        spec = opt.spec

        # Cooldowns
        if spec.id in last_effect_end and (sf - last_effect_end[spec.id]) < repeat_cd_f and (ef > last_effect_end[spec.id] - repeat_cd_f):
            if abs(sf - last_effect_end[spec.id]) < repeat_cd_f:
                return False, "repeat_effect_cooldown"
        if spec.family in last_family_end and abs(sf - last_family_end[spec.family]) < family_cd_f:
            return False, "same_family_cooldown"
        if spec.is_strong_transform and abs(sf - last_strong_end) < strong_cd_f:
            return False, "strong_accent_cooldown"

        # Overlap compatibility checks against already selected events
        concurrent_strong = 0
        concurrent_light = 0
        for ev in selected_events:
            overlaps = sf < ev.end_frame_exclusive and ef > ev.start_frame
            if not overlaps:
                continue
            ev_spec = reg.get(ev.effect_id)
            if ev.layer == spec.layer:
                return False, f"same_layer_overlap:{spec.layer}"
            if ev.layer in spec.incompatibilities or spec.layer in ev_spec.incompatibilities:
                return False, f"incompatible_layer_overlap:{spec.layer}+{ev.layer}"
            if ev.effect_id in spec.incompatibilities or spec.id in ev_spec.incompatibilities:
                return False, f"incompatible_effect_overlap:{spec.id}+{ev.effect_id}"
            if ev_spec.is_strong_transform:
                concurrent_strong += 1
            else:
                concurrent_light += 1

        if spec.is_strong_transform and concurrent_strong >= 1:
            return False, "max_one_concurrent_strong_transform"
        if not spec.is_strong_transform and concurrent_light >= 2:
            return False, "max_concurrent_light_accents"
        return True, "ok"

    def _commit(opt: ScoredOption, ev_idx: int) -> None:
        nonlocal last_strong_end
        spec = opt.spec
        params = dict(spec.default_params)
        base_int = float(params.get("intensity", 0.75))
        eff_int = min(spec.max_intensity, max(0.15, base_int * profile.intensity_multiplier * config.intensity_scale))
        params["intensity"] = round(eff_int, 4)
        params = reg.validate_params(spec.id, params)

        easing = "spring_damped" if spec.id == "settle_bounce" else (
            "ease_in_out_cubic" if spec.family in ("camera", "transition") else "ease_out_cubic"
        )
        ev = EffectEvent(
            id=f"ev_{ev_idx:03d}_{spec.code.lower()}",
            effect_id=spec.id,
            effect_version=spec.version,
            engine=opt.engine,
            start_frame=opt.start_frame,
            end_frame_exclusive=opt.end_frame_exclusive,
            layer=spec.layer,
            region=opt.region,
            parameters=params,
            easing=easing,
            priority=10 if spec.layer == "transition" else (20 if spec.layer in ("transform", "layout") else (30 if spec.layer == "grade" else 40)),
            reason_codes=opt.reasons,
            evidence_refs=[opt.candidate.id],
            fallback_id=spec.fallback_id,
            fallback_engine="ffmpeg",
        )
        selected_events.append(ev)
        used_counts[spec.id] = used_counts.get(spec.id, 0) + 1
        used_families[spec.family] = used_families.get(spec.family, 0) + 1
        last_effect_end[spec.id] = opt.end_frame_exclusive
        last_family_end[spec.family] = opt.end_frame_exclusive
        if spec.is_strong_transform:
            last_strong_end = opt.end_frame_exclusive

        explain_selected.append(
            {
                "event_id": ev.id,
                "effect_id": spec.id,
                "code": spec.code,
                "family": spec.family,
                "engine": opt.engine,
                "start_frame": opt.start_frame,
                "end_frame_exclusive": opt.end_frame_exclusive,
                "score": round(opt.score, 4),
                "terms": opt.terms,
                "reasons": opt.reasons,
            }
        )

    ev_counter = 1
    for opt in trans_options:
        if trans_selected >= max_transitions:
            break
        ok, why = _can_place(opt)
        if not ok:
            rejections.append({"candidate": opt.candidate.id, "effect": opt.spec.id, "reason": why})
            continue
        _commit(opt, ev_counter)
        ev_counter += 1
        trans_selected += 1

    # Greedy iterative selection with novelty & family-diversity bonuses so a 15s-30s clip gets balanced variety (>=3 families when safe)
    main_selected = 0
    remaining = list(main_options)
    while main_selected < max_main_events and remaining:
        # Dynamic re-score with novelty & crowding penalties
        best_opt: ScoredOption | None = None
        best_adj = -1e9
        next_remaining: list[ScoredOption] = []
        for opt in remaining:
            rep_pen = -0.22 * used_counts.get(opt.spec.id, 0)
            fam_bonus = 0.16 if used_families.get(opt.spec.family, 0) == 0 else (-0.06 * used_families.get(opt.spec.family, 0))
            adj = opt.score + rep_pen + fam_bonus
            if adj > best_adj or (
                abs(adj - best_adj) < 1e-9
                and best_opt is not None
                and (opt.start_frame, opt.spec.code, opt.candidate.id) < (best_opt.start_frame, best_opt.spec.code, best_opt.candidate.id)
            ):
                best_adj = adj
                best_opt = opt
            next_remaining.append(opt)

        if best_opt is None:
            break
        remaining.remove(best_opt)

        ok, why = _can_place(best_opt)
        if not ok:
            rejections.append({"candidate": best_opt.candidate.id, "effect": best_opt.spec.id, "reason": why})
            continue

        best_opt.terms["novelty_diversity"] = round(best_adj - best_opt.score, 4)
        best_opt.score = best_adj
        _commit(best_opt, ev_counter)
        ev_counter += 1
        main_selected += 1

    # Stable deterministic ordering
    selected_events.sort(key=lambda e: (e.start_frame, e.end_frame_exclusive, e.layer, e.priority, e.id))

    status = "PLANNED" if selected_events else "NO_SAFE_EFFECTS"
    if status == "NO_SAFE_EFFECTS":
        warnings.append("NO_SAFE_EFFECTS: no eligible safe effects met policy thresholds; passthrough only.")

    plan = EditPlan(
        schema_version=SCHEMA_VERSION,
        source_hash=media.input_hash,
        config_hash=config.config_hash(),
        seed=config.seed,
        profile=profile.name,
        requested_profile=config.profile,
        width=target_w,
        height=target_h,
        fps_num=media.fps_num,
        fps_den=media.fps_den,
        duration_frames=dur_frames,
        aspect_mode=config.aspect,
        aspect_bg_mode=config.aspect_bg_mode,
        audio_policy=audio_policy,
        engine_capabilities=engine_capabilities,
        events=selected_events,
        planner_version=PLANNER_VERSION,
        status=status,
        warnings=sorted(set(warnings)),
        layout_plan=layout_plan,
    )

    validate_plan(plan, registry=reg)

    explain = {
        "schema_version": SCHEMA_VERSION,
        "planner_version": PLANNER_VERSION,
        "plan_hash": plan.plan_hash(),
        "status": status,
        "profile_selected": profile.name,
        "profile_requested": config.profile,
        "profile_reasons": profile_reasons,
        "policy": profile.to_dict(),
        "budgets": {
            "max_main_events": max_main_events,
            "max_transitions": max_transitions,
            "selected_main_events": main_selected,
            "selected_transitions": trans_selected,
            "families_used": sorted(used_families.keys()),
        },
        "selected": explain_selected,
        "rejection_count": len(rejections),
        "rejections_sample": rejections[:120],
    }
    return plan, explain


def validate_plan(plan: EditPlan, registry: EffectRegistry | None = None) -> None:
    """Phase D6: Strict plan validator rejecting unknown effects, overlaps, bad bounds, or forbidden audio ops."""
    reg = registry or load_effects()

    if plan.width <= 0 or plan.height <= 0 or plan.width % 2 != 0 or plan.height % 2 != 0:
        raise ValueError(f"Invalid target dimensions: {plan.width}x{plan.height}")
    if plan.duration_frames <= 0:
        raise ValueError(f"Invalid duration_frames: {plan.duration_frames}")
    if plan.audio_policy.get("retime_allowed") or plan.audio_policy.get("filters_allowed"):
        raise ValueError("Forbidden audio policy: audio retime/filtering is prohibited in ML-01")

    seen_ids: set[str] = set()
    for ev in plan.events:
        if ev.id in seen_ids:
            raise ValueError(f"Duplicate event id '{ev.id}'")
        seen_ids.add(ev.id)
        ev.validate(plan.duration_frames)
        spec = reg.get(ev.effect_id)
        reg.validate_params(ev.effect_id, ev.parameters)
        if ev.layer != spec.layer:
            raise ValueError(f"Event '{ev.id}' layer '{ev.layer}' mismatches spec layer '{spec.layer}'")
        if ev.fallback_id is not None:
            reg.get(ev.fallback_id)

    # Verify no forbidden concurrent overlaps
    for i, a in enumerate(plan.events):
        spec_a = reg.get(a.effect_id)
        for b in plan.events[i + 1 :]:
            if a.start_frame < b.end_frame_exclusive and a.end_frame_exclusive > b.start_frame:
                spec_b = reg.get(b.effect_id)
                if a.layer == b.layer:
                    raise ValueError(f"Overlapping events on same layer '{a.layer}': {a.id} and {b.id}")
                if b.layer in spec_a.incompatibilities or a.layer in spec_b.incompatibilities:
                    raise ValueError(f"Incompatible overlapping layers: {a.id} ({a.layer}) and {b.id} ({b.layer})")
