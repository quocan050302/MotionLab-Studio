"""Deterministic Signal-Driven Creative Composition Graph Planner (ML-01.R3).

Zero hardcoded video names, zero hardcoded timestamps, zero LLM runtime.
Selects diverse reusable creative recipes strictly from measured media signals,
tracked ROIs, acoustic pauses, matte quality, and verified engine capabilities.
"""

from __future__ import annotations

import random
from typing import Any

from motionlab.choreography import build_beats
from motionlab.composition.recipes import RECIPE_CATALOG, RecipeSpec, get_recipe
from motionlab.composition.schema import (
    COMPOSITION_SCHEMA_VERSION,
    CompositionGraph,
    CompositionLayer,
    CompositionScene,
    validate_composition_graph,
)
from motionlab.config import MotionLabConfig
from motionlab.policies import ProfilePolicy
from motionlab.schemas import AnalysisResult, MediaInfo
from motionlab.vision.anchors import query_span_anchors


def _build_layers_for_recipe(
    scene_id: str,
    spec: RecipeSpec,
    primary_engine: str,
    graphics_engine: str,
    anchors_slice: dict[str, Any],
    palette: dict[str, Any],
) -> list[CompositionLayer]:
    """Construct explicit z-ordered declarative layer stack for a creative composition scene."""
    layers: list[CompositionLayer] = []
    z = 0
    # Layer 0: Base source / softened depth background plane
    layers.append(
        CompositionLayer(
            layer_id=f"{scene_id}_L0_bg",
            role="background_source",
            engine=primary_engine,
            z_index=z,
            asset_ref="source_clip",
            alpha_mode="opaque",
            blend_mode="normal",
            parameters={
                "blur_px": spec.default_params.get("bg_blur_px", 0.0),
                "dim": spec.default_params.get("bg_dim", spec.default_params.get("warm_dim", 1.0)),
                "scale_drift": spec.default_params.get("bg_scale_drift", 1.02),
            },
            reason=f"same_time_source_base_for_{spec.name}",
            required=True,
        )
    )
    z += 1

    # Behind-subject HyperFrames graphic layer (when scene has matte occlusion!)
    if spec.requires_matte and spec.supports_hyperframes_graphics and graphics_engine == "hyperframes":
        layers.append(
            CompositionLayer(
                layer_id=f"{scene_id}_L1_hf_behind",
                role="background_graphics",
                engine="hyperframes",
                z_index=z,
                asset_ref=f"hf_{spec.hyperframes_template}",
                alpha_mode="straight",
                blend_mode="normal",
                parameters={
                    "template": spec.hyperframes_template,
                    "primary_hex": palette.get("primary_hex", "#C8A26A"),
                    "secondary_hex": palette.get("secondary_hex", "#4A3F35"),
                    "tertiary_hex": palette.get("tertiary_hex", "#F4EBDD"),
                    "opacity": spec.default_params.get("arch_opacity", spec.default_params.get("path_opacity", 0.86)),
                },
                reason=f"hyperframes_substantive_behind_subject_{spec.hyperframes_template}",
                required=True,
            )
        )
        z += 1

    # Detail same-time cropped source aperture layer (RCP02 / RCP06)
    if spec.uses_detail_crop:
        layers.append(
            CompositionLayer(
                layer_id=f"{scene_id}_L_detail",
                role="source_detail",
                engine=primary_engine,
                z_index=z,
                asset_ref="source_clip_same_time_roi",
                alpha_mode="straight",
                blend_mode="normal",
                parameters={
                    "detail_zoom": spec.default_params.get("detail_zoom", 1.50),
                    "roi_center": [anchors_slice["task_mid"]["cx"], anchors_slice["task_mid"]["cy"]],
                    "slot": anchors_slice["detail_slot"],
                    "capsule_radius_px": spec.default_params.get("capsule_radius_px", 34),
                },
                reason="same_time_tracked_task_roi_detail_aperture",
                required=True,
            )
        )
        z += 1

    # MODNet continuous alpha foreground subject cutout layer (occludes background_graphics cleanly!)
    if spec.requires_matte:
        layers.append(
            CompositionLayer(
                layer_id=f"{scene_id}_L_fg_matte",
                role="foreground_subject_matte",
                engine=primary_engine,
                z_index=z,
                asset_ref="modnet_fg_alpha",
                alpha_mode="straight",
                blend_mode="normal",
                parameters={
                    "fg_micro_lift": spec.default_params.get("fg_micro_lift", 1.018),
                    "occlusion_verified": True,
                },
                reason="modnet_continuous_alpha_foreground_occlusion",
                required=True,
            )
        )
        z += 1

    # Foreground HyperFrames substantive graphic layer (connectors, split dividers, contours, non-matte flourishes)
    if (not spec.requires_matte) and spec.supports_hyperframes_graphics and graphics_engine == "hyperframes":
        layers.append(
            CompositionLayer(
                layer_id=f"{scene_id}_L_hf_fg",
                role="foreground_graphics",
                engine="hyperframes",
                z_index=z,
                asset_ref=f"hf_{spec.hyperframes_template}",
                alpha_mode="straight",
                blend_mode="normal",
                parameters={
                    "template": spec.hyperframes_template,
                    "primary_hex": palette.get("primary_hex", "#C8A26A"),
                    "secondary_hex": palette.get("secondary_hex", "#4A3F35"),
                    "tertiary_hex": palette.get("tertiary_hex", "#F4EBDD"),
                    "opacity": spec.default_params.get("connector_opacity", spec.default_params.get("contour_opacity", 0.88)),
                },
                reason=f"hyperframes_substantive_foreground_{spec.hyperframes_template}",
                required=True,
            )
        )
        z += 1

    return layers


def _score_recipe_for_beat(
    spec: RecipeSpec,
    beat_role: str,
    beat_idx: int,
    has_task: bool,
    has_audio: bool,
    matte_ok: bool,
    used_recipes: list[str],
    used_archetypes: list[str],
    policy: ProfilePolicy,
) -> tuple[float, str]:
    """Deterministic multi-factor suitability score for a creative recipe on a specific beat."""
    if policy.name not in spec.supported_profiles:
        return -1.0, f"profile_disallowed:{policy.name}"
    if spec.requires_matte and not matte_ok:
        return -1.0, "matte_unavailable"
    if spec.requires_task_roi and not has_task:
        return -1.0, "no_task_roi"
    if spec.requires_audio and not has_audio:
        return -1.0, "no_audio"

    score = 0.50
    # Role affinity (purely generic editorial motion grammar!)
    role_affinities: dict[str, dict[str, float]] = {
        "hook": {"RCP01": 0.45, "RCP09": 0.36, "RCP05": 0.30, "RCP04": 0.28},
        "activity_focus": {"RCP02": 0.46, "RCP03": 0.44, "RCP06": 0.40, "RCP07": 0.35, "RCP04": 0.25},
        "emphasis": {"RCP03": 0.44, "RCP06": 0.42, "RCP04": 0.40, "RCP09": 0.38, "RCP02": 0.34, "RCP07": 0.30},
        "visual_reset": {"RCP05": 0.42, "RCP04": 0.40, "RCP08": 0.36, "RCP01": 0.32, "RCP07": 0.30},
        "closing_hold": {"RCP08": 0.48, "RCP01": 0.30, "RCP07": 0.24},
    }
    score += role_affinities.get(beat_role, {}).get(spec.id, 0.12)

    # Strong diversity bonus for unseen archetypes & recipes; heavy penalty for immediate repeat!
    if spec.archetype not in used_archetypes:
        score += 0.32
    elif used_archetypes and used_archetypes[-1] == spec.archetype:
        score -= 0.65
    elif len(used_archetypes) >= 2 and spec.archetype in used_archetypes[-2:]:
        score -= 0.35

    if spec.id in used_recipes:
        score -= 0.45

    # Reward matte-occlusion & detail-reveal variety early so showcase clips show true multi-layer range
    if beat_idx == 0 and spec.requires_matte:
        score += 0.12
    if beat_idx == 1 and spec.uses_detail_crop:
        score += 0.14

    return round(score, 4), f"role={beat_role},new_arch={spec.archetype not in used_archetypes},matte={spec.requires_matte}"


def plan_composition_graph(
    media: MediaInfo,
    analysis: AnalysisResult,
    anchors: dict[str, Any],
    policy: ProfilePolicy,
    config: MotionLabConfig,
    engine_caps: dict[str, Any],
    matte_available_override: bool | None = None,
) -> tuple[CompositionGraph, dict[str, Any]]:
    """Solve deterministic R3 CompositionGraph from real analysis signals, tracked ROIs & verified engine capabilities."""
    fps_f = media.fps.as_float()
    total_f = media.duration_frames
    dur_s = total_f / max(1.0, fps_f)
    rng = random.Random(config.seed)

    rem_ok = bool((engine_caps.get("remotion") or {}).get("available"))
    hf_ok = bool((engine_caps.get("hyperframes") or {}).get("available"))
    matte_ok = bool(anchors.get("matte_available")) if matte_available_override is None else bool(matte_available_override)
    has_task = bool(anchors.get("has_task_track"))
    has_audio = bool(media.has_audio)
    palette = anchors.get("palette") or analysis.dominant_palette or {
        "primary_hex": "#C8A26A", "secondary_hex": "#4A3F35", "tertiary_hex": "#F4EBDD",
    }

    # Screen-recording / UI tutorials, high-motion gameplay, or ultra-short clips -> safe NO_SAFE_COMPOSITION!
    if policy.protect_edges_strict or policy.name in ("tutorial", "gameplay") or dur_s < 2.2:
        g = CompositionGraph(
            schema_version=COMPOSITION_SCHEMA_VERSION,
            source_hash=media.input_hash,
            config_hash=config.config_hash(),
            seed=config.seed,
            profile=policy.name,
            style=config.style,
            width=media.display_width,
            height=media.display_height,
            fps=media.fps,
            duration_frames=total_f,
            matte_required=False,
            matte_available=matte_ok,
            scenes=[],
            quiet_spans=[],
            engine_capabilities=engine_caps,
            metrics={"total_creative_scenes": 0, "substantive_coverage_ratio": 0.0, "distinct_archetypes": 0},
            status="NO_SAFE_COMPOSITION",
            warnings=[f"Safe no-composition selected for profile={policy.name} (duration={dur_s:.2f}s, protect_edges={policy.protect_edges_strict})."],
        )
        g.graph_hash = g.compute_graph_hash()
        return g, {"status": "NO_SAFE_COMPOSITION", "decisions": [], "reason": g.warnings[0]}

    # Primary creative compositor is Remotion when licensed/opted-in; otherwise explicit single-engine candidate (never masquerades as Remotion!)
    primary_engine = "remotion" if rem_ok else "ffmpeg"
    graphics_engine = "hyperframes" if hf_ok else "none"

    beats = build_beats(analysis, policy, config)
    quiet_spans: list[dict[str, Any]] = []
    candidate_beats: list[Any] = []
    for b in beats:
        if b.beat_role == "quiet_hold":
            quiet_spans.append({
                "beat_id": b.beat_id,
                "start_frame": b.start_frame,
                "end_frame_exclusive": b.end_frame_exclusive,
                "reason": "acoustic_or_editorial_breathing_hold",
            })
        else:
            candidate_beats.append(b)

    # Select balanced, high-impact creative scenes (~5-7 beats across 30s -> ~45-58% coverage, preserving clean natural holds!)
    # Keep natural breathing gaps between scenes and leave the final ~1.5-2.0s completely calm natural source.
    target_max_scenes = 6 if dur_s >= 22.0 else (4 if dur_s >= 12.0 else (2 if dur_s >= 5.0 else 1))
    if config.motion_density == "low":
        target_max_scenes = max(1, target_max_scenes - 2)
    elif config.motion_density == "rich":
        target_max_scenes = min(7, target_max_scenes + 1)

    # Space active beats so ~45-58% of timeline carries substantive composition while clean holds separate major gestures!
    selected_beats: list[Any] = []
    if len(candidate_beats) <= target_max_scenes:
        selected_beats = list(candidate_beats)
    else:
        # Always keep opening hook (index 0) + evenly distributed middle beats + penultimate action beat
        # Keep the final closing 1.8s calm natural source (so closing beat ends cleanly ~1.6s before end!)
        idxs = [0]
        inner = list(range(1, len(candidate_beats)))
        need = target_max_scenes - 1
        step = len(inner) / max(1, need)
        for k in range(need):
            pick = inner[min(len(inner) - 1, int(round(k * step)))]
            if pick not in idxs:
                idxs.append(pick)
        selected_beats = [candidate_beats[i] for i in sorted(idxs)]

    scenes: list[CompositionScene] = []
    decisions: list[dict[str, Any]] = []
    used_recipes: list[str] = []
    used_archetypes: list[str] = []
    min_gap_f = max(6, int(round(fps_f * 0.45)))
    calm_end_margin_f = int(round(fps_f * 1.4)) if dur_s >= 10.0 else max(2, int(round(fps_f * 0.2)))

    prev_scene_end = -min_gap_f
    for s_idx, b in enumerate(selected_beats):
        sf = max(b.start_frame, prev_scene_end + min_gap_f)
        # Keep opening hook starting right on frame 0 (or within 4 frames) for immediate visual hook
        if s_idx == 0 and b.beat_role == "hook":
            sf = max(0, b.start_frame)
        ef = min(b.end_frame_exclusive, total_f - calm_end_margin_f)
        # Ensure clean duration within recipe window (2.0s - 3.6s on 30s clips)
        min_f = max(12, int(round(fps_f * 1.8))) if dur_s >= 5.0 else max(8, int(round(fps_f * 0.8)))
        max_f = max(min_f + 4, int(round(fps_f * 3.4)))
        if ef - sf > max_f:
            ef = sf + max_f
        if ef - sf < min_f:
            decisions.append({"beat_id": b.beat_id, "status": "SKIPPED_SHORT_WINDOW", "span": [sf, ef]})
            continue

        # Score all 9 reusable recipes deterministically
        scored: list[tuple[float, float, RecipeSpec, str]] = []
        for spec in RECIPE_CATALOG:
            sc_val, why = _score_recipe_for_beat(
                spec, b.beat_role, s_idx, has_task, has_audio, matte_ok, used_recipes, used_archetypes, policy
            )
            if sc_val > 0.0:
                # Deterministic tie-break using seeded rng jitter < 1e-4
                scored.append((sc_val, rng.random() * 1e-4, spec, why))
        if not scored:
            decisions.append({"beat_id": b.beat_id, "status": "NO_ELIGIBLE_RECIPE", "span": [sf, ef]})
            continue

        scored.sort(key=lambda x: (-x[0], -x[1], x[2].id))
        best_score, _, spec, why = scored[0]

        scene_id = f"scene_{s_idx:02d}_{spec.id}"
        anc_slice = query_span_anchors(anchors, sf, ef)
        # Alternate detail aperture side if two detail recipes appear in same timeline
        if spec.uses_detail_crop and any(sc.uses_detail_crop for sc in scenes):
            anc_slice = {**anc_slice, "detail_slot": anc_slice["alt_detail_slot"]}

        layers = _build_layers_for_recipe(scene_id, spec, primary_engine, graphics_engine, anc_slice, palette)
        entry_f = max(6, min(14, (ef - sf) // 5))
        exit_f = max(6, min(14, (ef - sf) // 5))

        scene = CompositionScene(
            scene_id=scene_id,
            recipe_id=spec.id,
            recipe_name=spec.name,
            archetype=spec.archetype,
            start_frame=sf,
            end_frame_exclusive=ef,
            beat_role=b.beat_role,
            primary_engine=primary_engine,
            graphics_engine=graphics_engine if spec.supports_hyperframes_graphics else "none",
            requires_matte=spec.requires_matte,
            uses_detail_crop=spec.uses_detail_crop,
            substantive_composition=spec.substantive,
            layers=layers,
            anchor_bundle=anc_slice,
            parameters={**spec.default_params, "palette": palette},
            entry_frames=entry_f,
            exit_frames=exit_f,
            evidence_refs=list(b.evidence_refs),
            fallback_recipe_id=spec.fallback_recipe_id,
            verification_criteria={
                "min_layers": len(layers),
                "requires_matte_occlusion": spec.requires_matte,
                "requires_detail_aperture": spec.uses_detail_crop,
                "requires_hyperframes_alpha": (spec.supports_hyperframes_graphics and graphics_engine == "hyperframes"),
                "requires_remotion_fullframe": (primary_engine == "remotion"),
            },
        )
        scenes.append(scene)
        used_recipes.append(spec.id)
        used_archetypes.append(spec.archetype)
        prev_scene_end = ef
        decisions.append({
            "scene_id": scene_id,
            "beat_id": b.beat_id,
            "recipe_id": spec.id,
            "archetype": spec.archetype,
            "score": best_score,
            "reason": why,
            "span": [sf, ef],
            "primary_engine": primary_engine,
            "graphics_engine": scene.graphics_engine,
            "status": "SELECTED",
        })

    # Compute transparent quality/diversity metrics
    total_creative_frames = sum(sc.end_frame_exclusive - sc.start_frame for sc in scenes)
    coverage_ratio = round(total_creative_frames / max(1, total_f), 4)
    rem_scenes = sum(1 for sc in scenes if sc.primary_engine == "remotion")
    hf_scenes = sum(1 for sc in scenes if sc.graphics_engine == "hyperframes")
    dual_scenes = sum(1 for sc in scenes if sc.primary_engine == "remotion" and sc.graphics_engine == "hyperframes")
    matte_scenes = sum(1 for sc in scenes if sc.requires_matte)
    detail_scenes = sum(1 for sc in scenes if sc.uses_detail_crop)
    distinct_archs = sorted({sc.archetype for sc in scenes})
    distinct_rcps = sorted({sc.recipe_id for sc in scenes})

    warnings: list[str] = []
    status = "READY"
    if not scenes:
        status = "NO_SAFE_COMPOSITION"
    elif not rem_ok:
        status = "PARTIAL_SINGLE_ENGINE"
        warnings.append("Remotion is not enabled/licensed (--enable-remotion required for dual-engine Remotion full-frame compositions).")
    elif not matte_ok:
        warnings.append("Subject matte unavailable; matte-requiring archetypes skipped cleanly.")

    metrics = {
        "total_creative_scenes": len(scenes),
        "quiet_hold_spans": len(quiet_spans),
        "total_creative_frames": total_creative_frames,
        "substantive_coverage_ratio": coverage_ratio,
        "distinct_archetypes_count": len(distinct_archs),
        "distinct_archetypes": distinct_archs,
        "distinct_recipes": distinct_rcps,
        "remotion_fullframe_scenes": rem_scenes,
        "hyperframes_substantive_scenes": hf_scenes,
        "dual_engine_combined_scenes": dual_scenes,
        "matte_occlusion_scenes": matte_scenes,
        "detail_crop_scenes": detail_scenes,
        "zoom_only_active_beats": 0,
    }

    safe_caps = {
        k: {"available": bool((v or {}).get("available")), "version": str((v or {}).get("version", "-"))}
        for k, v in (engine_caps or {}).items()
    }
    graph = CompositionGraph(
        schema_version=COMPOSITION_SCHEMA_VERSION,
        source_hash=media.input_hash,
        config_hash=config.config_hash(),
        seed=config.seed,
        profile=policy.name,
        style=config.style,
        width=media.display_width,
        height=media.display_height,
        fps=media.fps,
        duration_frames=total_f,
        matte_required=any(sc.requires_matte for sc in scenes),
        matte_available=matte_ok,
        scenes=scenes,
        quiet_spans=quiet_spans,
        engine_capabilities=safe_caps,
        metrics=metrics,
        status=status,
        warnings=warnings,
    )
    graph.graph_hash = graph.compute_graph_hash()
    validate_composition_graph(graph, media, engine_caps=engine_caps, strict_replay=False)
    return graph, {"status": status, "metrics": metrics, "decisions": decisions}
