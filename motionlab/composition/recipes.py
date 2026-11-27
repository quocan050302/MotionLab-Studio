"""ML-01.R3 Reusable Creative Compositing Recipe Registry (9 Substantive Archetypes: RCP01..RCP09).

Every recipe is 100% reusable across footage types (zero hardcoded video/subject/timestamp logic),
declarative, deterministic, and strictly distinct from camera-only zoom or edge-line decorations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class RecipeSpec:
    id: str
    name: str
    archetype: str
    description: str
    supported_profiles: tuple[str, ...]
    requires_matte: bool
    requires_task_roi: bool
    requires_audio: bool
    uses_detail_crop: bool
    supports_hyperframes_graphics: bool
    hyperframes_template: str | None
    remotion_scene_kind: str
    min_duration_s: float
    max_duration_s: float
    max_crop_zoom: float
    substantive: bool
    fallback_recipe_id: str | None
    default_params: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


NON_UI_PROFILES = ("talking-head", "product", "storytelling", "cinematic", "neutral")

RECIPE_CATALOG: tuple[RecipeSpec, ...] = (
    RecipeSpec(
        id="RCP01",
        name="SubjectDepthStage",
        archetype="depth_stage",
        description=(
            "True foreground/background separation using MODNet continuous portrait matte: softens and subtly grades "
            "the real background plane while inserting a sculptural HyperFrames GSAP architectural arch & depth geometry "
            "BEHIND the speaker's head, hair, and shoulders with real foreground occlusion."
        ),
        supported_profiles=("talking-head", "storytelling", "neutral"),
        requires_matte=True,
        requires_task_roi=False,
        requires_audio=False,
        uses_detail_crop=False,
        supports_hyperframes_graphics=True,
        hyperframes_template="depth-stage",
        remotion_scene_kind="depth_stage",
        min_duration_s=1.8,
        max_duration_s=4.0,
        max_crop_zoom=1.06,
        substantive=True,
        fallback_recipe_id="RCP05",
        default_params={
            "bg_blur_px": 9.0,
            "bg_scale_drift": 1.045,
            "fg_micro_lift": 1.022,
            "arch_opacity": 0.86,
            "warm_dim": 0.90,
        },
    ),
    RecipeSpec(
        id="RCP02",
        name="AsymmetricDetailReveal",
        archetype="detail_reveal",
        description=(
            "Same-time organic curved-capsule detail aperture framing the tracked active foreground task ROI "
            "(moderate 1.48x–1.60x detail zoom without pixelation) tucked into verified negative space, paired with "
            "a HyperFrames GSAP animated vector callout arc connecting the live hands/task region to the detail card."
        ),
        supported_profiles=NON_UI_PROFILES,
        requires_matte=False,
        requires_task_roi=True,
        requires_audio=False,
        uses_detail_crop=True,
        supports_hyperframes_graphics=True,
        hyperframes_template="detail-connector",
        remotion_scene_kind="detail_aperture",
        min_duration_s=2.0,
        max_duration_s=4.2,
        max_crop_zoom=1.58,
        substantive=True,
        fallback_recipe_id="RCP07",
        default_params={
            "detail_zoom": 1.52,
            "main_shift_x": 0.04,
            "capsule_radius_px": 34,
            "shadow_opacity": 0.34,
            "connector_opacity": 0.88,
        },
    ),
    RecipeSpec(
        id="RCP03",
        name="TrackedActionPath",
        archetype="action_path",
        description=(
            "Action-driven HyperFrames GSAP/SVG hero curved trajectory path, travelling directional nodes, and soft "
            "origin/destination rings anchored to observed foreground motion vectors, composited in Remotion with "
            "true z-order occlusion behind the MODNet subject matte so geometry sweeps organically around the speaker."
        ),
        supported_profiles=("talking-head", "product", "storytelling", "neutral"),
        requires_matte=True,
        requires_task_roi=True,
        requires_audio=False,
        uses_detail_crop=False,
        supports_hyperframes_graphics=True,
        hyperframes_template="action-path",
        remotion_scene_kind="action_path",
        min_duration_s=2.0,
        max_duration_s=4.2,
        max_crop_zoom=1.05,
        substantive=True,
        fallback_recipe_id="RCP07",
        default_params={
            "bg_depth_dim": 0.94,
            "path_opacity": 0.90,
            "fg_occlusion": True,
            "bg_blur_px": 4.5,
        },
    ),
    RecipeSpec(
        id="RCP04",
        name="ForegroundPassThrough",
        archetype="foreground_passthrough",
        description=(
            "Sculptural editorial ribbon and layered geometric arc rendered in HyperFrames sweeping across the "
            "upper-mid plane BEHIND the speaker's hair and shoulders (occluded by MODNet alpha matte), while an "
            "independent subtle focal contour accompanies foreground hand motion — creating crisp layered depth."
        ),
        supported_profiles=("talking-head", "storytelling", "neutral"),
        requires_matte=True,
        requires_task_roi=False,
        requires_audio=False,
        uses_detail_crop=False,
        supports_hyperframes_graphics=True,
        hyperframes_template="pass-through",
        remotion_scene_kind="action_path",
        min_duration_s=2.0,
        max_duration_s=3.8,
        max_crop_zoom=1.05,
        substantive=True,
        fallback_recipe_id="RCP05",
        default_params={
            "bg_depth_dim": 0.92,
            "path_opacity": 0.88,
            "fg_occlusion": True,
            "bg_blur_px": 6.5,
        },
    ),
    RecipeSpec(
        id="RCP05",
        name="MatchShapeLayoutMorph",
        archetype="layout_morph",
        description=(
            "Duration-preserving organic curved-mask morph transitioning smoothly from full-bleed source into an "
            "elevated editorial recessed stage over a softly blurred warm ambient backdrop with independent layer "
            "parallax and a HyperFrames curved editorial accent frame, morphing cleanly back to full-bleed."
        ),
        supported_profiles=NON_UI_PROFILES,
        requires_matte=False,
        requires_task_roi=False,
        requires_audio=False,
        uses_detail_crop=False,
        supports_hyperframes_graphics=True,
        hyperframes_template="match-shape",
        remotion_scene_kind="layout_morph",
        min_duration_s=2.0,
        max_duration_s=3.8,
        max_crop_zoom=1.06,
        substantive=True,
        fallback_recipe_id=None,
        default_params={
            "stage_scale": 0.88,
            "stage_radius_px": 36,
            "bg_blur_px": 18.0,
            "bg_dim": 0.72,
        },
    ),
    RecipeSpec(
        id="RCP06",
        name="SplitFocusEditorial",
        archetype="split_focus",
        description=(
            "Asymmetric curved diagonal dual-focus layout: upper editorial pane retains the speaker's face and "
            "upper context while a curved lower-corner detail pane highlights the same-time tracked hands/task "
            "action with independent motion and a HyperFrames animated curved divider (never duplicating full bodies)."
        ),
        supported_profiles=("talking-head", "product", "storytelling", "neutral"),
        requires_matte=False,
        requires_task_roi=True,
        requires_audio=False,
        uses_detail_crop=True,
        supports_hyperframes_graphics=True,
        hyperframes_template="split-divider",
        remotion_scene_kind="split_focus",
        min_duration_s=2.2,
        max_duration_s=4.5,
        max_crop_zoom=1.55,
        substantive=True,
        fallback_recipe_id="RCP02",
        default_params={
            "detail_zoom": 1.48,
            "split_ratio": 0.62,
            "curve_depth_px": 42,
            "divider_opacity": 0.90,
        },
    ),
    RecipeSpec(
        id="RCP07",
        name="ContextualContourAccent",
        archetype="contextual_contour",
        description=(
            "Tracked organic contour arcs and soft radial editorial spotlight anchored dynamically to the active "
            "foreground ROI via HyperFrames GSAP SVG, paired with gentle peripheral depth separation in Remotion "
            "without obstructing hands, objects, or face."
        ),
        supported_profiles=NON_UI_PROFILES,
        requires_matte=False,
        requires_task_roi=True,
        requires_audio=False,
        uses_detail_crop=False,
        supports_hyperframes_graphics=True,
        hyperframes_template="contour-focus",
        remotion_scene_kind="contour_focus",
        min_duration_s=1.8,
        max_duration_s=3.8,
        max_crop_zoom=1.05,
        substantive=True,
        fallback_recipe_id=None,
        default_params={
            "surround_dim": 0.14,
            "contour_opacity": 0.88,
            "spotlight_radius_frac": 0.34,
        },
    ),
    RecipeSpec(
        id="RCP08",
        name="DepthRevealAndRelease",
        archetype="depth_release",
        description=(
            "Refined closing/resolution depth separation: lifts foreground speaker slightly against a softly "
            "defocused warm kitchen depth plane with a subtle behind-subject HyperFrames curved horizon halo, "
            "then glides effortlessly back to natural 1.0x full-bleed source before the calm ending hold."
        ),
        supported_profiles=("talking-head", "storytelling", "cinematic", "neutral"),
        requires_matte=True,
        requires_task_roi=False,
        requires_audio=False,
        uses_detail_crop=False,
        supports_hyperframes_graphics=True,
        hyperframes_template="depth-horizon",
        remotion_scene_kind="depth_stage",
        min_duration_s=1.8,
        max_duration_s=3.6,
        max_crop_zoom=1.04,
        substantive=True,
        fallback_recipe_id="RCP05",
        default_params={
            "bg_blur_px": 7.5,
            "bg_scale_drift": 1.03,
            "fg_micro_lift": 1.018,
            "arch_opacity": 0.78,
            "warm_dim": 0.93,
        },
    ),
    RecipeSpec(
        id="RCP09",
        name="AudioReactiveHeroGraphic",
        archetype="audio_stage",
        description=(
            "Semantic-neutral concentric sculptural behind-subject graphic arcs driven smoothly (damped, non-strobe) "
            "by verified PTS-aligned audio RMS energy in HyperFrames, occluded behind the MODNet subject matte in "
            "Remotion so acoustic rhythm radiates organically from behind the speaker."
        ),
        supported_profiles=("talking-head", "storytelling", "neutral"),
        requires_matte=True,
        requires_task_roi=False,
        requires_audio=True,
        uses_detail_crop=False,
        supports_hyperframes_graphics=True,
        hyperframes_template="audio-stage",
        remotion_scene_kind="depth_stage",
        min_duration_s=2.0,
        max_duration_s=4.0,
        max_crop_zoom=1.05,
        substantive=True,
        fallback_recipe_id="RCP01",
        default_params={
            "bg_blur_px": 8.0,
            "bg_scale_drift": 1.035,
            "fg_micro_lift": 1.02,
            "arch_opacity": 0.88,
            "warm_dim": 0.91,
        },
    ),
)

_RECIPE_BY_ID: dict[str, RecipeSpec] = {r.id: r for r in RECIPE_CATALOG}


def list_recipes() -> list[RecipeSpec]:
    return list(RECIPE_CATALOG)


def get_recipe(recipe_id: str) -> RecipeSpec:
    if recipe_id not in _RECIPE_BY_ID:
        raise KeyError(f"Unknown creative recipe_id: {recipe_id}")
    return _RECIPE_BY_ID[recipe_id]


def build_recipe_capability_matrix(
    engine_caps: dict[str, Any],
    vision_model_info: dict[str, Any],
) -> dict[str, Any]:
    """Build honest capability matrix matching real implemented Remotion/HyperFrames/MODNet adapters."""
    rem_ok = bool((engine_caps.get("remotion") or {}).get("available"))
    hf_ok = bool((engine_caps.get("hyperframes") or {}).get("available"))
    matte_ok = bool(vision_model_info.get("available"))
    rows: list[dict[str, Any]] = []
    for r in RECIPE_CATALOG:
        can_rem = rem_ok and (not r.requires_matte or matte_ok)
        can_hf = hf_ok if r.supports_hyperframes_graphics else False
        rows.append({
            "recipe_id": r.id,
            "name": r.name,
            "archetype": r.archetype,
            "substantive": r.substantive,
            "requires_matte": r.requires_matte,
            "requires_task_roi": r.requires_task_roi,
            "uses_detail_crop": r.uses_detail_crop,
            "remotion_scene_kind": r.remotion_scene_kind,
            "hyperframes_template": r.hyperframes_template,
            "remotion_ready": can_rem,
            "hyperframes_ready": can_hf,
            "dual_engine_ready": can_rem and can_hf,
            "blocked_reason": (
                None if (can_rem and can_hf)
                else ("matte_model_missing" if (r.requires_matte and not matte_ok)
                      else ("remotion_license_opt_in_required" if not rem_ok else "hyperframes_unavailable"))
            ),
        })
    return {
        "total_recipes": len(RECIPE_CATALOG),
        "remotion_available": rem_ok,
        "hyperframes_available": hf_ok,
        "matte_model_available": matte_ok,
        "recipes": rows,
    }
