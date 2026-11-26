"""Phase D2 & D7: Explicit JSON profile loading and conservative signal-based auto selection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from motionlab.config import PRESETS_DIR, VALID_PROFILES
from motionlab.schemas import AnalysisResult, SchemaValidationError


@dataclass(frozen=True)
class ProfilePolicy:
    name: str
    description: str
    max_accents_per_15s: int
    max_transitions_per_15s: int
    intensity_scale: float
    min_strong_spacing_s: float
    same_effect_cooldown_s: float
    same_family_cooldown_s: float
    allow_subject_zoom: bool
    allow_layout_masks: bool
    allow_split_layout: bool
    allow_audio_reactive: bool
    protect_edges_strict: bool
    preferred_families: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "max_accents_per_15s": self.max_accents_per_15s,
            "max_transitions_per_15s": self.max_transitions_per_15s,
            "intensity_scale": self.intensity_scale,
            "min_strong_spacing_s": self.min_strong_spacing_s,
            "same_effect_cooldown_s": self.same_effect_cooldown_s,
            "same_family_cooldown_s": self.same_family_cooldown_s,
            "allow_subject_zoom": self.allow_subject_zoom,
            "allow_layout_masks": self.allow_layout_masks,
            "allow_split_layout": self.allow_split_layout,
            "allow_audio_reactive": self.allow_audio_reactive,
            "protect_edges_strict": self.protect_edges_strict,
            "preferred_families": list(self.preferred_families),
        }


def load_profile(name: str, presets_dir: Path | None = None) -> ProfilePolicy:
    """Load an explicit JSON profile preset from motionlab/presets/."""
    if name == "auto" or name not in VALID_PROFILES:
        raise SchemaValidationError(f"Cannot load non-concrete profile: {name!r}")
    pdir = presets_dir or PRESETS_DIR
    path = pdir / f"{name}.json"
    if not path.is_file():
        raise SchemaValidationError(f"Preset JSON file missing: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return ProfilePolicy(
        name=str(data["name"]),
        description=str(data["description"]),
        max_accents_per_15s=int(data["max_accents_per_15s"]),
        max_transitions_per_15s=int(data["max_transitions_per_15s"]),
        intensity_scale=float(data["intensity_scale"]),
        min_strong_spacing_s=float(data["min_strong_spacing_s"]),
        same_effect_cooldown_s=float(data["same_effect_cooldown_s"]),
        same_family_cooldown_s=float(data["same_family_cooldown_s"]),
        allow_subject_zoom=bool(data["allow_subject_zoom"]),
        allow_layout_masks=bool(data["allow_layout_masks"]),
        allow_split_layout=bool(data["allow_split_layout"]),
        allow_audio_reactive=bool(data["allow_audio_reactive"]),
        protect_edges_strict=bool(data["protect_edges_strict"]),
        preferred_families=tuple(data["preferred_families"]),
    )


def select_profile(analysis: AnalysisResult, requested: str) -> tuple[ProfilePolicy, str]:
    """Select profile strictly from user choice or deterministic measurable media traits.

    NEVER claims semantic understanding of spoken words, narrative, product, or niche.
    Uncertain inputs always fall back to `neutral`.
    """
    if requested != "auto":
        policy = load_profile(requested)
        return policy, f"user_explicit_profile:{requested}"

    flags = set(analysis.quality_flags)
    stats = analysis.summary_stats
    total_f = max(1, analysis.duration_frames)
    face_cov = float(stats.get("face_coverage_frames", 0)) / total_f
    ui_edge = float(stats.get("ui_edge_density", 0.0))
    mean_motion = float(stats.get("mean_motion", 0.0))
    has_ambig_multi = "multi_subject_ambiguous" in flags

    # 1. Screen / dense UI protection takes top priority
    if "busy_ui_or_text_detected" in flags and ui_edge >= 0.075 and mean_motion < 0.25:
        return load_profile("tutorial"), (
            f"signal_heuristic:dense_ui_edges(ui_edge_density={ui_edge:.3f},mean_motion={mean_motion:.3f})"
        )

    # 2. High continuous motion -> conservative gameplay-like HUD/motion protection
    if "high_existing_motion" in flags or mean_motion >= 0.42:
        return load_profile("gameplay"), (
            f"signal_heuristic:high_existing_motion(mean_motion={mean_motion:.3f})"
        )

    # 3. Clear single-subject frontal face coverage -> talking-head framing safety
    if face_cov >= 0.30 and not has_ambig_multi:
        return load_profile("talking-head"), (
            f"signal_heuristic:stable_single_face_coverage(face_cov={face_cov:.2f})"
        )

    # 4. Default uncertain fallback -> conservative neutral
    return load_profile("neutral"), (
        f"signal_heuristic:default_neutral(face_cov={face_cov:.2f},mean_motion={mean_motion:.3f},ui_edge={ui_edge:.3f})"
    )
