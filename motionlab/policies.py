"""Phase D1/D2/D7 — Style profile loading and deterministic signal-based auto-profile selection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from motionlab.config import CONCRETE_PROFILES, VALID_PROFILES
from motionlab.schemas import AnalysisResult, MediaInfo

PRESETS_DIR = Path(__file__).resolve().parent / "presets"


@dataclass(frozen=True)
class ProfilePolicy:
    name: str
    description: str
    max_events_per_15s: int
    max_transitions_per_15s: int
    strong_accent_cooldown_s: float
    repeat_effect_cooldown_s: float
    same_family_cooldown_s: float
    intensity_multiplier: float
    allow_layout_masks: bool
    allow_face_transforms: bool
    protect_edges_strict: bool
    preferred_families: tuple[str, ...]
    blocked_effects: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "max_events_per_15s": self.max_events_per_15s,
            "max_transitions_per_15s": self.max_transitions_per_15s,
            "strong_accent_cooldown_s": self.strong_accent_cooldown_s,
            "repeat_effect_cooldown_s": self.repeat_effect_cooldown_s,
            "same_family_cooldown_s": self.same_family_cooldown_s,
            "intensity_multiplier": self.intensity_multiplier,
            "allow_layout_masks": self.allow_layout_masks,
            "allow_face_transforms": self.allow_face_transforms,
            "protect_edges_strict": self.protect_edges_strict,
            "preferred_families": list(self.preferred_families),
            "blocked_effects": list(self.blocked_effects),
        }


def load_profile(name: str) -> ProfilePolicy:
    if name not in CONCRETE_PROFILES:
        raise ValueError(f"Unknown concrete profile '{name}'. Available: {', '.join(CONCRETE_PROFILES)}")
    path = PRESETS_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing profile config: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return ProfilePolicy(
        name=str(data["name"]),
        description=str(data["description"]),
        max_events_per_15s=int(data["max_events_per_15s"]),
        max_transitions_per_15s=int(data["max_transitions_per_15s"]),
        strong_accent_cooldown_s=float(data["strong_accent_cooldown_s"]),
        repeat_effect_cooldown_s=float(data["repeat_effect_cooldown_s"]),
        same_family_cooldown_s=float(data["same_family_cooldown_s"]),
        intensity_multiplier=float(data["intensity_multiplier"]),
        allow_layout_masks=bool(data["allow_layout_masks"]),
        allow_face_transforms=bool(data["allow_face_transforms"]),
        protect_edges_strict=bool(data["protect_edges_strict"]),
        preferred_families=tuple(data["preferred_families"]),
        blocked_effects=tuple(data["blocked_effects"]),
    )


def load_all_profiles() -> dict[str, ProfilePolicy]:
    return {name: load_profile(name) for name in CONCRETE_PROFILES}


def select_profile(
    analysis: AnalysisResult,
    media: MediaInfo,
    requested: str = "auto",
) -> tuple[ProfilePolicy, list[str]]:
    """Select a concrete profile policy deterministically from measurable signals ONLY.

    Never claims semantic niche, spoken language, or narrative understanding.
    Uncertain cases default conservatively to `neutral`.
    """
    if requested not in VALID_PROFILES:
        raise ValueError(f"Unknown requested profile '{requested}'")
    if requested != "auto":
        return load_profile(requested), [f"explicit_user_profile:{requested}"]

    summary = analysis.signal_summary or {}
    reasons: list[str] = []

    mean_motion = float(summary.get("mean_motion", 0.0))
    static_ratio = float(summary.get("static_ratio", 0.0))
    mean_edge = float(summary.get("mean_edge_density", 0.0))
    border_edge = float(summary.get("mean_border_edge_density", 0.0))
    subj_cov = float(summary.get("subject_coverage", 0.0))
    subj_conf = float(summary.get("mean_subject_confidence", 0.0))
    multi_subj = float(summary.get("multi_subject_ratio", 0.0))
    has_audio = "audio" in analysis.capabilities
    is_landscape = media.display_width > media.display_height

    # Rule 1: Landscape + static + high crisp UI/text edge density -> tutorial (screen-safe protection)
    if is_landscape and static_ratio >= 0.65 and (mean_edge >= 0.14 or border_edge >= 0.14):
        reasons.append(
            f"auto_signal_screen_like:landscape+static={static_ratio:.2f}+edge={mean_edge:.2f}"
        )
        return load_profile("tutorial"), reasons

    # Rule 2: High motion + dense peripheral HUD edges -> gameplay (HUD-safe protection)
    if mean_motion >= 0.32 and border_edge >= 0.12:
        reasons.append(
            f"auto_signal_high_motion_hud:motion={mean_motion:.2f}+border_edge={border_edge:.2f}"
        )
        return load_profile("gameplay"), reasons

    # Rule 3: Confident single face detected consistently + audio present -> talking-head
    if has_audio and subj_cov >= 0.35 and subj_conf >= 0.50 and multi_subj < 0.30:
        reasons.append(
            f"auto_signal_single_subject_audio:cov={subj_cov:.2f}+conf={subj_conf:.2f}"
        )
        return load_profile("talking-head"), reasons

    # Rule 4: Uncertain / 3D / cartoon / multi-subject / general footage -> neutral (conservative safe default)
    reasons.append("auto_signal_conservative_default_neutral")
    return load_profile("neutral"), reasons
