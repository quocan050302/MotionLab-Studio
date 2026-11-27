"""Phase 2 (ML-01.R4): Reusable Niche Creative Direction Profiles & Visual/Sonic Style Tokens.

Separates semantic subject matter (finance, product, storytelling, education, tutorial, lifestyle, gameplay, cinematic)
from visual safety constraints (face/hands protection, UI/HUD no-cover). Zero hardcoded clip names.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from motionlab.schemas import AnalysisResult
from motionlab.semantic.schema import SemanticUnit


@dataclass(frozen=True)
class NicheStyleProfile:
    niche: str
    description: str
    palette_override: dict[str, str]  # blends harmonically with source palette
    preferred_vfx: tuple[str, ...]
    disallowed_vfx: tuple[str, ...]
    sfx_palette: tuple[str, ...]
    sfx_gain_offset_db: float
    protect_ui_strict: bool
    allow_grounded_labels: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


NICHE_PROFILES: dict[str, NicheStyleProfile] = {
    "finance": NicheStyleProfile(
        niche="finance",
        description="Premium editorial x modern commercial personal finance / budgeting: warm ivory/sage/espresso/antique-gold hierarchy, grounded breakdowns, tactile paper/cash foley, physical prop depth, zero neon/casino spam.",
        palette_override={"primary_hex": "#C8A26A", "secondary_hex": "#3B312A", "tertiary_hex": "#F7F1E6", "accent_sage": "#6E8B74"},
        preferred_vfx=("VFX01", "VFX02", "VFX05", "VFX04", "VFX03", "VFX07", "VFX08", "VFX09", "VFX11", "VFX12", "VFX06", "VFX10"),
        disallowed_vfx=(),
        sfx_palette=("cash_flick", "paper_rustle", "envelope_slide", "soft_whoosh", "reverse_whoosh", "pop_reveal", "tick_accent", "soft_impact", "shimmer_payoff", "bass_settle"),
        sfx_gain_offset_db=0.0,
        protect_ui_strict=False,
        allow_grounded_labels=True,
    ),
    "product": NicheStyleProfile(
        niche="product",
        description="Clean commercial product / SaaS demonstration: crisp prop isolation, macro details, tracked callout flows, restrained clicks & tactile pops, verified feature labels only.",
        palette_override={"primary_hex": "#D4AF7A", "secondary_hex": "#28303B", "tertiary_hex": "#F5F7FA", "accent_sage": "#5D9B8B"},
        preferred_vfx=("VFX02", "VFX03", "VFX04", "VFX08", "VFX06", "VFX09", "VFX11", "VFX12"),
        disallowed_vfx=(),
        sfx_palette=("pop_reveal", "tick_accent", "soft_whoosh", "reverse_whoosh", "soft_impact", "shimmer_payoff"),
        sfx_gain_offset_db=-1.0,
        protect_ui_strict=False,
        allow_grounded_labels=True,
    ),
    "storytelling": NicheStyleProfile(
        niche="storytelling",
        description="Narrative / history editorial: atmospheric depth stages, occlusion wipes, shape morphs, layered editorial collages, gentle rises & air sweeps.",
        palette_override={"primary_hex": "#C99A6B", "secondary_hex": "#2B2523", "tertiary_hex": "#F2E9DC", "accent_sage": "#7A8C7E"},
        preferred_vfx=("VFX01", "VFX07", "VFX06", "VFX09", "VFX08", "VFX12"),
        disallowed_vfx=(),
        sfx_palette=("soft_whoosh", "reverse_whoosh", "riser_build", "soft_impact", "bass_settle"),
        sfx_gain_offset_db=-1.5,
        protect_ui_strict=False,
        allow_grounded_labels=True,
    ),
    "education": NicheStyleProfile(
        niche="education",
        description="Explainer / science / education: clean benefit/concept diagrams, tracked flow curves, macro focus, subtle ticks & soft reveals.",
        palette_override={"primary_hex": "#7FA8C9", "secondary_hex": "#243040", "tertiary_hex": "#F4F7FB", "accent_sage": "#6CA693"},
        preferred_vfx=("VFX05", "VFX04", "VFX01", "VFX03", "VFX08", "VFX12"),
        disallowed_vfx=(),
        sfx_palette=("tick_accent", "pop_reveal", "soft_whoosh", "shimmer_payoff"),
        sfx_gain_offset_db=-1.5,
        protect_ui_strict=False,
        allow_grounded_labels=True,
    ),
    "tutorial": NicheStyleProfile(
        niche="tutorial",
        description="Screen recording / software tutorial: strictly protects UI chrome and text legibility; blocks occlusive VFX.",
        palette_override={"primary_hex": "#6BA3D6", "secondary_hex": "#1E2530", "tertiary_hex": "#F2F5F9", "accent_sage": "#62A88A"},
        preferred_vfx=(),
        disallowed_vfx=("VFX01", "VFX02", "VFX03", "VFX04", "VFX05", "VFX06", "VFX07", "VFX08", "VFX09", "VFX10", "VFX11", "VFX12"),
        sfx_palette=("tick_accent",),
        sfx_gain_offset_db=-4.0,
        protect_ui_strict=True,
        allow_grounded_labels=False,
    ),
    "lifestyle": NicheStyleProfile(
        niche="lifestyle",
        description="Natural talking-head / lifestyle: warm depth stages, light gesture emphasis, subtle tactile foley, generous breathing space.",
        palette_override={"primary_hex": "#D0AE82", "secondary_hex": "#453A32", "tertiary_hex": "#FAF4EC", "accent_sage": "#7D9682"},
        preferred_vfx=("VFX01", "VFX03", "VFX07", "VFX09", "VFX12"),
        disallowed_vfx=(),
        sfx_palette=("soft_whoosh", "pop_reveal", "paper_rustle", "shimmer_payoff"),
        sfx_gain_offset_db=-2.0,
        protect_ui_strict=False,
        allow_grounded_labels=True,
    ),
    "gameplay": NicheStyleProfile(
        niche="gameplay",
        description="High-motion gameplay: strictly protects HUD and active motion; blocks occlusive VFX.",
        palette_override={"primary_hex": "#8EB4D9", "secondary_hex": "#1B202A", "tertiary_hex": "#F0F4FA", "accent_sage": "#6CA693"},
        preferred_vfx=(),
        disallowed_vfx=("VFX01", "VFX02", "VFX03", "VFX04", "VFX05", "VFX06", "VFX07", "VFX08", "VFX09", "VFX10", "VFX11", "VFX12"),
        sfx_palette=(),
        sfx_gain_offset_db=-6.0,
        protect_ui_strict=True,
        allow_grounded_labels=False,
    ),
    "cinematic": NicheStyleProfile(
        niche="cinematic",
        description="General / neutral cinematic editorial baseline when niche is ambiguous or silent.",
        palette_override={"primary_hex": "#C8A677", "secondary_hex": "#342D28", "tertiary_hex": "#F5EFE6", "accent_sage": "#748C7C"},
        preferred_vfx=("VFX01", "VFX03", "VFX06", "VFX07", "VFX09", "VFX12"),
        disallowed_vfx=(),
        sfx_palette=("soft_whoosh", "reverse_whoosh", "soft_impact", "bass_settle"),
        sfx_gain_offset_db=-2.0,
        protect_ui_strict=False,
        allow_grounded_labels=False,
    ),
    "tech": NicheStyleProfile(
        niche="tech",
        description="Tech / SaaS product & workflow demo: clean screen clarity, evidence-led feature highlights, crisp restrained ticks.",
        palette_override={"primary_hex": "#7FA8C9", "secondary_hex": "#1E2530", "tertiary_hex": "#F4F7FB", "accent_sage": "#5D9B8B"},
        preferred_vfx=("VFX02", "VFX04", "VFX05", "VFX08", "VFX12"),
        disallowed_vfx=(),
        sfx_palette=("tick_accent", "pop_reveal", "soft_whoosh"),
        sfx_gain_offset_db=-2.0,
        protect_ui_strict=False,
        allow_grounded_labels=True,
    ),
    "comedy": NicheStyleProfile(
        niche="comedy",
        description="Comedy / punchline storytelling: protect setup/reaction beats, minimal intrusive graphics, crisp timing.",
        palette_override={"primary_hex": "#D4A359", "secondary_hex": "#2B2523", "tertiary_hex": "#FAF5EC", "accent_sage": "#7D9682"},
        preferred_vfx=("VFX01", "VFX03", "VFX09"),
        disallowed_vfx=(),
        sfx_palette=("pop_reveal", "soft_whoosh", "soft_impact"),
        sfx_gain_offset_db=-2.5,
        protect_ui_strict=False,
        allow_grounded_labels=True,
    ),
    "wellness": NicheStyleProfile(
        niche="wellness",
        description="Wellness / calm reflective content: generous breathing pauses, soft typography, zero frantic cuts or loud impacts.",
        palette_override={"primary_hex": "#A9BFA8", "secondary_hex": "#2F3832", "tertiary_hex": "#F6F5F0", "accent_sage": "#84977A"},
        preferred_vfx=("VFX01", "VFX07", "VFX09"),
        disallowed_vfx=(),
        sfx_palette=("soft_whoosh", "bass_settle"),
        sfx_gain_offset_db=-4.0,
        protect_ui_strict=False,
        allow_grounded_labels=True,
    ),
    "silent_art": NicheStyleProfile(
        niche="silent_art",
        description="Silent aesthetic / ASMR / nonverbal showcase: protect natural ambient sound and unhurried visual cadence.",
        palette_override={"primary_hex": "#C8A677", "secondary_hex": "#292822", "tertiary_hex": "#F5F0E6", "accent_sage": "#84977A"},
        preferred_vfx=(),
        disallowed_vfx=("VFX01", "VFX02", "VFX03", "VFX04", "VFX05", "VFX06", "VFX07", "VFX08", "VFX09", "VFX10", "VFX11", "VFX12"),
        sfx_palette=(),
        sfx_gain_offset_db=-12.0,
        protect_ui_strict=True,
        allow_grounded_labels=False,
    ),
    "mixed": NicheStyleProfile(
        niche="mixed",
        description="Mixed multi-part narrative combining story/demo/explanation beats with global continuity.",
        palette_override={"primary_hex": "#C8A677", "secondary_hex": "#2C2A26", "tertiary_hex": "#F5F2EB", "accent_sage": "#748C7C"},
        preferred_vfx=("VFX01", "VFX03", "VFX05", "VFX08", "VFX09"),
        disallowed_vfx=(),
        sfx_palette=("soft_whoosh", "pop_reveal", "tick_accent", "soft_impact"),
        sfx_gain_offset_db=-2.0,
        protect_ui_strict=False,
        allow_grounded_labels=True,
    ),
    "unknown": NicheStyleProfile(
        niche="unknown",
        description="Ambiguous / low-confidence content: conservative grounded presentation without forcing niche cliches.",
        palette_override={"primary_hex": "#C8A677", "secondary_hex": "#342D28", "tertiary_hex": "#F5EFE6", "accent_sage": "#748C7C"},
        preferred_vfx=("VFX01", "VFX03", "VFX09"),
        disallowed_vfx=(),
        sfx_palette=("soft_whoosh", "tick_accent"),
        sfx_gain_offset_db=-3.0,
        protect_ui_strict=False,
        allow_grounded_labels=False,
    ),
}

_NICHE_ALIASES = {
    "personal-finance": "finance",
    "personal_finance": "finance",
    "budgeting": "finance",
    "money": "finance",
    "finance": "finance",
    "saas": "product",
    "product": "product",
    "ecommerce": "product",
    "history": "storytelling",
    "storytelling": "storytelling",
    "science": "education",
    "education": "education",
    "explainer": "education",
    "tutorial": "tutorial",
    "screen": "tutorial",
    "lifestyle": "lifestyle",
    "talking-head": "lifestyle",
    "vlog": "lifestyle",
    "gameplay": "gameplay",
    "gaming": "gameplay",
    "cinematic": "cinematic",
    "general": "cinematic",
    "neutral": "cinematic",
}


def select_semantic_profile(
    llm_niche_raw: str,
    llm_confidence: float,
    units: list[SemanticUnit],
    analysis: AnalysisResult,
    requested_style: str = "auto",
) -> tuple[NicheStyleProfile, float, str]:
    """Select reusable niche creative profile combining local LLM classification + visual safety invariants."""
    flags = set(analysis.quality_flags or [])
    stats = analysis.summary_stats or {}
    total_f = max(1, analysis.duration_frames)
    face_cov = float(stats.get("face_coverage_frames", 0)) / total_f
    ui_edge = float(stats.get("ui_edge_density", 0.0))
    mean_mot = float(stats.get("mean_motion", 0.0))

    # Visual safety overrides semantic flair ONLY on genuine screen recordings (high UI edge + low human face)
    # or high-motion HUD gameplay without a presenter!
    if "busy_ui_or_text_detected" in flags and ui_edge >= 0.075 and mean_mot < 0.25 and face_cov < 0.25:
        return NICHE_PROFILES["tutorial"], 0.95, "visual_safety_override:screen_tutorial_detected"
    if "high_existing_motion" in flags and face_cov < 0.15:
        return NICHE_PROFILES["gameplay"], 0.90, "visual_safety_override:high_existing_motion"

    req = (requested_style or "auto").strip().lower()
    if req in _NICHE_ALIASES and req not in ("auto", "editorial", "minimal", "dynamic"):
        chosen = _NICHE_ALIASES[req]
        return NICHE_PROFILES[chosen], 1.0, f"user_explicit_style:{chosen}"

    norm_llm = _NICHE_ALIASES.get((llm_niche_raw or "").strip().lower(), "")
    conf = max(0.0, min(1.0, float(llm_confidence) / 100.0 if float(llm_confidence) > 1.0 else float(llm_confidence)))
    # Guard: if local LLM classifies a human presenter talking about money/budgeting/bills as "tutorial" or generic "education",
    # disambiguate cleanly to "finance" when strong personal-finance domain signals appear in the real transcript!
    if norm_llm in ("tutorial", "education") and face_cov >= 0.25 and "busy_ui_or_text_detected" not in flags:
        joined = " ".join(u.verbatim_quote.lower() for u in units)
        fin_hits = sum(
            1 for k in ("$", "€", "£", "bank", "paycheck", "payday", "budget", "bills", "dollar", "money", "spend", "groceries", "income", "saving", "debt")
            if k in joined
        )
        if fin_hits >= 2:
            return NICHE_PROFILES["finance"], round(conf, 3), f"semantic_disambiguation:finance_presenter_domain(from={norm_llm},hits={fin_hits})"
        if norm_llm == "tutorial":
            return NICHE_PROFILES["education"], round(conf, 3), "semantic_disambiguation:presenter_education_not_screen_ui"

    if norm_llm and conf >= 0.55:
        return NICHE_PROFILES[norm_llm], round(conf, 3), f"local_llm_semantic_niche:{norm_llm}(conf={conf:.2f})"

    # Conservative fallback to neutral cinematic if ambiguous or no speech
    return NICHE_PROFILES["cinematic"], round(max(0.4, conf), 3), f"conservative_fallback:low_or_ambiguous_niche({llm_niche_raw!r})"
