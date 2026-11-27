"""Phase R1-A4 (ML-01.R5.R1): Two-Pass Semantic Sound Director & Animation-Marker Alignment.

Replaces keyword/recipe-only sound triggers with narrative, material, and marker-grounded sound design:
  1. Explicit SoundRole taxonomy:
     DIEGETIC_FOLEY | ILLUSTRATIVE_FOLEY | GRAPHIC_MOTION | COGNITIVE_EMPHASIS |
     TRANSITION_BRIDGE | PRODUCT_TEXTURE | AMBIENCE | INTENTIONAL_SILENCE
  2. Every planned cue carries supporting word_ids, quote, story_intent, emotional_role,
     visual_element_id, animation marker_id/marker_type, rationale, AND explicit rejected_candidates.
  3. Explicit QuietSpan records document intentional silence (breathing pauses, CTA clarity,
     negation intelligibility, low-confidence abstention).
  4. Fail-closed semantic grounding validator rejects diegetic claims without visual contact confidence,
     happy cash sounds on negated/non-monetary clauses, engine sounds on financial "gas", and marker drift.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

from motionlab.audio.assets import inspect_foley_library
from motionlab.audio.cues import frame_to_sample_exact
from motionlab.audio.schema import AUDIO_SCHEMA_VERSION, AudioTimeline, SoundEvent
from motionlab.audio.synthesis import SFX_FAMILIES, synthesize_sfx_waveform
from motionlab.composition.generative_schema import SceneSpec, SoundIntent, StoryboardSpec
from motionlab.schemas import canonical_hash, sanitize_native
from motionlab.semantic.story_graph import StoryBeatNode, StoryGraph

SONIC_SCHEMA_VERSION = "2.0.0"

VALID_SOUND_ROLES = (
    "DIEGETIC_FOLEY",
    "ILLUSTRATIVE_FOLEY",
    "GRAPHIC_MOTION",
    "COGNITIVE_EMPHASIS",
    "TRANSITION_BRIDGE",
    "PRODUCT_TEXTURE",
    "AMBIENCE",
    "INTENTIONAL_SILENCE",
)

# Material & semantic metadata for every registered sound family
SOUND_FAMILY_CATALOG: dict[str, dict[str, Any]] = {
    "cash_flick": {
        "material": "crisp_banknote_paper",
        "texture": "dry_fibrous_snap",
        "allowed_roles": ("DIEGETIC_FOLEY", "ILLUSTRATIVE_FOLEY"),
        "pre_roll_ms": 0.0,
        "finance_specific": True,
    },
    "paper_rustle": {
        "material": "paper_sheet",
        "texture": "soft_organic_shuffle",
        "allowed_roles": ("DIEGETIC_FOLEY", "ILLUSTRATIVE_FOLEY", "PRODUCT_TEXTURE"),
        "pre_roll_ms": 15.0,
        "finance_specific": False,
    },
    "envelope_slide": {
        "material": "kraft_cardstock_envelope",
        "texture": "smooth_pocket_glide",
        "allowed_roles": ("DIEGETIC_FOLEY", "ILLUSTRATIVE_FOLEY", "PRODUCT_TEXTURE"),
        "pre_roll_ms": 20.0,
        "finance_specific": False,
    },
    "page_turn": {
        "material": "workbook_paper_page",
        "texture": "resonant_page_flip",
        "allowed_roles": ("DIEGETIC_FOLEY", "ILLUSTRATIVE_FOLEY", "PRODUCT_TEXTURE", "TRANSITION_BRIDGE"),
        "pre_roll_ms": 30.0,
        "finance_specific": False,
    },
    "soft_whoosh": {
        "material": "editorial_air",
        "texture": "velvet_low_mid_sweep",
        "allowed_roles": ("GRAPHIC_MOTION", "TRANSITION_BRIDGE"),
        "pre_roll_ms": 110.0,
        "finance_specific": False,
    },
    "reverse_whoosh": {
        "material": "editorial_air_reverse",
        "texture": "restrained_pre_reveal_inhale",
        "allowed_roles": ("GRAPHIC_MOTION", "TRANSITION_BRIDGE", "COGNITIVE_EMPHASIS"),
        "pre_roll_ms": 140.0,
        "finance_specific": False,
    },
    "pop_reveal": {
        "material": "muted_wood_organic",
        "texture": "warm_soft_pop",
        "allowed_roles": ("GRAPHIC_MOTION", "COGNITIVE_EMPHASIS"),
        "pre_roll_ms": 0.0,
        "finance_specific": False,
    },
    "tick_accent": {
        "material": "muted_wood_tactile",
        "texture": "crisp_numeric_settle_tick",
        "allowed_roles": ("GRAPHIC_MOTION", "COGNITIVE_EMPHASIS", "PRODUCT_TEXTURE"),
        "pre_roll_ms": 0.0,
        "finance_specific": False,
    },
    "soft_impact": {
        "material": "warm_felt_low_end",
        "texture": "restrained_editorial_thump",
        "allowed_roles": ("COGNITIVE_EMPHASIS", "TRANSITION_BRIDGE"),
        "pre_roll_ms": 0.0,
        "finance_specific": False,
    },
    "shimmer_payoff": {
        "material": "warm_harmonic_tail",
        "texture": "delicate_fifth_resolve",
        "allowed_roles": ("COGNITIVE_EMPHASIS", "TRANSITION_BRIDGE"),
        "pre_roll_ms": 0.0,
        "finance_specific": False,
    },
    "bass_settle": {
        "material": "warm_sub_resolve",
        "texture": "calm_editorial_closure",
        "allowed_roles": ("COGNITIVE_EMPHASIS", "TRANSITION_BRIDGE"),
        "pre_roll_ms": 0.0,
        "finance_specific": False,
    },
    "riser_build": {
        "material": "breath_tension_air",
        "texture": "gentle_swell",
        "allowed_roles": ("TRANSITION_BRIDGE", "COGNITIVE_EMPHASIS"),
        "pre_roll_ms": 160.0,
        "finance_specific": False,
    },
}

# Disallowed novelty/meme sounds explicitly tracked in rejected candidate logs
DISALLOWED_CLICHE_FAMILIES = ("cash_register_chaching", "slot_machine_coins", "error_buzzer", "engine_rev", "notification_ding")


@dataclass
class AnimationMarker:
    marker_id: str
    scene_id: str
    marker_type: str  # anticipate | reveal_start | release | allocation_arrival | counter_settle | page_contact | transition_peak | resolve | exit
    local_frame: int
    global_frame: int
    source_element_id: str
    engine: str
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return sanitize_native(asdict(self))


@dataclass
class SonicOpportunity:
    opportunity_id: str
    beat_id: str
    scene_id: str
    speech_act: str
    quote: str
    word_ids: list[int]
    start_frame: int
    end_frame_exclusive: int
    negated: bool
    has_ledger_step: bool
    ledger_step_type: str
    ledger_category: str
    has_non_monetary_count: bool
    observed_contact_frame: int | None
    observed_contact_confidence: float
    source_audio_rms_dbfs: float
    recommended_role: str
    candidate_families: list[str]
    rejected_candidates: list[dict[str, str]] = field(default_factory=list)
    silence_candidate: bool = False
    silence_rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return sanitize_native(asdict(self))


@dataclass
class QuietSpan:
    start_frame: int
    end_frame_exclusive: int
    start_s: float
    end_s: float
    reason: str
    scene_id: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return sanitize_native(asdict(self))


@dataclass
class PlannedSonicCue:
    cue_id: str
    beat_id: str
    scene_id: str
    semantic_event_id: str
    word_ids: list[int]
    quote: str
    story_intent: str
    emotional_role: str
    visual_element_id: str
    action_ref: str
    action_confidence: float
    marker_id: str
    marker_type: str
    sound_role: str
    material: str
    texture: str
    family: str
    variant: int
    candidate_families: list[str]
    rejected_candidates: list[dict[str, str]]
    reason: str
    target_frame: int
    pre_roll_ms: float
    gain_db: float
    pan: float
    priority: int
    asset_id: str = ""
    asset_provenance: str = "procedural_dsp_synthesis"
    asset_license: str = ""
    asset_sha256: str = ""
    fallback_status: str = "PARTIAL_FOLEY_DSP_FALLBACK"

    def to_dict(self) -> dict[str, Any]:
        return sanitize_native(asdict(self))


@dataclass
class SonicScore:
    schema_version: str
    mode: str
    story_graph_hash: str
    storyboard_hash: str
    foley_library_status: str
    recorded_cues_used: int
    procedural_cues_used: int
    cues: list[PlannedSonicCue] = field(default_factory=list)
    quiet_spans: list[QuietSpan] = field(default_factory=list)
    rejected_decisions: list[dict[str, Any]] = field(default_factory=list)
    markers: list[AnimationMarker] = field(default_factory=list)
    validation_errors: list[str] = field(default_factory=list)
    score_hash: str = ""

    def compute_hash(self) -> str:
        payload = self.to_dict()
        payload.pop("score_hash", None)
        return canonical_hash(payload)

    def to_dict(self) -> dict[str, Any]:
        return sanitize_native(asdict(self))


def extract_rendered_animation_markers(
    storyboard: StoryboardSpec,
    story_graph: StoryGraph,
) -> list[AnimationMarker]:
    """Emit exact deterministic AnimationMarkers from compiled Remotion & HyperFrames keyframes.

    Sound alignment MUST consume these markers rather than hardcoded beat midpoints so any
    timing revision in animation automatically propagates to sample-exact SFX placement.
    """
    markers: list[AnimationMarker] = []
    fps_f = storyboard.fps_num / max(1, storyboard.fps_den)
    step_map = {st.step_id: st for st in story_graph.ledger.steps}

    for sc in storyboard.scenes:
        dur = max(2, sc.end_frame_exclusive - sc.start_frame)
        # 1. Entrance transition peak marker (if scene has transition_in)
        if sc.transition_in.kind != "none" and sc.transition_in.duration_frames > 0:
            tr_mid = max(1, min(dur - 2, sc.transition_in.duration_frames // 2))
            markers.append(AnimationMarker(
                marker_id=f"{sc.scene_id}:transition_peak",
                scene_id=sc.scene_id,
                marker_type="transition_peak",
                local_frame=tr_mid,
                global_frame=sc.start_frame + tr_mid,
                source_element_id=sc.transition_in.shared_element_id or f"transition_{sc.transition_in.kind}",
                engine="remotion",
                rationale=f"Mid-peak of {sc.transition_in.kind} entrance transition",
            ))

        # 2. Primary reveal / release / contact / settle markers derived from actual layer keyframes
        st = step_map.get(sc.persistent_ledger_step_id or "") if sc.persistent_ledger_step_id else None
        primary_el = sc.layers[0].primitive_id if sc.layers else sc.stage_mode
        primary_eng = sc.layers[0].engine if sc.layers else "remotion"

        # Anchor primary reveal to spoken ledger step onset when inside scene, else animation entry settle
        if st is not None and sc.start_frame <= st.start_frame < sc.end_frame_exclusive - 4:
            rel_reveal = max(2, st.start_frame - sc.start_frame + 1)
        else:
            rel_reveal = min(dur - 4, max(3, int(round(fps_f * 0.16))))

        if sc.role == "hook_contrast":
            markers.append(AnimationMarker(
                marker_id=f"{sc.scene_id}:reveal_start",
                scene_id=sc.scene_id,
                marker_type="reveal_start",
                local_frame=rel_reveal,
                global_frame=sc.start_frame + rel_reveal,
                source_element_id="ledger_balance_hero",
                engine=primary_eng,
                rationale="Hero opening balance reveal + illustrative banknote arrival",
            ))
            settle_rel = min(dur - 6, rel_reveal + int(round(min(0.65, dur / fps_f * 0.34) * fps_f)))
            markers.append(AnimationMarker(
                marker_id=f"{sc.scene_id}:counter_settle",
                scene_id=sc.scene_id,
                marker_type="counter_settle",
                local_frame=settle_rel,
                global_frame=sc.start_frame + settle_rel,
                source_element_id="ledger_balance_hero",
                engine=primary_eng,
                rationale="Counter lands cleanly on opening balance value",
            ))
        elif sc.role == "negation_tension":
            # StrikethroughReplace starts strike at delay_frames (default 6) + half strike_frames (6) = 12
            strike_delay = 6
            strike_dur = 12
            for ly in sc.layers:
                if ly.primitive_id in ("rem_ru_strikethrough", "hf_strikethrough_replace"):
                    strike_delay = int(ly.params.get("delay_frames", 6))
                    strike_dur = int(ly.params.get("strike_frames", 12))
            strike_rel = min(dur - 6, max(4, strike_delay + strike_dur // 2))
            markers.append(AnimationMarker(
                marker_id=f"{sc.scene_id}:impact",
                scene_id=sc.scene_id,
                marker_type="impact",
                local_frame=strike_rel,
                global_frame=sc.start_frame + strike_rel,
                source_element_id="negation_strike_symbol",
                engine=primary_eng,
                rationale="Exact strikethrough sweep landing on contradiction contrast",
            ))
        elif sc.role == "allocation_step":
            markers.append(AnimationMarker(
                marker_id=f"{sc.scene_id}:release",
                scene_id=sc.scene_id,
                marker_type="release",
                local_frame=rel_reveal,
                global_frame=sc.start_frame + rel_reveal,
                source_element_id=f"allocation_{st.category.lower() if st else 'step'}",
                engine=primary_eng,
                rationale="Allocation token release & proportional bar fill start",
            ))
            settle_rel = min(dur - 5, rel_reveal + max(10, int(round(fps_f * 0.58))))
            markers.append(AnimationMarker(
                marker_id=f"{sc.scene_id}:counter_settle",
                scene_id=sc.scene_id,
                marker_type="counter_settle",
                local_frame=settle_rel,
                global_frame=sc.start_frame + settle_rel,
                source_element_id="persistent_ledger_balance",
                engine=primary_eng,
                rationale="Running remaining balance settles after deduction",
            ))
        elif sc.role == "payoff_reveal":
            impact_rel = min(dur - 8, max(5, sc.transition_in.duration_frames if sc.transition_in.kind != "none" else 6))
            markers.append(AnimationMarker(
                marker_id=f"{sc.scene_id}:impact",
                scene_id=sc.scene_id,
                marker_type="impact",
                local_frame=impact_rel,
                global_frame=sc.start_frame + impact_rel,
                source_element_id="payoff_remaining_hero",
                engine="remotion",
                rationale="Main payoff visual hero transform arrival",
            ))
            res_rel = min(dur - 10, impact_rel + int(round(fps_f * 0.85)))
            markers.append(AnimationMarker(
                marker_id=f"{sc.scene_id}:resolve",
                scene_id=sc.scene_id,
                marker_type="resolve",
                local_frame=res_rel,
                global_frame=sc.start_frame + res_rel,
                source_element_id="payoff_waterfall",
                engine="remotion",
                rationale="Payoff waterfall & remaining hero settle",
            ))
        elif sc.role == "product_intro":
            markers.append(AnimationMarker(
                marker_id=f"{sc.scene_id}:reveal_start",
                scene_id=sc.scene_id,
                marker_type="reveal_start",
                local_frame=min(dur - 6, max(4, int(round(fps_f * 0.18)))),
                global_frame=sc.start_frame + min(dur - 6, max(4, int(round(fps_f * 0.18)))),
                source_element_id="product_title_motif",
                engine=primary_eng,
                rationale="Editorial sweep reveals product title",
            ))
        elif sc.role == "product_benefit":
            flip_delay = max(10, int(round(dur * 0.28)))
            for ly in sc.layers:
                if ly.primitive_id == "rem_rc_flip_page":
                    flip_delay = int(ly.params.get("flip_delay", flip_delay))
            flip_contact = min(dur - 10, max(6, flip_delay + 4))
            markers.append(AnimationMarker(
                marker_id=f"{sc.scene_id}:page_contact",
                scene_id=sc.scene_id,
                marker_type="page_contact",
                local_frame=flip_contact,
                global_frame=sc.start_frame + flip_contact,
                source_element_id="conceptual_workbook_page",
                engine="remotion",
                rationale="3D workbook page turn crest / leaf turn contact",
            ))
            org_rel = min(dur - 8, flip_contact + int(round(fps_f * 1.15)))
            markers.append(AnimationMarker(
                marker_id=f"{sc.scene_id}:resolve",
                scene_id=sc.scene_id,
                marker_type="resolve",
                local_frame=org_rel,
                global_frame=sc.start_frame + org_rel,
                source_element_id="product_specs_list",
                engine="hyperframes",
                rationale="Organized components settle cleanly",
            ))
        else:
            # closing_takeaway / cta / generic statement
            markers.append(AnimationMarker(
                marker_id=f"{sc.scene_id}:resolve",
                scene_id=sc.scene_id,
                marker_type="resolve",
                local_frame=min(dur - 12, max(5, int(round(fps_f * 0.25)))),
                global_frame=sc.start_frame + min(dur - 12, max(5, int(round(fps_f * 0.25)))),
                source_element_id=primary_el,
                engine=primary_eng,
                rationale="Editorial takeaway phrase settles cleanly while preserving calm ending",
            ))

    return markers


def extract_sonic_opportunities(
    story_graph: StoryGraph,
    transcript: Any,
    observed_actions: dict[str, Any] | None = None,
    source_audio_events: list[dict[str, Any]] | None = None,
    storyboard: StoryboardSpec | None = None,
) -> list[SonicOpportunity]:
    """Pass 1: Derive semantically grounded sonic opportunities from StoryGraph, ROI actions, and storyboard."""
    step_map = {st.step_id: st for st in story_graph.ledger.steps}
    roi_frames = (observed_actions or {}).get("keyframes") or (observed_actions or {}).get("frames") or []
    non_money_spans = [
        (int(nm.get("start_frame", -1)), int(nm.get("end_frame_exclusive", -1)))
        for nm in story_graph.ledger.non_monetary_counts
    ]

    # Map storyboard scenes to beats by frame overlap
    scenes_list = storyboard.scenes if storyboard is not None else []

    def _find_scene(sf: int, ef: int) -> SceneSpec | None:
        best: SceneSpec | None = None
        best_ov = 0
        for sc in scenes_list:
            ov = min(ef, sc.end_frame_exclusive) - max(sf, sc.start_frame)
            if ov > best_ov:
                best_ov = ov
                best = sc
        return best

    def _best_action_contact(sf: int, ef: int) -> tuple[int | None, float]:
        best_f: int | None = None
        best_c = 0.0
        for r in roi_frames:
            f = int(r.get("frame", -1))
            if sf <= f < ef:
                tbox = r.get("task") or r.get("task_box")
                energy = float(r.get("action_energy", r.get("motion_energy", 0.0)))
                conf = float(tbox.get("confidence", 0.0)) if isinstance(tbox, dict) else 0.0
                score = min(1.0, conf * 0.7 + min(1.0, energy) * 0.45)
                if score > best_c:
                    best_c = score
                    best_f = f
        return best_f, round(best_c, 3)

    opportunities: list[SonicOpportunity] = []
    for idx, beat in enumerate(story_graph.beats):
        sc = _find_scene(beat.start_frame, beat.end_frame_exclusive)
        scene_id = sc.scene_id if sc else f"beat_span_{idx:02d}"
        st = step_map.get(beat.ledger_step_id or "") if beat.ledger_step_id else None
        contact_f, contact_conf = _best_action_contact(beat.start_frame, beat.end_frame_exclusive)
        has_non_money = any(
            min(beat.end_frame_exclusive, nm1) > max(beat.start_frame, nm0)
            for nm0, nm1 in non_money_spans
        )
        quote_low = beat.verbatim_quote.lower()
        rejected: list[dict[str, str]] = []

        if beat.speech_act == "breathing_bridge":
            opportunities.append(SonicOpportunity(
                opportunity_id=f"opp_{idx:02d}_{beat.beat_id}",
                beat_id=beat.beat_id,
                scene_id=scene_id,
                speech_act=beat.speech_act,
                quote=beat.verbatim_quote,
                word_ids=list(beat.word_ids),
                start_frame=beat.start_frame,
                end_frame_exclusive=beat.end_frame_exclusive,
                negated=beat.negated,
                has_ledger_step=False,
                ledger_step_type="",
                ledger_category="",
                has_non_monetary_count=has_non_money,
                observed_contact_frame=contact_f,
                observed_contact_confidence=contact_conf,
                source_audio_rms_dbfs=-28.0,
                recommended_role="INTENTIONAL_SILENCE",
                candidate_families=[],
                rejected_candidates=[{"family": "soft_whoosh", "reason": "Preserve natural speaker breathing pause without filler sound"}],
                silence_candidate=True,
                silence_rationale="Natural breathing pause lets preceding payoff register cleanly before solution setup.",
            ))
            continue

        if beat.negated or beat.speech_act == "negation_tension":
            rejected.append({"family": "cash_flick", "reason": "Negated logical contradiction ('doesn't mean ... to spend'): happy cash foley contradicts meaning"})
            rejected.append({"family": "error_buzzer", "reason": "Meme error buzzer violates editorial tone"})
            rec_role = "COGNITIVE_EMPHASIS"
            cands = ["reverse_whoosh", "tick_accent"]
        elif beat.speech_act == "hook_contrast" and st is not None:
            rejected.append({"family": "cash_register_chaching", "reason": "Avoid cartoon casino/register cliche on opening bank balance"})
            rec_role = "DIEGETIC_FOLEY" if (contact_f is not None and contact_conf >= 0.62) else "ILLUSTRATIVE_FOLEY"
            cands = ["cash_flick", "paper_rustle", "soft_impact"]
        elif beat.speech_act == "allocation_step" and st is not None:
            cat_l = st.category.lower()
            if "gas" in cat_l or "gas" in quote_low:
                rejected.append({"family": "engine_rev", "reason": "'gas' is a financial budget category, not a vehicle scene"})
            if "grocer" in cat_l or "grocer" in quote_low:
                rejected.append({"family": "checkout_beep", "reason": "No physical grocery scanner visible; avoid literal supermarket cliche"})
            rec_role = "DIEGETIC_FOLEY" if (contact_f is not None and contact_conf >= 0.60) else "ILLUSTRATIVE_FOLEY"
            cands = ["cash_flick", "paper_rustle", "envelope_slide", "tick_accent"]
        elif beat.speech_act == "payoff_reveal":
            rejected.append({"family": "slot_machine_coins", "reason": "Remaining $100 is an example constraint, NOT guaranteed savings or jackpot"})
            rejected.append({"family": "cash_flick", "reason": "Use cognitive low impact + restrained resolve rather than repetitive banknote flick"})
            rec_role = "COGNITIVE_EMPHASIS"
            cands = ["soft_impact", "shimmer_payoff"]
        elif beat.speech_act == "product_intro":
            rejected.append({"family": "cash_flick", "reason": "Product introduction transitions from problem to tool; no cash transfer occurring"})
            rec_role = "TRANSITION_BRIDGE"
            cands = ["soft_whoosh", "page_turn"]
        elif beat.speech_act == "product_benefit" or has_non_money:
            if has_non_money:
                rejected.append({"family": "cash_flick", "reason": "Non-monetary count ('56-page workbook') must NEVER trigger money/cash foley"})
            rec_role = "PRODUCT_TEXTURE"
            cands = ["page_turn", "paper_rustle", "tick_accent"]
        elif beat.speech_act == "cta":
            rejected.append({"family": "notification_ding", "reason": "Avoid synthetic notification ping or fake UI tap on spoken CTA; keep voice clean"})
            opportunities.append(SonicOpportunity(
                opportunity_id=f"opp_{idx:02d}_{beat.beat_id}",
                beat_id=beat.beat_id,
                scene_id=scene_id,
                speech_act=beat.speech_act,
                quote=beat.verbatim_quote,
                word_ids=list(beat.word_ids),
                start_frame=beat.start_frame,
                end_frame_exclusive=beat.end_frame_exclusive,
                negated=beat.negated,
                has_ledger_step=False,
                ledger_step_type="",
                ledger_category="",
                has_non_monetary_count=has_non_money,
                observed_contact_frame=contact_f,
                observed_contact_confidence=contact_conf,
                source_audio_rms_dbfs=-18.0,
                recommended_role="INTENTIONAL_SILENCE",
                candidate_families=[],
                rejected_candidates=rejected,
                silence_candidate=True,
                silence_rationale="Keep final CTA syllables ('Link in my profile') completely clear and unmasked.",
            ))
            continue
        elif beat.speech_act == "closing_takeaway":
            rejected.append({"family": "cash_flick", "reason": "Phrase 'give every dollar a purpose' is conceptual closure, not physical banknote handling"})
            rec_role = "COGNITIVE_EMPHASIS"
            cands = ["bass_settle", "paper_rustle"]
        else:
            # Generic non-finance / product / storytelling / education beat
            if not story_graph.ledger.active:
                rejected.append({"family": "cash_flick", "reason": "Non-finance narrative must never receive currency foley"})
            rec_role = "GRAPHIC_MOTION"
            cands = ["soft_whoosh", "pop_reveal", "tick_accent"]

        opportunities.append(SonicOpportunity(
            opportunity_id=f"opp_{idx:02d}_{beat.beat_id}",
            beat_id=beat.beat_id,
            scene_id=scene_id,
            speech_act=beat.speech_act,
            quote=beat.verbatim_quote,
            word_ids=list(beat.word_ids),
            start_frame=beat.start_frame,
            end_frame_exclusive=beat.end_frame_exclusive,
            negated=beat.negated,
            has_ledger_step=st is not None,
            ledger_step_type=st.step_type if st else "",
            ledger_category=st.category if st else "",
            has_non_monetary_count=has_non_money,
            observed_contact_frame=contact_f,
            observed_contact_confidence=contact_conf,
            source_audio_rms_dbfs=-18.0,
            recommended_role=rec_role,
            candidate_families=cands,
            rejected_candidates=rejected,
        ))

    return opportunities


def resolve_sound_asset(
    family: str,
    variant: int,
    foley_lib: dict[str, Any] | None = None,
    sr: int = 44100,
    seed: int = 42,
) -> dict[str, Any]:
    """Resolve a semantically chosen SFX family/variant against verified recorded foley first, else honest DSP."""
    lib = foley_lib if foley_lib is not None else inspect_foley_library()
    # Exact (family, variant) match first, then any verified recorded variant of the same family
    rec_matches = [a for a in lib.get("assets", []) if a.get("family") == family]
    chosen_rec = None
    if rec_matches:
        chosen_rec = next((a for a in rec_matches if int(a.get("variant", 0)) == int(variant)), None)
        if chosen_rec is None:
            chosen_rec = rec_matches[int(variant) % len(rec_matches)]

    if chosen_rec is not None:
        return {
            "asset_id": str(chosen_rec["asset_id"]),
            "family": family,
            "variant": int(chosen_rec.get("variant", variant)),
            "provenance": "recorded_local_foley",
            "license": str(chosen_rec.get("license", "")),
            "sha256": str(chosen_rec.get("sha256", "")),
            "fallback_status": "RECORDED_LOCAL_FOLEY",
            "resolved_path": str(chosen_rec.get("resolved_path", "")),
        }

    _, spec = synthesize_sfx_waveform(family, variant=variant % 3, sr=sr, seed=seed)
    return {
        "asset_id": spec.asset_id,
        "family": family,
        "variant": variant % 3,
        "provenance": "procedural_dsp_synthesis",
        "license": spec.license,
        "sha256": spec.sha256,
        "fallback_status": "PARTIAL_FOLEY_DSP_FALLBACK",
        "resolved_path": "",
    }


def plan_sonic_score(
    storyboard: StoryboardSpec,
    opportunities: list[SonicOpportunity],
    story_graph: StoryGraph,
    markers: list[AnimationMarker],
    *,
    niche_style: str | None = None,
    model_hints: dict[str, str] | None = None,
    foley_dir: Path | None = None,
    seed: int = 42,
    sr: int = 44100,
) -> SonicScore:
    """Pass 2: Synthesize a validated, marker-linked SonicScore with explicit roles, rationales, and QuietSpans."""
    foley_lib = inspect_foley_library(foley_dir=foley_dir)
    fps_f = storyboard.fps_num / max(1, storyboard.fps_den)
    markers_by_scene: dict[str, list[AnimationMarker]] = {}
    for m in markers:
        markers_by_scene.setdefault(m.scene_id, []).append(m)

    cues: list[PlannedSonicCue] = []
    quiet_spans: list[QuietSpan] = []
    rejected_decisions: list[dict[str, Any]] = []
    last_fam_variant: dict[str, tuple[int, int]] = {}  # family -> (last_frame, last_variant)
    recent_fams: list[str] = []
    niche = (niche_style or storyboard.niche or "cinematic").lower()

    # Track non-scene gaps >= 1.2s as explicit QuietSpans (e.g. breathing bridge)
    prev_end = 0
    for sc in storyboard.scenes:
        if sc.start_frame - prev_end >= int(round(fps_f * 1.0)):
            quiet_spans.append(QuietSpan(
                start_frame=prev_end,
                end_frame_exclusive=sc.start_frame,
                start_s=round(prev_end / fps_f, 3),
                end_s=round(sc.start_frame / fps_f, 3),
                reason="natural_breathing_pause",
                scene_id="live_gap",
                notes="Intentional acoustic breathing space between beats; voice/room tone preserved untouched.",
            ))
        prev_end = max(prev_end, sc.end_frame_exclusive)

    alloc_idx = 0
    for opp in opportunities:
        if opp.rejected_candidates:
            for rc in opp.rejected_candidates:
                rejected_decisions.append({"beat_id": opp.beat_id, "scene_id": opp.scene_id, **rc})

        if opp.silence_candidate or opp.recommended_role == "INTENTIONAL_SILENCE":
            if not any(q.start_frame <= opp.start_frame and q.end_frame_exclusive >= opp.end_frame_exclusive for q in quiet_spans):
                quiet_spans.append(QuietSpan(
                    start_frame=opp.start_frame,
                    end_frame_exclusive=opp.end_frame_exclusive,
                    start_s=round(opp.start_frame / fps_f, 3),
                    end_s=round(opp.end_frame_exclusive / fps_f, 3),
                    reason="avoid_cta_masking" if opp.speech_act == "cta" else "natural_breathing_pause",
                    scene_id=opp.scene_id,
                    notes=opp.silence_rationale,
                ))
            continue

        sc_markers = markers_by_scene.get(opp.scene_id, [])
        if not sc_markers:
            # No rendered visual scene owns this beat (e.g. strict UI protection tutorial/gameplay -> stay quiet!)
            quiet_spans.append(QuietSpan(
                start_frame=opp.start_frame,
                end_frame_exclusive=opp.end_frame_exclusive,
                start_s=round(opp.start_frame / fps_f, 3),
                end_s=round(opp.end_frame_exclusive / fps_f, 3),
                reason="ui_or_clean_footage_protection",
                scene_id=opp.scene_id,
                notes="No graphic motion scheduled in this beat; abstaining from gratuitous SFX.",
            ))
            continue

        # Select primary marker + semantically justified family (incorporating safe LLM hint if semantically valid!)
        hint_fam = (model_hints or {}).get(opp.beat_id, "")
        chosen_fam = ""
        if hint_fam and hint_fam in opp.candidate_families and hint_fam in SOUND_FAMILY_CATALOG:
            chosen_fam = hint_fam
        elif opp.speech_act == "allocation_step":
            # Distinct tactile texture across sequential allocations (banknote flick -> paper rustle -> envelope slide)
            seq = ["cash_flick", "paper_rustle", "envelope_slide"]
            chosen_fam = seq[alloc_idx % len(seq)]
            alloc_idx += 1
        elif opp.candidate_families:
            chosen_fam = opp.candidate_families[0]
        else:
            continue

        # Prevent monotonous 3-in-a-row family repetition
        if len(recent_fams) >= 2 and recent_fams[-1] == chosen_fam and recent_fams[-2] == chosen_fam:
            alt = next((f for f in opp.candidate_families if f != chosen_fam), None)
            if alt:
                rejected_decisions.append({
                    "beat_id": opp.beat_id, "scene_id": opp.scene_id, "family": chosen_fam,
                    "reason": "Avoided 3x consecutive identical sound family repetition",
                })
                chosen_fam = alt

        # Choose exact AnimationMarker matching this beat's primary semantic action
        primary_marker = sc_markers[0]
        for pref_type in ("impact", "page_contact", "release", "reveal_start", "resolve", "transition_peak"):
            hit = next((m for m in sc_markers if m.marker_type == pref_type), None)
            if hit is not None:
                primary_marker = hit
                break

        # Determine sound role strictly from evidence (never claim DIEGETIC_FOLEY without observed contact!)
        sound_role = opp.recommended_role
        action_ref = f"marker:{primary_marker.marker_id}"
        target_f = primary_marker.global_frame
        if sound_role == "DIEGETIC_FOLEY":
            if opp.observed_contact_frame is not None and opp.observed_contact_confidence >= 0.60:
                # Only snap to visual contact when within the scene window; otherwise anchor to graphic marker as ILLUSTRATIVE_FOLEY
                if abs(opp.observed_contact_frame - primary_marker.global_frame) <= 12:
                    action_ref = f"observed_hand_contact:f{opp.observed_contact_frame}"
                else:
                    sound_role = "ILLUSTRATIVE_FOLEY"
            else:
                sound_role = "ILLUSTRATIVE_FOLEY"

        # Ensure chosen_fam matches sound_role compatibility
        fmeta = SOUND_FAMILY_CATALOG.get(chosen_fam, SOUND_FAMILY_CATALOG["soft_whoosh"])
        if sound_role not in fmeta["allowed_roles"]:
            sound_role = fmeta["allowed_roles"][0]

        # Non-repeating variant selection
        prev_f, prev_v = last_fam_variant.get(chosen_fam, (-9999, -1))
        var = (prev_v + 1) % 3 if (target_f - prev_f) < int(round(fps_f * 8.0)) else (len(cues) % 3)
        last_fam_variant[chosen_fam] = (target_f, var)
        recent_fams.append(chosen_fam)

        asset_info = resolve_sound_asset(chosen_fam, variant=var, foley_lib=foley_lib, sr=sr, seed=seed)
        gain_db = -5.0 if opp.speech_act in ("payoff_reveal", "hook_contrast", "product_benefit") else -6.0
        pan = 0.0 if chosen_fam in ("soft_impact", "bass_settle", "shimmer_payoff") else ((-0.12, 0.12)[len(cues) % 2])

        cues.append(PlannedSonicCue(
            cue_id=f"scue_{len(cues):02d}_{chosen_fam}",
            beat_id=opp.beat_id,
            scene_id=opp.scene_id,
            semantic_event_id=opp.opportunity_id,
            word_ids=list(opp.word_ids[:8]),
            quote=opp.quote[:160],
            story_intent=f"{opp.speech_act}:{opp.ledger_category or niche}",
            emotional_role=_emotional_role_for_act(opp.speech_act, opp.negated),
            visual_element_id=primary_marker.source_element_id,
            action_ref=action_ref,
            action_confidence=opp.observed_contact_confidence if sound_role == "DIEGETIC_FOLEY" else 0.95,
            marker_id=primary_marker.marker_id,
            marker_type=primary_marker.marker_type,
            sound_role=sound_role,
            material=fmeta["material"],
            texture=fmeta["texture"],
            family=chosen_fam,
            variant=asset_info["variant"],
            candidate_families=list(opp.candidate_families),
            rejected_candidates=list(opp.rejected_candidates),
            reason=_build_cue_reason(opp, chosen_fam, sound_role, primary_marker),
            target_frame=target_f,
            pre_roll_ms=float(fmeta["pre_roll_ms"]),
            gain_db=gain_db,
            pan=pan,
            priority=1,
            asset_id=asset_info["asset_id"],
            asset_provenance=asset_info["provenance"],
            asset_license=asset_info["license"],
            asset_sha256=asset_info["sha256"],
            fallback_status=asset_info["fallback_status"],
        ))

        # Add ONE tasteful secondary settle/resolve marker cue ONLY when a distinct secondary marker exists >=12f later
        sec_marker = next(
            (m for m in sc_markers if m.marker_id != primary_marker.marker_id and (m.global_frame - primary_marker.global_frame) >= 12),
            None,
        )
        if sec_marker is not None and opp.speech_act in ("allocation_step", "payoff_reveal"):
            sec_fam = "tick_accent" if opp.speech_act == "allocation_step" else "shimmer_payoff"
            sec_meta = SOUND_FAMILY_CATALOG[sec_fam]
            s_prev_f, s_prev_v = last_fam_variant.get(sec_fam, (-9999, -1))
            s_var = (s_prev_v + 1) % 3 if (sec_marker.global_frame - s_prev_f) < int(round(fps_f * 8.0)) else (len(cues) % 3)
            last_fam_variant[sec_fam] = (sec_marker.global_frame, s_var)
            s_asset = resolve_sound_asset(sec_fam, variant=s_var, foley_lib=foley_lib, sr=sr, seed=seed)
            cues.append(PlannedSonicCue(
                cue_id=f"scue_{len(cues):02d}_{sec_fam}",
                beat_id=opp.beat_id,
                scene_id=opp.scene_id,
                semantic_event_id=f"{opp.opportunity_id}_settle",
                word_ids=list(opp.word_ids[-3:]),
                quote=opp.quote[:160],
                story_intent=f"{opp.speech_act}_settle",
                emotional_role="clarity_settle" if opp.speech_act == "allocation_step" else "restrained_payoff_tail",
                visual_element_id=sec_marker.source_element_id,
                action_ref=f"marker:{sec_marker.marker_id}",
                action_confidence=0.95,
                marker_id=sec_marker.marker_id,
                marker_type=sec_marker.marker_type,
                sound_role="COGNITIVE_EMPHASIS" if opp.speech_act == "payoff_reveal" else "GRAPHIC_MOTION",
                material=sec_meta["material"],
                texture=sec_meta["texture"],
                family=sec_fam,
                variant=s_asset["variant"],
                candidate_families=[sec_fam],
                rejected_candidates=[],
                reason=f"Aligned to {sec_marker.marker_type} ({sec_marker.rationale}) so numeric/payoff landing feels crisp.",
                target_frame=sec_marker.global_frame,
                pre_roll_ms=0.0,
                gain_db=-7.5 if sec_fam == "tick_accent" else -8.0,
                pan=-pan,
                priority=2,
                asset_id=s_asset["asset_id"],
                asset_provenance=s_asset["provenance"],
                asset_license=s_asset["license"],
                asset_sha256=s_asset["sha256"],
                fallback_status=s_asset["fallback_status"],
            ))

    recorded_n = sum(1 for c in cues if c.asset_provenance == "recorded_local_foley")
    proc_n = sum(1 for c in cues if c.asset_provenance != "recorded_local_foley")

    score = SonicScore(
        schema_version=SONIC_SCHEMA_VERSION,
        mode=storyboard.mode,
        story_graph_hash=story_graph.story_graph_hash,
        storyboard_hash=storyboard.storyboard_hash or storyboard.compute_hash(),
        foley_library_status=foley_lib["status"],
        recorded_cues_used=recorded_n,
        procedural_cues_used=proc_n,
        cues=cues,
        quiet_spans=quiet_spans,
        rejected_decisions=rejected_decisions,
        markers=markers,
    )
    score.validation_errors = validate_semantic_sound_grounding(score, story_graph, None, markers)
    score.score_hash = score.compute_hash()
    return score


def validate_semantic_sound_grounding(
    score: SonicScore,
    story_graph: StoryGraph,
    observed_actions: dict[str, Any] | None,
    markers: list[AnimationMarker],
) -> list[str]:
    """Fail-closed semantic validator preventing ungrounded diegetic foley, cliché mismatches, or marker drift."""
    errors: list[str] = []
    marker_map = {m.marker_id: m for m in markers}
    beat_map = {b.beat_id: b for b in story_graph.beats}
    non_money_spans = [
        (int(nm.get("start_frame", -1)), int(nm.get("end_frame_exclusive", -1)))
        for nm in story_graph.ledger.non_monetary_counts
    ]

    last_pair: dict[tuple[str, int], int] = {}
    for c in score.cues:
        if c.sound_role not in VALID_SOUND_ROLES:
            errors.append(f"invalid_sound_role:{c.cue_id}:{c.sound_role}")
        if c.family in DISALLOWED_CLICHE_FAMILIES:
            errors.append(f"disallowed_cliche_sound:{c.cue_id}:{c.family}")
        if c.family not in SFX_FAMILIES:
            errors.append(f"unknown_sfx_family:{c.cue_id}:{c.family}")
        if not c.reason.strip():
            errors.append(f"missing_rationale:{c.cue_id}")

        # Diegetic foley requires real observed action confidence >= 0.55
        if c.sound_role == "DIEGETIC_FOLEY" and c.action_confidence < 0.55:
            errors.append(f"unsupported_diegetic_foley:{c.cue_id}:conf={c.action_confidence}")

        # Marker contract check: cue must bind to an existing marker and stay within <=1 frame (graphic) / <=2 (action)
        m = marker_map.get(c.marker_id)
        if m is None:
            errors.append(f"missing_animation_marker:{c.cue_id}:{c.marker_id}")
        else:
            tol = 2 if c.sound_role == "DIEGETIC_FOLEY" else 1
            if abs(int(c.target_frame) - int(m.global_frame)) > tol:
                errors.append(f"marker_alignment_drift:{c.cue_id}:cue_f={c.target_frame},marker_f={m.global_frame},tol={tol}")

        # Semantic contradiction guards
        b = beat_map.get(c.beat_id)
        if b is not None:
            if (b.negated or b.speech_act == "negation_tension") and c.family in ("cash_flick", "shimmer_payoff"):
                errors.append(f"negated_beat_money_cue_forbidden:{c.cue_id}:{c.family}")
            if any(nm0 <= c.target_frame < nm1 for nm0, nm1 in non_money_spans) and c.family == "cash_flick":
                errors.append(f"non_monetary_count_cash_cue_forbidden:{c.cue_id}")
            if not story_graph.ledger.active and c.family == "cash_flick":
                errors.append(f"non_finance_cash_cue_forbidden:{c.cue_id}")

        # Back-to-back exact variant spam guard (within 60 frames = 2.5s @ 24fps)
        pk = (c.family, int(c.variant))
        if pk in last_pair and (c.target_frame - last_pair[pk]) < 60:
            errors.append(f"duplicate_exact_variant_spam:{c.cue_id}:{pk}")
        last_pair[pk] = c.target_frame

    return errors


def align_sound_to_marker(
    score: SonicScore,
    *,
    fps_num: int,
    fps_den: int,
    sample_rate: int,
    total_frames: int,
    source_hash: str,
    priming_offset_s: float = 0.0,
    audio_mode: str = "mix",
    sfx_enabled: bool = True,
    music_enabled: bool = False,
    seed: int = 42,
) -> tuple[AudioTimeline, list[dict[str, Any]], dict[str, Any]]:
    """Convert validated SonicScore + AnimationMarkers onto exact Rational Fraction sample positions."""
    dur_samples = frame_to_sample_exact(total_frames, fps_num, fps_den, sample_rate)
    if audio_mode == "preserve-exact" or not sfx_enabled:
        tl = AudioTimeline(
            schema_version=AUDIO_SCHEMA_VERSION,
            source_hash=source_hash,
            audio_mode=audio_mode,
            sfx_enabled=False,
            music_enabled=False,
            sample_rate=sample_rate,
            channels=1,
            duration_samples=dur_samples,
            fps_num=fps_num,
            fps_den=fps_den,
            pts_priming_offset_s=round(priming_offset_s, 6),
            events=[],
        )
        tl.timeline_hash = tl.compute_hash()
        return tl, [], {"used_assets": [], "recorded_cues_used": 0, "procedural_cues_used": 0}

    sorted_cues = sorted(score.cues, key=lambda c: (c.target_frame, c.priority))
    events: list[SoundEvent] = []
    alignment_rows: list[dict[str, Any]] = []
    used_assets: dict[str, dict[str, Any]] = {}
    last_any_peak = -sample_rate
    last_fam_peak: dict[str, int] = {}

    for c in sorted_cues:
        clamped_f = max(1, min(total_frames - 3, int(c.target_frame)))
        peak_samp = frame_to_sample_exact(clamped_f, fps_num, fps_den, sample_rate)
        if peak_samp - last_any_peak < int(sample_rate * 0.32):
            continue
        if c.family in last_fam_peak and (peak_samp - last_fam_peak[c.family]) < int(sample_rate * 0.80):
            continue

        wav, spec = synthesize_sfx_waveform(c.family, variant=int(c.variant) % 3, sr=sample_rate, seed=seed)
        used_assets[spec.asset_id] = spec.to_dict()

        trans_samp = int(round(spec.transient_offset_s * sample_rate))
        pre_roll_samp = int(round((c.pre_roll_ms / 1000.0) * sample_rate)) if c.sound_role == "TRANSITION_BRIDGE" else 0
        desired_peak = max(trans_samp, peak_samp - pre_roll_samp // 3)
        start_samp = max(0, desired_peak - trans_samp)
        end_samp = min(dur_samples, start_samp + wav.size)
        if end_samp - start_samp < int(sample_rate * 0.05):
            continue

        actual_peak_samp = min(end_samp - 1, start_samp + trans_samp)
        ev_id = f"sfx_{len(events):02d}_{c.family}"
        events.append(SoundEvent(
            id=ev_id,
            linked_visual_event_id=c.scene_id,
            semantic_span_ids=[str(w) for w in c.word_ids],
            observed_action_id=c.action_ref,
            asset_id=c.asset_id or spec.asset_id,
            family=c.family,
            variant=int(c.variant) % 3,
            asset_sha256=c.asset_sha256 or spec.sha256,
            start_sample=start_samp,
            peak_sample=actual_peak_samp,
            end_sample=end_samp,
            peak_frame=clamped_f,
            gain_db=float(c.gain_db),
            pan=float(c.pan),
            role=c.sound_role,
            confidence=float(c.action_confidence),
            reason=c.reason,
            fallback=c.fallback_status,
        ))
        alignment_rows.append({
            "event_id": ev_id,
            "cue_id": c.cue_id,
            "scene_id": c.scene_id,
            "marker_id": c.marker_id,
            "marker_type": c.marker_type,
            "sound_role": c.sound_role,
            "family": c.family,
            "variant": int(c.variant) % 3,
            "provenance": c.asset_provenance,
            "expected_frame": clamped_f,
            "expected_sample": peak_samp,
            "placed_peak_sample": actual_peak_samp,
            "delta_samples": actual_peak_samp - peak_samp,
            "delta_frames": round((actual_peak_samp - peak_samp) / (sample_rate * float(Fraction(fps_den, fps_num))), 3),
            "start_sample": start_samp,
            "end_sample": end_samp,
        })
        last_any_peak = actual_peak_samp
        last_fam_peak[c.family] = actual_peak_samp
        if len(events) >= 16:
            break

    tl = AudioTimeline(
        schema_version=AUDIO_SCHEMA_VERSION,
        source_hash=source_hash,
        audio_mode=audio_mode,
        sfx_enabled=True,
        music_enabled=bool(music_enabled),
        sample_rate=sample_rate,
        channels=2,
        duration_samples=dur_samples,
        fps_num=fps_num,
        fps_den=fps_den,
        pts_priming_offset_s=round(priming_offset_s, 6),
        events=events,
    )
    tl.timeline_hash = tl.compute_hash()
    return tl, alignment_rows, {
        "used_assets": list(used_assets.values()),
        "recorded_cues_used": score.recorded_cues_used,
        "procedural_cues_used": score.procedural_cues_used,
    }


def _emotional_role_for_act(speech_act: str, negated: bool) -> str:
    if negated or speech_act == "negation_tension":
        return "cognitive_tension_contrast"
    mapping = {
        "hook_contrast": "curiosity_opening_hook",
        "allocation_step": "tactile_subtraction_clarity",
        "payoff_reveal": "decisive_reality_payoff",
        "product_intro": "calm_solution_handoff",
        "product_benefit": "organized_tactile_craft",
        "closing_takeaway": "warm_editorial_resolution",
    }
    return mapping.get(speech_act, "editorial_support")


def _build_cue_reason(opp: SonicOpportunity, family: str, role: str, marker: AnimationMarker) -> str:
    if opp.negated or opp.speech_act == "negation_tension":
        return f"{role} ({family}) accents contradiction symbol at {marker.marker_id} while keeping negated speech clear."
    if opp.speech_act == "allocation_step":
        return f"{role} ({family}) marks {opp.ledger_category or 'budget'} transfer at {marker.marker_id} matching proportional deduction."
    if opp.speech_act == "payoff_reveal":
        return f"{role} ({family}) punctuates remaining balance contrast at {marker.marker_id} without celebratory cliché."
    if opp.speech_act == "product_benefit":
        return f"{role} ({family}) aligns tactile page/paper motion at {marker.marker_id} for conceptual workbook overview."
    return f"{role} ({family}) supports {opp.speech_act} aligned to {marker.marker_id}."
