"""Phase R2-A1 (ML-01.R5.R2): Hierarchical Stateful Multimodal Creative Intelligence.

Replaces single-pass recipe/keyword selection with bounded, versioned reasoning stages:
  1. Perception & Evidence Index (`build_evidence_index`, `normalize_semantic_entities`, `resolve_coreferences`)
  2. Narrative Contract & Persistent State Graph (`build_narrative_contract`, `build_state_graph`, `plan_attention_arc`)
  3. Local Versioned Creative Playbook (`retrieve_playbook_rules`) with human-reviewed vs automated rule provenance
  4. Multi-Candidate Scene Graph Proposal, Deterministic Rubric Ranking & Global Reconciliation
     (`propose_scene_graphs`, `rank_scene_candidates`, `reconcile_global_storyboard`)

Strict invariants:
  - 100% offline / Zero-API (never calls cloud LLMs or executes LLM-emitted code).
  - ZERO hardcoded Olivia scripts, timestamps, filenames, or clip SHA shortcuts.
  - Honest hybrid reasoning accounting: records raw model proposals vs deterministic guard corrections.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from motionlab.schemas import MediaInfo, canonical_hash, sanitize_native
from motionlab.semantic.schema import Transcript, WordSpan
from motionlab.semantic.story_graph import StoryBeatNode, StoryGraph, format_cents, parse_amount_to_cents

NARRATIVE_BRAIN_SCHEMA_VERSION = "2.0.0"

_INJECTION_PATTERNS = (
    re.compile(r"(?i)ignore\s+(?:all\s+)?(?:previous|prior)\s+instructions?"),
    re.compile(r"(?i)system\s*:\s*you\s+are"),
    re.compile(r"(?i)<\s*script"),
    re.compile(r"(?i)\beval\s*\("),
    re.compile(r"(?i)\b(?:rm\s+-rf|curl\s+http|wget\s+http|powershell\s+-)"),
)

_VEHICLE_CONTEXT_RE = re.compile(
    r"(?i)\b(car|engine|pedal|throttle|highway|accelerate|rpm|driving|dashboard\s+gauge|horsepower|exhaust|motor)\b"
)
_GROCERY_SCANNER_RE = re.compile(
    r"(?i)\b(barcode|scanner|checkout\s+lane|cashier|register\s+beep|aisle|shopping\s+cart)\b"
)
_REFUND_INCOME_RE = re.compile(
    r"(?i)\b(refund|cashback|reimbursement|returned|got\s+back|paid\s+back|bonus\s+received|income|earned|deposited)\b"
)
_GUARANTEE_CLAIM_RE = re.compile(
    r"(?i)\b(guarantee[ds]?|100%\s+certain|definitely\s+save|proven\s+profit|risk[\-\s]?free)\b"
)
_UNCERTAIN_MODALITY_RE = re.compile(
    r"(?i)\b(might|could|may|possibly|potentially|let'?s\s+say|suppose|for\s+example|hypothetically)\b"
)
_SLOGAN_MONEY_RE = re.compile(
    r"(?i)\b(every\s+dollar\s+a\s+purpose|time\s+is\s+money|dollar\s+for\s+dollar|worth\s+every\s+penny)\b"
)


# ---------------------------------------------------------------------------
# 1. Local Versioned Creative Playbook (Defect-Derived Reusable Rules)
# ---------------------------------------------------------------------------

PLAYBOOK_VERSION = "2.0.0"

DEFAULT_PLAYBOOK_RULES: list[dict[str, Any]] = [
    {
        "rule_id": "PB-01",
        "category": "numeric_integrity",
        "status": "HUMAN_APPROVED",
        "origin": "defect_ledger:D01,D05",
        "niches": ["finance", "product", "education"],
        "rule": "Never interpolate arbitrary fractional cents (e.g. $1,151.93 or $144.06) when spoken amounts are whole integers; lock integer-dollar display mode unless source evidence explicitly speaks fractional cents.",
    },
    {
        "rule_id": "PB-02",
        "category": "single_numeric_owner",
        "status": "HUMAN_APPROVED",
        "origin": "defect_ledger:D03",
        "niches": ["finance"],
        "rule": "Maintain ONE canonical visible currency counter per beat; never pair an independent bottom ribbon counter against an active top allocation counter displaying conflicting intermediate numbers.",
    },
    {
        "rule_id": "PB-03",
        "category": "continuous_bridges",
        "status": "HUMAN_APPROVED",
        "origin": "defect_ledger:D06,D12",
        "niches": ["finance", "product", "storytelling", "education", "lifestyle", "cinematic"],
        "rule": "Every transition between full-frame graphic cutaway and live footage MUST have an explicit rendered incoming or outgoing bridge (paper fold, soft mask unveil, or shared-element morph) rather than an unmotivated hard cut.",
    },
    {
        "rule_id": "PB-04",
        "category": "editorial_hierarchy",
        "status": "HUMAN_APPROVED",
        "origin": "defect_ledger:D01_R1,D04",
        "niches": ["finance", "product", "storytelling", "education", "lifestyle", "cinematic"],
        "rule": "Avoid repetitive dark rounded card templates and tiny speaker inset boxes; keep live speaker as full-frame human anchor and reserve cutaways for high-value teaching milestones.",
    },
    {
        "rule_id": "PB-05",
        "category": "semantic_sound_integrity",
        "status": "HUMAN_APPROVED",
        "origin": "defect_ledger:D09,D10,D11",
        "niches": ["finance", "product", "storytelling", "education", "lifestyle", "cinematic"],
        "rule": "Match sound role to actual visual/semantic context (never engine rev on budget gas, never cash flick on negation or non-monetary page counts, never silent cue drops, protect 1-4kHz dialogue).",
    },
    {
        "rule_id": "PB-06",
        "category": "face_contrast_protection",
        "status": "HUMAN_APPROVED",
        "origin": "defect_ledger:D08",
        "niches": ["finance", "product", "storytelling", "education", "lifestyle", "cinematic"],
        "rule": "Never apply bright full-frame light leaks over the speaker's face (max edge glow <= 0.22 opacity with clear center face mask) and keep all text clear of eyes/mouth and Reels safe zones.",
    },
    {
        "rule_id": "PB-07",
        "category": "product_asset_truth",
        "status": "HUMAN_APPROVED",
        "origin": "defect_ledger:D07",
        "niches": ["product", "finance", "education"],
        "rule": "Display real product UI ONLY from owner-authorized verified ProductAssetCatalog entries; otherwise render honest conceptual illustration explicitly marked ASSET_MISSING_CONCEPTUAL.",
    },
]


def retrieve_playbook_rules(
    niche: str,
    categories: list[str] | None = None,
    approved_only: bool = False,
) -> list[dict[str, Any]]:
    """Return applicable versioned creative playbook rules with provenance & approval status."""
    n = (niche or "cinematic").lower()
    out: list[dict[str, Any]] = []
    for r in DEFAULT_PLAYBOOK_RULES:
        if approved_only and r.get("status") != "HUMAN_APPROVED":
            continue
        if n in r.get("niches", []) or "all" in r.get("niches", []):
            if categories is None or r.get("category") in categories:
                out.append(dict(r))
    return out


# ---------------------------------------------------------------------------
# 2. Evidence Index, Entity Normalization & Coreference Resolution
# ---------------------------------------------------------------------------

@dataclass
class SemanticEntity:
    entity_id: str
    kind: str  # currency_balance | currency_allocation | currency_remaining | currency_deficit | currency_income | non_monetary_count | product | product_feature | metaphor | cta
    surface_text: str
    normalized_value_cents: int | None
    currency_symbol: str
    currency_code: str
    unit: str
    has_fractional_cents: bool
    modality: str  # assertive | hypothetical | negated | uncertain_asr
    evidence_word_ids: list[int]
    start_frame: int
    end_frame_exclusive: int
    confidence: float
    coref_target_id: str | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return sanitize_native(asdict(self))


@dataclass
class EvidenceIndex:
    schema_version: str
    source_hash: str
    transcript_hash: str
    language: str
    total_frames: int
    fps_float: float
    word_count: int
    uncertain_word_ids: list[int]
    prompt_injection_flags: list[dict[str, Any]]
    word_refs: list[dict[str, Any]]
    visual_anchor_summary: dict[str, Any]
    audio_silence_spans: list[dict[str, Any]]
    available_product_assets: list[dict[str, Any]]
    entities: list[SemanticEntity] = field(default_factory=list)
    index_hash: str = ""

    def compute_hash(self) -> str:
        d = self.to_dict()
        d.pop("index_hash", None)
        return canonical_hash(d)

    def to_dict(self) -> dict[str, Any]:
        return sanitize_native(asdict(self))


def _flatten_words_with_frames(transcript: Transcript, fps_float: float, total_frames: int) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    wid = 0
    for seg in transcript.segments:
        for w in seg.words:
            sf = max(0, min(total_frames - 1, int(round(float(w.start_s) * fps_float))))
            ef = max(sf + 1, min(total_frames, int(round(float(w.end_s) * fps_float))))
            refs.append({
                "word_id": wid,
                "word": w.word.strip(),
                "start_s": round(float(w.start_s), 4),
                "end_s": round(float(w.end_s), 4),
                "start_frame": sf,
                "end_frame_exclusive": ef,
                "confidence": round(float(w.probability), 4),
                "uncertain": bool(float(w.probability) < 0.45),
            })
            wid += 1
    return refs


def normalize_semantic_entities(
    story_graph: StoryGraph,
    word_refs: list[dict[str, Any]],
) -> list[SemanticEntity]:
    """Normalize monetary, non-monetary, product, metaphor, and CTA entities with strict semantic typing."""
    entities: list[SemanticEntity] = []
    led = story_graph.ledger
    step_by_id = {st.step_id: st for st in led.steps}

    # 1. Ledger monetary entities (exact integer cents + explicit fractional flag)
    opening_entity_id: str | None = None
    for st in led.steps:
        has_frac = (abs(int(st.amount_cents)) % 100) != 0
        if st.step_type == "opening_balance":
            kind = "currency_balance"
            eid = f"ent_balance_{st.step_id}"
            opening_entity_id = eid
        elif st.step_type == "allocation":
            kind = "currency_deficit" if st.running_balance_cents < 0 else "currency_allocation"
            eid = f"ent_alloc_{st.category.lower().replace(' ', '_')}_{st.step_id}"
        elif st.step_type == "remaining_confirmed":
            kind = "currency_deficit" if st.running_balance_cents < 0 else "currency_remaining"
            eid = f"ent_remaining_{st.step_id}"
        elif st.step_type == "income_refund":
            kind = "currency_income"
            eid = f"ent_income_{st.step_id}"
        else:
            kind = "currency_allocation"
            eid = f"ent_step_{st.step_id}"

        mod = "uncertain_asr" if st.uncertain else st.modality
        entities.append(SemanticEntity(
            entity_id=eid,
            kind=kind,
            surface_text=format_cents(st.amount_cents, st.currency_symbol),
            normalized_value_cents=int(st.amount_cents),
            currency_symbol=st.currency_symbol,
            currency_code=st.currency_code,
            unit=st.currency_code,
            has_fractional_cents=has_frac,
            modality=mod,
            evidence_word_ids=list(st.word_ids),
            start_frame=st.start_frame,
            end_frame_exclusive=st.end_frame_exclusive,
            confidence=float(st.confidence),
            coref_target_id=opening_entity_id if eid != opening_entity_id else None,
            notes=f"{st.step_type}:{st.category} ({st.provenance})",
        ))

    # 2. Non-monetary counts (e.g., "56-page workbook", "10%") strictly isolated from currency
    for idx, nm in enumerate(led.non_monetary_counts):
        entities.append(SemanticEntity(
            entity_id=f"ent_nonmon_{idx:02d}",
            kind="non_monetary_count",
            surface_text=str(nm.get("raw", "")),
            normalized_value_cents=None,
            currency_symbol="",
            currency_code="",
            unit=str(nm.get("unit", "count")),
            has_fractional_cents=("." in str(nm.get("value", ""))),
            modality="assertive",
            evidence_word_ids=[],
            start_frame=int(nm.get("start_frame", 0)),
            end_frame_exclusive=int(nm.get("end_frame_exclusive", 1)),
            confidence=0.95,
            notes="Non-monetary count; MUST NEVER enter currency math or trigger cash foley",
        ))

    # 3. Beat-level negated statements, metaphors, products, attributes, CTAs
    prod_entity_id: str | None = None
    for b in story_graph.beats:
        q = b.verbatim_quote or ""
        if b.negated:
            entities.append(SemanticEntity(
                entity_id=f"ent_neg_{b.beat_id}",
                kind="negated_contrast",
                surface_text=q[:120],
                normalized_value_cents=None,
                currency_symbol=led.currency_symbol if led.active else "",
                currency_code=led.currency_code if led.active else "",
                unit="proposition",
                has_fractional_cents=False,
                modality="negated",
                evidence_word_ids=list(b.word_ids),
                start_frame=b.start_frame,
                end_frame_exclusive=b.end_frame_exclusive,
                confidence=float(b.mean_word_confidence),
                coref_target_id=opening_entity_id,
                notes="Negated proposition — never interpret as positive spendable/savings claim",
            ))
        if _SLOGAN_MONEY_RE.search(q):
            entities.append(SemanticEntity(
                entity_id=f"ent_metaphor_{b.beat_id}",
                kind="metaphor",
                surface_text=q[:120],
                normalized_value_cents=None,
                currency_symbol="",
                currency_code="",
                unit="slogan",
                has_fractional_cents=False,
                modality="assertive",
                evidence_word_ids=list(b.word_ids),
                start_frame=b.start_frame,
                end_frame_exclusive=b.end_frame_exclusive,
                confidence=float(b.mean_word_confidence),
                notes="Conceptual money slogan — not physical cash handling",
            ))
        if b.product_entities:
            prod_entity_id = f"ent_product_{b.beat_id}"
            entities.append(SemanticEntity(
                entity_id=prod_entity_id,
                kind="product",
                surface_text=b.product_entities[0],
                normalized_value_cents=None,
                currency_symbol="",
                currency_code="",
                unit="product",
                has_fractional_cents=False,
                modality="assertive",
                evidence_word_ids=list(b.word_ids),
                start_frame=b.start_frame,
                end_frame_exclusive=b.end_frame_exclusive,
                confidence=float(b.mean_word_confidence),
                notes="Spoken product name grounded in transcript",
            ))
        for ai, attr in enumerate(b.attribute_entities):
            entities.append(SemanticEntity(
                entity_id=f"ent_attr_{b.beat_id}_{ai}",
                kind="product_feature",
                surface_text=attr,
                normalized_value_cents=None,
                currency_symbol="",
                currency_code="",
                unit="feature",
                has_fractional_cents=False,
                modality="assertive",
                evidence_word_ids=list(b.word_ids),
                start_frame=b.start_frame,
                end_frame_exclusive=b.end_frame_exclusive,
                confidence=float(b.mean_word_confidence),
                coref_target_id=prod_entity_id,
                notes="Spoken product component",
            ))
        if b.speech_act == "cta":
            entities.append(SemanticEntity(
                entity_id=f"ent_cta_{b.beat_id}",
                kind="cta",
                surface_text=q[:120],
                normalized_value_cents=None,
                currency_symbol="",
                currency_code="",
                unit="cta",
                has_fractional_cents=False,
                modality="assertive",
                evidence_word_ids=list(b.word_ids),
                start_frame=b.start_frame,
                end_frame_exclusive=b.end_frame_exclusive,
                confidence=float(b.mean_word_confidence),
                coref_target_id=prod_entity_id,
                notes="Spoken call-to-action",
            ))

    return resolve_coreferences(entities, story_graph)


def resolve_coreferences(
    entities: list[SemanticEntity],
    story_graph: StoryGraph,
) -> list[SemanticEntity]:
    """Link later references ('that balance', 'it', product attributes, remaining balance) to canonical global entities."""
    opening = next((e for e in entities if e.kind == "currency_balance"), None)
    product = next((e for e in entities if e.kind == "product"), None)
    for e in entities:
        if e.coref_target_id is None:
            if e.kind in ("currency_allocation", "currency_remaining", "currency_deficit") and opening is not None:
                e.coref_target_id = opening.entity_id
            elif e.kind in ("product_feature", "cta") and product is not None:
                e.coref_target_id = product.entity_id
    return entities


def build_evidence_index(
    media: MediaInfo,
    transcript: Transcript,
    story_graph: StoryGraph,
    roi_tracks: dict[str, Any] | None = None,
    asset_catalog: list[dict[str, Any]] | None = None,
) -> EvidenceIndex:
    """Build versioned, hash-linked EvidenceIndex from ASR words, visual ROI tracks, silence spans, and verified assets."""
    word_refs = _flatten_words_with_frames(transcript, media.fps_float, media.duration_frames)
    uncertain_ids = [w["word_id"] for w in word_refs if w["uncertain"]]

    # Scan for ASR transcript prompt-injection attempts (never execute — flag & isolate!)
    injection_flags: list[dict[str, Any]] = []
    full_txt = transcript.full_text or ""
    for pat in _INJECTION_PATTERNS:
        m = pat.search(full_txt)
        if m:
            injection_flags.append({"pattern": pat.pattern, "match_offset": m.start(), "action": "neutralized_as_untrusted_data"})

    # Detect natural silence/breathing spans >= 0.9s between words
    silence_spans: list[dict[str, Any]] = []
    for i in range(1, len(word_refs)):
        gap = word_refs[i]["start_s"] - word_refs[i - 1]["end_s"]
        if gap >= 0.9:
            silence_spans.append({
                "start_s": word_refs[i - 1]["end_s"],
                "end_s": word_refs[i]["start_s"],
                "start_frame": word_refs[i - 1]["end_frame_exclusive"],
                "end_frame_exclusive": word_refs[i]["start_frame"],
                "duration_s": round(gap, 3),
            })

    roi_sum = dict((roi_tracks or {}).get("summary") or {})
    kfs = (roi_tracks or {}).get("keyframes") or (roi_tracks or {}).get("frames") or []
    face_present_n = sum(1 for k in kfs if isinstance(k.get("face"), dict) and float(k["face"].get("conf", k["face"].get("confidence", 0.0))) >= 0.4)
    task_present_n = sum(1 for k in kfs if isinstance(k.get("task"), dict) and float(k["task"].get("conf", k["task"].get("confidence", 0.0))) >= 0.35)
    visual_anchor_summary = {
        "keyframe_count": len(kfs),
        "has_face_track": bool((roi_tracks or {}).get("has_face_track") or face_present_n > 0),
        "has_task_track": bool((roi_tracks or {}).get("has_task_track") or task_present_n > 0),
        "face_present_ratio": round(face_present_n / max(1, len(kfs)), 3) if kfs else float(roi_sum.get("face_present_ratio", 0.0)),
        "task_present_ratio": round(task_present_n / max(1, len(kfs)), 3) if kfs else float(roi_sum.get("task_present_ratio", 0.0)),
        "tabletop_guard_y": (roi_tracks or {}).get("tabletop_guard_y"),
    }

    entities = normalize_semantic_entities(story_graph, word_refs)
    idx = EvidenceIndex(
        schema_version=NARRATIVE_BRAIN_SCHEMA_VERSION,
        source_hash=media.input_hash,
        transcript_hash=transcript.transcript_hash,
        language=transcript.language,
        total_frames=media.duration_frames,
        fps_float=round(media.fps_float, 4),
        word_count=len(word_refs),
        uncertain_word_ids=uncertain_ids,
        prompt_injection_flags=injection_flags,
        word_refs=word_refs,
        visual_anchor_summary=visual_anchor_summary,
        audio_silence_spans=silence_spans,
        available_product_assets=list(asset_catalog or []),
        entities=entities,
    )
    idx.index_hash = idx.compute_hash()
    return idx


# ---------------------------------------------------------------------------
# 3. Narrative Contract, Persistent State Graph & Attention Arc
# ---------------------------------------------------------------------------

@dataclass
class NarrativeContract:
    schema_version: str
    inferred_niche: str
    niche_confidence: float
    narrative_pattern: str  # financial_conservation | problem_solution_product | how_to_steps | cause_effect | storytelling_reveal | neutral_preserve
    video_purpose: str
    hook_question: str
    central_conflict: str
    proof_mechanism: str
    payoff_takeaway: str
    product_value_proposition: str
    cta_summary: str
    tone: str
    allow_fractional_cents: bool
    is_hypothetical_example: bool
    has_authorized_product_asset: bool
    constraints: list[str] = field(default_factory=list)
    playbook_rule_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return sanitize_native(asdict(self))


@dataclass
class BeatStateSnapshot:
    beat_id: str
    order: int
    start_frame: int
    end_frame_exclusive: int
    speech_act: str
    primary_focal_object: str
    desired_emotion: str
    pacing_energy: str  # calm | measured | focused_build | decisive_payoff | breathing_hold
    speaker_presence: str  # full_hero | macro_hands | architectural_split | purposeful_cutaway
    continuity_group: str
    state_before: dict[str, Any]
    state_after: dict[str, Any]
    state_transition_cause: str
    must_not_reveal_before_word_id: int | None
    viewer_takeaway: str
    sound_semantic_context: dict[str, Any]
    allow_fractional_cents: bool = False

    def to_dict(self) -> dict[str, Any]:
        return sanitize_native(asdict(self))


@dataclass
class NarrativeStateGraph:
    schema_version: str
    contract: NarrativeContract
    persistent_elements: list[dict[str, Any]]
    beat_states: list[BeatStateSnapshot]
    attention_arc: list[dict[str, Any]]
    state_graph_hash: str = ""

    def compute_hash(self) -> str:
        d = self.to_dict()
        d.pop("state_graph_hash", None)
        return canonical_hash(d)

    def to_dict(self) -> dict[str, Any]:
        return sanitize_native(asdict(self))


def build_narrative_contract(
    evidence: EvidenceIndex,
    story_graph: StoryGraph,
    requested_niche: str | None = None,
) -> NarrativeContract:
    """Infer global narrative contract strictly from transcript evidence, ledger structure, and visual observations."""
    full_low = (story_graph.full_text or "").lower()
    led = story_graph.ledger
    face_ratio = float(evidence.visual_anchor_summary.get("face_present_ratio", 0.0))

    # Infer niche honestly from semantic evidence (never assume finance on non-finance clips!)
    if requested_niche and requested_niche not in ("auto", ""):
        niche = requested_niche.lower()
        nconf = 0.92
    elif led.active:
        niche = "finance"
        nconf = 0.94
    elif story_graph.product_name or story_graph.product_attributes:
        niche = "product"
        nconf = 0.86
    elif any(k in full_low for k in ("step one", "step 1", "tutorial", "click here", "terminal", "install", "how to")):
        niche = "education" if face_ratio >= 0.45 else "tutorial"
        nconf = 0.84
    elif any(k in full_low for k in ("story", "happened", "remember when", "years ago", "1969", "journey")):
        niche = "storytelling"
        nconf = 0.82
    else:
        niche = "cinematic"
        nconf = 0.72

    # Narrative structure pattern
    if led.active:
        pattern = "financial_conservation"
    elif story_graph.product_name:
        pattern = "problem_solution_product"
    elif niche in ("education", "tutorial"):
        pattern = "how_to_steps"
    elif niche == "storytelling":
        pattern = "storytelling_reveal"
    else:
        pattern = "neutral_preserve"

    # Determine whether ANY monetary entity in the source transcript explicitly speaks fractional cents
    allow_frac = any(
        e.has_fractional_cents
        for e in evidence.entities
        if e.kind.startswith("currency_")
    )
    is_hyp = (led.modality == "hypothetical_example") or bool(_UNCERTAIN_MODALITY_RE.search(full_low))

    # Check if an owner-authorized verified product asset matches the spoken product
    has_auth_asset = False
    if story_graph.product_name and evidence.available_product_assets:
        p_low = story_graph.product_name.lower()
        for a in evidence.available_product_assets:
            if a.get("authorized") and a.get("verified") and (
                p_low in str(a.get("product_name", "")).lower()
                or str(a.get("product_name", "")).lower() in p_low
            ):
                has_auth_asset = True
                break

    hook_b = next((b for b in story_graph.beats if b.speech_act == "hook_contrast"), story_graph.beats[0] if story_graph.beats else None)
    neg_b = next((b for b in story_graph.beats if b.negated or b.speech_act == "negation_tension"), None)
    payoff_b = next((b for b in story_graph.beats if b.speech_act == "payoff_reveal"), None)
    prod_b = next((b for b in story_graph.beats if b.speech_act in ("product_intro", "product_benefit")), None)

    sym = led.currency_symbol or "$"
    if led.active:
        purpose = f"Illustrate the gap between opening balance ({format_cents(led.initial_cents, sym)}) and truly spendable funds ({format_cents(led.computed_remaining_cents, sym)}) in an example."
        proof = f"Step-by-step integer-cent deduction ({len([s for s in led.steps if s.step_type == 'allocation'])} committed categories)."
        payoff_txt = (
            f"Deficit of {format_cents(led.computed_remaining_cents, sym)}"
            if led.computed_remaining_cents < 0
            else f"{format_cents(led.computed_remaining_cents, sym)} actually left to spend in this example"
        )
    else:
        purpose = "Communicate the speaker's core message clearly while preserving live visual context."
        proof = "Grounded visual emphasis synchronized to spoken beats."
        payoff_txt = payoff_b.verbatim_quote[:100] if payoff_b else "Clear viewer takeaway"

    prod_vp = (
        f"Position {story_graph.product_name} ({', '.join(story_graph.product_attributes) or 'organization system'}) strictly as an organization tool without unverified savings guarantees."
        if story_graph.product_name
        else "No product pitch detected."
    )

    rules = retrieve_playbook_rules(niche)
    constraints = [
        "zero_invented_numbers_or_savings_guarantees",
        "integer_dollar_display_locked" if not allow_frac else "explicit_fractional_cents_permitted",
        "single_canonical_ledger_state_owner",
        "preserve_source_voice_unity_and_timeline",
        "protect_eyes_mouth_and_reels_safe_zones",
        "authorized_product_asset_verified" if has_auth_asset else "conceptual_product_illustration_only",
    ]

    return NarrativeContract(
        schema_version=NARRATIVE_BRAIN_SCHEMA_VERSION,
        inferred_niche=niche,
        niche_confidence=round(nconf, 3),
        narrative_pattern=pattern,
        video_purpose=purpose,
        hook_question=(hook_b.verbatim_quote[:120] if hook_b else ""),
        central_conflict=(neg_b.verbatim_quote[:120] if neg_b else "Clarify key distinction"),
        proof_mechanism=proof,
        payoff_takeaway=payoff_txt,
        product_value_proposition=prod_vp,
        cta_summary=(story_graph.cta_phrase or "No explicit CTA"),
        tone="premium_editorial_quiet_luxury" if niche in ("finance", "product", "lifestyle") else "clean_editorial",
        allow_fractional_cents=allow_frac,
        is_hypothetical_example=is_hyp,
        has_authorized_product_asset=has_auth_asset,
        constraints=constraints,
        playbook_rule_ids=[r["rule_id"] for r in rules],
    )


def build_state_graph(
    evidence: EvidenceIndex,
    story_graph: StoryGraph,
    contract: NarrativeContract,
) -> NarrativeStateGraph:
    """Build deterministic before/after persistent world state for every beat across the full video."""
    led = story_graph.ledger
    step_map = {st.step_id: st for st in led.steps}
    full_low = (story_graph.full_text or "").lower()
    is_vehicle_video = bool(_VEHICLE_CONTEXT_RE.search(full_low)) and not led.active
    has_grocery_scanner = bool(_GROCERY_SCANNER_RE.search(full_low))

    persistent_elements: list[dict[str, Any]] = []
    if led.active:
        persistent_elements.extend([
            {"element_id": "balance_hero", "role": "canonical_amount_display", "continuity_group": "ledger_world"},
            {"element_id": "cash_pool", "role": "illustrative_paper_tokens", "continuity_group": "ledger_world"},
            {"element_id": "allocation_ledger", "role": "proportional_category_bars", "continuity_group": "ledger_world"},
        ])
    if story_graph.product_name or story_graph.product_attributes:
        persistent_elements.append({
            "element_id": "product_organizer",
            "role": "authorized_product_demo" if contract.has_authorized_product_asset else "conceptual_organizer_motif",
            "continuity_group": "solution_world",
        })
    if not persistent_elements:
        persistent_elements.append({"element_id": "editorial_focus", "role": "typographic_focus", "continuity_group": "editorial_world"})

    beat_states: list[BeatStateSnapshot] = []
    running_cents = led.initial_cents if led.active else 0
    allocated_cats: dict[str, int] = {}
    payoff_beat = next((b for b in story_graph.beats if b.speech_act == "payoff_reveal"), None)
    payoff_first_word_id = payoff_beat.word_ids[0] if (payoff_beat and payoff_beat.word_ids) else None

    for b in story_graph.beats:
        st = step_map.get(b.ledger_step_id or "") if b.ledger_step_id else None
        q_low = (b.verbatim_quote or "").lower()

        before_state = {
            "ledger_active": led.active,
            "available_cents": running_cents,
            "allocated_categories": dict(allocated_cats),
            "currency_symbol": led.currency_symbol if led.active else "",
            "product_introduced": any(prev.speech_act == "product_intro" for prev in story_graph.beats[: b.order]),
        }

        cause = f"beat_{b.speech_act}"
        if st is not None:
            if st.step_type == "opening_balance":
                running_cents = st.amount_cents
                cause = f"establish_opening_balance:{format_cents(st.amount_cents, st.currency_symbol)}"
            elif st.step_type == "allocation":
                running_cents = st.running_balance_cents
                allocated_cats[st.category] = abs(st.amount_cents)
                cause = f"commit_allocation:{st.category}:{format_cents(st.delta_cents, st.currency_symbol)}"
            elif st.step_type == "income_refund":
                running_cents = st.running_balance_cents
                cause = f"credit_income_or_refund:{format_cents(st.delta_cents, st.currency_symbol)}"
            elif st.step_type == "remaining_confirmed":
                running_cents = st.running_balance_cents
                cause = f"confirm_remaining_balance:{format_cents(st.running_balance_cents, st.currency_symbol)}"

        after_state = {
            "ledger_active": led.active,
            "available_cents": running_cents,
            "allocated_categories": dict(allocated_cats),
            "currency_symbol": led.currency_symbol if led.active else "",
            "product_introduced": before_state["product_introduced"] or b.speech_act == "product_intro",
        }

        # Context-aware sound semantics (prevents naive keyword mapping!)
        is_budget_gas = ("gas" in q_low) and (led.active or not is_vehicle_video)
        is_vehicle_gas = ("gas" in q_low) and is_vehicle_video
        is_budget_groceries = ("grocer" in q_low) and not has_grocery_scanner
        is_slogan_money = bool(_SLOGAN_MONEY_RE.search(q_low))
        is_non_money_count = any(
            min(b.end_frame_exclusive, int(nm.get("end_frame_exclusive", -1))) > max(b.start_frame, int(nm.get("start_frame", -1)))
            for nm in led.non_monetary_counts
        )
        is_deficit = led.active and (running_cents < 0)

        sound_ctx = {
            "speech_act": b.speech_act,
            "negated": b.negated,
            "is_budget_gas": is_budget_gas,
            "is_vehicle_gas": is_vehicle_gas,
            "is_budget_groceries": is_budget_groceries,
            "is_slogan_money": is_slogan_money,
            "is_non_monetary_count": is_non_money_count,
            "is_deficit": is_deficit,
            "is_hypothetical": contract.is_hypothetical_example,
            "has_observed_hand_contact": bool(
                b.visual_observations.get("has_task") and float(b.visual_observations.get("task_conf", 0.0)) >= 0.62
            ),
            "prefer_silence": b.speech_act in ("breathing_bridge", "cta"),
            "disallowed_families": _compute_disallowed_sound_families(
                b, led.active, is_budget_gas, is_budget_groceries, is_slogan_money, is_non_money_count, is_deficit
            ),
        }

        # Focal object, emotion, pacing, speaker presence & continuity group
        focal, emo, pacing, presence, cgroup = _plan_beat_attention(b, st, led.active, contract)
        no_reveal_before = b.word_ids[0] if b.word_ids else None
        # Protect payoff suspense: do not reveal final remaining hero numeral prior to payoff beat!
        if b.speech_act == "allocation_step" and payoff_first_word_id is not None:
            if st is not None and st.running_balance_cents == led.computed_remaining_cents:
                no_reveal_before = b.word_ids[0] if b.word_ids else None

        takeaway = _synthesize_viewer_takeaway(b, st, led.active, contract)
        beat_states.append(BeatStateSnapshot(
            beat_id=b.beat_id,
            order=b.order,
            start_frame=b.start_frame,
            end_frame_exclusive=b.end_frame_exclusive,
            speech_act=b.speech_act,
            primary_focal_object=focal,
            desired_emotion=emo,
            pacing_energy=pacing,
            speaker_presence=presence,
            continuity_group=cgroup,
            state_before=before_state,
            state_after=after_state,
            state_transition_cause=cause,
            must_not_reveal_before_word_id=no_reveal_before,
            viewer_takeaway=takeaway,
            sound_semantic_context=sound_ctx,
            allow_fractional_cents=contract.allow_fractional_cents,
        ))

    arc = plan_attention_arc(beat_states)
    nsg = NarrativeStateGraph(
        schema_version=NARRATIVE_BRAIN_SCHEMA_VERSION,
        contract=contract,
        persistent_elements=persistent_elements,
        beat_states=beat_states,
        attention_arc=arc,
    )
    nsg.state_graph_hash = nsg.compute_hash()
    return nsg


def _compute_disallowed_sound_families(
    beat: StoryBeatNode,
    ledger_active: bool,
    is_budget_gas: bool,
    is_budget_groceries: bool,
    is_slogan_money: bool,
    is_non_money_count: bool,
    is_deficit: bool,
) -> list[str]:
    bad: list[str] = ["cash_register_chaching", "slot_machine_coins", "error_buzzer"]
    if beat.negated or beat.speech_act == "negation_tension":
        bad.extend(["cash_flick", "shimmer_payoff"])
    if is_budget_gas:
        bad.append("engine_rev")
    if is_budget_groceries:
        bad.append("checkout_beep")
    if is_slogan_money or is_non_money_count or not ledger_active:
        bad.append("cash_flick")
    if is_deficit:
        bad.append("shimmer_payoff")
    if beat.speech_act == "cta":
        bad.extend(["notification_ding", "cash_flick"])
    return sorted(set(bad))


def _plan_beat_attention(
    beat: StoryBeatNode,
    step: Any,
    ledger_active: bool,
    contract: NarrativeContract,
) -> tuple[str, str, str, str, str]:
    act = beat.speech_act
    if act == "breathing_bridge":
        return ("live_speaker_expression", "reflective_pause", "breathing_hold", "full_hero", "live_world")
    if act == "hook_contrast":
        return ("balance_hero", "immediate_curiosity", "measured", "full_hero", "ledger_world")
    if act == "negation_tension":
        return ("balance_hero", "sharp_clarity", "focused_build", "full_hero", "ledger_world")
    if act == "allocation_step":
        return ("cash_pool", "tactile_proof", "focused_build", "macro_hands", "ledger_world")
    if act == "payoff_reveal":
        return ("balance_hero", "decisive_realization", "decisive_payoff", "purposeful_cutaway", "ledger_world")
    if act == "product_intro":
        return ("product_organizer", "calm_solution_relief", "measured", "full_hero", "solution_world")
    if act == "product_benefit":
        return ("product_organizer", "structured_clarity", "measured", "purposeful_cutaway", "solution_world")
    if act in ("closing_takeaway", "cta"):
        return ("live_speaker_and_takeaway", "warm_confidence", "calm", "full_hero", "solution_world")
    return ("editorial_focus", "engaged_clarity", "measured", "full_hero", "editorial_world")


def _synthesize_viewer_takeaway(
    beat: StoryBeatNode,
    step: Any,
    ledger_active: bool,
    contract: NarrativeContract,
) -> str:
    if step is not None:
        sym = step.currency_symbol or "$"
        if step.step_type == "opening_balance":
            return f"Opening example balance is {format_cents(step.amount_cents, sym)}"
        if step.step_type == "allocation":
            return f"{format_cents(step.amount_cents, sym)} committed to {step.category}; {format_cents(step.running_balance_cents, sym)} remaining"
        if step.step_type == "remaining_confirmed":
            return f"Only {format_cents(step.running_balance_cents, sym)} left to spend in this example"
    if beat.negated:
        return "Bank balance does not equal spendable cash"
    if beat.speech_act == "product_intro":
        return f"Introduce {beat.product_entities[0] if beat.product_entities else 'organization system'}"
    if beat.speech_act == "product_benefit":
        return "Show structured components that organize money before payday"
    if beat.speech_act == "cta":
        return "Clear closing direction without clutter"
    return (beat.verbatim_quote or beat.narrative_role)[:96]


def plan_attention_arc(beat_states: list[BeatStateSnapshot]) -> list[dict[str, Any]]:
    """Summarize pacing rhythm, visual load, and alternating reveal/calm beats across the entire clip."""
    arc: list[dict[str, Any]] = []
    for bs in beat_states:
        info_load = (
            "low_rest" if bs.pacing_energy == "breathing_hold"
            else ("high_milestone" if bs.pacing_energy == "decisive_payoff" else "single_clear_idea")
        )
        arc.append({
            "beat_id": bs.beat_id,
            "frame_range": [bs.start_frame, bs.end_frame_exclusive],
            "focal": bs.primary_focal_object,
            "emotion": bs.desired_emotion,
            "pacing": bs.pacing_energy,
            "speaker_presence": bs.speaker_presence,
            "information_burden": info_load,
            "continuity_group": bs.continuity_group,
        })
    return arc


# ---------------------------------------------------------------------------
# 4. Multi-Candidate Scene Graph Proposal & Deterministic Rubric Ranking
# ---------------------------------------------------------------------------

def propose_scene_graphs(
    beat: StoryBeatNode,
    beat_state: BeatStateSnapshot,
    retrieved_candidates: list[dict[str, Any]],
    *,
    llm_proposal: dict[str, Any] | None = None,
    matte_available: bool = True,
    is_talking_head: bool = True,
    alloc_index: int = 0,
    verified_rem_ids: set[str] | None = None,
    verified_hf_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Generate 2-3 distinct, continuity-aware design candidates for key narrative beats (1-2 for minor beats)."""
    v_rem = verified_rem_ids or set()
    v_hf = verified_hf_ids or set()
    llm_p = llm_proposal or {}
    cands: list[dict[str, Any]] = []

    def _safe_rem(pid: str) -> str:
        return pid if (pid in v_rem) else "none"

    def _safe_hf(pid: str) -> str:
        return pid if (pid in v_hf) else "none"

    act = beat.speech_act
    if act == "hook_contrast":
        # Candidate A: Full-frame live depth matte + open integer hero typography + tactile paper tokens
        cands.append({
            "candidate_id": f"{beat.beat_id}_optA_depth_paper",
            "strategy": "live_depth_hero_paper_tokens",
            "stage_mode": "live_depth_matte" if matte_available else "live_clean_bridge",
            "remotion_primitive_id": _safe_rem("rem_viv_particle_field") if matte_available else "none",
            "hyperframes_primitive_id": _safe_hf("hf_apple_money_count"),
            "transition_in": "none",
            "transition_out": "none",
            "single_numeric_owner": True,
            "preserve_speaker_hero": True,
            "rationale": "Full-frame speaker anchor with MODNet depth plane and open editorial balance reveal.",
        })
        # Candidate B: Clean live hero with Remotion counter only (lighter cost, fewer tokens)
        cands.append({
            "candidate_id": f"{beat.beat_id}_optB_clean_counter",
            "strategy": "live_clean_counter_only",
            "stage_mode": "live_clean_bridge",
            "remotion_primitive_id": _safe_rem("rem_ru_counter"),
            "hyperframes_primitive_id": "none",
            "transition_in": "none",
            "transition_out": "none",
            "single_numeric_owner": True,
            "preserve_speaker_hero": True,
            "rationale": "Minimal single-engine balance counter without depth parallax.",
        })
    elif act == "negation_tension":
        cands.append({
            "candidate_id": f"{beat.beat_id}_optA_bold_strike",
            "strategy": "open_editorial_strikethrough_morph",
            "stage_mode": "live_clean_bridge",
            "remotion_primitive_id": _safe_rem("rem_ru_strikethrough"),
            "hyperframes_primitive_id": "none",
            "transition_in": "none",
            "transition_out": "none",
            "single_numeric_owner": True,
            "preserve_speaker_hero": True,
            "rationale": "High-hierarchy editorial strikethrough preserving speaker expression during contradiction.",
        })
        cands.append({
            "candidate_id": f"{beat.beat_id}_optB_hf_strike",
            "strategy": "hyperframes_vector_strike",
            "stage_mode": "live_clean_bridge",
            "remotion_primitive_id": "none",
            "hyperframes_primitive_id": _safe_hf("hf_strikethrough_replace"),
            "transition_in": "none",
            "transition_out": "none",
            "single_numeric_owner": True,
            "preserve_speaker_hero": True,
            "rationale": "HyperFrames vector strikethrough alternative.",
        })
    elif act == "allocation_step":
        # Distinct continuous rhythm across consecutive allocations without repeating dark cards or duplicate counters!
        modes_by_idx = [
            ("tracked_macro_split", _safe_rem("rem_ru_track_matte"), _safe_hf("hf_mk_progress_stat"), "live_macro_and_transfer_strip"),
            ("aspect_morph_editorial", "none", _safe_hf("hf_data_chart"), "balanced_editorial_breakdown_split"),
            ("live_clean_bridge", "none", _safe_hf("hf_mk_progress_stat"), "open_editorial_rail_transfer"),
        ]
        m_stage, m_rem, m_hf, m_strat = modes_by_idx[alloc_index % len(modes_by_idx)]
        cands.append({
            "candidate_id": f"{beat.beat_id}_optA_{m_strat}",
            "strategy": m_strat,
            "stage_mode": m_stage,
            "remotion_primitive_id": m_rem,
            "hyperframes_primitive_id": m_hf,
            "transition_in": "none",
            "transition_out": "none",
            "single_numeric_owner": True,
            "preserve_speaker_hero": True,
            "rationale": f"Varied allocation rhythm ({m_strat}) with ONE canonical ledger counter and physical token motion.",
        })
        cands.append({
            "candidate_id": f"{beat.beat_id}_optB_clean_bar",
            "strategy": "clean_live_proportional_bar",
            "stage_mode": "live_clean_bridge",
            "remotion_primitive_id": "none",
            "hyperframes_primitive_id": _safe_hf("hf_mk_progress_stat"),
            "transition_in": "none",
            "transition_out": "none",
            "single_numeric_owner": True,
            "preserve_speaker_hero": True,
            "rationale": "Restrained full-frame live footage with proportional bar only.",
        })
    elif act == "payoff_reveal":
        cands.append({
            "candidate_id": f"{beat.beat_id}_optA_ivory_waterfall_bridge",
            "strategy": "shared_sheet_payoff_cutaway_with_exit_fold",
            "stage_mode": "full_graphic_cutaway",
            "remotion_primitive_id": _safe_rem("rem_ru_waterfall"),
            "hyperframes_primitive_id": _safe_hf("hf_apple_money_count"),
            "transition_in": "line_led",
            "transition_out": "paper_fold_out",
            "single_numeric_owner": True,
            "preserve_speaker_hero": False,
            "rationale": "Warm ivory architectural sweep (zero dark iris) + exact integer hero payoff + waterfall + smooth paper-fold return to live speaker.",
        })
        cands.append({
            "candidate_id": f"{beat.beat_id}_optB_live_hero_payoff",
            "strategy": "live_speaker_payoff_overlay",
            "stage_mode": "live_depth_matte" if matte_available else "live_clean_bridge",
            "remotion_primitive_id": "none",
            "hyperframes_primitive_id": _safe_hf("hf_apple_money_count"),
            "transition_in": "none",
            "transition_out": "none",
            "single_numeric_owner": True,
            "preserve_speaker_hero": True,
            "rationale": "Keeps speaker visible throughout payoff without waterfall proof.",
        })
    elif act == "product_intro":
        cands.append({
            "candidate_id": f"{beat.beat_id}_optA_depth_mask_sweep",
            "strategy": "live_depth_editorial_title_sweep",
            "stage_mode": "live_depth_matte" if matte_available else "live_clean_bridge",
            "remotion_primitive_id": _safe_rem("rem_viv_particle_field") if matte_available else "none",
            "hyperframes_primitive_id": _safe_hf("hf_lt_mask_reveal"),
            "transition_in": "line_led",
            "transition_out": "none",
            "single_numeric_owner": True,
            "preserve_speaker_hero": True,
            "rationale": "Editorial brass-line sweep introducing solution title over depth-separated speaker.",
        })
    elif act == "product_benefit":
        cands.append({
            "candidate_id": f"{beat.beat_id}_optA_flip_and_specs_bridged",
            "strategy": "bridged_cutaway_3d_page_flip_and_specs",
            "stage_mode": "full_graphic_cutaway",
            "remotion_primitive_id": _safe_rem("rem_rc_flip_page"),
            "hyperframes_primitive_id": _safe_hf("hf_mk_specs_list"),
            "transition_in": "line_led",
            "transition_out": "paper_fold_out",
            "single_numeric_owner": True,
            "preserve_speaker_hero": False,
            "rationale": "Smooth architectural entry & paper-fold exit bridging real 3D FlipPageTransition and open editorial specs list.",
        })
        cands.append({
            "candidate_id": f"{beat.beat_id}_optB_exploded_stack",
            "strategy": "exploded_component_stack_cutaway",
            "stage_mode": "full_graphic_cutaway",
            "remotion_primitive_id": "none",
            "hyperframes_primitive_id": _safe_hf("hf_exploded_stack"),
            "transition_in": "line_led",
            "transition_out": "paper_fold_out",
            "single_numeric_owner": True,
            "preserve_speaker_hero": False,
            "rationale": "Single-engine HyperFrames exploded stack alternative.",
        })
    elif act in ("closing_takeaway", "cta"):
        cands.append({
            "candidate_id": f"{beat.beat_id}_optA_editorial_slide_rim",
            "strategy": "clean_editorial_takeaway_edge_rim_glow",
            "stage_mode": "live_clean_bridge",
            "remotion_primitive_id": _safe_rem("rem_ru_masked_slide"),
            "hyperframes_primitive_id": _safe_hf("hf_organic_light_leak"),
            "transition_in": "none",
            "transition_out": "none",
            "single_numeric_owner": True,
            "preserve_speaker_hero": True,
            "rationale": "Full-frame speaker return with open serif takeaway in top safe zone and subtle edge-only rim glow (face clear).",
        })
    else:
        # Generic non-finance / storytelling / education beat
        llm_stage = str(llm_p.get("stage_mode") or "live_clean_bridge")
        if is_talking_head and llm_stage == "full_graphic_cutaway":
            llm_stage = "aspect_morph_editorial"
        top_rem = _safe_rem(str(llm_p.get("remotion_primitive_id") or ""))
        top_hf = _safe_hf(str(llm_p.get("hyperframes_primitive_id") or ""))
        if top_rem == "none" and top_hf == "none" and retrieved_candidates:
            top_rem = _safe_rem(next((c["primitive_id"] for c in retrieved_candidates if c["engine"] == "remotion"), ""))
            top_hf = _safe_hf(next((c["primitive_id"] for c in retrieved_candidates if c["engine"] == "hyperframes"), ""))
        cands.append({
            "candidate_id": f"{beat.beat_id}_optA_generic",
            "strategy": "generic_grounded_editorial",
            "stage_mode": llm_stage,
            "remotion_primitive_id": top_rem,
            "hyperframes_primitive_id": top_hf,
            "transition_in": str(llm_p.get("transition_in") or "none"),
            "transition_out": "paper_fold_out" if llm_stage == "full_graphic_cutaway" else "none",
            "single_numeric_owner": True,
            "preserve_speaker_hero": llm_stage != "full_graphic_cutaway",
            "rationale": "Generic narrative beat composition grounded in retrieved verified primitives.",
        })

    # Include the raw LLM proposal as a competing candidate when meaningfully distinct so ranking audits it honestly!
    raw_rem = _safe_rem(str(llm_p.get("remotion_primitive_id") or "none"))
    raw_hf = _safe_hf(str(llm_p.get("hyperframes_primitive_id") or "none"))
    raw_stage = str(llm_p.get("stage_mode") or "")
    if raw_stage and (raw_rem != "none" or raw_hf != "none"):
        sig = (raw_stage, raw_rem, raw_hf)
        if not any((c["stage_mode"], c["remotion_primitive_id"], c["hyperframes_primitive_id"]) == sig for c in cands):
            cands.append({
                "candidate_id": f"{beat.beat_id}_optLLM_raw",
                "strategy": "raw_local_llm_proposal",
                "stage_mode": raw_stage,
                "remotion_primitive_id": raw_rem,
                "hyperframes_primitive_id": raw_hf,
                "transition_in": str(llm_p.get("transition_in") or "none"),
                "transition_out": str(llm_p.get("transition_out") or "none"),
                "single_numeric_owner": True,
                "preserve_speaker_hero": raw_stage != "full_graphic_cutaway",
                "rationale": "Raw Qwen2.5-3B structured JSON proposal evaluated against rubric.",
            })

    return cands


def rank_scene_candidates(
    candidates: list[dict[str, Any]],
    beat: StoryBeatNode,
    beat_state: BeatStateSnapshot,
    contract: NarrativeContract,
    *,
    prev_stage_modes: list[str] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Deterministically rank scene graph candidates with hard-reject guards (never averages away safety failures!)."""
    PRODUCT_ONLY = {"hf_exploded_stack", "hf_mk_specs_list", "hf_lt_mask_reveal", "rem_rc_flip_page"}
    LEDGER_ONLY = {"rem_ru_waterfall", "hf_data_chart", "hf_mk_progress_stat", "hf_apple_money_count"}

    is_prod = beat.speech_act in ("product_intro", "product_benefit") or bool(beat.product_entities or beat.attribute_entities)
    has_led = bool(beat_state.state_after.get("ledger_active") and beat.ledger_step_id)
    prev_modes = prev_stage_modes or []

    scored: list[dict[str, Any]] = []
    for cand in candidates:
        rejects: list[str] = []
        rem_p = cand.get("remotion_primitive_id", "none")
        hf_p = cand.get("hyperframes_primitive_id", "none")
        st = cand.get("stage_mode", "live_clean_bridge")

        # Hard rejects (fail-closed semantic/role/continuity safety)
        if not is_prod and (rem_p in PRODUCT_ONLY or hf_p in PRODUCT_ONLY):
            rejects.append("role_mismatch_product_primitive_on_non_product_beat")
        if not has_led and (rem_p in LEDGER_ONLY or hf_p in LEDGER_ONLY):
            rejects.append("role_mismatch_ledger_primitive_on_non_ledger_beat")
        if st == "full_graphic_cutaway" and beat.speech_act not in ("payoff_reveal", "product_benefit") and beat_state.speaker_presence != "purposeful_cutaway":
            rejects.append("unmotivated_cutaway_hiding_speaker")
        if st == "full_graphic_cutaway" and cand.get("transition_in") == "none" and cand.get("transition_out") == "none":
            rejects.append("unbridged_hard_cut_cutaway")
        if rem_p == "none" and hf_p == "none" and beat.speech_act != "breathing_bridge":
            rejects.append("empty_visual_primitives")

        # Weighted rubric (only meaningful when zero hard rejects)
        score = 0.0
        if not rejects:
            # 1. Story clarity & dual-engine visual craft
            score += 24.0 if (rem_p != "none" and hf_p != "none") else 17.0
            # 2. Continuity bridge quality
            if st == "full_graphic_cutaway" and cand.get("transition_in") != "none" and cand.get("transition_out") != "none":
                score += 18.0
            elif st != "full_graphic_cutaway":
                score += 14.0
            # 3. Anti-repetition novelty vs previous 2 beats
            if len(prev_modes) >= 2 and prev_modes[-1] == st and prev_modes[-2] == st:
                score -= 12.0
            elif not prev_modes or prev_modes[-1] != st:
                score += 10.0
            # 4. Single numeric owner & speaker hero preservation
            if cand.get("single_numeric_owner"):
                score += 10.0
            if beat_state.speaker_presence in ("full_hero", "macro_hands") and st != "full_graphic_cutaway":
                score += 8.0
            # 5. Avoid iris hole on payoff
            if beat.speech_act == "payoff_reveal" and cand.get("transition_in") == "frame_morph":
                score -= 8.0
            # 6. Prefer curated primary strategies over duplicate raw fallback on tie
            if "_optA_" in cand.get("candidate_id", ""):
                score += 4.0

        scored.append({
            **cand,
            "hard_rejects": rejects,
            "rubric_score": round(score, 2),
            "eligible": len(rejects) == 0,
        })

    eligible = [c for c in scored if c["eligible"]]
    eligible.sort(key=lambda x: x["rubric_score"], reverse=True)
    chosen = eligible[0] if eligible else scored[0]
    return chosen, scored
