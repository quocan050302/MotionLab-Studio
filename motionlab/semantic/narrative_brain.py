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

NARRATIVE_BRAIN_SCHEMA_VERSION = "3.0.0"

_INJECTION_PATTERNS = (
    re.compile(r"(?i)ignore\s+(?:all\s+)?(?:previous|prior)\s+instructions?"),
    re.compile(r"(?i)system\s*:\s*you\s+are"),
    re.compile(r"(?i)<\s*script"),
    re.compile(r"(?i)\beval\s*\("),
    re.compile(r"(?i)(?:\.\./|\.\.\\)"),
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
    r"(?i)\b(might|could|may|possibly|potentially|let'?s\s+say|suppose|for\s+example|hypothetically|if\s+you\s+had)\b"
)
_SLOGAN_MONEY_RE = re.compile(
    r"(?i)\b(every\s+dollar\s+a\s+purpose|time\s+is\s+money|dollar\s+for\s+dollar|worth\s+every\s+penny|burning\s+[\$€£]?\d[\d,]*|throwing\s+money\s+away)\b"
)
_EXPLOSION_METAPHOR_RE = re.compile(
    r"(?i)\b(explosion|exploded|blew\s+up|bomb|fireworks)\b"
)
_QUIET_EMOTIONAL_RE = re.compile(
    r"(?i)\b(whisper|quietly|softly|breathe|take\s+a\s+breath|pause\s+and\s+think|gently|honestly|in\s+silence)\b"
)


# ---------------------------------------------------------------------------
# 1. Local Versioned Creative Playbook (Defect-Derived & Human-Approved Rules)
# ---------------------------------------------------------------------------

PLAYBOOK_VERSION = "3.0.0"

DEFAULT_PLAYBOOK_RULES: list[dict[str, Any]] = [
    {
        "rule_id": "PB-01",
        "category": "numeric_integrity",
        "status": "HUMAN_APPROVED",
        "provenance_class": "defect_derived_mechanical",
        "scope": "global_generic",
        "origin": "defect_ledger:D01_R1,D05_R1",
        "niches": ["finance", "product", "education"],
        "rule": "Never interpolate arbitrary fractional cents when spoken amounts are whole integers; lock integer-dollar display mode unless source evidence explicitly speaks fractional cents.",
    },
    {
        "rule_id": "PB-02",
        "category": "single_numeric_owner",
        "status": "HUMAN_APPROVED",
        "provenance_class": "defect_derived_mechanical",
        "scope": "global_generic",
        "origin": "defect_ledger:D03_R1",
        "niches": ["finance"],
        "rule": "Maintain ONE canonical visible currency counter per beat; never pair an independent bottom ribbon counter against an active top allocation counter displaying conflicting intermediate numbers.",
    },
    {
        "rule_id": "PB-03",
        "category": "continuous_bridges",
        "status": "HUMAN_APPROVED",
        "provenance_class": "defect_derived_mechanical",
        "scope": "global_generic",
        "origin": "defect_ledger:D06_R1,D11_R3",
        "niches": ["finance", "product", "storytelling", "education", "lifestyle", "cinematic"],
        "rule": "Every transition between full-frame graphic cutaway and live footage MUST have an explicit rendered incoming or outgoing bridge (paper fold, soft mask unveil, or shared-element morph) rather than an unmotivated hard cut.",
    },
    {
        "rule_id": "PB-04",
        "category": "editorial_hierarchy",
        "status": "HUMAN_APPROVED",
        "provenance_class": "human_approved_aesthetic",
        "scope": "global_generic",
        "origin": "editorial_rubric:VQ01",
        "niches": ["finance", "product", "storytelling", "education", "lifestyle", "cinematic"],
        "rule": "Avoid repetitive dark rounded card templates and tiny speaker inset boxes; keep live speaker as full-frame human anchor and reserve cutaways for high-value teaching milestones.",
    },
    {
        "rule_id": "PB-05",
        "category": "semantic_sound_integrity",
        "status": "HUMAN_APPROVED",
        "provenance_class": "defect_derived_mechanical",
        "scope": "global_generic",
        "origin": "defect_ledger:D07_R3,D08_R3",
        "niches": ["finance", "product", "storytelling", "education", "lifestyle", "cinematic"],
        "rule": "Match sound role to actual visual/semantic context (never engine rev on budget gas, never cash flick on negation or non-monetary counts, never silent cue drops, protect 1-4kHz dialogue).",
    },
    {
        "rule_id": "PB-06",
        "category": "face_contrast_protection",
        "status": "HUMAN_APPROVED",
        "provenance_class": "defect_derived_mechanical",
        "scope": "global_generic",
        "origin": "defect_ledger:D08_R1",
        "niches": ["finance", "product", "storytelling", "education", "lifestyle", "cinematic"],
        "rule": "Never apply bright full-frame light leaks over the speaker's face (max edge glow <= 0.22 opacity with clear center face mask) and keep all text clear of eyes/mouth and phone safe zones.",
    },
    {
        "rule_id": "PB-07",
        "category": "product_asset_truth",
        "status": "HUMAN_APPROVED",
        "provenance_class": "defect_derived_mechanical",
        "scope": "global_generic",
        "origin": "defect_ledger:D05_R3,D06_R3",
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
    # R5.R3 Multimodal Grounding & Honest Timestamp Provenance (A2 / D03)
    timestamp_quality: str = "WORD_ALIGNED_APPROX"  # WORD_ALIGNED_APPROX | SEGMENT_APPROX | BEAT_LEVEL_APPROX
    word_timestamps_exact: bool = True
    source_audio_hash: str = ""
    language_probability: float = 0.0
    asr_model_id: str = ""
    provenance_summary: dict[str, str] = field(default_factory=dict)
    roi_frame_index: list[dict[str, Any]] = field(default_factory=list)
    index_hash: str = ""

    def query_visible_at_word(self, word_id: int) -> dict[str, Any]:
        """Cross-modal query: what visual anchors/safe space are observed when word_id is spoken?"""
        wref = next((w for w in self.word_refs if int(w.get("word_id", -1)) == int(word_id)), None)
        if wref is None:
            return {"found": False, "provenance": "UNKNOWN"}
        sf, ef = int(wref["start_frame"]), int(wref["end_frame_exclusive"])
        return {
            "found": True,
            "word_id": int(word_id),
            "word": wref["word"],
            "frame_range": [sf, ef],
            "timestamp_quality": self.timestamp_quality,
            "free_safe_zone": self.query_free_safe_zone(sf, ef),
            "action_support": self.query_action_support(sf, ef),
            "provenance": "OBSERVED_PIXEL" if self.roi_frame_index else "ASR_WORD",
        }

    def query_free_safe_zone(self, start_frame: int, end_frame: int) -> dict[str, Any]:
        """Determine free negative space avoiding tracked face/eyes/mouth and task ROI."""
        in_win = [r for r in self.roi_frame_index if start_frame <= int(r.get("frame", -1)) < end_frame]
        if not in_win and self.roi_frame_index:
            mid = (start_frame + end_frame) // 2
            in_win = [min(self.roi_frame_index, key=lambda r: abs(int(r.get("frame", 0)) - mid))]
        face_top = 0.25
        face_bot = 0.50
        has_face = False
        for r in in_win:
            fb = r.get("face")
            if isinstance(fb, dict) and float(fb.get("conf", fb.get("confidence", 0.0))) >= 0.35:
                has_face = True
                face_top = min(face_top, float(fb.get("y", 0.25)))
                face_bot = max(face_bot, float(fb.get("y", 0.25)) + float(fb.get("h", 0.25)))
        if has_face and face_top >= 0.20:
            return {"region": "upper_negative_space", "y_min": 0.07, "y_max": round(max(0.18, face_top - 0.02), 3), "provenance": "OBSERVED_PIXEL"}
        return {"region": "upper_editorial_band", "y_min": 0.07, "y_max": 0.24, "provenance": "OBSERVED_PIXEL" if has_face else "ILLUSTRATIVE"}

    def query_action_support(self, start_frame: int, end_frame: int) -> dict[str, Any]:
        """Query whether observed hand/prop action energy supports diegetic interaction in [start_frame, end_frame)."""
        in_win = [r for r in self.roi_frame_index if start_frame <= int(r.get("frame", -1)) < end_frame]
        confs = [
            float(r["task"].get("conf", r["task"].get("confidence", 0.0)))
            for r in in_win
            if isinstance(r.get("task"), dict)
        ]
        max_c = round(max(confs), 3) if confs else 0.0
        return {
            "observed_contact_supported": bool(max_c >= 0.60),
            "max_task_confidence": max_c,
            "frames_sampled": len(in_win),
            "provenance": "OBSERVED_PIXEL" if in_win else "UNKNOWN",
        }

    def query_available_assets(self, product_name: str | None) -> list[dict[str, Any]]:
        """Strict normalized exact match against verified product assets (zero empty substring matching!)."""
        if not product_name or not product_name.strip():
            return []
        target = re.sub(r"\s+", " ", product_name.strip().lower())
        hits: list[dict[str, Any]] = []
        for a in self.available_product_assets:
            if not (a.get("authorized") and a.get("verified")):
                continue
            pname = re.sub(r"\s+", " ", str(a.get("product_name", "")).strip().lower())
            pid = re.sub(r"\s+", " ", str(a.get("product_id", "")).strip().lower())
            aliases = [re.sub(r"\s+", " ", str(x).strip().lower()) for x in (a.get("approved_aliases") or []) if str(x).strip()]
            if target and (target == pname or target == pid or target in aliases):
                hits.append(a)
        return hits

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

    # Determine honest timestamp provenance (D03 fix: NEVER claim exact word timestamps on coarse/beat-wide spans!)
    distinct_word_starts = len({round(float(w["start_s"]), 3) for w in word_refs})
    all_beat_wide_dup = (len(word_refs) >= 6 and distinct_word_starts <= max(1, len(story_graph.beats)))
    if all_beat_wide_dup or getattr(transcript, "model_id", "") == "story_graph_beat_approx":
        ts_quality = "BEAT_LEVEL_APPROX"
        wt_exact = False
    elif not bool(getattr(transcript, "word_timestamps_exact", True)):
        ts_quality = "SEGMENT_APPROX"
        wt_exact = False
    else:
        ts_quality = "WORD_ALIGNED_APPROX"
        wt_exact = True

    compact_roi_idx = [
        {
            "frame": int(k["frame"]),
            "face": k.get("face") or k.get("face_box"),
            "task": k.get("task") or k.get("task_box"),
        }
        for k in kfs
        if isinstance(k, dict) and "frame" in k
    ]
    prov_summary = {
        "speech_text": "ASR_WORD" if wt_exact else "DERIVED",
        "visual_rois": "OBSERVED_PIXEL" if compact_roi_idx else "UNKNOWN",
        "silence_spans": "OBSERVED_AUDIO" if silence_spans else "DERIVED",
        "visual_motion_planning": "ILLUSTRATIVE",
        "vlm_visual_inspection": "UNKNOWN",
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
        timestamp_quality=ts_quality,
        word_timestamps_exact=wt_exact,
        source_audio_hash=str(getattr(transcript, "source_audio_hash", "") or ""),
        language_probability=round(float(getattr(transcript, "language_probability", 0.0)), 4),
        asr_model_id=str(getattr(transcript, "model_id", "") or ""),
        provenance_summary=prov_summary,
        roi_frame_index=compact_roi_idx,
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

    # Infer niche honestly from multi-cue semantic evidence (never force finance or product on unrelated clips!)
    if requested_niche and requested_niche not in ("auto", ""):
        niche = requested_niche.lower()
        nconf = 0.92
    elif not full_low.strip():
        niche = "silent_art"
        nconf = 0.90
    elif led.active:
        niche = "finance"
        nconf = 0.94
    elif any(k in full_low for k in ("boss fight", "boss combo", "stamina bar", "hud", "respawn", "headshot", "combo meter", "minimap", "gameplay")):
        niche = "gameplay"
        nconf = 0.90
    elif any(k in full_low for k in ("inhale", "exhale", "breathe deeply", "mindful", "meditation", "calmly let", "nervous system", "posture balance")):
        niche = "wellness"
        nconf = 0.86
    elif any(k in full_low for k in ("punchline", "joke", "hilarious", "wait for it", "comedy", "laughed so hard")):
        niche = "comedy"
        nconf = 0.84
    elif any(k in full_low for k in ("step one", "step 1", "step 2", "tutorial", "click here", "terminal", "install", "how to", "lesson", "first,", "second,", "third,", "photosynthesis", "explain how")):
        niche = "education" if face_ratio >= 0.45 else "tutorial"
        nconf = 0.86
    elif any(k in full_low for k in ("api latency", "deploy", "saas", "workflow automation", "cloud dashboard", "benchmark fps", "compiler")):
        niche = "tech"
        nconf = 0.85
    elif story_graph.product_name or story_graph.product_attributes:
        has_story_intro = any(k in full_low for k in ("years ago", "1969", "remember when", "story"))
        niche = "mixed" if has_story_intro else "product"
        nconf = 0.86
    elif any(k in full_low for k in ("story", "happened", "remember when", "years ago", "1969", "journey", "whisper", "quietly", "history", "century")):
        niche = "storytelling"
        nconf = 0.82
    else:
        niche = "cinematic"
        nconf = 0.72

    # Narrative structure pattern
    if led.active:
        pattern = "financial_conservation"
    elif niche == "gameplay":
        pattern = "neutral_preserve"
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

    # Strict exact normalized match for owner-authorized verified product assets (fixes D06 substring leak!)
    matched_prod_assets = evidence.query_available_assets(story_graph.product_name)
    has_auth_asset = bool(matched_prod_assets)

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
        is_explosion_talk = bool(_EXPLOSION_METAPHOR_RE.search(q_low))
        is_emotional_quiet = bool(_QUIET_EMOTIONAL_RE.search(q_low))
        act_sup = evidence.query_action_support(b.start_frame, b.end_frame_exclusive)
        has_obs_contact = bool(
            (b.visual_observations.get("has_task") and float(b.visual_observations.get("task_conf", 0.0)) >= 0.62)
            or act_sup.get("observed_contact_supported", False)
        )
        disallowed = _compute_disallowed_sound_families(
            b, led.active, is_budget_gas, is_budget_groceries, is_slogan_money, is_non_money_count, is_deficit
        )
        if is_explosion_talk and not has_obs_contact:
            disallowed = sorted(set(disallowed + ["explosion_sfx", "diegetic_explosion"]))

        sound_ctx = {
            "speech_act": b.speech_act,
            "negated": b.negated,
            "is_budget_gas": is_budget_gas,
            "is_vehicle_gas": is_vehicle_gas,
            "is_budget_groceries": is_budget_groceries,
            "is_slogan_money": is_slogan_money,
            "is_non_monetary_count": is_non_money_count,
            "is_deficit": is_deficit,
            "is_explosion_without_observed_action": bool(is_explosion_talk and not has_obs_contact),
            "is_emotional_quiet": is_emotional_quiet,
            "is_hypothetical": contract.is_hypothetical_example,
            "has_observed_hand_contact": has_obs_contact,
            "prefer_silence": bool(b.speech_act in ("breathing_bridge", "cta") or is_emotional_quiet),
            "disallowed_families": disallowed,
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
# 4. Multi-Candidate Scene Graph Proposal, Fail-Closed Ranker & Global Beam Search (A4 / D01 / D02)
# ---------------------------------------------------------------------------

NO_ELIGIBLE_CANDIDATE_ID = "NO_ELIGIBLE_CANDIDATE"


@dataclass
class SelectedScenePlan:
    """Causal director decision object compiled directly into SceneSpec/LayerSpec/TransitionSpec/SoundIntent (Fixes D01)."""
    scene_id: str
    beat_ids: list[str]
    chosen_candidate_id: str
    strategy: str
    stage_mode: str
    remotion_primitive_id: str
    hyperframes_primitive_id: str
    hf_layers: list[dict[str, Any]]  # Ordered multi-HF-layer specs (supports 0, 1, or >=2 distinct HF layers!)
    transition_in: str
    transition_out: str
    continuity_group: str
    viewer_takeaway: str
    evidence_word_ids: list[int]
    selection_rationale: str
    candidate_provenance: str  # "LLM_DIRECT" | "HYBRID_GLOBAL_RANKED" | "REPLAN_SAFE_FALLBACK" | "CAUSAL_OVERRIDE"
    rubric_score: float
    global_path_score: float
    all_candidates_considered: list[dict[str, Any]] = field(default_factory=list)
    replan_attempts: int = 0
    intervention_type: str = "none"  # "none" | "safety" | "factual" | "artistic" | "resource"

    def to_dict(self) -> dict[str, Any]:
        return sanitize_native(asdict(self))


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
    """Generate 2-3 meaningfully distinct, continuity-aware visual candidates per major beat (zero _optA_ bias!)."""
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
        # Strategy 1 (Multi-HF-layer true depth separation!): Paper tokens behind MODNet matte + crisp foreground typography above matte
        if matte_available and "hf_apple_money_count" in v_hf:
            cands.append({
                "candidate_id": f"{beat.beat_id}_depth_multilayer_paper",
                "strategy": "live_depth_dual_hf_paper_and_hero_type",
                "stage_mode": "live_depth_matte",
                "remotion_primitive_id": _safe_rem("rem_viv_particle_field"),
                "hyperframes_primitive_id": "hf_apple_money_count",
                "hf_layers": [
                    {
                        "layer_suffix": "hf_bg_tokens",
                        "primitive_id": "hf_apple_money_count",
                        "layer_role": "background_graphics",
                        "z_order": 15,
                        "param_overrides": {"render_mode": "paper_tokens_only", "token_count": 4, "dark_plate": True},
                    },
                    {
                        "layer_suffix": "hf_fg_hero",
                        "primitive_id": "hf_apple_money_count",
                        "layer_role": "foreground_graphics",
                        "z_order": 35,
                        "param_overrides": {"render_mode": "typography_only", "token_count": 0, "dark_plate": True},
                    },
                ],
                "transition_in": "none",
                "transition_out": "none",
                "single_numeric_owner": True,
                "preserve_speaker_hero": True,
                "visual_clarity_rating": 9.4,
                "continuity_rating": 9.0,
                "rationale": "True 2-layer depth separation: curved paper tokens float behind MODNet foreground subject while clean hero amount floats above in upper negative space.",
            })
        # Strategy 2: Single-HF open typography + depth plane
        cands.append({
            "candidate_id": f"{beat.beat_id}_depth_single_hf",
            "strategy": "live_depth_hero_paper_tokens",
            "stage_mode": "live_depth_matte" if matte_available else "live_clean_bridge",
            "remotion_primitive_id": _safe_rem("rem_viv_particle_field") if matte_available else "none",
            "hyperframes_primitive_id": _safe_hf("hf_apple_money_count"),
            "hf_layers": [],
            "transition_in": "none",
            "transition_out": "none",
            "single_numeric_owner": True,
            "preserve_speaker_hero": True,
            "visual_clarity_rating": 8.6,
            "continuity_rating": 8.4,
            "rationale": "Full-frame speaker anchor with depth plane and unified balance hero.",
        })
        # Strategy 3: Minimal single-engine live Remotion counter
        cands.append({
            "candidate_id": f"{beat.beat_id}_clean_counter",
            "strategy": "live_clean_counter_only",
            "stage_mode": "live_clean_bridge",
            "remotion_primitive_id": _safe_rem("rem_ru_counter"),
            "hyperframes_primitive_id": "none",
            "hf_layers": [],
            "transition_in": "none",
            "transition_out": "none",
            "single_numeric_owner": True,
            "preserve_speaker_hero": True,
            "visual_clarity_rating": 7.8,
            "continuity_rating": 7.6,
            "rationale": "Minimal single-engine live balance counter without depth parallax.",
        })
    elif act == "negation_tension":
        cands.append({
            "candidate_id": f"{beat.beat_id}_bold_strike",
            "strategy": "open_editorial_strikethrough_morph",
            "stage_mode": "live_clean_bridge",
            "remotion_primitive_id": _safe_rem("rem_ru_strikethrough"),
            "hyperframes_primitive_id": "none",
            "hf_layers": [],
            "transition_in": "none",
            "transition_out": "none",
            "single_numeric_owner": True,
            "preserve_speaker_hero": True,
            "visual_clarity_rating": 9.2,
            "continuity_rating": 8.8,
            "rationale": "Open atmospheric editorial strikethrough preserving speaker expression during contradiction.",
        })
        cands.append({
            "candidate_id": f"{beat.beat_id}_hf_strike",
            "strategy": "hyperframes_vector_strike",
            "stage_mode": "live_clean_bridge",
            "remotion_primitive_id": "none",
            "hyperframes_primitive_id": _safe_hf("hf_strikethrough_replace"),
            "hf_layers": [],
            "transition_in": "none",
            "transition_out": "none",
            "single_numeric_owner": True,
            "preserve_speaker_hero": True,
            "visual_clarity_rating": 8.0,
            "continuity_rating": 8.0,
            "rationale": "HyperFrames vector strikethrough alternative.",
        })
    elif act == "allocation_step":
        # Offer distinct visual allocation approaches; for short beats (<2.1s, fixes R5.R4 D07),
        # prefer continuous live money-world over a brief jarring full-screen chart flip!
        beat_dur_s = max(0.2, float(beat.end_s) - float(beat.start_s))
        is_short_alloc = beat_dur_s < 2.1
        cands.append({
            "candidate_id": f"{beat.beat_id}_macro_transfer_strip",
            "strategy": "live_macro_and_transfer_strip",
            "stage_mode": "tracked_macro_split",
            "remotion_primitive_id": _safe_rem("rem_ru_track_matte"),
            "hyperframes_primitive_id": _safe_hf("hf_mk_progress_stat"),
            "hf_layers": [],
            "transition_in": "none",
            "transition_out": "none",
            "single_numeric_owner": True,
            "preserve_speaker_hero": True,
            "layout_variant": "top_strip",
            "visual_clarity_rating": 9.1 if alloc_index == 0 else 8.4,
            "continuity_rating": 8.9,
            "rationale": "Live detail aperture macro paired with proportional deduction strip and single canonical ledger owner.",
        })
        cands.append({
            "candidate_id": f"{beat.beat_id}_editorial_breakdown_morph",
            "strategy": "balanced_editorial_breakdown_split",
            "stage_mode": "aspect_morph_editorial",
            "remotion_primitive_id": "none",
            "hyperframes_primitive_id": _safe_hf("hf_data_chart"),
            "hf_layers": [],
            "transition_in": "none",
            "transition_out": "none",
            "single_numeric_owner": True,
            "preserve_speaker_hero": True,
            "visual_clarity_rating": 7.2 if is_short_alloc else (9.1 if alloc_index == 1 else 8.3),
            "continuity_rating": 7.0 if is_short_alloc else 8.8,
            "rationale": (
                "Full-frame aspect morph chart discouraged on short allocation beat (<2.1s) to prevent jarring rapid flip (D07)."
                if is_short_alloc
                else "Balanced editorial morph framing live speaker cleanly below proportional cumulative breakdown chart."
            ),
        })
        cands.append({
            "candidate_id": f"{beat.beat_id}_open_editorial_rail",
            "strategy": "open_editorial_rail_transfer",
            "stage_mode": "live_clean_bridge",
            "remotion_primitive_id": "none",
            "hyperframes_primitive_id": _safe_hf("hf_mk_progress_stat"),
            "hf_layers": [],
            "transition_in": "none",
            "transition_out": "none",
            "single_numeric_owner": True,
            "preserve_speaker_hero": True,
            "layout_variant": "compact_editorial_rail" if (alloc_index >= 2 or not is_short_alloc) else "top_strip",
            "visual_clarity_rating": 9.3 if (is_short_alloc or alloc_index >= 2) else 8.4,
            "continuity_rating": 9.2 if is_short_alloc else 8.8,
            "rationale": "Continuous full-frame live money world with clean editorial allocation rail (zero jarring full-screen chart flash on short beats).",
        })
    elif act == "payoff_reveal":
        cands.append({
            "candidate_id": f"{beat.beat_id}_ivory_waterfall_bridge",
            "strategy": "shared_sheet_payoff_cutaway_with_exit_fold",
            "stage_mode": "full_graphic_cutaway",
            "remotion_primitive_id": _safe_rem("rem_ru_waterfall"),
            "hyperframes_primitive_id": _safe_hf("hf_apple_money_count"),
            "hf_layers": [],
            "transition_in": "sheet_sweep",
            "transition_out": "paper_fold_out",
            "single_numeric_owner": True,
            "preserve_speaker_hero": False,
            "visual_clarity_rating": 9.5,
            "continuity_rating": 9.4,
            "rationale": "Warm ivory architectural sheet sweep + discrete integer payoff + full waterfall + smooth paper-fold return to live speaker.",
        })
        cands.append({
            "candidate_id": f"{beat.beat_id}_line_led_waterfall",
            "strategy": "line_led_payoff_cutaway_with_exit_fold",
            "stage_mode": "full_graphic_cutaway",
            "remotion_primitive_id": _safe_rem("rem_ru_waterfall"),
            "hyperframes_primitive_id": _safe_hf("hf_apple_money_count"),
            "hf_layers": [],
            "transition_in": "line_led",
            "transition_out": "paper_fold_out",
            "single_numeric_owner": True,
            "preserve_speaker_hero": False,
            "visual_clarity_rating": 9.0,
            "continuity_rating": 9.0,
            "rationale": "Brass line-led aperture sweep into waterfall payoff with paper-fold exit.",
        })
        cands.append({
            "candidate_id": f"{beat.beat_id}_live_hero_payoff",
            "strategy": "live_speaker_payoff_overlay",
            "stage_mode": "live_depth_matte" if matte_available else "live_clean_bridge",
            "remotion_primitive_id": "none",
            "hyperframes_primitive_id": _safe_hf("hf_apple_money_count"),
            "hf_layers": [],
            "transition_in": "none",
            "transition_out": "none",
            "single_numeric_owner": True,
            "preserve_speaker_hero": True,
            "visual_clarity_rating": 8.0,
            "continuity_rating": 8.4,
            "rationale": "Keeps speaker visible throughout payoff without full waterfall diagram.",
        })
    elif act == "product_intro":
        cands.append({
            "candidate_id": f"{beat.beat_id}_depth_mask_sweep",
            "strategy": "live_depth_editorial_title_sweep",
            "stage_mode": "live_depth_matte" if matte_available else "live_clean_bridge",
            "remotion_primitive_id": _safe_rem("rem_viv_particle_field") if matte_available else "none",
            "hyperframes_primitive_id": _safe_hf("hf_lt_mask_reveal"),
            "hf_layers": [],
            "transition_in": "line_led",
            "transition_out": "none",
            "single_numeric_owner": True,
            "preserve_speaker_hero": True,
            "visual_clarity_rating": 9.2,
            "continuity_rating": 9.0,
            "rationale": "Editorial brass-line sweep introducing solution title over depth-separated speaker.",
        })
        cands.append({
            "candidate_id": f"{beat.beat_id}_clean_title_slide",
            "strategy": "live_clean_masked_title",
            "stage_mode": "live_clean_bridge",
            "remotion_primitive_id": _safe_rem("rem_ru_masked_slide"),
            "hyperframes_primitive_id": "none",
            "hf_layers": [],
            "transition_in": "none",
            "transition_out": "none",
            "single_numeric_owner": True,
            "preserve_speaker_hero": True,
            "visual_clarity_rating": 8.1,
            "continuity_rating": 8.2,
            "rationale": "Restrained single-engine live title slide without depth matte.",
        })
    elif act == "product_benefit":
        cands.append({
            "candidate_id": f"{beat.beat_id}_flip_and_specs_bridged",
            "strategy": "bridged_cutaway_3d_page_flip_and_specs",
            "stage_mode": "full_graphic_cutaway",
            "remotion_primitive_id": _safe_rem("rem_rc_flip_page"),
            "hyperframes_primitive_id": _safe_hf("hf_mk_specs_list"),
            "hf_layers": [],
            "transition_in": "line_led",
            "transition_out": "paper_fold_out",
            "single_numeric_owner": True,
            "preserve_speaker_hero": False,
            "visual_clarity_rating": 9.4,
            "continuity_rating": 9.2,
            "rationale": "Smooth architectural entry & paper-fold exit bridging real 3D FlipPageTransition (authorized asset or honest conceptual schematic) and open editorial specs list.",
        })
        cands.append({
            "candidate_id": f"{beat.beat_id}_exploded_stack",
            "strategy": "exploded_component_stack_cutaway",
            "stage_mode": "full_graphic_cutaway",
            "remotion_primitive_id": "none",
            "hyperframes_primitive_id": _safe_hf("hf_exploded_stack"),
            "hf_layers": [],
            "transition_in": "sheet_sweep",
            "transition_out": "paper_fold_out",
            "single_numeric_owner": True,
            "preserve_speaker_hero": False,
            "visual_clarity_rating": 8.2,
            "continuity_rating": 8.4,
            "rationale": "Single-engine HyperFrames isometric conceptual stack alternative.",
        })
    elif act in ("closing_takeaway", "cta"):
        cands.append({
            "candidate_id": f"{beat.beat_id}_editorial_slide_rim",
            "strategy": "clean_editorial_takeaway_edge_rim_glow",
            "stage_mode": "live_clean_bridge",
            "remotion_primitive_id": _safe_rem("rem_ru_masked_slide"),
            "hyperframes_primitive_id": _safe_hf("hf_organic_light_leak"),
            "hf_layers": [],
            "transition_in": "none",
            "transition_out": "none",
            "single_numeric_owner": True,
            "preserve_speaker_hero": True,
            "visual_clarity_rating": 9.2,
            "continuity_rating": 9.1,
            "rationale": "Full-frame speaker return with open serif takeaway in upper negative space and face-masked peripheral rim glow.",
        })
        cands.append({
            "candidate_id": f"{beat.beat_id}_callout_highlight",
            "strategy": "speech_synced_callout_takeaway",
            "stage_mode": "live_clean_bridge",
            "remotion_primitive_id": "none",
            "hyperframes_primitive_id": _safe_hf("hf_mk_callout_highlight"),
            "hf_layers": [],
            "transition_in": "none",
            "transition_out": "none",
            "single_numeric_owner": True,
            "preserve_speaker_hero": True,
            "visual_clarity_rating": 8.0,
            "continuity_rating": 8.1,
            "rationale": "Word-highlighting callout alternative for closing line.",
        })
    else:
        # Generic non-finance / storytelling / education / tech beat
        llm_stage = str(llm_p.get("stage_mode") or "live_clean_bridge")
        if is_talking_head and llm_stage == "full_graphic_cutaway":
            llm_stage = "aspect_morph_editorial"
        top_rem = _safe_rem(str(llm_p.get("remotion_primitive_id") or ""))
        top_hf = _safe_hf(str(llm_p.get("hyperframes_primitive_id") or ""))
        stage_only = {"rem_viv_aspect_morph", "rem_ru_parallax", "rem_gtk_frame_morph", "rem_gtk_line_led"}
        if top_rem in stage_only:
            top_rem = "none"
        if top_rem == "none" and top_hf == "none" and retrieved_candidates:
            top_rem = _safe_rem(next((c["primitive_id"] for c in retrieved_candidates if c["engine"] == "remotion" and c["primitive_id"] not in stage_only), ""))
            top_hf = _safe_hf(next((c["primitive_id"] for c in retrieved_candidates if c["engine"] == "hyperframes"), ""))
        cands.append({
            "candidate_id": f"{beat.beat_id}_generic_editorial",
            "strategy": "generic_grounded_editorial",
            "stage_mode": llm_stage,
            "remotion_primitive_id": top_rem,
            "hyperframes_primitive_id": top_hf,
            "hf_layers": [],
            "transition_in": str(llm_p.get("transition_in") or "none"),
            "transition_out": "paper_fold_out" if llm_stage == "full_graphic_cutaway" else "none",
            "single_numeric_owner": True,
            "preserve_speaker_hero": llm_stage != "full_graphic_cutaway",
            "visual_clarity_rating": 8.5,
            "continuity_rating": 8.5,
            "rationale": "Generic narrative beat composition grounded in retrieved verified primitives.",
        })
        cands.append({
            "candidate_id": f"{beat.beat_id}_generic_minimal_live",
            "strategy": "minimal_live_kinetic_slide",
            "stage_mode": "live_clean_bridge",
            "remotion_primitive_id": _safe_rem("rem_ru_masked_slide"),
            "hyperframes_primitive_id": "none",
            "hf_layers": [],
            "transition_in": "none",
            "transition_out": "none",
            "single_numeric_owner": True,
            "preserve_speaker_hero": True,
            "visual_clarity_rating": 8.0,
            "continuity_rating": 8.2,
            "rationale": "Conservative full-frame live speaker preservation with crisp kinetic typography.",
        })

    # Include the raw LLM proposal as a genuine competing candidate when meaningfully distinct!
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
                "hf_layers": [],
                "transition_in": str(llm_p.get("transition_in") or "none"),
                "transition_out": str(llm_p.get("transition_out") or "none"),
                "single_numeric_owner": True,
                "preserve_speaker_hero": raw_stage != "full_graphic_cutaway",
                "visual_clarity_rating": 8.6,
                "continuity_rating": 8.4,
                "from_raw_llm": True,
                "rationale": "Raw local LLM structured JSON proposal evaluated directly under the same rubric and hard guards.",
            })

    return cands


def _check_hard_constraints(
    cand: dict[str, Any],
    beat: StoryBeatNode,
    beat_state: BeatStateSnapshot,
) -> list[str]:
    """Fail-closed semantic, spatial, role, and continuity hard constraints (never averaged away by score!)."""
    PRODUCT_ONLY = {"hf_exploded_stack", "hf_mk_specs_list", "hf_lt_mask_reveal", "rem_rc_flip_page"}
    LEDGER_ONLY = {"rem_ru_waterfall", "hf_data_chart", "hf_mk_progress_stat", "hf_apple_money_count"}
    STAGE_ONLY_PRIMS = {"rem_viv_aspect_morph", "rem_ru_parallax", "rem_gtk_frame_morph", "rem_gtk_line_led"}
    COMPATIBLE_CUTAWAY_PAIRS = {
        ("rem_ru_waterfall", "hf_apple_money_count"),
        ("rem_ru_counter", "hf_data_chart"),
        ("rem_rc_flip_page", "hf_mk_specs_list"),
    }

    is_prod = beat.speech_act in ("product_intro", "product_benefit") or bool(beat.product_entities or beat.attribute_entities)
    has_led = bool(beat_state.state_after.get("ledger_active") and beat.ledger_step_id)

    rejects: list[str] = []
    rem_p = str(cand.get("remotion_primitive_id") or "none")
    hf_p = str(cand.get("hyperframes_primitive_id") or "none")
    hf_layers = cand.get("hf_layers") or []
    all_hf_pids = [str(l.get("primitive_id", "")) for l in hf_layers] if hf_layers else ([hf_p] if hf_p != "none" else [])
    st = str(cand.get("stage_mode") or "live_clean_bridge")

    if rem_p in STAGE_ONLY_PRIMS:
        rejects.append(f"stage_only_primitive_misused_as_layer:{rem_p}")
    if not is_prod and (rem_p in PRODUCT_ONLY or any(h in PRODUCT_ONLY for h in all_hf_pids)):
        rejects.append("role_mismatch_product_primitive_on_non_product_beat")
    if not has_led and (rem_p in LEDGER_ONLY or any(h in LEDGER_ONLY for h in all_hf_pids)):
        rejects.append("role_mismatch_ledger_primitive_on_non_ledger_beat")
    if st == "full_graphic_cutaway" and beat.speech_act not in ("payoff_reveal", "product_benefit") and beat_state.speaker_presence != "purposeful_cutaway":
        rejects.append("unmotivated_cutaway_hiding_speaker")
    if st == "full_graphic_cutaway" and cand.get("transition_in", "none") == "none" and cand.get("transition_out", "none") == "none":
        rejects.append("unbridged_hard_cut_cutaway")
    if rem_p == "none" and not all_hf_pids and beat.speech_act != "breathing_bridge":
        rejects.append("empty_visual_primitives")
    # Spatial collision guard for arbitrary Remotion + HyperFrames pair
    if rem_p != "none" and all_hf_pids:
        non_colliding = (
            rem_p in ("rem_viv_particle_field", "rem_ru_track_matte")
            or all(h == "hf_organic_light_leak" for h in all_hf_pids)
            or (st == "full_graphic_cutaway" and all((rem_p, h) in COMPATIBLE_CUTAWAY_PAIRS for h in all_hf_pids))
        )
        if not non_colliding:
            rejects.append(f"spatial_overlap_collision:{rem_p}+{all_hf_pids}")
    return rejects


def rank_scene_candidates(
    candidates: list[dict[str, Any]],
    beat: StoryBeatNode,
    beat_state: BeatStateSnapshot,
    contract: NarrativeContract,
    *,
    prev_stage_modes: list[str] | None = None,
    score_overrides: dict[str, float] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Fixes D02: Fail-closed deterministic candidate ranker with ZERO _optA_ bias & ZERO blind dual-engine bonus!

    Returns `(NO_ELIGIBLE_CANDIDATE_sentinel, scored)` when `candidates` is empty or every candidate fails hard guards
    (NEVER returns ineligible `scored[0]` and NEVER raises `IndexError`!).
    """
    prev_modes = prev_stage_modes or []
    overrides = score_overrides or {}

    if not candidates:
        return (
            {
                "candidate_id": NO_ELIGIBLE_CANDIDATE_ID,
                "status": NO_ELIGIBLE_CANDIDATE_ID,
                "eligible": False,
                "hard_rejects": ["empty_candidate_list"],
                "rubric_score": 0.0,
                "rationale": "No scene candidates were provided for this beat.",
            },
            [],
        )

    scored: list[dict[str, Any]] = []
    for cand in candidates:
        rejects = _check_hard_constraints(cand, beat, beat_state)
        rem_p = str(cand.get("remotion_primitive_id") or "none")
        hf_p = str(cand.get("hyperframes_primitive_id") or "none")
        hf_layers = cand.get("hf_layers") or []
        st = str(cand.get("stage_mode") or "live_clean_bridge")

        score = 0.0
        if not rejects:
            # 1. Story clarity, focal hierarchy & visual craft (NOT blind dual-engine count or _optA_ name!)
            score += float(cand.get("visual_clarity_rating", 8.0)) * 2.5
            score += float(cand.get("continuity_rating", 8.0)) * 1.5
            # 2. Continuity bridge quality
            if st == "full_graphic_cutaway":
                if cand.get("transition_in", "none") != "none" and cand.get("transition_out", "none") != "none":
                    score += 14.0
                else:
                    score += 6.0
            else:
                score += 11.0
            # 3. Anti-repetition novelty vs previous beats
            if len(prev_modes) >= 2 and prev_modes[-1] == st and prev_modes[-2] == st:
                score -= 14.0
            elif prev_modes and prev_modes[-1] == st:
                score -= 4.0
            elif not prev_modes or prev_modes[-1] != st:
                score += 6.0
            # 4. Single numeric owner & speaker hero preservation
            if cand.get("single_numeric_owner", True):
                score += 8.0
            if beat_state.speaker_presence in ("full_hero", "macro_hands") and st != "full_graphic_cutaway":
                score += 7.0
            # 5. Depth plane value when true foreground/background separation adds legible space
            if st == "live_depth_matte" and len(hf_layers) >= 2:
                score += 4.5
            # 6. Avoid circular iris on payoff
            if beat.speech_act == "payoff_reveal" and cand.get("transition_in") == "frame_morph":
                score -= 9.0
            # 7. Respect controlled override for causal proof testing or human preference tuning
            cid = str(cand.get("candidate_id", ""))
            if cid in overrides:
                score += float(overrides[cid])

        scored.append({
            **cand,
            "status": "ELIGIBLE" if not rejects else "REJECTED_HARD_CONSTRAINT",
            "hard_rejects": rejects,
            "rubric_score": round(score, 2),
            "eligible": len(rejects) == 0,
        })

    eligible = [c for c in scored if c["eligible"]]
    if not eligible:
        all_reasons = sorted({r for c in scored for r in c.get("hard_rejects", [])})
        return (
            {
                "candidate_id": NO_ELIGIBLE_CANDIDATE_ID,
                "status": NO_ELIGIBLE_CANDIDATE_ID,
                "eligible": False,
                "hard_rejects": all_reasons or ["all_candidates_rejected"],
                "rubric_score": 0.0,
                "rationale": f"All {len(scored)} candidate(s) rejected by hard constraints: {all_reasons}",
            },
            scored,
        )

    eligible.sort(key=lambda x: (-x["rubric_score"], str(x.get("candidate_id", ""))))
    return eligible[0], scored


def _build_replan_candidates(
    beat: StoryBeatNode,
    beat_state: BeatStateSnapshot,
    rejection_reasons: list[str],
    *,
    verified_rem_ids: set[str],
    verified_hf_ids: set[str],
    attempt: int,
) -> list[dict[str, Any]]:
    """Bounded replanning generator using prior rejection reasons before resorting to minimal source-preserving fallback."""
    replan: list[dict[str, Any]] = []
    has_led = bool(beat_state.state_after.get("ledger_active") and beat.ledger_step_id)
    if attempt == 1:
        if has_led and "hf_mk_progress_stat" in verified_hf_ids:
            replan.append({
                "candidate_id": f"{beat.beat_id}_replan1_live_ledger",
                "strategy": "replan_live_single_owner_ledger",
                "stage_mode": "live_clean_bridge",
                "remotion_primitive_id": "none",
                "hyperframes_primitive_id": "hf_mk_progress_stat",
                "hf_layers": [],
                "transition_in": "none",
                "transition_out": "none",
                "single_numeric_owner": True,
                "preserve_speaker_hero": True,
                "visual_clarity_rating": 8.0,
                "continuity_rating": 8.0,
                "rationale": f"Replan attempt 1 resolving {rejection_reasons[:2]} via clean live ledger strip.",
            })
        if "rem_ru_masked_slide" in verified_rem_ids:
            replan.append({
                "candidate_id": f"{beat.beat_id}_replan1_live_slide",
                "strategy": "replan_live_clean_typography",
                "stage_mode": "live_clean_bridge",
                "remotion_primitive_id": "rem_ru_masked_slide",
                "hyperframes_primitive_id": "none",
                "hf_layers": [],
                "transition_in": "none",
                "transition_out": "none",
                "single_numeric_owner": True,
                "preserve_speaker_hero": True,
                "visual_clarity_rating": 7.6,
                "continuity_rating": 8.0,
                "rationale": f"Replan attempt 1 resolving {rejection_reasons[:2]} via speaker-preserving kinetic title.",
            })
    else:
        # Attempt 2: minimal source-preserving verified fallback (explicitly labeled SAFE_FALLBACK)
        prim_r = "rem_ru_masked_slide" if "rem_ru_masked_slide" in verified_rem_ids else next(iter(sorted(verified_rem_ids - {"rem_viv_aspect_morph", "rem_ru_parallax", "rem_gtk_frame_morph", "rem_gtk_line_led", "rem_rc_flip_page", "rem_ru_waterfall"})), "none")
        replan.append({
            "candidate_id": f"{beat.beat_id}_replan2_safe_source_preserve",
            "strategy": "SAFE_FALLBACK_SOURCE_PRESERVING",
            "stage_mode": "live_clean_bridge",
            "remotion_primitive_id": prim_r,
            "hyperframes_primitive_id": "none",
            "hf_layers": [],
            "transition_in": "none",
            "transition_out": "none",
            "single_numeric_owner": True,
            "preserve_speaker_hero": True,
            "visual_clarity_rating": 6.5,
            "continuity_rating": 7.5,
            "is_safe_fallback": True,
            "rationale": f"Bounded replan safe source-preserving fallback after rejecting {rejection_reasons[:3]}.",
        })
    return replan


def rank_global_candidates(
    beat_entries: list[dict[str, Any]],
    contract: NarrativeContract,
    *,
    verified_rem_ids: set[str],
    verified_hf_ids: set[str],
    beam_width: int = 8,
    score_overrides: dict[str, float] | None = None,
    max_replan_attempts: int = 2,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Phase A4 Global Coherent Beam Search over full beat sequence (NOT greedy independent shots!).

    Optimizes global pacing, visual variety across consecutive beats, engine balance without blind dual-engine padding,
    and bridge continuity while strictly respecting hard-constraint filtering and bounded replanning on NO_ELIGIBLE_CANDIDATE.
    """
    overrides = score_overrides or {}
    if not beat_entries:
        return [], {"status": "EMPTY_BEATS", "beam_width": beam_width, "global_score": 0.0, "replan_events": []}

    # Step 1: Hard-filter and individually score every beat's candidates; trigger bounded replan if NO_ELIGIBLE_CANDIDATE!
    eligible_per_beat: list[list[dict[str, Any]]] = []
    all_scored_per_beat: list[list[dict[str, Any]]] = []
    replan_events: list[dict[str, Any]] = []

    for entry in beat_entries:
        beat = entry["beat"]
        bs = entry["beat_state"]
        cands = list(entry["candidates"])
        chosen_local, scored = rank_scene_candidates(cands, beat, bs, contract, prev_stage_modes=[], score_overrides=overrides)
        elig = [c for c in scored if c["eligible"]]
        attempts_used = 0

        while not elig and attempts_used < max_replan_attempts:
            attempts_used += 1
            rej_reasons = chosen_local.get("hard_rejects", [])
            replan_cands = _build_replan_candidates(
                beat, bs, rej_reasons,
                verified_rem_ids=verified_rem_ids,
                verified_hf_ids=verified_hf_ids,
                attempt=attempts_used,
            )
            chosen_local, r_scored = rank_scene_candidates(replan_cands, beat, bs, contract, prev_stage_modes=[], score_overrides=overrides)
            scored.extend(r_scored)
            elig = [c for c in r_scored if c["eligible"]]
            replan_events.append({
                "beat_id": beat.beat_id,
                "attempt": attempts_used,
                "prior_rejects": rej_reasons,
                "resolved": bool(elig),
                "chosen_after_replan": elig[0]["candidate_id"] if elig else NO_ELIGIBLE_CANDIDATE_ID,
            })

        if not elig:
            return [], {
                "status": "BLOCKED_NO_ELIGIBLE_CANDIDATE",
                "blocked_beat_id": beat.beat_id,
                "rejection_reasons": chosen_local.get("hard_rejects", []),
                "replan_events": replan_events,
            }

        for c in elig:
            c["_replan_attempts"] = attempts_used
        eligible_per_beat.append(elig)
        all_scored_per_beat.append(scored)

    # Step 2: Bounded Beam Search optimizing GLOBAL trajectory across the entire beat sequence!
    # State tuple: (cum_score, chosen_list, stage_history, prim_history, used_engines_frozenset)
    beams: list[tuple[float, list[dict[str, Any]], list[str], list[str], set[str]]] = [(0.0, [], [], [], set())]

    for idx, elig_list in enumerate(eligible_per_beat):
        next_beams: list[tuple[float, list[dict[str, Any]], list[str], list[str], set[str]]] = []
        for cum_score, path, st_hist, pr_hist, eng_set in beams:
            for cand in elig_list:
                st = str(cand.get("stage_mode", "live_clean_bridge"))
                rem_p = str(cand.get("remotion_primitive_id", "none"))
                hf_p = str(cand.get("hyperframes_primitive_id", "none"))
                sig_p = f"{rem_p}|{hf_p}"

                step_bonus = float(cand["rubric_score"])
                # Global trajectory coherence terms:
                # a) Strong penalty for 3 identical stage modes in a row, moderate penalty for 2 identical live stage modes back-to-back
                if len(st_hist) >= 2 and st_hist[-1] == st and st_hist[-2] == st:
                    step_bonus -= 22.0
                elif len(st_hist) >= 1 and st_hist[-1] == st and st != "live_clean_bridge":
                    step_bonus -= 9.5
                elif len(st_hist) >= 1 and st_hist[-1] != st:
                    step_bonus += 5.5

                # b) Penalize repeating exact same primitive signature on consecutive beats
                if pr_hist and pr_hist[-1] == sig_p:
                    step_bonus -= 10.0
                elif pr_hist.count(sig_p) >= 2:
                    step_bonus -= 4.5

                # c) Penalize back-to-back full graphic cutaways (protect speaker rhythm!)
                if st_hist and st_hist[-1] == "full_graphic_cutaway" and st == "full_graphic_cutaway":
                    step_bonus -= 16.0

                # d) Modest global-level bonus for achieving genuine multi-engine variety across the video (only first time each engine joins!)
                new_engs = set(eng_set)
                if rem_p != "none" or st in ("live_depth_matte", "aspect_morph_editorial"):
                    if "remotion" not in new_engs:
                        step_bonus += 3.0
                    new_engs.add("remotion")
                if hf_p != "none" or cand.get("hf_layers"):
                    if "hyperframes" not in new_engs:
                        step_bonus += 3.0
                    new_engs.add("hyperframes")

                next_beams.append((
                    round(cum_score + step_bonus, 3),
                    path + [cand],
                    st_hist + [st],
                    pr_hist + [sig_p],
                    new_engs,
                ))

        # Deterministic tie-breaking by candidate_id sequence
        next_beams.sort(key=lambda b: (-b[0], [str(c.get("candidate_id", "")) for c in b[1]]))
        beams = next_beams[: max(1, int(beam_width))]

    best_score, best_path, _, _, best_engines = beams[0]
    selected_rows: list[dict[str, Any]] = []
    for idx, chosen in enumerate(best_path):
        selected_rows.append({
            "beat_id": beat_entries[idx]["beat"].beat_id,
            "chosen": chosen,
            "candidates": all_scored_per_beat[idx],
            "replan_attempts": int(chosen.get("_replan_attempts", 0)),
        })

    return selected_rows, {
        "status": "OK",
        "beam_width": beam_width,
        "global_score": best_score,
        "engines_covered_globally": sorted(best_engines),
        "replan_events": replan_events,
    }
