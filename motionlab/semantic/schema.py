"""Typed schemas for ML-01.R4 local ASR transcript, semantic units, claim grounding, and AI Creative Director plan."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from motionlab.schemas import canonical_hash

SEMANTIC_SCHEMA_VERSION = "1.0.0"

VALID_INTENTS = frozenset({
    "HOOK", "PROBLEM", "DEMONSTRATION", "ALLOCATION", "BENEFIT",
    "CONTRAST", "PRODUCT", "RESULT", "CTA", "HOLD",
})


@dataclass
class WordSpan:
    word: str
    start_s: float
    end_s: float
    probability: float
    uncertain: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "WordSpan":
        return cls(
            word=str(d["word"]),
            start_s=float(d["start_s"]),
            end_s=float(d["end_s"]),
            probability=float(d["probability"]),
            uncertain=bool(d.get("uncertain", False)),
        )


@dataclass
class TranscriptSegment:
    segment_id: str
    start_s: float
    end_s: float
    text: str
    avg_logprob: float
    mean_word_prob: float
    words: list[WordSpan] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["words"] = [w.to_dict() for w in self.words]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TranscriptSegment":
        return cls(
            segment_id=str(d["segment_id"]),
            start_s=float(d["start_s"]),
            end_s=float(d["end_s"]),
            text=str(d["text"]),
            avg_logprob=float(d.get("avg_logprob", 0.0)),
            mean_word_prob=float(d.get("mean_word_prob", 0.0)),
            words=[WordSpan.from_dict(w) for w in d.get("words", [])],
        )


@dataclass
class Transcript:
    schema_version: str
    source_hash: str
    source_audio_hash: str
    language: str
    language_probability: float
    duration_s: float
    sample_rate_asr: int
    model_id: str
    model_sha256: str
    asr_params: dict[str, Any]
    word_timestamps_exact: bool
    pts_priming_offset_s: float
    pts_anomalies: list[dict[str, Any]]
    status: str  # OK, NO_SPEECH, LOW_CONFIDENCE, BLOCKED_NO_MODEL
    full_text: str
    segments: list[TranscriptSegment] = field(default_factory=list)
    uncertain_spans: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0
    cache_hit: bool = False
    transcript_hash: str = ""

    def compute_hash(self) -> str:
        payload = self.to_dict()
        payload.pop("transcript_hash", None)
        payload.pop("elapsed_s", None)
        payload.pop("cache_hit", None)
        return canonical_hash(payload)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["segments"] = [s.to_dict() for s in self.segments]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Transcript":
        obj = cls(
            schema_version=str(d.get("schema_version", SEMANTIC_SCHEMA_VERSION)),
            source_hash=str(d["source_hash"]),
            source_audio_hash=str(d.get("source_audio_hash", "")),
            language=str(d.get("language", "unknown")),
            language_probability=float(d.get("language_probability", 0.0)),
            duration_s=float(d.get("duration_s", 0.0)),
            sample_rate_asr=int(d.get("sample_rate_asr", 16000)),
            model_id=str(d.get("model_id", "")),
            model_sha256=str(d.get("model_sha256", "")),
            asr_params=dict(d.get("asr_params", {})),
            word_timestamps_exact=bool(d.get("word_timestamps_exact", False)),
            pts_priming_offset_s=float(d.get("pts_priming_offset_s", 0.0)),
            pts_anomalies=list(d.get("pts_anomalies", [])),
            status=str(d.get("status", "OK")),
            full_text=str(d.get("full_text", "")),
            segments=[TranscriptSegment.from_dict(s) for s in d.get("segments", [])],
            uncertain_spans=list(d.get("uncertain_spans", [])),
            warnings=list(d.get("warnings", [])),
            elapsed_s=float(d.get("elapsed_s", 0.0)),
            cache_hit=bool(d.get("cache_hit", False)),
            transcript_hash=str(d.get("transcript_hash", "")),
        )
        if not obj.transcript_hash:
            obj.transcript_hash = obj.compute_hash()
        return obj


@dataclass
class SemanticUnit:
    unit_id: str
    start_s: float
    end_s: float
    start_sample: int
    end_sample: int
    start_frame: int
    end_frame_exclusive: int
    transcript_span_ids: list[str]
    verbatim_quote: str
    mean_confidence: float
    has_uncertain_words: bool
    spoken_numbers: list[dict[str, Any]]
    observed_motion_energy: float
    intent: str = "DEMONSTRATION"
    subject: str = ""
    action: str = ""
    object_ref: str = ""
    benefit_type: str = ""
    evidence_refs: list[str] = field(default_factory=list)
    candidate_visual_intent: str = ""
    audio_accent_role: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SemanticUnit":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


@dataclass
class GroundedClaim:
    claim_id: str
    unit_id: str
    claim_kind: str  # quotation, verified_figure, grounded_paraphrase, abstract_visual
    display_label: str
    verbatim_quote: str
    numeric_values: list[str]
    confidence: float
    accepted: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SemanticBeat:
    beat_id: str
    unit_ids: list[str]
    start_frame: int
    end_frame_exclusive: int
    payoff_frame: int
    verbatim_quote: str
    intent: str
    recipe_id: str  # VFX01..VFX12 or QUIET_HOLD
    transition_in: str
    display_label: str
    secondary_labels: list[str]
    benefit_rationale: str
    sfx_roles: list[dict[str, Any]]
    grounded_claim_ids: list[str]
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SemanticDirectorPlan:
    schema_version: str
    source_hash: str
    transcript_hash: str
    llm_model_id: str
    llm_model_sha256: str
    seed: int
    niche: str
    niche_confidence: float
    niche_reason: str
    core_value_proposition: str
    style_profile: str
    status: str  # READY, PARTIAL_LOW_CONFIDENCE, NO_SPEECH_FALLBACK, BLOCKED_NO_MODEL
    beats: list[SemanticBeat] = field(default_factory=list)
    accepted_claims: list[GroundedClaim] = field(default_factory=list)
    rejected_claims: list[GroundedClaim] = field(default_factory=list)
    rejected_candidates: list[dict[str, Any]] = field(default_factory=list)
    llm_elapsed_s: float = 0.0
    cache_hit: bool = False
    warnings: list[str] = field(default_factory=list)
    semantic_plan_hash: str = ""

    def compute_hash(self) -> str:
        payload = self.to_dict()
        payload.pop("semantic_plan_hash", None)
        payload.pop("llm_elapsed_s", None)
        payload.pop("cache_hit", None)
        return canonical_hash(payload)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["beats"] = [b.to_dict() for b in self.beats]
        d["accepted_claims"] = [c.to_dict() for c in self.accepted_claims]
        d["rejected_claims"] = [c.to_dict() for c in self.rejected_claims]
        return d
