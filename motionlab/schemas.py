"""Public strongly-typed contracts & strict schema validators for MotionLab AI ML-01.

Every public contract:
  - MediaInfo
  - AnalysisResult
  - CandidateEvent
  - EffectSpec
  - EffectEvent
  - EditPlan
  - RenderResult
  - ValidationResult

Rejects NaN/Inf, path traversal, unknown effects, invalid frame bounds, and invalid fields.
Produces canonical deterministic JSON so same input/config/seed/versions -> identical plan hash.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field, asdict
from fractions import Fraction
from typing import Any

from motionlab import SCHEMA_VERSION, PLANNER_VERSION, ANALYZER_VERSION
from motionlab.config import (
    atomic_write_json,
    atomic_write_text,
    canonical_json_dumps,
    ensure_safe_subpath,
    validate_no_path_traversal,
)


class SchemaValidationError(ValueError):
    """Raised when a declarative schema/graph/plan fails strict fail-closed validation."""


def sanitize_native(obj: Any) -> Any:
    """Convert NumPy scalars/arrays and Path objects recursively into JSON-safe native Python types."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return obj.item()
        except Exception:
            pass
    if isinstance(obj, dict):
        return {str(k): sanitize_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_native(v) for v in obj]
    if isinstance(obj, Fraction):
        return {"num": obj.numerator, "den": obj.denominator}
    return str(obj)


def canonical_hash(payload: Any) -> str:
    """Compute canonical SHA-256 hex digest over any JSON-serializable structure."""
    raw = canonical_json_dumps(sanitize_native(payload), indent=None)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def ensure_finite(obj: Any, path: str = "root") -> None:
    _assert_finite_tree(obj, path)


def ensure_contained_path(candidate: Any, root: Any) -> Any:
    from pathlib import Path as _Path

    return ensure_safe_subpath(_Path(root), _Path(candidate))


class RationalFraction(Fraction):
    """Exact Fraction subclass also implementing R3/R4 Rational helper methods cleanly."""

    @property
    def num(self) -> int:
        return int(self.numerator)

    @property
    def den(self) -> int:
        return int(self.denominator)

    def as_fraction(self) -> Fraction:
        return Fraction(self.numerator, self.denominator)

    def as_float(self) -> float:
        return float(self)

    def to_ffmpeg(self) -> str:
        return f"{self.numerator}/{self.denominator}"

    def to_dict(self) -> dict[str, int]:
        return {"num": int(self.numerator), "den": int(self.denominator)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RationalFraction":
        return cls(int(d["num"]), max(1, int(d.get("den", 1))))


Rational = RationalFraction


@dataclass
class ProtectedZone:
    """Spatial protected zone (face, active task, UI text, letterbox) for R2/R3/R4 choreography."""

    start_frame: int
    end_frame_exclusive: int
    x: float
    y: float
    w: float
    h: float
    reason: str
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _assert_finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"Field '{name}' must be a finite number, got {value!r}")
    return float(value)


def _assert_finite_tree(obj: Any, path: str = "root") -> None:
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError(f"Non-finite float at {path}: {obj!r}")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _assert_finite_tree(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for idx, item in enumerate(obj):
            _assert_finite_tree(item, f"{path}[{idx}]")


@dataclass
class AudioStreamInfo:
    index: int
    audio_index: int  # 0-based among audio streams
    codec_name: str
    sample_rate: int
    channels: int
    channel_layout: str
    duration_s: float
    start_pts: int
    start_time_s: float
    time_base: str
    bit_rate: int | None = None
    initial_padding: int = 0
    language: str = "und"
    mp4_copy_compatible: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ColorMetadata:
    color_space: str = "unknown"
    color_transfer: str = "unknown"
    color_primaries: str = "unknown"
    color_range: str = "unknown"
    is_hdr: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MediaInfo:
    source_path: str
    input_hash: str
    file_size_bytes: int
    duration_pts: int
    duration_s: float
    time_base_num: int
    time_base_den: int
    width: int
    height: int
    display_width: int
    display_height: int
    rotation: int
    sar_num: int
    sar_den: int
    fps_num: int
    fps_den: int
    vfr: bool
    nb_frames: int
    stream_start_pts: int
    stream_start_time_s: float
    video_codec: str
    pixel_format: str
    field_order: str
    is_interlaced: bool
    has_alpha: bool
    color_metadata: ColorMetadata
    audio_streams: list[AudioStreamInfo] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def fps(self) -> RationalFraction:
        return RationalFraction(self.fps_num, self.fps_den)

    @property
    def fps_rational(self) -> RationalFraction:
        return RationalFraction(self.fps_num, self.fps_den)

    @property
    def fps_float(self) -> float:
        return self.fps_num / self.fps_den

    @property
    def path(self) -> str:
        return self.source_path

    @property
    def duration_frames(self) -> int:
        return int(self.nb_frames)

    @property
    def has_audio(self) -> bool:
        return len(self.audio_streams) > 0

    @property
    def pts_summary(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = [
            {
                "codec_type": "video",
                "start_pts": self.stream_start_pts,
                "start_time_s": self.stream_start_time_s,
                "anomalies": [w for w in self.warnings if "pts" in w or "vfr" in w],
            }
        ]
        for a in self.audio_streams:
            priming_s = float(a.initial_padding) / max(1, a.sample_rate) if a.initial_padding else 0.0
            out.append(
                {
                    "codec_type": "audio",
                    "audio_index": a.audio_index,
                    "start_pts": a.start_pts,
                    "start_time_s": a.start_time_s,
                    "initial_padding": a.initial_padding,
                    "priming_offset_s": round(priming_s, 6),
                    "anomalies": [w for w in self.warnings if f"audio_track_{a.audio_index}" in w],
                }
            )
        return out

    def frame_to_seconds(self, frame_idx: int) -> float:
        return float(Fraction(frame_idx * self.fps_den, self.fps_num))

    def seconds_to_frame(self, seconds: float) -> int:
        _assert_finite(seconds, "seconds")
        return max(0, int(round(seconds * self.fps_num / self.fps_den)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "input_hash": self.input_hash,
            "file_size_bytes": self.file_size_bytes,
            "duration_pts": self.duration_pts,
            "duration_s": round(self.duration_s, 6),
            "time_base": f"{self.time_base_num}/{self.time_base_den}",
            "time_base_num": self.time_base_num,
            "time_base_den": self.time_base_den,
            "width": self.width,
            "height": self.height,
            "display_width": self.display_width,
            "display_height": self.display_height,
            "rotation": self.rotation,
            "sar": f"{self.sar_num}:{self.sar_den}",
            "sar_num": self.sar_num,
            "sar_den": self.sar_den,
            "fps": f"{self.fps_num}/{self.fps_den}",
            "fps_num": self.fps_num,
            "fps_den": self.fps_den,
            "vfr": self.vfr,
            "nb_frames": self.nb_frames,
            "stream_start_pts": self.stream_start_pts,
            "stream_start_time_s": round(self.stream_start_time_s, 6),
            "video_codec": self.video_codec,
            "pixel_format": self.pixel_format,
            "field_order": self.field_order,
            "is_interlaced": self.is_interlaced,
            "has_alpha": self.has_alpha,
            "color_metadata": self.color_metadata.to_dict(),
            "audio_streams": [a.to_dict() for a in self.audio_streams],
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MediaInfo":
        cm = ColorMetadata(**data["color_metadata"])
        audio_streams = [AudioStreamInfo(**a) for a in data.get("audio_streams", [])]
        return cls(
            source_path=data["source_path"],
            input_hash=data["input_hash"],
            file_size_bytes=int(data["file_size_bytes"]),
            duration_pts=int(data["duration_pts"]),
            duration_s=float(data["duration_s"]),
            time_base_num=int(data["time_base_num"]),
            time_base_den=int(data["time_base_den"]),
            width=int(data["width"]),
            height=int(data["height"]),
            display_width=int(data["display_width"]),
            display_height=int(data["display_height"]),
            rotation=int(data["rotation"]),
            sar_num=int(data["sar_num"]),
            sar_den=int(data["sar_den"]),
            fps_num=int(data["fps_num"]),
            fps_den=int(data["fps_den"]),
            vfr=bool(data["vfr"]),
            nb_frames=int(data["nb_frames"]),
            stream_start_pts=int(data["stream_start_pts"]),
            stream_start_time_s=float(data["stream_start_time_s"]),
            video_codec=str(data["video_codec"]),
            pixel_format=str(data["pixel_format"]),
            field_order=str(data["field_order"]),
            is_interlaced=bool(data["is_interlaced"]),
            has_alpha=bool(data["has_alpha"]),
            color_metadata=cm,
            audio_streams=audio_streams,
            warnings=list(data.get("warnings", [])),
        )


@dataclass
class SceneCut:
    frame: int
    time_s: float
    confidence: float
    boundary_type: str  # "detected" | "inferred"
    hist_diff: float
    luma_diff: float
    protected_existing_transition: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame": int(self.frame),
            "time_s": round(float(self.time_s), 4),
            "confidence": round(float(self.confidence), 4),
            "boundary_type": str(self.boundary_type),
            "hist_diff": round(float(self.hist_diff), 4),
            "luma_diff": round(float(self.luma_diff), 4),
            "protected_existing_transition": bool(self.protected_existing_transition),
        }


@dataclass
class MotionWindow:
    start_frame: int
    end_frame_exclusive: int
    motion_score: float  # 0.0 (static) .. 1.0 (very high motion)
    stability_score: float  # 0.0 .. 1.0
    motion_class: str  # "near_static" | "moderate" | "high"
    source_class: str  # "camera" | "subject" | "unknown"
    saliency_center_x: float  # normalized 0..1
    saliency_center_y: float  # normalized 0..1
    edge_density: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_frame": self.start_frame,
            "end_frame_exclusive": self.end_frame_exclusive,
            "motion_score": round(self.motion_score, 4),
            "stability_score": round(self.stability_score, 4),
            "motion_class": self.motion_class,
            "source_class": self.source_class,
            "saliency_center_x": round(self.saliency_center_x, 4),
            "saliency_center_y": round(self.saliency_center_y, 4),
            "edge_density": round(self.edge_density, 4),
        }


@dataclass
class AudioEnvelopePoint:
    frame: int
    time_s: float
    rms_norm: float  # robust percentile normalized 0..1
    peak_norm: float  # 0..1
    db: float

    @property
    def normalized_rms(self) -> float:
        return self.rms_norm

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame,
            "time_s": round(self.time_s, 4),
            "rms_norm": round(self.rms_norm, 4),
            "peak_norm": round(self.peak_norm, 4),
            "db": round(self.db, 2),
        }


@dataclass
class EnergyPeak:
    frame: int
    time_s: float
    strength: float  # 0..1
    onset_flux: float
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame,
            "time_s": round(self.time_s, 4),
            "strength": round(self.strength, 4),
            "onset_flux": round(self.onset_flux, 4),
            "confidence": round(self.confidence, 4),
        }


@dataclass
class PauseWindow:
    start_frame: int
    end_frame_exclusive: int
    duration_s: float
    mean_db: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_frame": self.start_frame,
            "end_frame_exclusive": self.end_frame_exclusive,
            "duration_s": round(self.duration_s, 4),
            "mean_db": round(self.mean_db, 2),
        }


@dataclass
class SubjectRegion:
    frame: int
    x: float  # normalized 0..1 left
    y: float  # normalized 0..1 top
    w: float  # normalized 0..1 width
    h: float  # normalized 0..1 height
    confidence: float
    multi_subject: bool = False
    detector: str = "opencv_haar_frontalface_default"

    @property
    def center_x(self) -> float:
        return self.x + self.w * 0.5

    @property
    def center_y(self) -> float:
        return self.y + self.h * 0.5

    @property
    def start_frame(self) -> int:
        return int(self.frame)

    @property
    def end_frame_exclusive(self) -> int:
        return int(self.frame) + 1

    @property
    def ambiguous(self) -> bool:
        return bool(self.multi_subject)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame,
            "x": round(self.x, 4),
            "y": round(self.y, 4),
            "w": round(self.w, 4),
            "h": round(self.h, 4),
            "confidence": round(self.confidence, 4),
            "multi_subject": self.multi_subject,
            "detector": self.detector,
        }


@dataclass
class ProtectedInterval:
    start_frame: int
    end_frame_exclusive: int
    reason: str
    severity: float  # 0..1

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_frame": self.start_frame,
            "end_frame_exclusive": self.end_frame_exclusive,
            "reason": self.reason,
            "severity": round(self.severity, 4),
        }


@dataclass
class CandidateEvent:
    id: str
    frame: int
    end_frame_exclusive: int
    event_type: str  # scene_boundary | sustained_static | audio_onset | energy_rise | motion_change | safe_hold
    confidence: float
    reason: str
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "frame": self.frame,
            "end_frame_exclusive": self.end_frame_exclusive,
            "event_type": self.event_type,
            "confidence": round(self.confidence, 4),
            "reason": self.reason,
            "metrics": {k: round(float(v), 4) for k, v in sorted(self.metrics.items())},
        }


@dataclass
class AnalysisResult:
    source_hash: str
    config_hash: str
    analyzer_version: str
    timebase: str
    fps_num: int
    fps_den: int
    duration_frames: int
    cuts: list[SceneCut] = field(default_factory=list)
    motion_windows: list[MotionWindow] = field(default_factory=list)
    audio_envelope: list[AudioEnvelopePoint] = field(default_factory=list)
    energy_peaks: list[EnergyPeak] = field(default_factory=list)
    pauses: list[PauseWindow] = field(default_factory=list)
    subject_regions: list[SubjectRegion] = field(default_factory=list)
    protected_intervals: list[ProtectedInterval] = field(default_factory=list)
    safe_zones: dict[str, Any] = field(default_factory=dict)
    quality_flags: list[str] = field(default_factory=list)
    capabilities: list[str] = field(default_factory=list)
    signal_summary: dict[str, Any] = field(default_factory=dict)
    candidates: list[CandidateEvent] = field(default_factory=list)

    @property
    def summary_stats(self) -> dict[str, Any]:
        return self.signal_summary

    @property
    def dominant_palette(self) -> dict[str, str]:
        return dict(self.signal_summary.get("dominant_palette") or {})

    @property
    def fps(self) -> RationalFraction:
        return RationalFraction(self.fps_num, self.fps_den)

    @property
    def protected_zones(self) -> list[ProtectedZone]:
        return []

    @property
    def action_windows(self) -> list[Any]:
        return []

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "source_hash": self.source_hash,
            "config_hash": self.config_hash,
            "analyzer_version": self.analyzer_version,
            "timebase": self.timebase,
            "fps_num": self.fps_num,
            "fps_den": self.fps_den,
            "duration_frames": self.duration_frames,
            "cuts": [c.to_dict() for c in self.cuts],
            "motion_windows": [m.to_dict() for m in self.motion_windows],
            "audio_envelope": [a.to_dict() for a in self.audio_envelope],
            "energy_peaks": [p.to_dict() for p in self.energy_peaks],
            "pauses": [p.to_dict() for p in self.pauses],
            "subject_regions": [s.to_dict() for s in self.subject_regions],
            "protected_intervals": [p.to_dict() for p in self.protected_intervals],
            "safe_zones": self.safe_zones,
            "quality_flags": sorted(set(self.quality_flags)),
            "capabilities": sorted(set(self.capabilities)),
            "signal_summary": self.signal_summary,
            "candidates": [c.to_dict() for c in self.candidates],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnalysisResult":
        return cls(
            source_hash=str(data["source_hash"]),
            config_hash=str(data["config_hash"]),
            analyzer_version=str(data.get("analyzer_version", ANALYZER_VERSION)),
            timebase=str(data["timebase"]),
            fps_num=int(data["fps_num"]),
            fps_den=int(data["fps_den"]),
            duration_frames=int(data["duration_frames"]),
            cuts=[SceneCut(**c) for c in data.get("cuts", [])],
            motion_windows=[MotionWindow(**m) for m in data.get("motion_windows", [])],
            audio_envelope=[AudioEnvelopePoint(**a) for a in data.get("audio_envelope", [])],
            energy_peaks=[EnergyPeak(**p) for p in data.get("energy_peaks", [])],
            pauses=[PauseWindow(**p) for p in data.get("pauses", [])],
            subject_regions=[SubjectRegion(**s) for s in data.get("subject_regions", [])],
            protected_intervals=[ProtectedInterval(**p) for p in data.get("protected_intervals", [])],
            safe_zones=dict(data.get("safe_zones", {})),
            quality_flags=list(data.get("quality_flags", [])),
            capabilities=list(data.get("capabilities", [])),
            signal_summary=dict(data.get("signal_summary", {})),
            candidates=[CandidateEvent(**c) for c in data.get("candidates", [])],
        )


@dataclass(frozen=True)
class EffectSpec:
    id: str
    code: str  # e.g., C01, L01, T01, V01, G01
    version: str
    family: str  # camera | mask_layout | transition | color_optical | motion_graphics
    description: str
    engine: str  # primary engine: ffmpeg | remotion | hyperframes
    supported_engines: tuple[str, ...]
    layer: str  # transform | layout | transition | grade | overlay
    required_features: tuple[str, ...]  # e.g., ("audio",), ("confident_subject",), ("scene_cut",)
    supported_profiles: tuple[str, ...]
    parameter_schema: dict[str, dict[str, Any]]
    default_params: dict[str, Any]
    min_duration_s: float
    max_duration_s: float
    max_intensity: float
    safety_class: str  # "low_risk" | "subject_sensitive" | "boundary_only" | "explicit_only"
    incompatibilities: tuple[str, ...]  # layers or effect IDs
    fallback_id: str | None
    estimated_cost: float  # relative 1.0..5.0
    is_core: bool = False
    is_strong_transform: bool = False
    uses_neighbor_frames: bool = False
    excluded_cases: tuple[str, ...] = ()
    license_note: str = "Local engine (FFmpeg/OpenCV/Pillow); check Remotion/HyperFrames notices when used."

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "code": self.code,
            "version": self.version,
            "family": self.family,
            "description": self.description,
            "engine": self.engine,
            "supported_engines": list(self.supported_engines),
            "layer": self.layer,
            "required_features": list(self.required_features),
            "supported_profiles": list(self.supported_profiles),
            "parameter_schema": self.parameter_schema,
            "default_params": self.default_params,
            "min_duration_s": self.min_duration_s,
            "max_duration_s": self.max_duration_s,
            "max_intensity": self.max_intensity,
            "safety_class": self.safety_class,
            "incompatibilities": list(self.incompatibilities),
            "fallback_id": self.fallback_id,
            "estimated_cost": self.estimated_cost,
            "is_core": self.is_core,
            "is_strong_transform": self.is_strong_transform,
            "uses_neighbor_frames": self.uses_neighbor_frames,
            "excluded_cases": list(self.excluded_cases),
            "license_note": self.license_note,
        }


@dataclass
class EffectEvent:
    id: str
    effect_id: str
    effect_version: str
    engine: str
    start_frame: int
    end_frame_exclusive: int
    layer: str
    region: dict[str, float]  # normalized anchor/box: {"cx": 0.5, "cy": 0.5, "w": 1.0, "h": 1.0}
    parameters: dict[str, Any]
    easing: str
    priority: int
    reason_codes: list[str]
    evidence_refs: list[str]
    fallback_id: str | None = None
    fallback_engine: str | None = "ffmpeg"

    def validate(self, duration_frames: int) -> None:
        if not self.id or not self.effect_id:
            raise ValueError("EffectEvent requires non-empty id and effect_id")
        if self.start_frame < 0 or self.end_frame_exclusive <= self.start_frame:
            raise ValueError(
                f"Invalid frame bounds for {self.id}: [{self.start_frame}, {self.end_frame_exclusive})"
            )
        if self.end_frame_exclusive > duration_frames:
            raise ValueError(
                f"Event {self.id} end_frame_exclusive ({self.end_frame_exclusive}) > duration_frames ({duration_frames})"
            )
        _assert_finite_tree(self.region, f"{self.id}.region")
        _assert_finite_tree(self.parameters, f"{self.id}.parameters")
        if self.easing not in ("ease_out_cubic", "ease_in_out_cubic", "spring_damped", "linear_bounded"):
            raise ValueError(f"Unsupported easing '{self.easing}' in event {self.id}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "effect_id": self.effect_id,
            "effect_version": self.effect_version,
            "engine": self.engine,
            "start_frame": self.start_frame,
            "end_frame_exclusive": self.end_frame_exclusive,
            "layer": self.layer,
            "region": {k: round(float(v), 5) for k, v in sorted(self.region.items())},
            "parameters": self.parameters,
            "easing": self.easing,
            "priority": self.priority,
            "reason_codes": list(self.reason_codes),
            "evidence_refs": list(self.evidence_refs),
            "fallback_id": self.fallback_id,
            "fallback_engine": self.fallback_engine,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EffectEvent":
        return cls(
            id=str(data["id"]),
            effect_id=str(data["effect_id"]),
            effect_version=str(data["effect_version"]),
            engine=str(data["engine"]),
            start_frame=int(data["start_frame"]),
            end_frame_exclusive=int(data["end_frame_exclusive"]),
            layer=str(data["layer"]),
            region=dict(data["region"]),
            parameters=dict(data["parameters"]),
            easing=str(data["easing"]),
            priority=int(data["priority"]),
            reason_codes=list(data.get("reason_codes", [])),
            evidence_refs=list(data.get("evidence_refs", [])),
            fallback_id=data.get("fallback_id"),
            fallback_engine=data.get("fallback_engine", "ffmpeg"),
        )


@dataclass
class EditPlan:
    schema_version: str
    source_hash: str
    config_hash: str
    seed: int
    profile: str
    requested_profile: str
    width: int
    height: int
    fps_num: int
    fps_den: int
    duration_frames: int
    aspect_mode: str
    aspect_bg_mode: str
    audio_policy: dict[str, Any]
    engine_capabilities: dict[str, Any]
    events: list[EffectEvent]
    planner_version: str = PLANNER_VERSION
    status: str = "PLANNED"  # PLANNED | NO_SAFE_EFFECTS
    warnings: list[str] = field(default_factory=list)
    layout_plan: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        sorted_events = sorted(
            self.events,
            key=lambda e: (e.start_frame, e.end_frame_exclusive, e.layer, e.priority, e.id),
        )
        return {
            "schema_version": self.schema_version,
            "source_hash": self.source_hash,
            "config_hash": self.config_hash,
            "seed": self.seed,
            "profile": self.profile,
            "requested_profile": self.requested_profile,
            "width": self.width,
            "height": self.height,
            "fps": f"{self.fps_num}/{self.fps_den}",
            "fps_num": self.fps_num,
            "fps_den": self.fps_den,
            "duration_frames": self.duration_frames,
            "aspect_mode": self.aspect_mode,
            "aspect_bg_mode": self.aspect_bg_mode,
            "audio_policy": self.audio_policy,
            "engine_capabilities": self.engine_capabilities,
            "events": [e.to_dict() for e in sorted_events],
            "planner_version": self.planner_version,
            "status": self.status,
            "warnings": list(self.warnings),
            "layout_plan": self.layout_plan,
        }

    def plan_hash(self) -> str:
        # Canonical hash over deterministic decision payload (excluding transient host paths in capabilities)
        d = self.to_dict()
        d.pop("engine_capabilities", None)
        payload = canonical_json_dumps(d, indent=None)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EditPlan":
        for req in ("schema_version", "source_hash", "config_hash", "seed", "width", "height", "fps_num", "fps_den", "duration_frames"):
            if req not in data:
                raise ValueError(f"EditPlan missing required field '{req}'")
        _assert_finite_tree(data, "EditPlan")
        events = [EffectEvent.from_dict(e) for e in data.get("events", [])]
        plan = cls(
            schema_version=str(data["schema_version"]),
            source_hash=str(data["source_hash"]),
            config_hash=str(data["config_hash"]),
            seed=int(data["seed"]),
            profile=str(data.get("profile", "neutral")),
            requested_profile=str(data.get("requested_profile", "auto")),
            width=int(data["width"]),
            height=int(data["height"]),
            fps_num=int(data["fps_num"]),
            fps_den=int(data["fps_den"]),
            duration_frames=int(data["duration_frames"]),
            aspect_mode=str(data.get("aspect_mode", "source")),
            aspect_bg_mode=str(data.get("aspect_bg_mode", "blur")),
            audio_policy=dict(data.get("audio_policy", {})),
            engine_capabilities=dict(data.get("engine_capabilities", {})),
            events=events,
            planner_version=str(data.get("planner_version", PLANNER_VERSION)),
            status=str(data.get("status", "PLANNED")),
            warnings=list(data.get("warnings", [])),
            layout_plan=dict(data.get("layout_plan", {})),
        )
        if plan.width <= 0 or plan.height <= 0 or plan.width % 2 != 0 or plan.height % 2 != 0:
            raise ValueError(f"EditPlan dimensions must be positive even integers, got {plan.width}x{plan.height}")
        if plan.duration_frames <= 0:
            raise ValueError(f"EditPlan duration_frames must be > 0, got {plan.duration_frames}")
        for ev in plan.events:
            ev.validate(plan.duration_frames)
        return plan


@dataclass
class RenderResult:
    artifact_path: str
    engines_used: list[str]
    event_backends: dict[str, str] = field(default_factory=dict)
    timings_s: dict[str, float] = field(default_factory=dict)
    ffprobe_info: dict[str, Any] = field(default_factory=dict)
    status: str = "PASS"  # PASS | FAIL | BLOCKED
    audio_mode: str = "stream_copy"  # stream_copy | transcode_compat | mix | none
    warnings: list[str] = field(default_factory=list)
    fallback_history: list[dict[str, Any]] = field(default_factory=list)
    peak_rss_mb: float | None = None
    peak_temp_bytes: int = 0
    event_engine_map: dict[str, str] = field(default_factory=dict)
    encoding_settings: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.event_engine_map and not self.event_backends:
            self.event_backends = dict(self.event_engine_map)
        elif self.event_backends and not self.event_engine_map:
            self.event_engine_map = dict(self.event_backends)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_path": self.artifact_path,
            "engines_used": sorted(set(self.engines_used)),
            "event_backends": dict(sorted(self.event_backends.items())),
            "timings_s": {k: round(float(v), 4) for k, v in sorted(self.timings_s.items())},
            "ffprobe_info": self.ffprobe_info,
            "status": self.status,
            "audio_mode": self.audio_mode,
            "warnings": list(self.warnings),
            "fallback_history": list(self.fallback_history),
            "peak_rss_mb": round(self.peak_rss_mb, 2) if self.peak_rss_mb is not None else None,
            "peak_temp_bytes": self.peak_temp_bytes,
            "encoding_settings": dict(self.encoding_settings),
        }


@dataclass
class ValidationCheck:
    name: str
    status: str  # PASS | FAIL | SKIP | BLOCKED
    expected: Any
    actual: Any
    evidence_path: str | None = None
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        if self.evidence_path:
            validate_no_path_traversal(self.evidence_path)
        return {
            "name": self.name,
            "status": self.status,
            "expected": self.expected,
            "actual": self.actual,
            "evidence_path": self.evidence_path,
            "message": self.message,
        }


@dataclass
class ValidationResult:
    checks: list[ValidationCheck]
    overall_status: str  # TECHNICAL_PASS | FAIL | BLOCKED
    aesthetic_status: str = "AESTHETIC_PENDING"
    summary: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        counts = {"PASS": 0, "FAIL": 0, "SKIP": 0, "BLOCKED": 0}
        for c in self.checks:
            counts[c.status] = counts.get(c.status, 0) + 1
        return {
            "schema_version": SCHEMA_VERSION,
            "overall_status": self.overall_status,
            "aesthetic_status": self.aesthetic_status,
            "summary": counts,
            "checks": [c.to_dict() for c in self.checks],
            "warnings": list(self.warnings),
        }
