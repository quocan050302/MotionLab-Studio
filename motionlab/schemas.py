"""Public data contracts, strict validation, and canonical atomic JSON I/O."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

from motionlab import ANALYZER_VERSION, PLANNER_VERSION, SCHEMA_VERSION

SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
FORBIDDEN_URL_PREFIXES = ("http://", "https://", "ftp://", "rtsp://", "rtmp://", "file://")


class SchemaValidationError(ValueError):
    """Raised when a contract object or JSON payload violates ML-01 invariants."""


def ensure_finite(value: Any, path: str = "root") -> None:
    """Recursively reject NaN, +Inf, and -Inf in any payload."""
    if isinstance(value, bool):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SchemaValidationError(f"Non-finite float at {path}: {value!r}")
    elif isinstance(value, dict):
        for k, v in value.items():
            ensure_finite(v, f"{path}.{k}")
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            ensure_finite(v, f"{path}[{i}]")


def validate_local_path_string(raw: str, allow_nonexistent: bool = False) -> Path:
    """Reject remote URLs, null bytes, and suspicious traversal components."""
    if not isinstance(raw, str) or not raw.strip():
        raise SchemaValidationError("Path must be a non-empty string")
    if "\x00" in raw:
        raise SchemaValidationError("Path contains null byte")
    lower = raw.strip().lower()
    if any(lower.startswith(prefix) for prefix in FORBIDDEN_URL_PREFIXES):
        raise SchemaValidationError(f"Remote URL inputs are forbidden in offline ML-01: {raw}")
    p = Path(raw)
    if not allow_nonexistent and not p.exists():
        raise SchemaValidationError(f"Path does not exist: {raw}")
    return p.resolve()


def ensure_contained_path(target: Path, root: Path) -> Path:
    """Verify target is strictly inside root without escaping via .. or symlinks."""
    resolved_root = root.resolve()
    # Check raw lexical traversal and resolved path containment
    resolved_target = target.resolve()
    try:
        resolved_target.relative_to(resolved_root)
    except ValueError as exc:
        raise SchemaValidationError(
            f"Path escape detected: {target} resolves outside {resolved_root}"
        ) from exc
    return resolved_target


def canonical_json_bytes(data: Any) -> bytes:
    """Serialize deterministically with strict NaN rejection and sorted keys."""
    ensure_finite(data)
    text = json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return text.encode("utf-8")


def canonical_hash(data: Any) -> str:
    """Return SHA-256 hex digest of canonical JSON representation."""
    return hashlib.sha256(canonical_json_bytes(data)).hexdigest()


def atomic_write_json(path: Path, data: Any, indent: int = 2) -> str:
    """Atomically write validated JSON to disk and return its canonical SHA-256."""
    ensure_finite(data)
    digest = canonical_hash(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(
        data,
        sort_keys=True,
        indent=indent,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(rendered)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return digest


def atomic_write_text(path: Path, content: str) -> None:
    """Atomically write UTF-8 text to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


@dataclass(frozen=True)
class Rational:
    num: int
    den: int = 1

    def __post_init__(self) -> None:
        if self.den <= 0 or self.num <= 0:
            raise SchemaValidationError(f"Rational must be positive: {self.num}/{self.den}")

    def as_fraction(self) -> Fraction:
        return Fraction(self.num, self.den)

    def as_float(self) -> float:
        return self.num / self.den

    def frame_to_seconds(self, frame_idx: int) -> float:
        return float(Fraction(frame_idx, 1) / self.as_fraction())

    def seconds_to_frame(self, seconds: float) -> int:
        if not math.isfinite(seconds):
            raise SchemaValidationError(f"Non-finite timestamp: {seconds}")
        return int(round(seconds * self.num / self.den))

    def to_ffmpeg(self) -> str:
        f = self.as_fraction()
        return f"{f.numerator}/{f.denominator}"

    def to_dict(self) -> dict[str, int]:
        f = self.as_fraction()
        return {"num": f.numerator, "den": f.denominator}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Rational":
        return cls(num=int(data["num"]), den=int(data["den"]))


@dataclass
class AudioStreamInfo:
    stream_index: int
    audio_track_index: int
    codec_name: str
    sample_rate: int
    channels: int
    channel_layout: str
    bit_rate: int | None
    start_pts: int
    start_time_s: float
    duration_s: float | None
    language: str | None
    mp4_copy_compatible: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MediaInfo:
    path: str
    input_hash: str
    file_size_bytes: int
    container_format: str
    duration_pts: int
    duration_s: float
    duration_frames: int
    width: int
    height: int
    display_width: int
    display_height: int
    rotation: int
    sar: Rational
    fps: Rational
    timebase: Rational
    vfr: bool
    stream_start_pts: int
    video_start_time_s: float
    video_codec: str
    pixel_format: str
    color_metadata: dict[str, str | None]
    interlaced: bool
    has_alpha: bool
    is_hdr: bool
    audio_streams: list[AudioStreamInfo] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def has_audio(self) -> bool:
        return len(self.audio_streams) > 0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["sar"] = self.sar.to_dict()
        d["fps"] = self.fps.to_dict()
        d["timebase"] = self.timebase.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MediaInfo":
        return cls(
            path=str(d["path"]),
            input_hash=str(d["input_hash"]),
            file_size_bytes=int(d["file_size_bytes"]),
            container_format=str(d["container_format"]),
            duration_pts=int(d["duration_pts"]),
            duration_s=float(d["duration_s"]),
            duration_frames=int(d["duration_frames"]),
            width=int(d["width"]),
            height=int(d["height"]),
            display_width=int(d["display_width"]),
            display_height=int(d["display_height"]),
            rotation=int(d["rotation"]),
            sar=Rational.from_dict(d["sar"]),
            fps=Rational.from_dict(d["fps"]),
            timebase=Rational.from_dict(d["timebase"]),
            vfr=bool(d["vfr"]),
            stream_start_pts=int(d["stream_start_pts"]),
            video_start_time_s=float(d["video_start_time_s"]),
            video_codec=str(d["video_codec"]),
            pixel_format=str(d["pixel_format"]),
            color_metadata=dict(d.get("color_metadata") or {}),
            interlaced=bool(d.get("interlaced", False)),
            has_alpha=bool(d.get("has_alpha", False)),
            is_hdr=bool(d.get("is_hdr", False)),
            audio_streams=[AudioStreamInfo(**a) for a in d.get("audio_streams", [])],
            warnings=list(d.get("warnings", [])),
        )


@dataclass
class SceneCut:
    frame: int
    confidence: float
    boundary_type: str  # "detected" | "inferred"
    hist_diff: float
    luma_diff: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MotionWindow:
    start_frame: int
    end_frame_exclusive: int
    motion_score: float
    stability: float
    near_static: bool
    motion_class: str  # "camera" | "subject" | "unknown" | "static"
    saliency_center_x: float
    saliency_center_y: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AudioEnvelopePoint:
    frame: int
    rms: float
    peak: float
    normalized_rms: float
    clipped: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EnergyPeak:
    frame: int
    strength: float
    onset_flux: float
    rms_norm: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PauseWindow:
    start_frame: int
    end_frame_exclusive: int
    mean_rms_db: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SubjectRegion:
    start_frame: int
    end_frame_exclusive: int
    x: float  # normalized [0, 1]
    y: float
    w: float
    h: float
    confidence: float
    subject_count: int
    ambiguous: bool
    detector: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProtectedZone:
    start_frame: int
    end_frame_exclusive: int
    x: float
    y: float
    w: float
    h: float
    reason: str  # "face" | "ui_text" | "letterbox" | "existing_effect"
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnalysisResult:
    schema_version: str
    analyzer_version: str
    source_hash: str
    timebase: Rational
    fps: Rational
    duration_frames: int
    cuts: list[SceneCut]
    motion_windows: list[MotionWindow]
    audio_envelope: list[AudioEnvelopePoint]
    energy_peaks: list[EnergyPeak]
    pauses: list[PauseWindow]
    subject_regions: list[SubjectRegion]
    protected_zones: list[ProtectedZone]
    quality_flags: list[str]
    capabilities: list[str]
    summary_stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["timebase"] = self.timebase.to_dict()
        d["fps"] = self.fps.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AnalysisResult":
        return cls(
            schema_version=str(d.get("schema_version", SCHEMA_VERSION)),
            analyzer_version=str(d.get("analyzer_version", ANALYZER_VERSION)),
            source_hash=str(d["source_hash"]),
            timebase=Rational.from_dict(d["timebase"]),
            fps=Rational.from_dict(d["fps"]),
            duration_frames=int(d["duration_frames"]),
            cuts=[SceneCut(**c) for c in d.get("cuts", [])],
            motion_windows=[MotionWindow(**m) for m in d.get("motion_windows", [])],
            audio_envelope=[AudioEnvelopePoint(**a) for a in d.get("audio_envelope", [])],
            energy_peaks=[EnergyPeak(**e) for e in d.get("energy_peaks", [])],
            pauses=[PauseWindow(**p) for p in d.get("pauses", [])],
            subject_regions=[SubjectRegion(**s) for s in d.get("subject_regions", [])],
            protected_zones=[ProtectedZone(**z) for z in d.get("protected_zones", [])],
            quality_flags=list(d.get("quality_flags", [])),
            capabilities=list(d.get("capabilities", [])),
            summary_stats=dict(d.get("summary_stats", {})),
        )


@dataclass
class CandidateEvent:
    id: str
    frame: int
    start_frame: int
    end_frame_exclusive: int
    event_type: str  # scene_boundary | sustained_static | audio_onset | energy_rise | motion_change | safe_hold
    confidence: float
    reason: str
    evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ParameterBounds:
    kind: str  # "float" | "int" | "str" | "bool"
    default: Any
    min_val: float | None = None
    max_val: float | None = None
    choices: tuple[str, ...] | None = None

    def validate(self, name: str, value: Any) -> Any:
        if self.kind == "bool":
            if not isinstance(value, bool):
                raise SchemaValidationError(f"Param {name} must be bool, got {type(value).__name__}")
            return value
        if self.kind == "str":
            if not isinstance(value, str):
                raise SchemaValidationError(f"Param {name} must be str")
            if self.choices and value not in self.choices:
                raise SchemaValidationError(f"Param {name}={value!r} not in {self.choices}")
            return value
        if self.kind == "int":
            if isinstance(value, bool) or not isinstance(value, int):
                raise SchemaValidationError(f"Param {name} must be int")
            if self.min_val is not None and value < self.min_val:
                raise SchemaValidationError(f"Param {name}={value} < {self.min_val}")
            if self.max_val is not None and value > self.max_val:
                raise SchemaValidationError(f"Param {name}={value} > {self.max_val}")
            return value
        if self.kind == "float":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise SchemaValidationError(f"Param {name} must be float")
            fval = float(value)
            if not math.isfinite(fval):
                raise SchemaValidationError(f"Param {name} is non-finite: {value}")
            if self.min_val is not None and fval < self.min_val:
                raise SchemaValidationError(f"Param {name}={fval} < {self.min_val}")
            if self.max_val is not None and fval > self.max_val:
                raise SchemaValidationError(f"Param {name}={fval} > {self.max_val}")
            return fval
        raise SchemaValidationError(f"Unknown param kind: {self.kind}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "default": self.default,
            "min": self.min_val,
            "max": self.max_val,
            "choices": list(self.choices) if self.choices else None,
        }


@dataclass(frozen=True)
class EffectSpec:
    id: str
    name: str
    family: str  # camera | layout | transition | color | graphics
    version: str
    milestone: str  # core16 | expand28 | expand40
    description: str
    engine: str  # primary engine: ffmpeg | remotion | hyperframes
    supported_engines: tuple[str, ...]
    required_features: tuple[str, ...]
    supported_profiles: tuple[str, ...]
    parameter_schema: dict[str, ParameterBounds]
    min_duration_s: float
    max_duration_s: float
    max_intensity: float
    layer: str  # transform | mask | transition | color | overlay
    safety_class: str  # safe_all | subject_aware | boundary_only | opt_in_layout | audio_reactive
    incompatibilities: tuple[str, ...]
    fallback_id: str | None
    estimated_cost: float
    uses_neighbor_frames: bool = False
    applicability: str = ""
    excluded_cases: str = ""
    is_2d_illusion: bool = False

    def validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        unknown = set(params.keys()) - set(self.parameter_schema.keys())
        if unknown:
            raise SchemaValidationError(f"Unknown parameters for {self.id}: {sorted(unknown)}")
        validated: dict[str, Any] = {}
        for key, bounds in self.parameter_schema.items():
            val = params.get(key, bounds.default)
            validated[key] = bounds.validate(key, val)
        if "intensity" in validated and validated["intensity"] > self.max_intensity:
            raise SchemaValidationError(
                f"Effect {self.id} intensity {validated['intensity']} exceeds max {self.max_intensity}"
            )
        return validated

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "family": self.family,
            "version": self.version,
            "milestone": self.milestone,
            "description": self.description,
            "engine": self.engine,
            "supported_engines": list(self.supported_engines),
            "required_features": list(self.required_features),
            "supported_profiles": list(self.supported_profiles),
            "parameter_schema": {k: v.to_dict() for k, v in self.parameter_schema.items()},
            "min_duration_s": self.min_duration_s,
            "max_duration_s": self.max_duration_s,
            "max_intensity": self.max_intensity,
            "layer": self.layer,
            "safety_class": self.safety_class,
            "incompatibilities": list(self.incompatibilities),
            "fallback_id": self.fallback_id,
            "estimated_cost": self.estimated_cost,
            "uses_neighbor_frames": self.uses_neighbor_frames,
            "applicability": self.applicability,
            "excluded_cases": self.excluded_cases,
            "is_2d_illusion": self.is_2d_illusion,
        }


@dataclass
class EffectEvent:
    id: str
    effect_id: str
    effect_version: str
    start_frame: int
    end_frame_exclusive: int
    layer: str
    region: dict[str, float]
    parameters: dict[str, Any]
    easing: str
    priority: int
    engine: str
    reason_codes: list[str]
    evidence_refs: list[str]
    fallback_id: str | None
    uses_neighbor_frames: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EditPlan:
    schema_version: str
    planner_version: str
    source_hash: str
    config_hash: str
    seed: int
    profile: str
    profile_selection_reason: str
    width: int
    height: int
    fps: Rational
    duration_frames: int
    aspect_policy: str  # preserve | pad_blur_9x16 | pad_solid_9x16 | smart_crop_9x16
    audio_policy: dict[str, Any]
    engine_capabilities: dict[str, Any]
    events: list[EffectEvent]
    status: str  # READY | NO_SAFE_EFFECTS
    warnings: list[str] = field(default_factory=list)
    plan_hash: str = ""

    def compute_plan_hash(self) -> str:
        payload = self.to_dict()
        payload.pop("plan_hash", None)
        return canonical_hash(payload)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["fps"] = self.fps.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EditPlan":
        return cls(
            schema_version=str(d.get("schema_version", SCHEMA_VERSION)),
            planner_version=str(d.get("planner_version", PLANNER_VERSION)),
            source_hash=str(d["source_hash"]),
            config_hash=str(d["config_hash"]),
            seed=int(d["seed"]),
            profile=str(d["profile"]),
            profile_selection_reason=str(d.get("profile_selection_reason", "")),
            width=int(d["width"]),
            height=int(d["height"]),
            fps=Rational.from_dict(d["fps"]),
            duration_frames=int(d["duration_frames"]),
            aspect_policy=str(d["aspect_policy"]),
            audio_policy=dict(d["audio_policy"]),
            engine_capabilities=dict(d.get("engine_capabilities", {})),
            events=[EffectEvent(**e) for e in d.get("events", [])],
            status=str(d.get("status", "READY")),
            warnings=list(d.get("warnings", [])),
            plan_hash=str(d.get("plan_hash", "")),
        )


@dataclass
class RenderResult:
    artifact_path: str
    engines_used: list[str]
    event_engine_map: dict[str, str]
    timings_s: dict[str, float]
    ffprobe_info: dict[str, Any]
    status: str  # PASS | FAIL | BLOCKED
    warnings: list[str] = field(default_factory=list)
    fallback_history: list[dict[str, Any]] = field(default_factory=list)
    encoding_settings: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationCheck:
    name: str
    status: str  # PASS | FAIL | SKIP | BLOCKED
    expected: Any
    actual: Any
    evidence_path: str | None = None
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationResult:
    schema_version: str
    overall_status: str  # PASS | FAIL | SKIP | BLOCKED
    technical_status: str
    aesthetic_status: str  # PENDING_HUMAN_REVIEW | APPROVED
    checks: list[ValidationCheck]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
