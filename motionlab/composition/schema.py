"""Declarative Composition Graph Schema, Hashing & Strict Fail-Closed Validator (ML-01.R3)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from motionlab.schemas import (
    MediaInfo,
    Rational,
    SchemaValidationError,
    canonical_hash,
    ensure_finite,
    sanitize_native,
)

COMPOSITION_SCHEMA_VERSION = "2.0.0"
VALID_LAYER_ROLES = (
    "background_source",
    "background_plate",
    "background_graphics",
    "source_detail",
    "foreground_subject_matte",
    "prop_cutout_matte",
    "foreground_graphics",
)
VALID_ALPHA_MODES = ("opaque", "straight", "premultiplied")
VALID_BLEND_MODES = ("normal", "multiply", "screen")
FORBIDDEN_PARAM_PATTERNS = ("http://", "https://", "javascript:", "<script", "eval(", "Function(", "..")


@dataclass
class CompositionLayer:
    layer_id: str
    role: str
    engine: str  # "remotion" | "hyperframes" | "ffmpeg"
    z_index: int
    asset_ref: str
    alpha_mode: str = "straight"
    blend_mode: str = "normal"
    parameters: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    required: bool = True

    def to_dict(self) -> dict[str, Any]:
        return sanitize_native(asdict(self))

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CompositionLayer":
        return cls(**d)


@dataclass
class CompositionScene:
    scene_id: str
    recipe_id: str
    recipe_name: str
    archetype: str
    start_frame: int
    end_frame_exclusive: int
    beat_role: str
    primary_engine: str  # "remotion" | "ffmpeg"
    graphics_engine: str  # "hyperframes" | "none"
    requires_matte: bool
    uses_detail_crop: bool
    substantive_composition: bool
    layers: list[CompositionLayer]
    anchor_bundle: dict[str, Any]
    parameters: dict[str, Any]
    entry_frames: int = 10
    exit_frames: int = 10
    evidence_refs: list[str] = field(default_factory=list)
    fallback_recipe_id: str | None = None
    verification_criteria: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return sanitize_native(asdict(self))

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CompositionScene":
        return cls(
            **{
                **d,
                "layers": [CompositionLayer.from_dict(x) for x in d.get("layers", [])],
            }
        )


@dataclass
class CompositionGraph:
    schema_version: str
    source_hash: str
    config_hash: str
    seed: int
    profile: str
    style: str
    width: int
    height: int
    fps: Rational
    duration_frames: int
    matte_required: bool
    matte_available: bool
    scenes: list[CompositionScene]
    quiet_spans: list[dict[str, Any]]
    engine_capabilities: dict[str, Any]
    metrics: dict[str, Any] = field(default_factory=dict)
    status: str = "READY"
    warnings: list[str] = field(default_factory=list)
    graph_hash: str = ""

    def compute_graph_hash(self) -> str:
        payload = self.to_dict()
        payload.pop("graph_hash", None)
        return canonical_hash(payload)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["fps"] = self.fps.to_dict()
        return sanitize_native(d)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CompositionGraph":
        return cls(
            schema_version=str(d["schema_version"]),
            source_hash=str(d["source_hash"]),
            config_hash=str(d["config_hash"]),
            seed=int(d["seed"]),
            profile=str(d["profile"]),
            style=str(d["style"]),
            width=int(d["width"]),
            height=int(d["height"]),
            fps=Rational.from_dict(d["fps"]),
            duration_frames=int(d["duration_frames"]),
            matte_required=bool(d.get("matte_required", False)),
            matte_available=bool(d.get("matte_available", False)),
            scenes=[CompositionScene.from_dict(x) for x in d.get("scenes", [])],
            quiet_spans=list(d.get("quiet_spans", [])),
            engine_capabilities=dict(d.get("engine_capabilities", {})),
            metrics=dict(d.get("metrics", {})),
            status=str(d.get("status", "READY")),
            warnings=list(d.get("warnings", [])),
            graph_hash=str(d.get("graph_hash", "")),
        )


def _scan_forbidden_strings(obj: Any, path: str = "root") -> list[str]:
    errs: list[str] = []
    if isinstance(obj, str):
        low = obj.lower()
        for pat in FORBIDDEN_PARAM_PATTERNS:
            if pat in low:
                errs.append(f"Forbidden token '{pat}' in declarative composition graph at {path}")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            errs.extend(_scan_forbidden_strings(v, f"{path}.{k}"))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            errs.extend(_scan_forbidden_strings(v, f"{path}[{i}]"))
    return errs


def validate_composition_graph(
    graph: CompositionGraph,
    media: MediaInfo,
    engine_caps: dict[str, Any] | None = None,
    strict_replay: bool = False,
) -> list[str]:
    """Fail-closed validation of CompositionGraph: schema, hash, bounds, z-order, safety, and engine eligibility."""
    ensure_finite(graph.to_dict())
    errors: list[str] = []

    if graph.schema_version != COMPOSITION_SCHEMA_VERSION:
        errors.append(f"Unsupported composition schema_version: {graph.schema_version!r}")
    if graph.source_hash != media.input_hash:
        errors.append(f"Composition source_hash mismatch: graph={graph.source_hash[:12]} vs media={media.input_hash[:12]}")
    if graph.duration_frames != media.duration_frames:
        errors.append(f"Composition duration_frames mismatch: {graph.duration_frames} vs {media.duration_frames}")
    if graph.fps.as_fraction() != media.fps.as_fraction():
        errors.append(f"Composition fps mismatch: {graph.fps.to_ffmpeg()} vs {media.fps.to_ffmpeg()}")
    if graph.width <= 0 or graph.height <= 0 or graph.width % 2 != 0 or graph.height % 2 != 0:
        errors.append(f"Composition dimensions must be positive even ints: {graph.width}x{graph.height}")

    # Security scan: no remote URLs, path traversal, or executable JS strings inside graph
    errors.extend(_scan_forbidden_strings(graph.to_dict()))

    if strict_replay:
        if not graph.graph_hash:
            errors.append("Strict replay requires canonical graph_hash.")
        elif graph.graph_hash != graph.compute_graph_hash():
            errors.append("Composition graph_hash mismatch (tampered composition graph).")
    elif graph.graph_hash and graph.graph_hash != graph.compute_graph_hash():
        errors.append("Composition graph_hash mismatch.")

    seen_scene_ids: set[str] = set()
    prev_end = -1
    for sc in sorted(graph.scenes, key=lambda s: s.start_frame):
        if sc.scene_id in seen_scene_ids:
            errors.append(f"Duplicate scene_id: {sc.scene_id}")
        seen_scene_ids.add(sc.scene_id)
        if sc.start_frame < 0 or sc.end_frame_exclusive > graph.duration_frames or sc.end_frame_exclusive <= sc.start_frame:
            errors.append(f"Illegal frame bounds on scene {sc.scene_id}: [{sc.start_frame}, {sc.end_frame_exclusive})")
        if sc.start_frame < prev_end:
            errors.append(f"Overlapping composition scene spans at {sc.scene_id} (start={sc.start_frame} < prev_end={prev_end})")
        prev_end = sc.end_frame_exclusive

        if sc.requires_matte and not graph.matte_available:
            errors.append(f"Scene {sc.scene_id} ({sc.recipe_id}) requires matte, but matte_available=False")

        if strict_replay and engine_caps is not None:
            for eng_req in (sc.primary_engine, sc.graphics_engine):
                if eng_req and eng_req != "none":
                    if not bool((engine_caps.get(eng_req) or {}).get("available")):
                        errors.append(f"Replay blocked: scene {sc.scene_id} requires unavailable/unlicensed engine '{eng_req}'")

        if not sc.layers:
            errors.append(f"Scene {sc.scene_id} has 0 layers.")
        seen_lids: set[str] = set()
        z_vals: list[int] = []
        for lyr in sc.layers:
            if lyr.layer_id in seen_lids:
                errors.append(f"Duplicate layer_id '{lyr.layer_id}' in {sc.scene_id}")
            seen_lids.add(lyr.layer_id)
            if lyr.role not in VALID_LAYER_ROLES:
                errors.append(f"Unknown layer role '{lyr.role}' in {sc.scene_id}")
            if lyr.alpha_mode not in VALID_ALPHA_MODES:
                errors.append(f"Invalid alpha_mode '{lyr.alpha_mode}' in {sc.scene_id}")
            if lyr.blend_mode not in VALID_BLEND_MODES:
                errors.append(f"Invalid blend_mode '{lyr.blend_mode}' in {sc.scene_id}")
            z_vals.append(lyr.z_index)
        if z_vals != sorted(z_vals):
            errors.append(f"Layers in {sc.scene_id} are not monotonically z-ordered: {z_vals}")
        bg_g_zs = [lyr.z_index for lyr in sc.layers if lyr.role == "background_graphics"]
        fg_m_zs = [lyr.z_index for lyr in sc.layers if lyr.role == "foreground_subject_matte"]
        if bg_g_zs and fg_m_zs and min(fg_m_zs) <= max(bg_g_zs):
            errors.append(
                f"Invalid occlusion z-order in {sc.scene_id}: foreground_subject_matte ({fg_m_zs}) "
                f"must sit strictly above background_graphics ({bg_g_zs})"
            )

    if errors:
        raise SchemaValidationError("CompositionGraph validation failed:\n- " + "\n- ".join(errors))
    return []
