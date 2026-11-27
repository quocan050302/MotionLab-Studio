"""Phase A3 (ML-01.R5): Typed Generative Storyboard & SceneSpec DSL + Fail-Closed Security & Grounding Validator.

Replaces fixed VFX01..12 lookup tables with an expressive, safe, typed scene graph that a local LLM
composes from VERIFIED imported Remotion (MIT) & HyperFrames (Apache-2.0) primitives.
Zero arbitrary code execution (`eval`/`exec`/`<script`/remote URLs/path traversal all fail closed).
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from motionlab.composition.primitive_registry import UnifiedPrimitiveRegistry, load_unified_registry
from motionlab.schemas import SchemaValidationError, canonical_hash, ensure_finite, sanitize_native
from motionlab.semantic.story_graph import StoryGraph, validate_claim_graph

GENERATIVE_SCHEMA_VERSION = "3.0.0"

VALID_STAGE_MODES = (
    "live_depth_matte",
    "aspect_morph_editorial",
    "full_graphic_cutaway",
    "tracked_macro_split",
    "live_clean_bridge",
)
VALID_LAYER_ROLES = ("background_graphics", "foreground_graphics", "hero_stage")
VALID_TRANSITIONS = ("none", "frame_morph", "line_led", "flip_page", "sheet_sweep")
VALID_EASINGS = ("linear", "ease_out_cubic", "ease_in_out_cubic", "spring_editorial", "bezier_reveal")

MAX_SCENES = 20
MAX_LAYERS_PER_SCENE = 24
MAX_KEYFRAMES_PER_SCENE = 1500
MAX_PARTICLES = 120
MAX_STRING_LEN = 480

_FORBIDDEN_TOKEN_PATTERNS = (
    re.compile(r"(?i)https?://"),
    re.compile(r"(?i)<\s*script"),
    re.compile(r"(?i)javascript\s*:"),
    re.compile(r"(?i)\beval\s*\("),
    re.compile(r"(?i)\bnew\s+Function\s*\("),
    re.compile(r"(?i)\b__import__\s*\("),
    re.compile(r"(?i)data\s*:\s*text/html"),
    re.compile(r"(?:\.\./|\.\.\\)"),
)


@dataclass
class KeyframePoint:
    frame: int
    property: str
    value: float
    easing: str = "ease_out_cubic"

    def to_dict(self) -> dict[str, Any]:
        return sanitize_native(asdict(self))


@dataclass
class MotionTrack:
    track_id: str
    target_layer_id: str
    keyframes: list[KeyframePoint] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return sanitize_native(asdict(self))


@dataclass
class ClaimRef:
    claim_id: str
    kind: str  # "ledger_step" | "product_name" | "product_attribute" | "negated_contrast" | "cta" | "statement"
    verbatim_quote: str
    provenance: str  # "verbatim_spoken" | "derived_arithmetic"
    word_ids: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return sanitize_native(asdict(self))


@dataclass
class AssetRef:
    asset_id: str
    asset_kind: str  # "source_video" | "modnet_matte" | "prop_matte" | "conceptual_schematic" | "local_media"
    provenance: str
    status: str = "VERIFIED"  # "VERIFIED" | "ASSET_MISSING_CONCEPTUAL"

    def to_dict(self) -> dict[str, Any]:
        return sanitize_native(asdict(self))


@dataclass
class LayerSpec:
    layer_id: str
    primitive_id: str
    engine: str  # "remotion" | "hyperframes"
    layer_role: str  # "background_graphics" | "foreground_graphics" | "hero_stage"
    z_order: int
    params: dict[str, Any] = field(default_factory=dict)
    motion_tracks: list[MotionTrack] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return sanitize_native(asdict(self))


@dataclass
class TransitionSpec:
    kind: str = "none"  # "none" | "frame_morph" | "line_led" | "flip_page"
    duration_frames: int = 12
    shared_element_id: str = ""
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return sanitize_native(asdict(self))


@dataclass
class SoundIntent:
    cue_id: str
    family: str
    target_frame: int
    variant: int = 0
    gain_db: float = -5.5
    pan: float = 0.0
    rationale: str = ""
    word_ids: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return sanitize_native(asdict(self))


@dataclass
class SceneSpec:
    scene_id: str
    start_frame: int
    end_frame_exclusive: int
    role: str
    stage_mode: str
    viewer_takeaway: str
    unit_ids: list[str] = field(default_factory=list)
    word_ids: list[int] = field(default_factory=list)
    claim_refs: list[ClaimRef] = field(default_factory=list)
    asset_refs: list[AssetRef] = field(default_factory=list)
    layers: list[LayerSpec] = field(default_factory=list)
    transition_in: TransitionSpec = field(default_factory=TransitionSpec)
    sound_intents: list[SoundIntent] = field(default_factory=list)
    persistent_ledger_step_id: str | None = None
    show_persistent_ribbon: bool = False
    requires_matte: bool = False
    intentional_cutaway: bool = False
    parameters: dict[str, Any] = field(default_factory=dict)
    expected_visual_change: str = ""
    rationale: str = ""
    # R5.R3 Causal Selection & Product Grounding Provenance (backward-compatible defaults)
    chosen_candidate_id: str = ""
    candidate_provenance: str = ""  # "LLM_DIRECT" | "HYBRID_GLOBAL_RANKED" | "REPLAN_SAFE_FALLBACK" | "ART_DIRECTED_QA"
    selection_rationale: str = ""
    product_asset_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return sanitize_native(asdict(self))


@dataclass
class StoryboardSpec:
    schema_version: str
    mode: str  # "AUTO_GENERATIVE" | "ART_DIRECTED_BENCHMARK" | "ART_DIRECTED_QA"
    source_hash: str
    transcript_hash: str
    story_graph_hash: str
    niche: str
    niche_confidence: float
    core_value_proposition: str
    width: int
    height: int
    fps_num: int
    fps_den: int
    total_frames: int
    seed: int
    palette: dict[str, str]
    scenes: list[SceneSpec] = field(default_factory=list)
    retrieved_candidates: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    director_model_id: str = ""
    raw_director_response_sha256: str = ""
    validation_notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    storyboard_hash: str = ""
    # R5.R3 Causal Global Selection Lock & Retrieval Provenance
    selected_graph_hash: str = ""
    selection_trace: list[dict[str, Any]] = field(default_factory=list)
    retrieval_mode: str = "ontology_hybrid_sparse"
    catalog_index_hash: str = ""

    def compute_hash(self) -> str:
        payload = self.to_dict()
        payload.pop("storyboard_hash", None)
        return canonical_hash(payload)

    def to_dict(self) -> dict[str, Any]:
        return sanitize_native(asdict(self))


def _scan_forbidden_recursive(obj: Any, path: str = "$") -> list[str]:
    errs: list[str] = []
    if isinstance(obj, str):
        if len(obj) > MAX_STRING_LEN and not path.endswith("core_value_proposition"):
            errs.append(f"string_too_long:{path}({len(obj)}>{MAX_STRING_LEN})")
        for pat in _FORBIDDEN_TOKEN_PATTERNS:
            if pat.search(obj):
                errs.append(f"forbidden_token:{path}:{pat.pattern}")
    elif isinstance(obj, float):
        if not math.isfinite(obj):
            errs.append(f"non_finite_float:{path}:{obj}")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            errs.extend(_scan_forbidden_recursive(k, f"{path}.key"))
            errs.extend(_scan_forbidden_recursive(v, f"{path}.{k}"))
    elif isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            errs.extend(_scan_forbidden_recursive(item, f"{path}[{i}]"))
    return errs


def validate_storyboard_spec(
    spec: StoryboardSpec,
    story_graph: StoryGraph | None = None,
    registry: UnifiedPrimitiveRegistry | None = None,
    require_verified_primitives: bool = True,
) -> list[str]:
    """Fail-closed validator for StoryboardSpec: security, frame clocks, verified primitives, caps & semantic grounding."""
    reg = registry or load_unified_registry()
    errors: list[str] = []

    if spec.width < 120 or spec.height < 120 or spec.total_frames < 2:
        errors.append(f"invalid_canvas_or_frames:{spec.width}x{spec.height}@{spec.total_frames}")
    if len(spec.scenes) > MAX_SCENES:
        errors.append(f"too_many_scenes:{len(spec.scenes)}>{MAX_SCENES}")

    # Security & finite numeric scan over entire tree
    errors.extend(_scan_forbidden_recursive(spec.to_dict()))

    prev_end = 0
    seen_ids: set[str] = set()
    all_texts: list[str] = []

    for sc in spec.scenes:
        if sc.scene_id in seen_ids:
            errors.append(f"duplicate_scene_id:{sc.scene_id}")
        seen_ids.add(sc.scene_id)

        if sc.start_frame < 0 or sc.end_frame_exclusive > spec.total_frames or sc.start_frame >= sc.end_frame_exclusive:
            errors.append(f"out_of_bounds_scene:{sc.scene_id}:[{sc.start_frame},{sc.end_frame_exclusive})/{spec.total_frames}")
        if sc.start_frame < prev_end:
            errors.append(f"overlapping_scene:{sc.scene_id}:start={sc.start_frame}<prev_end={prev_end}")
        prev_end = max(prev_end, sc.end_frame_exclusive)

        if sc.stage_mode not in VALID_STAGE_MODES:
            errors.append(f"invalid_stage_mode:{sc.scene_id}:{sc.stage_mode}")
        if sc.transition_in.kind not in VALID_TRANSITIONS:
            errors.append(f"invalid_transition:{sc.scene_id}:{sc.transition_in.kind}")
        if sc.transition_in.duration_frames < 0 or sc.transition_in.duration_frames > 24:
            errors.append(f"invalid_transition_duration:{sc.scene_id}:{sc.transition_in.duration_frames}")

        if len(sc.layers) > MAX_LAYERS_PER_SCENE:
            errors.append(f"too_many_layers:{sc.scene_id}:{len(sc.layers)}>{MAX_LAYERS_PER_SCENE}")

        kf_total = 0
        for ly in sc.layers:
            if not reg.has(ly.primitive_id):
                errors.append(f"unknown_primitive:{sc.scene_id}:{ly.primitive_id}")
                continue
            pspec = reg.get(ly.primitive_id)
            if require_verified_primitives and pspec.status not in ("VERIFIED", "USED_IN_CANONICAL"):
                errors.append(f"unverified_primitive:{sc.scene_id}:{ly.primitive_id}:status={pspec.status}")
            if ly.engine != pspec.engine:
                errors.append(f"engine_mismatch:{sc.scene_id}:{ly.primitive_id}:got={ly.engine},expected={pspec.engine}")
            if ly.layer_role not in VALID_LAYER_ROLES:
                errors.append(f"invalid_layer_role:{sc.scene_id}:{ly.layer_role}")

            # Particle cap enforcement
            if ly.primitive_id == "rem_viv_particle_field":
                cnt = int(ly.params.get("count", 48))
                if cnt > MAX_PARTICLES or cnt < 0:
                    errors.append(f"particle_cap_exceeded:{sc.scene_id}:{cnt}>{MAX_PARTICLES}")

            for tr in ly.motion_tracks:
                kf_total += len(tr.keyframes)
                for kf in tr.keyframes:
                    if not math.isfinite(float(kf.value)):
                        errors.append(f"non_finite_float:{sc.scene_id}.{ly.layer_id}.{kf.property}")
                    if kf.easing not in VALID_EASINGS:
                        errors.append(f"invalid_easing:{sc.scene_id}:{kf.easing}")
                    if kf.frame < 0 or kf.frame > (sc.end_frame_exclusive - sc.start_frame):
                        errors.append(f"keyframe_out_of_scene:{sc.scene_id}:{kf.frame}")

            # Collect display text strings for semantic grounding & consumer QA-jargon audit
            for tk in ("eyebrow", "title", "subtitle", "caption", "headline", "text", "from_text", "to_text", "label_before", "label_after", "category"):
                v = ly.params.get(tk)
                if isinstance(v, str) and v.strip():
                    all_texts.append(v.strip())
                    if "(derived)" in v.lower() or "derived_arithmetic" in v.lower() or "verbatim_spoken" in v.lower():
                        errors.append(f"consumer_qa_jargon_leak:{sc.scene_id}.{ly.layer_id}.{tk}")
            if isinstance(ly.params.get("lines"), list):
                all_texts.extend(str(x) for x in ly.params["lines"] if x)

        if kf_total > MAX_KEYFRAMES_PER_SCENE:
            errors.append(f"too_many_keyframes:{sc.scene_id}:{kf_total}>{MAX_KEYFRAMES_PER_SCENE}")

        for sfx in sc.sound_intents:
            if sfx.target_frame < 0 or sfx.target_frame >= spec.total_frames:
                errors.append(f"sfx_out_of_bounds:{sc.scene_id}:{sfx.cue_id}:{sfx.target_frame}")

    # Grounding & anti-hallucination check against StoryGraph
    if story_graph is not None and all_texts:
        g_errs = validate_claim_graph(story_graph, all_texts)
        errors.extend(g_errs)

    return errors


def validate_canonical_reuse_gate(
    spec: StoryboardSpec,
    actual_rendered_engines: set[str],
    actual_rendered_primitives: list[str],
    min_distinct_total: int = 4,
    min_per_engine: int = 2,
    registry: UnifiedPrimitiveRegistry | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Enforce Section 02B.1 Gate G2: real verified imported mechanisms across BOTH Remotion & HyperFrames (no aliases!)."""
    reg = registry or load_unified_registry()
    unique_prims = sorted(set(actual_rendered_primitives))
    rem_prims: list[str] = []
    hf_prims: list[str] = []
    origin_repos: set[str] = set()
    unverified: list[str] = []

    for pid in unique_prims:
        if not reg.has(pid):
            unverified.append(pid)
            continue
        p = reg.get(pid)
        if p.status not in ("VERIFIED", "USED_IN_CANONICAL"):
            unverified.append(f"{pid}({p.status})")
            continue
        origin_repos.add(p.repo)
        if p.engine == "remotion":
            rem_prims.append(pid)
        elif p.engine == "hyperframes":
            hf_prims.append(pid)

    passed = bool(
        not unverified
        and len(rem_prims) + len(hf_prims) >= min_distinct_total
        and len(rem_prims) >= min_per_engine
        and len(hf_prims) >= min_per_engine
        and "remotion" in actual_rendered_engines
        and "hyperframes" in actual_rendered_engines
    )
    report = {
        "passed": passed,
        "distinct_verified_total": len(rem_prims) + len(hf_prims),
        "remotion_primitives": rem_prims,
        "hyperframes_primitives": hf_prims,
        "distinct_origin_repos": sorted(origin_repos),
        "actual_rendered_engines": sorted(actual_rendered_engines),
        "unverified_or_unknown": unverified,
        "requirements": {
            "min_distinct_total": min_distinct_total,
            "min_per_engine": min_per_engine,
        },
    }
    return passed, report
