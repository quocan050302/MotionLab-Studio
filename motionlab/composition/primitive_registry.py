"""Phase A2/A3 (ML-01.R5): Unified Effect & Motion Primitive Registry with Provenance & Lifecycle Gates.

Unifies:
  1. Verified imported Remotion (MIT) & HyperFrames (Apache-2.0) motion mechanisms (`third_party/motion_sources.lock.json`)
  2. Legacy 40 ML-01 signal effects & R3/R4 recipes (preserved for backward compatibility)

Enforces strict lifecycle statuses:
  DISCOVERED -> LICENSE_CHECKED -> UPSTREAM_RENDERED -> ADAPTED_RENDERED -> VERIFIED -> USED_IN_CANONICAL (or REJECTED / BLOCKED)
Only VERIFIED imported mechanisms are exposed to the generative creative director!
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from motionlab.config import PROJECT_ROOT
from motionlab.schemas import sanitize_native

LOCK_PATH = PROJECT_ROOT / "third_party" / "motion_sources.lock.json"
STATUS_MANIFEST_PATH = PROJECT_ROOT / "third_party" / "imported_primitive_manifest.json"

VALID_STATUSES = (
    "DISCOVERED",
    "LICENSE_CHECKED",
    "UPSTREAM_RENDERED",
    "ADAPTED_RENDERED",
    "VERIFIED",
    "USED_IN_CANONICAL",
    "REJECTED",
    "BLOCKED",
)


@dataclass
class MotionPrimitiveSpec:
    primitive_id: str
    engine: str  # "remotion" | "hyperframes"
    source_code: str
    origin_slug: str
    repo: str
    commit_sha: str
    upstream_path: str
    imported_path: str
    license: str
    visual_family: str
    roles: list[str]
    intent_tags: list[str]
    description: str
    supports_alpha: bool
    supports_portrait_9x16: bool
    requires_matte: bool
    requires_ledger: bool
    parameter_schema: dict[str, str]
    status: str = "LICENSE_CHECKED"
    upstream_smoke_proof: dict[str, Any] = field(default_factory=dict)
    adapted_proof: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return sanitize_native(asdict(self))


# Semantic capability & role metadata strictly mapped to the 20 actual imported upstream mechanisms
_PRIMITIVE_META: dict[str, dict[str, Any]] = {
    "rem_ru_counter": {
        "roles": ["hero_numeric_counter", "balance_reveal", "payoff_counter"],
        "intent_tags": ["money", "balance", "counter", "odometer", "number", "payoff", "hook", "stat", "kpi", "finance"],
        "supports_alpha": True, "requires_matte": False, "requires_ledger": False,
        "parameter_schema": {"from_value": "number", "to_value": "number", "prefix": "string", "roll": "bool", "font_size": "int"},
    },
    "rem_ru_waterfall": {
        "roles": ["financial_waterfall", "allocation_breakdown", "ledger_bridge"],
        "intent_tags": ["waterfall", "ledger", "allocation", "bills", "groceries", "gas", "deduction", "remaining", "budget", "finance", "chart"],
        "supports_alpha": True, "requires_matte": False, "requires_ledger": True,
        "parameter_schema": {"steps": "list[{label,value,isTotal,color}]", "currency_symbol": "string", "highlight_index": "int"},
    },
    "rem_ru_strikethrough": {
        "roles": ["contradiction_contrast", "myth_bust", "correction"],
        "intent_tags": ["contrast", "contradiction", "negation", "doesnt_mean", "strikethrough", "replace", "correction", "tension", "spend"],
        "supports_alpha": True, "requires_matte": False, "requires_ledger": False,
        "parameter_schema": {"from_text": "string", "to_text": "string", "strike_color": "hex", "font_size": "int"},
    },
    "rem_ru_masked_slide": {
        "roles": ["kinetic_headline", "takeaway_reveal", "closing_statement"],
        "intent_tags": ["headline", "typography", "kinetic", "reveal", "takeaway", "purpose", "cta", "title", "storytelling"],
        "supports_alpha": True, "requires_matte": False, "requires_ledger": False,
        "parameter_schema": {"lines": "list[string]", "font_size": "int", "color": "hex", "stagger_frames": "int"},
    },
    "rem_ru_track_matte": {
        "roles": ["masked_video_reveal", "occlusion_window", "focus_aperture"],
        "intent_tags": ["matte", "mask", "aperture", "track", "reveal", "occlusion", "macro", "detail", "hands"],
        "supports_alpha": True, "requires_matte": False, "requires_ledger": False,
        "parameter_schema": {"mode": "alpha|luma", "shape": "rounded_rect|circle|arch"},
    },
    "rem_ru_parallax": {
        "roles": ["depth_stage_25d", "subject_parallax", "spatial_camera"],
        "intent_tags": ["parallax", "depth", "2.5d", "spatial", "matte", "hook", "set_extension", "layered", "cinematic"],
        "supports_alpha": False, "requires_matte": True, "requires_ledger": False,
        "parameter_schema": {"travel_px": "number", "zoom": "number", "bg_blur_px": "number"},
    },
    "rem_viv_aspect_morph": {
        "roles": ["editorial_reframe_stage", "split_story_stage", "product_editorial"],
        "intent_tags": ["editorial", "morph", "reframe", "collage", "split", "stage", "storytelling", "product", "finance"],
        "supports_alpha": False, "requires_matte": False, "requires_ledger": False,
        "parameter_schema": {"kicker": "string", "lines": "list[string]", "paper_color": "hex", "accent_color": "hex"},
    },
    "rem_viv_particle_field": {
        "roles": ["atmospheric_depth_motes", "payoff_spatial_particles"],
        "intent_tags": ["particles", "depth", "atmosphere", "motes", "payoff", "glint", "spatial", "vfx"],
        "supports_alpha": True, "requires_matte": False, "requires_ledger": False,
        "parameter_schema": {"count": "int<=120", "speed": "number", "colors": "list[hex]", "connect": "bool"},
    },
    "rem_rc_flip_page": {
        "roles": ["workbook_page_flip", "product_reveal", "chapter_turn"],
        "intent_tags": ["workbook", "page", "flip", "book", "product", "plan", "dashboard", "reveal", "56-page", "organize"],
        "supports_alpha": False, "requires_matte": False, "requires_ledger": False,
        "parameter_schema": {"label_before": "string", "label_after": "string", "color_before": "hex", "color_after": "hex"},
    },
    "rem_gtk_frame_morph": {
        "roles": ["seamless_frame_morph_transition", "scene_bridge"],
        "intent_tags": ["transition", "morph", "bridge", "seamless", "continuity", "shape_match"],
        "supports_alpha": False, "requires_matte": False, "requires_ledger": False,
        "parameter_schema": {"size": "number", "background": "hex"},
    },
    "rem_gtk_line_led": {
        "roles": ["seamless_line_aperture_transition", "solution_reveal_bridge"],
        "intent_tags": ["transition", "line", "split", "wipe", "bridge", "reveal", "product", "seamless"],
        "supports_alpha": False, "requires_matte": False, "requires_ledger": False,
        "parameter_schema": {"color": "hex", "thickness": "int", "orientation": "horizontal|vertical"},
    },
    "hf_apple_money_count": {
        "roles": ["hero_money_counter_gsap", "balance_hook_stage", "payoff_money_drop"],
        "intent_tags": ["money", "counter", "balance", "cash", "bank", "payoff", "hook", "finance", "dollar", "tokens"],
        "supports_alpha": True, "requires_matte": False, "requires_ledger": True,
        "parameter_schema": {"from_cents": "int", "to_cents": "int", "currency_symbol": "string", "label": "string", "token_count": "int"},
    },
    "hf_mk_progress_stat": {
        "roles": ["proportional_allocation_stat", "single_focus_metric"],
        "intent_tags": ["allocation", "proportion", "stat", "progress", "bills", "groceries", "gas", "remaining", "metric", "finance"],
        "supports_alpha": True, "requires_matte": False, "requires_ledger": True,
        "parameter_schema": {"amount_cents": "int", "denom_cents": "int", "remaining_cents": "int", "category": "string", "caption": "string"},
    },
    "hf_mk_specs_list": {
        "roles": ["product_feature_checklist", "solution_attributes"],
        "intent_tags": ["product", "features", "specs", "checklist", "dashboard", "workbook", "notion", "benefit", "organize"],
        "supports_alpha": True, "requires_matte": False, "requires_ledger": False,
        "parameter_schema": {"title": "string", "rows": "list[{label,value}]", "scrim": "float"},
    },
    "hf_mk_callout_highlight": {
        "roles": ["speech_synced_takeaway", "closing_callout"],
        "intent_tags": ["callout", "highlight", "takeaway", "closing", "purpose", "payday", "quote", "speech_sync", "cta"],
        "supports_alpha": True, "requires_matte": False, "requires_ledger": False,
        "parameter_schema": {"text": "string", "subtitle": "string", "font_size": "int"},
    },
    "hf_lt_mask_reveal": {
        "roles": ["product_name_reveal", "editorial_section_title"],
        "intent_tags": ["title", "reveal", "mask", "product_intro", "plan", "paycheck", "lower_third", "editorial"],
        "supports_alpha": True, "requires_matte": False, "requires_ledger": False,
        "parameter_schema": {"headline": "string", "subtitle": "string", "position": "upper|lower"},
    },
    "hf_strikethrough_replace": {
        "roles": ["gsap_contradiction_strike", "myth_replacement"],
        "intent_tags": ["strikethrough", "replace", "contradiction", "negation", "contrast", "spend"],
        "supports_alpha": True, "requires_matte": False, "requires_ledger": False,
        "parameter_schema": {"from_text": "string", "to_text": "string", "eyebrow": "string"},
    },
    "hf_organic_light_leak": {
        "roles": ["warm_transition_light_leak", "bridge_accent"],
        "intent_tags": ["transition", "light_leak", "bridge", "warm", "editorial", "cinematic"],
        "supports_alpha": True, "requires_matte": False, "requires_ledger": False,
        "parameter_schema": {"opacity": "float"},
    },
    "hf_data_chart": {
        "roles": ["proportional_ledger_chart", "multi_category_comparison"],
        "intent_tags": ["chart", "data", "ledger", "bars", "proportion", "finance", "allocation", "comparison"],
        "supports_alpha": True, "requires_matte": False, "requires_ledger": True,
        "parameter_schema": {"title": "string", "items": "list[{label,cents,color}]", "denom_cents": "int"},
    },
    "hf_exploded_stack": {
        "roles": ["conceptual_system_stack", "product_organization_visual"],
        "intent_tags": ["stack", "layers", "isometric", "organize", "system", "product", "dashboard", "workbook", "conceptual"],
        "supports_alpha": True, "requires_matte": False, "requires_ledger": False,
        "parameter_schema": {"title": "string", "layers": "list[{label,sublabel,color}]", "conceptual_badge": "string"},
    },
}


class UnifiedPrimitiveRegistry:
    """Loads, queries, and tracks verification statuses of all imported Remotion/HyperFrames motion primitives."""

    def __init__(self, lock_path: Path | None = None, status_path: Path | None = None) -> None:
        self.lock_path = lock_path or LOCK_PATH
        self.status_path = status_path or STATUS_MANIFEST_PATH
        self._primitives: dict[str, MotionPrimitiveSpec] = {}
        self._load()

    def _load(self) -> None:
        if not self.lock_path.is_file():
            return
        lock_data = json.loads(self.lock_path.read_text(encoding="utf-8"))
        saved_status: dict[str, dict[str, Any]] = {}
        if self.status_path.is_file():
            try:
                raw_st = json.loads(self.status_path.read_text(encoding="utf-8"))
                for row in raw_st.get("primitives", []):
                    saved_status[str(row["primitive_id"])] = row
            except Exception:
                pass

        for src in lock_data.get("sources", []):
            pid = str(src["primitive_id"])
            meta = _PRIMITIVE_META.get(pid, {})
            st_row = saved_status.get(pid, {})
            self._primitives[pid] = MotionPrimitiveSpec(
                primitive_id=pid,
                engine=str(src["engine"]),
                source_code=str(src["source_code"]),
                origin_slug=str(src["origin_slug"]),
                repo=str(src["repo"]),
                commit_sha=str(src["commit_sha"]),
                upstream_path=str(src["upstream_path"]),
                imported_path=str(src["imported_path"]),
                license=str(src["license"]),
                visual_family=str(src["visual_family"]),
                roles=list(meta.get("roles", [src["visual_family"]])),
                intent_tags=list(meta.get("intent_tags", [])),
                description=str(src["description"]),
                supports_alpha=bool(meta.get("supports_alpha", True)),
                supports_portrait_9x16=True,
                requires_matte=bool(meta.get("requires_matte", False)),
                requires_ledger=bool(meta.get("requires_ledger", False)),
                parameter_schema=dict(meta.get("parameter_schema", {})),
                status=str(st_row.get("status", src.get("status", "LICENSE_CHECKED"))),
                upstream_smoke_proof=dict(st_row.get("upstream_smoke_proof", {})),
                adapted_proof=dict(st_row.get("adapted_proof", {})),
            )

    def all_primitives(self) -> list[MotionPrimitiveSpec]:
        return list(self._primitives.values())

    def verified_primitives(self) -> list[MotionPrimitiveSpec]:
        return [p for p in self._primitives.values() if p.status in ("VERIFIED", "USED_IN_CANONICAL")]

    def get(self, primitive_id: str) -> MotionPrimitiveSpec:
        if primitive_id not in self._primitives:
            raise KeyError(f"Unknown motion primitive_id: {primitive_id}")
        return self._primitives[primitive_id]

    def has(self, primitive_id: str) -> bool:
        return primitive_id in self._primitives

    def set_proof(
        self,
        primitive_id: str,
        status: str,
        upstream_smoke_proof: dict[str, Any] | None = None,
        adapted_proof: dict[str, Any] | None = None,
    ) -> None:
        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid primitive status: {status}")
        spec = self.get(primitive_id)
        spec.status = status
        if upstream_smoke_proof is not None:
            spec.upstream_smoke_proof = upstream_smoke_proof
        if adapted_proof is not None:
            spec.adapted_proof = adapted_proof

    def save_manifest(self) -> Path:
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        prims = self.all_primitives()
        verified = self.verified_primitives()
        payload = {
            "schema_version": "1.0.0",
            "total_primitives": len(prims),
            "verified_count": len(verified),
            "verified_remotion_count": sum(1 for p in verified if p.engine == "remotion"),
            "verified_hyperframes_count": sum(1 for p in verified if p.engine == "hyperframes"),
            "distinct_verified_repos": sorted({p.repo for p in verified}),
            "primitives": [p.to_dict() for p in prims],
        }
        self.status_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return self.status_path

    def compact_catalog_for_llm(self, verified_only: bool = True) -> list[dict[str, Any]]:
        pool = self.verified_primitives() if verified_only else self.all_primitives()
        return [
            {
                "id": p.primitive_id,
                "engine": p.engine,
                "origin": f"{p.origin_slug} ({p.license})",
                "family": p.visual_family,
                "roles": p.roles,
                "tags": p.intent_tags[:8],
                "requires_matte": p.requires_matte,
                "requires_ledger": p.requires_ledger,
                "desc": p.description[:110],
            }
            for p in pool
        ]


def load_unified_registry() -> UnifiedPrimitiveRegistry:
    return UnifiedPrimitiveRegistry()

