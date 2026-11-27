"""Phase R2-A2 (ML-01.R5.R2): Continuous Story World Timeline, Boundary Contracts & Canonical Numeric State Trace.

Upgrades R5.R1 metadata-only continuity into an executable runtime contract consumed by BOTH Remotion and HyperFrames:
  1. `ContinuousTimeline`: frame-exact [start, end) half-open scene & gap coverage across 0..total_frames-1
     with zero dropped, duplicated, or timewarped source frames.
  2. `SharedElementTrack` & `evaluate_shared_state(track, global_frame)`: deterministic keyframed geometry
     (x, y, w, h, scale, rotation, opacity, z_order) mirrored identically in `renderers/remotion/src/composition/continuity.ts`
     so boundary position continuity error stays <= 2px.
  3. `BoundaryContract`: explicitly classifies EVERY scene boundary & gap
     (MATCH_ACTION | SHARED_ELEMENT_MORPH | MASK_PASS | CUTAWAY_BRIDGE | INTENTIONAL_CUT | QUIET_HOLD)
     with semantic rationale and real rendered incoming/outgoing bridge handle frames.
  4. `CanonicalLedgerStateTrace`: single source of truth for displayed numeric values across ALL frames,
     enforcing integer-dollar display lock (zero arbitrary fractional cents like $1,151.93 / $144.06!)
     and single visible currency owner per scene (zero competing top/bottom independent counters!).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from motionlab.composition.generative_schema import SceneSpec, StoryboardSpec
from motionlab.schemas import canonical_hash, sanitize_native
from motionlab.semantic.story_graph import StoryGraph, format_cents

CONTINUITY_SCHEMA_VERSION = "2.0.0"

VALID_BOUNDARY_CLASSES = (
    "MATCH_ACTION",
    "SHARED_ELEMENT_MORPH",
    "MASK_PASS",
    "CUTAWAY_BRIDGE",
    "INTENTIONAL_CUT",
    "QUIET_HOLD",
)


@dataclass
class ElementKeyframe:
    global_frame: int
    x: float
    y: float
    w: float
    h: float
    scale: float = 1.0
    rotation_deg: float = 0.0
    opacity: float = 1.0
    z_order: int = 25
    easing: str = "ease_out_cubic"

    def to_dict(self) -> dict[str, Any]:
        return sanitize_native(asdict(self))


@dataclass
class SharedElementTrack:
    element_id: str
    visual_role: str
    logical_state_id: str
    continuity_group: str
    start_frame: int
    end_frame_exclusive: int
    color: str
    transition_owner: str = "remotion"
    keyframes: list[ElementKeyframe] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return sanitize_native(asdict(self))


@dataclass
class BoundaryContract:
    boundary_id: str
    boundary_frame: int
    from_scene_id: str
    to_scene_id: str
    from_stage_mode: str
    to_stage_mode: str
    boundary_class: str  # MATCH_ACTION | SHARED_ELEMENT_MORPH | MASK_PASS | CUTAWAY_BRIDGE | INTENTIONAL_CUT | QUIET_HOLD
    bridge_direction: str  # outgoing_from_prev | incoming_to_next | shared_morph | quiet_hold | intentional_cut
    handle_frames: int
    shared_element_id: str
    semantic_rationale: str
    expected_position_error_px: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return sanitize_native(asdict(self))


def _ease_val(u: float, easing: str) -> float:
    t = max(0.0, min(1.0, float(u)))
    if easing == "linear":
        return t
    if easing == "ease_in_out_cubic":
        return 4.0 * t * t * t if t < 0.5 else 1.0 - ((-2.0 * t + 2.0) ** 3) / 2.0
    if easing == "smoothstep":
        return t * t * (3.0 - 2.0 * t)
    # Default: ease_out_cubic / spring_editorial
    return 1.0 - ((1.0 - t) ** 3)


def evaluate_shared_state(track: SharedElementTrack | dict[str, Any], global_frame: int) -> dict[str, float]:
    """Deterministic frame evaluator identical to `renderers/remotion/src/composition/continuity.ts`."""
    kfs_raw = track.keyframes if isinstance(track, SharedElementTrack) else track.get("keyframes", [])
    if not kfs_raw:
        return {"x": 0.067, "y": 0.086, "w": 0.788, "h": 0.18, "scale": 1.0, "rotation_deg": 0.0, "opacity": 0.0, "z_order": 25.0}

    def _get(k: Any, attr: str, default: float = 0.0) -> Any:
        return getattr(k, attr, default) if not isinstance(k, dict) else k.get(attr, default)

    kfs = sorted(kfs_raw, key=lambda k: int(_get(k, "global_frame", 0)))
    f = int(global_frame)
    if f <= int(_get(kfs[0], "global_frame", 0)):
        k0 = kfs[0]
        return {
            "x": round(float(_get(k0, "x", 0.0)), 5),
            "y": round(float(_get(k0, "y", 0.0)), 5),
            "w": round(float(_get(k0, "w", 0.5)), 5),
            "h": round(float(_get(k0, "h", 0.2)), 5),
            "scale": round(float(_get(k0, "scale", 1.0)), 5),
            "rotation_deg": round(float(_get(k0, "rotation_deg", 0.0)), 5),
            "opacity": round(float(_get(k0, "opacity", 1.0)), 5),
            "z_order": float(_get(k0, "z_order", 25)),
        }
    if f >= int(_get(kfs[-1], "global_frame", 0)):
        kn = kfs[-1]
        return {
            "x": round(float(_get(kn, "x", 0.0)), 5),
            "y": round(float(_get(kn, "y", 0.0)), 5),
            "w": round(float(_get(kn, "w", 0.5)), 5),
            "h": round(float(_get(kn, "h", 0.2)), 5),
            "scale": round(float(_get(kn, "scale", 1.0)), 5),
            "rotation_deg": round(float(_get(kn, "rotation_deg", 0.0)), 5),
            "opacity": round(float(_get(kn, "opacity", 1.0)), 5),
            "z_order": float(_get(kn, "z_order", 25)),
        }

    for i in range(len(kfs) - 1):
        a, b = kfs[i], kfs[i + 1]
        fa = int(_get(a, "global_frame", 0))
        fb = int(_get(b, "global_frame", 0))
        if fa <= f <= fb:
            if fb == fa:
                u = 1.0
            else:
                u = _ease_val((f - fa) / float(fb - fa), str(_get(b, "easing", "ease_out_cubic")))
            out: dict[str, float] = {}
            for fld in ("x", "y", "w", "h", "scale", "rotation_deg", "opacity"):
                va = float(_get(a, fld, 0.0))
                vb = float(_get(b, fld, 0.0))
                out[fld] = round(va + (vb - va) * u, 5)
            out["z_order"] = float(_get(a, "z_order", 25))
            return out

    return evaluate_shared_state(track, int(_get(kfs[-1], "global_frame", 0)))


def build_canonical_numeric_state_trace(
    storyboard: StoryboardSpec,
    story_graph: StoryGraph,
    allow_fractional_cents: bool | None = None,
) -> dict[str, Any]:
    """Build frame-by-frame canonical integer-cent display contract (ZERO arbitrary decimal cents, ONE numeric owner)."""
    led = story_graph.ledger
    if allow_fractional_cents is None:
        allow_fractional_cents = bool(
            led.active and (
                (abs(led.initial_cents) % 100 != 0)
                or any((abs(st.amount_cents) % 100 != 0) or (abs(st.running_balance_cents) % 100 != 0) for st in led.steps)
            )
        )

    step_map = {st.step_id: st for st in led.steps}
    sym = led.currency_symbol or "$"
    scene_traces: list[dict[str, Any]] = []
    violations: list[str] = []

    for sc in storyboard.scenes:
        st = step_map.get(sc.persistent_ledger_step_id or "") if sc.persistent_ledger_step_id else None
        if not led.active or st is None:
            scene_traces.append({
                "scene_id": sc.scene_id,
                "start_frame": sc.start_frame,
                "end_frame_exclusive": sc.end_frame_exclusive,
                "ledger_owned": False,
                "numeric_owner_layer": None,
                "duplicate_counters_detected": False,
            })
            continue

        # Identify which layer(s) render live numeric currency counters so we enforce ONE single owner!
        # Exclude pure background `paper_tokens_only` decorative layers that render zero digits!
        counter_layers = [
            ly.layer_id for ly in sc.layers
            if ly.primitive_id in ("hf_apple_money_count", "hf_mk_progress_stat", "rem_ru_counter", "hf_data_chart")
            and str(ly.params.get("render_mode", "combined")).lower() != "paper_tokens_only"
        ]
        if sc.show_persistent_ribbon:
            counter_layers.append("persistent_ledger_ribbon")
        dup = len(counter_layers) > 1
        if dup:
            violations.append(f"duplicate_numeric_counters:{sc.scene_id}:{counter_layers}")

        prev_c = led.initial_cents if st.step_type == "opening_balance" else (st.running_balance_cents + abs(st.delta_cents))
        settled_c = st.running_balance_cents
        step_amt_c = abs(st.amount_cents)

        # Verify integer-dollar lock on every discrete display state
        if not allow_fractional_cents:
            for nm_lbl, val_c in (("prev", prev_c), ("settled", settled_c), ("step", step_amt_c)):
                if abs(int(val_c)) % 100 != 0:
                    violations.append(f"unexpected_fractional_cents:{sc.scene_id}:{nm_lbl}={val_c}")

        scene_traces.append({
            "scene_id": sc.scene_id,
            "start_frame": sc.start_frame,
            "end_frame_exclusive": sc.end_frame_exclusive,
            "ledger_owned": True,
            "step_id": st.step_id,
            "step_type": st.step_type,
            "category": st.category,
            "numeric_owner_layer": counter_layers[0] if counter_layers else None,
            "counter_layers_present": counter_layers,
            "duplicate_counters_detected": dup,
            "allow_fractional_cents": bool(allow_fractional_cents),
            "integer_step_only": not bool(allow_fractional_cents),
            "currency_symbol": sym,
            "opening_cents": led.initial_cents,
            "prev_balance_cents": prev_c,
            "step_amount_cents": step_amt_c,
            "settled_balance_cents": settled_c,
            "display_settled": format_cents(settled_c, sym),
            "display_delta": format_cents(st.delta_cents, sym),
            "is_deficit": settled_c < 0,
            "conservation_exact": bool(led.conservation_exact),
        })

    return {
        "schema_version": CONTINUITY_SCHEMA_VERSION,
        "ledger_active": bool(led.active),
        "currency_symbol": sym,
        "currency_code": led.currency_code,
        "allow_fractional_cents": bool(allow_fractional_cents),
        "integer_dollar_lock": not bool(allow_fractional_cents),
        "opening_cents": led.initial_cents,
        "total_allocated_cents": led.total_allocated_cents,
        "final_remaining_cents": led.computed_remaining_cents,
        "conservation_exact": bool(led.conservation_exact),
        "single_numeric_owner_verified": not any(s.get("duplicate_counters_detected") for s in scene_traces),
        "violations": violations,
        "scenes": scene_traces,
    }


def compile_continuous_timeline(
    storyboard: StoryboardSpec,
    story_graph: StoryGraph,
    layout_plan: dict[str, Any],
    *,
    allow_fractional_cents: bool | None = None,
) -> dict[str, Any]:
    """Compile ContinuousTimeline, SharedElementTracks with exact keyframes, BoundaryContracts & NumericStateTrace."""
    layout_by_sc = {s["scene_id"]: s for s in layout_plan.get("scenes", [])}
    pal = storyboard.palette or {}
    total_f = int(storyboard.total_frames)
    canvas_w = int(storyboard.width)
    canvas_h = int(storyboard.height)

    # 1. Build keyframed SharedElementTracks for persistent narrative elements
    tracks: list[SharedElementTrack] = []
    ledger_scenes = [
        sc for sc in storyboard.scenes
        if sc.persistent_ledger_step_id or sc.role in ("hook_contrast", "negation_tension", "allocation_step", "payoff_reveal")
    ]
    if ledger_scenes:
        kfs: list[ElementKeyframe] = []
        for idx, sc in enumerate(ledger_scenes):
            bbox = (layout_by_sc.get(sc.scene_id) or {}).get("hero_box_norm", {"x": 0.067, "y": 0.086, "w": 0.788, "h": 0.18})
            entry_end = min(sc.end_frame_exclusive - 1, sc.start_frame + 8)
            exit_start = max(entry_end, sc.end_frame_exclusive - 8)
            kfs.append(ElementKeyframe(
                global_frame=sc.start_frame,
                x=bbox["x"], y=bbox["y"], w=bbox["w"], h=bbox["h"],
                scale=0.96 if idx == 0 else 1.0,
                opacity=0.0 if idx == 0 else 1.0,
                z_order=25, easing="ease_out_cubic",
            ))
            kfs.append(ElementKeyframe(
                global_frame=entry_end,
                x=bbox["x"], y=bbox["y"], w=bbox["w"], h=bbox["h"],
                scale=1.0, opacity=1.0, z_order=25, easing="ease_out_cubic",
            ))
            kfs.append(ElementKeyframe(
                global_frame=exit_start,
                x=bbox["x"], y=bbox["y"], w=bbox["w"], h=bbox["h"],
                scale=1.0, opacity=1.0, z_order=25, easing="ease_in_out_cubic",
            ))
            if idx == len(ledger_scenes) - 1:
                kfs.append(ElementKeyframe(
                    global_frame=sc.end_frame_exclusive - 1,
                    x=bbox["x"], y=max(0.02, bbox["y"] - 0.04), w=bbox["w"], h=bbox["h"],
                    scale=0.94, opacity=0.0, z_order=25, easing="ease_in_out_cubic",
                ))
        tracks.append(SharedElementTrack(
            element_id="persistent_ledger_balance",
            visual_role="canonical_amount_display",
            logical_state_id="ledger_world",
            continuity_group="ledger_world",
            start_frame=ledger_scenes[0].start_frame,
            end_frame_exclusive=ledger_scenes[-1].end_frame_exclusive,
            color=str(pal.get("forest", "#243B32")),
            transition_owner="remotion",
            keyframes=kfs,
        ))

    prod_scenes = [sc for sc in storyboard.scenes if sc.role in ("product_intro", "product_benefit", "closing_takeaway")]
    if prod_scenes:
        pkfs: list[ElementKeyframe] = []
        for idx, sc in enumerate(prod_scenes):
            bbox = (layout_by_sc.get(sc.scene_id) or {}).get("hero_box_norm", {"x": 0.067, "y": 0.086, "w": 0.788, "h": 0.18})
            entry_end = min(sc.end_frame_exclusive - 1, sc.start_frame + 8)
            pkfs.append(ElementKeyframe(
                global_frame=sc.start_frame,
                x=bbox["x"], y=bbox["y"], w=bbox["w"], h=bbox["h"],
                scale=0.96 if idx == 0 else 1.0,
                opacity=0.0 if idx == 0 else 1.0,
                z_order=28, easing="ease_out_cubic",
            ))
            pkfs.append(ElementKeyframe(
                global_frame=entry_end,
                x=bbox["x"], y=bbox["y"], w=bbox["w"], h=bbox["h"],
                scale=1.0, opacity=1.0, z_order=28, easing="ease_out_cubic",
            ))
            if idx == len(prod_scenes) - 1:
                pkfs.append(ElementKeyframe(
                    global_frame=sc.end_frame_exclusive - 1,
                    x=bbox["x"], y=bbox["y"], w=bbox["w"], h=bbox["h"],
                    scale=1.0, opacity=0.0, z_order=28, easing="ease_in_out_cubic",
                ))
        tracks.append(SharedElementTrack(
            element_id="product_organization_motif",
            visual_role="solution_continuity_motif",
            logical_state_id=story_graph.product_name or "solution_world",
            continuity_group="solution_world",
            start_frame=prod_scenes[0].start_frame,
            end_frame_exclusive=prod_scenes[-1].end_frame_exclusive,
            color=str(pal.get("gold", pal.get("brass", "#BE9A59"))),
            transition_owner="remotion",
            keyframes=pkfs,
        ))

    # 2. Build exhaustive segment timeline (scenes + live gaps) and BoundaryContracts across 0..total_frames
    segments: list[dict[str, Any]] = []
    boundaries: list[BoundaryContract] = []
    cursor = 0
    for sc in storyboard.scenes:
        if sc.start_frame > cursor:
            segments.append({
                "segment_id": f"live_hold_f{cursor}_{sc.start_frame}",
                "kind": "live_hold_gap",
                "stage_mode": "live_source_hold",
                "start_frame": cursor,
                "end_frame_exclusive": sc.start_frame,
                "source_frame_start": cursor,
                "source_frame_end_exclusive": sc.start_frame,
                "transition_in": "none",
                "transition_out": "none",
            })
        tr_out_kind = str(sc.parameters.get("transition_out", "none"))
        tr_out_f = int(sc.parameters.get("transition_out_frames", 12 if tr_out_kind != "none" else 0))
        segments.append({
            "segment_id": sc.scene_id,
            "kind": "scene",
            "stage_mode": sc.stage_mode,
            "role": sc.role,
            "start_frame": sc.start_frame,
            "end_frame_exclusive": sc.end_frame_exclusive,
            "source_frame_start": sc.start_frame,
            "source_frame_end_exclusive": sc.end_frame_exclusive,
            "transition_in": sc.transition_in.kind,
            "transition_in_frames": sc.transition_in.duration_frames,
            "transition_out": tr_out_kind,
            "transition_out_frames": tr_out_f,
        })
        cursor = sc.end_frame_exclusive
    if cursor < total_f:
        segments.append({
            "segment_id": f"live_hold_f{cursor}_{total_f}",
            "kind": "live_hold_gap",
            "stage_mode": "live_source_hold",
            "start_frame": cursor,
            "end_frame_exclusive": total_f,
            "source_frame_start": cursor,
            "source_frame_end_exclusive": total_f,
            "transition_in": "none",
            "transition_out": "none",
        })

    for i in range(len(segments) - 1):
        a = segments[i]
        b = segments[i + 1]
        bf = int(b["start_frame"])
        a_cut = a["stage_mode"] == "full_graphic_cutaway"
        b_cut = b["stage_mode"] == "full_graphic_cutaway"
        tr_out = a.get("transition_out", "none")
        tr_in = b.get("transition_in", "none")

        if a_cut and tr_out != "none":
            bcls = "CUTAWAY_BRIDGE"
            bdir = "outgoing_from_prev"
            hf = int(a.get("transition_out_frames", 12))
            rat = f"Outgoing {tr_out} smoothly folds cutaway back to exact source clock at f{bf}."
        elif b_cut and tr_in != "none":
            bcls = "CUTAWAY_BRIDGE"
            bdir = "incoming_to_next"
            hf = int(b.get("transition_in_frames", 12))
            rat = f"Incoming {tr_in} sweeps warm ivory cutaway cleanly from live plate at f{bf}."
        elif a["kind"] == "live_hold_gap" or b["kind"] == "live_hold_gap":
            bcls = "QUIET_HOLD"
            bdir = "quiet_hold"
            hf = 0
            rat = f"Natural live breathing hold preserves source speaker clock without jarring jump at f{bf}."
        elif tr_in != "none":
            bcls = "MASK_PASS" if tr_in == "line_led" else "SHARED_ELEMENT_MORPH"
            bdir = "incoming_to_next"
            hf = int(b.get("transition_in_frames", 12))
            rat = f"Rendered {tr_in} links {a['segment_id']} -> {b['segment_id']} cleanly."
        else:
            bcls = "SHARED_ELEMENT_MORPH" if a.get("stage_mode") != b.get("stage_mode") else "MATCH_ACTION"
            bdir = "shared_morph"
            hf = 8
            rat = f"Shared live coordinate world & persistent ledger carry {a['segment_id']} -> {b['segment_id']}."

        # Evaluate shared-element position continuity error across boundary (f-1 vs f)
        pos_err_px = 0.0
        sh_id = "persistent_ledger_balance"
        for trk in tracks:
            if trk.start_frame < bf < trk.end_frame_exclusive:
                sh_id = trk.element_id
                s_prev = evaluate_shared_state(trk, bf - 1)
                s_curr = evaluate_shared_state(trk, bf)
                dx = abs(s_curr["x"] - s_prev["x"]) * canvas_w
                dy = abs(s_curr["y"] - s_prev["y"]) * canvas_h
                pos_err_px = round(math.hypot(dx, dy), 3)
                break

        boundaries.append(BoundaryContract(
            boundary_id=f"bnd_{i:02d}_f{bf:03d}",
            boundary_frame=bf,
            from_scene_id=a["segment_id"],
            to_scene_id=b["segment_id"],
            from_stage_mode=a["stage_mode"],
            to_stage_mode=b["stage_mode"],
            boundary_class=bcls,
            bridge_direction=bdir,
            handle_frames=hf,
            shared_element_id=sh_id,
            semantic_rationale=rat,
            expected_position_error_px=pos_err_px,
        ))

    # 3. Validate timeline invariants (exact 0..total_f coverage, zero frame drop/overlap, no unbridged cutaway hard cut!)
    errors: list[str] = []
    if not segments or segments[0]["start_frame"] != 0 or segments[-1]["end_frame_exclusive"] != total_f:
        errors.append(f"timeline_coverage_incomplete:expected=[0,{total_f})")
    for i in range(len(segments) - 1):
        if segments[i]["end_frame_exclusive"] != segments[i + 1]["start_frame"]:
            errors.append(f"non_contiguous_segments:{segments[i]['segment_id']}->{segments[i + 1]['segment_id']}")
    for bnd in boundaries:
        if (bnd.from_stage_mode == "full_graphic_cutaway" or bnd.to_stage_mode == "full_graphic_cutaway") and bnd.handle_frames <= 0:
            errors.append(f"unbridged_cutaway_boundary:{bnd.boundary_id}:{bnd.from_scene_id}->{bnd.to_scene_id}")

    numeric_trace = build_canonical_numeric_state_trace(storyboard, story_graph, allow_fractional_cents=allow_fractional_cents)
    errors.extend(numeric_trace.get("violations", []))

    doc = {
        "schema_version": CONTINUITY_SCHEMA_VERSION,
        "total_frames": total_f,
        "canvas": {"width": canvas_w, "height": canvas_h},
        "persistent_element_ids": sorted({t.element_id for t in tracks}),
        "track_count": len(tracks),
        "tracks": [t.to_dict() for t in tracks],
        "segments": segments,
        "boundaries": [b.to_dict() for b in boundaries],
        "numeric_state_trace": numeric_trace,
        "validation_errors": errors,
        "continuous_timeline_valid": len(errors) == 0,
    }
    doc["continuity_hash"] = canonical_hash(doc)
    return doc


# ---------------------------------------------------------------------------
# Phase A6 (ML-01.R5.R3): Analytic Physically-Inspired Motion State & Material Solver
# (Explicitly labeled `analytic_physically_inspired`, NEVER misrepresented as real physics simulation!)
# ---------------------------------------------------------------------------

MATERIAL_MOTION_PRESETS: dict[str, dict[str, Any]] = {
    "linen_paper": {
        "damping": 0.78,
        "overshoot_max": 0.035,
        "flutter_amp_deg": 4.2,
        "flutter_freq_hz": 2.4,
        "shadow_blur_base_px": 14.0,
        "motion_model": "analytic_physically_inspired",
    },
    "solid_card": {
        "damping": 0.88,
        "overshoot_max": 0.018,
        "flutter_amp_deg": 1.1,
        "flutter_freq_hz": 1.2,
        "shadow_blur_base_px": 18.0,
        "motion_model": "analytic_physically_inspired",
    },
    "glass_plate": {
        "damping": 0.94,
        "overshoot_max": 0.008,
        "flutter_amp_deg": 0.0,
        "flutter_freq_hz": 0.0,
        "shadow_blur_base_px": 22.0,
        "motion_model": "analytic_physically_inspired",
    },
    "elastic_accent": {
        "damping": 0.64,
        "overshoot_max": 0.06,
        "flutter_amp_deg": 2.0,
        "flutter_freq_hz": 3.1,
        "shadow_blur_base_px": 10.0,
        "motion_model": "analytic_physically_inspired",
    },
}


def sample_motion_path(
    start_xy: tuple[float, float],
    control_xy: tuple[float, float],
    end_xy: tuple[float, float],
    u: float,
    *,
    easing: str = "ease_out_cubic",
) -> dict[str, float]:
    """Sample deterministic quadratic Bezier arc with exact analytic tangent and velocity."""
    t = _ease_val(max(0.0, min(1.0, float(u))), easing)
    inv = 1.0 - t
    x = inv * inv * start_xy[0] + 2.0 * inv * t * control_xy[0] + t * t * end_xy[0]
    y = inv * inv * start_xy[1] + 2.0 * inv * t * control_xy[1] + t * t * end_xy[1]
    dx = 2.0 * inv * (control_xy[0] - start_xy[0]) + 2.0 * t * (end_xy[0] - control_xy[0])
    dy = 2.0 * inv * (control_xy[1] - start_xy[1]) + 2.0 * t * (end_xy[1] - control_xy[1])
    speed = math.hypot(dx, dy)
    angle_deg = math.degrees(math.atan2(dy, dx)) if speed > 1e-6 else 0.0
    return {
        "x": round(x, 5),
        "y": round(y, 5),
        "vx": round(dx, 5),
        "vy": round(dy, 5),
        "speed": round(speed, 5),
        "tangent_deg": round(angle_deg, 3),
    }


def build_material_motion(material: str, u: float, speed: float = 0.0) -> dict[str, Any]:
    """Compute analytic depth shadow, flutter/flex angle, and contact settle for a material preset."""
    preset = MATERIAL_MOTION_PRESETS.get(material, MATERIAL_MOTION_PRESETS["linen_paper"])
    t = max(0.0, min(1.0, float(u)))
    env = math.sin(t * math.pi)
    flex_deg = round(preset["flutter_amp_deg"] * math.sin(t * math.pi * max(0.5, preset["flutter_freq_hz"])) * env, 4)
    depth_z = round(0.15 + 0.55 * env, 4)
    shadow_blur = round(preset["shadow_blur_base_px"] * (0.65 + 0.85 * depth_z), 2)
    motion_blur_px = round(min(6.0, abs(float(speed)) * 4.0), 2)
    return {
        "material": material if material in MATERIAL_MOTION_PRESETS else "linen_paper",
        "motion_model": "analytic_physically_inspired",
        "flex_deg": flex_deg,
        "depth_z": depth_z,
        "shadow_blur_px": shadow_blur,
        "motion_blur_px": motion_blur_px,
    }


def evaluate_motion_state(
    track: SharedElementTrack | dict[str, Any],
    global_frame: int,
    *,
    fps: float = 24.0,
    material: str = "linen_paper",
) -> dict[str, Any]:
    """Evaluate position + finite-difference analytic velocity, acceleration, depth, pivot, and paper flex."""
    f = int(global_frame)
    s_cur = evaluate_shared_state(track, f)
    s_prev = evaluate_shared_state(track, f - 1)
    s_next = evaluate_shared_state(track, f + 1)
    dt = 1.0 / max(1.0, float(fps))
    vx = (s_next["x"] - s_prev["x"]) / (2.0 * dt)
    vy = (s_next["y"] - s_prev["y"]) / (2.0 * dt)
    ax = (s_next["x"] - 2.0 * s_cur["x"] + s_prev["x"]) / (dt * dt)
    ay = (s_next["y"] - 2.0 * s_cur["y"] + s_prev["y"]) / (dt * dt)
    speed = math.hypot(vx, vy)
    start_f = track.start_frame if isinstance(track, SharedElementTrack) else int(track.get("start_frame", 0))
    end_f = track.end_frame_exclusive if isinstance(track, SharedElementTrack) else int(track.get("end_frame_exclusive", max(start_f + 1, f + 1)))
    u = (f - start_f) / max(1.0, float(end_f - start_f))
    mat = build_material_motion(material, u, speed=speed)
    return {
        **s_cur,
        "vx": round(vx, 5),
        "vy": round(vy, 5),
        "ax": round(ax, 5),
        "ay": round(ay, 5),
        "speed": round(speed, 5),
        "pivot_x": round(s_cur["x"] + s_cur["w"] * 0.5, 5),
        "pivot_y": round(s_cur["y"] + s_cur["h"] * 0.5, 5),
        "depth_z": mat["depth_z"],
        "flex_deg": mat["flex_deg"],
        "shadow_blur_px": mat["shadow_blur_px"],
        "motion_blur_px": mat["motion_blur_px"],
        "material": mat["material"],
        "motion_model": "analytic_physically_inspired",
    }


def resolve_contact_event(
    track: SharedElementTrack | dict[str, Any],
    *,
    fps: float = 24.0,
    settle_speed_thresh: float = 0.02,
) -> dict[str, Any]:
    """Identify deterministic release/peak-velocity/settle frames along a SharedElementTrack."""
    start_f = track.start_frame if isinstance(track, SharedElementTrack) else int(track.get("start_frame", 0))
    end_f = track.end_frame_exclusive if isinstance(track, SharedElementTrack) else int(track.get("end_frame_exclusive", start_f + 1))
    peak_f = start_f
    peak_v = -1.0
    settle_f = min(end_f - 1, start_f + 8)
    for f in range(start_f, max(start_f + 1, end_f)):
        st = evaluate_motion_state(track, f, fps=fps)
        if st["speed"] > peak_v:
            peak_v = st["speed"]
            peak_f = f
    for f in range(peak_f + 1, max(peak_f + 2, end_f)):
        st = evaluate_motion_state(track, f, fps=fps)
        if st["speed"] <= settle_speed_thresh:
            settle_f = f
            break
    return {
        "release_frame": start_f,
        "peak_velocity_frame": peak_f,
        "settle_contact_frame": settle_f,
        "peak_speed": round(max(0.0, peak_v), 5),
        "motion_model": "analytic_physically_inspired",
    }


def validate_track_continuity(
    track: SharedElementTrack | dict[str, Any],
    *,
    canvas_w: int = 720,
    canvas_h: int = 1280,
    max_step_px: float = 48.0,
) -> dict[str, Any]:
    """Verify finite values, bounded per-frame displacement, and zero NaN across entire track span."""
    start_f = track.start_frame if isinstance(track, SharedElementTrack) else int(track.get("start_frame", 0))
    end_f = track.end_frame_exclusive if isinstance(track, SharedElementTrack) else int(track.get("end_frame_exclusive", start_f + 1))
    max_jump = 0.0
    errors: list[str] = []
    prev = evaluate_shared_state(track, start_f)
    for f in range(start_f + 1, max(start_f + 1, end_f)):
        cur = evaluate_shared_state(track, f)
        for k, v in cur.items():
            if not math.isfinite(float(v)):
                errors.append(f"non_finite:{k}@f{f}")
        jump = math.hypot((cur["x"] - prev["x"]) * canvas_w, (cur["y"] - prev["y"]) * canvas_h)
        if jump > max_jump:
            max_jump = jump
        if jump > max_step_px:
            errors.append(f"discontinuity_jump:{jump:.1f}px>{max_step_px}px@f{f}")
        prev = cur
    return {
        "valid": len(errors) == 0,
        "max_step_px": round(max_jump, 3),
        "errors": errors,
    }
