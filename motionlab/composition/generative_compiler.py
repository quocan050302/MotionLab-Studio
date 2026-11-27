"""Phase A3 (ML-01.R5): Deterministic Compiler from Validated StoryboardSpec -> Remotion Props, HyperFrames HTML & Rational AudioTimeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
from typing import Any

from motionlab.audio.cues import frame_to_sample_exact
from motionlab.audio.schema import AUDIO_SCHEMA_VERSION, AudioTimeline, SoundEvent
from motionlab.audio.synthesis import SFX_FAMILIES, synthesize_sfx_waveform
from motionlab.composition.generative_schema import SceneSpec, StoryboardSpec, validate_storyboard_spec
from motionlab.composition.primitive_adapters import build_adapted_hyperframes_html
from motionlab.composition.primitive_registry import UnifiedPrimitiveRegistry, load_unified_registry
from motionlab.schemas import MediaInfo, SchemaValidationError, sanitize_native
from motionlab.semantic.story_graph import StoryGraph, format_cents

SUPPORTED_RATIONAL_FRACTIONAL_FPS = {
    (24000, 1001),
    (30000, 1001),
    (60000, 1001),
    (24000, 1000),
    (25000, 1000),
    (30000, 1000),
}


@dataclass
class CompiledOverlay:
    """Typed multi-layer HyperFrames overlay artifact contract (Fixes D04 — never silently drops layers!)."""
    layer_id: str
    scene_id: str
    primitive_id: str
    layer_role: str  # "background_graphics" | "foreground_graphics"
    z_order: int
    start_frame: int
    end_frame_exclusive: int
    fps_num: int
    fps_den: int
    opacity: float
    render_mode: str
    html: str
    html_sha256: str

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("html", None)
        return sanitize_native(d)


def validate_rational_fps_clock(fps_num: int, fps_den: int) -> dict[str, Any]:
    """Fixes D13: Validate exact rational FPS clock without silent rounding.

    Supports exact integer rates (1..120 fps) and standard broadcast NTSC rational rates (24000/1001, 30000/1001, 60000/1001).
    Fails closed (`SchemaValidationError`) on non-positive or unsupported arbitrary fractional timebases.
    """
    if int(fps_num) <= 0 or int(fps_den) <= 0:
        raise SchemaValidationError(f"Invalid rational FPS clock: {fps_num}/{fps_den}")
    frac = Fraction(int(fps_num), int(fps_den))
    fps_f = float(frac)
    if fps_f < 1.0 or fps_f > 240.0:
        raise SchemaValidationError(f"Unsupported FPS out of [1, 240] range: {fps_num}/{fps_den} ({fps_f:.4f})")

    if frac.denominator == 1:
        return {
            "fps_num": int(fps_num),
            "fps_den": int(fps_den),
            "fps_int_remotion": int(frac.numerator),
            "fps_float": fps_f,
            "is_fractional": False,
            "clock_mode": "exact_integer",
            "supported": True,
        }

    if (int(fps_num), int(fps_den)) in SUPPORTED_RATIONAL_FRACTIONAL_FPS or (frac.numerator, frac.denominator) in SUPPORTED_RATIONAL_FRACTIONAL_FPS:
        # Remotion accepts integer fps for frame index count while FFmpeg/Audio preserve exact rational PTS!
        return {
            "fps_num": int(fps_num),
            "fps_den": int(fps_den),
            "fps_int_remotion": int(round(fps_f)),
            "fps_float": fps_f,
            "is_fractional": True,
            "clock_mode": "rational_broadcast_ntsc_frame_indexed",
            "supported": True,
        }

    raise SchemaValidationError(
        f"Unsupported arbitrary fractional FPS clock {fps_num}/{fps_den} ({fps_f:.6f} fps): "
        "exact frame-clock parity cannot be guaranteed without silent rounding (fail-closed D13)."
    )


def _build_persistent_ledger_payload(
    scene: SceneSpec,
    story_graph: StoryGraph,
    palette: dict[str, str],
) -> dict[str, Any] | None:
    """Resolve verified integer-cent ledger carryover state for a scene (NEVER invented float math)."""
    led = story_graph.ledger
    if not led.active or led.initial_cents <= 0:
        return None
    step_id = scene.persistent_ledger_step_id
    if not step_id:
        return None

    steps = led.steps
    target_idx = -1
    for i, st in enumerate(steps):
        if st.step_id == step_id:
            target_idx = i
            break
    if target_idx < 0:
        return None

    cur = steps[target_idx]
    prev_bal = led.initial_cents if target_idx == 0 else steps[target_idx - 1].running_balance_cents

    # Build exact cumulative waterfall step rows strictly from verified integer cents
    wf_steps: list[dict[str, Any]] = [
        {"label": "Balance", "value": led.initial_cents / 100.0, "isTotal": True, "color": palette.get("espresso", "#292822")}
    ]
    cat_colors = [palette.get("clay", "#B85D43"), palette.get("brass", "#C6A46A"), palette.get("sage", "#A4B39A")]
    alloc_i = 0
    for st in steps[: target_idx + 1]:
        if st.step_type == "allocation":
            wf_steps.append({
                "label": st.category,
                "value": -abs(st.amount_cents) / 100.0,
                "color": cat_colors[alloc_i % len(cat_colors)],
            })
            alloc_i += 1
        elif st.step_type == "remaining_confirmed":
            wf_steps.append({
                "label": "Left",
                "value": st.running_balance_cents / 100.0,
                "isTotal": True,
                "color": palette.get("forest", "#365B49"),
            })

    return {
        "active": True,
        "currencySymbol": led.currency_symbol or "$",
        "initialCents": led.initial_cents,
        "prevBalanceCents": prev_bal,
        "nextBalanceCents": cur.running_balance_cents,
        "currentCategory": cur.category if cur.step_type == "allocation" else "",
        "currentDeltaCents": cur.delta_cents,
        "steps": wf_steps,
        "provenanceNote": f"{cur.running_balance_provenance}: {cur.formula}",
    }


def compile_scene_hyperframes_layers(
    scene: SceneSpec,
    storyboard: StoryboardSpec,
    story_graph: StoryGraph,
) -> list[CompiledOverlay]:
    """Fixes D04: Compile ALL HyperFrames layers in `scene.layers` sorted by `z_order` (zero silent drops!)."""
    clock = validate_rational_fps_clock(storyboard.fps_num, storyboard.fps_den)
    hf_layers = sorted(
        [ly for ly in scene.layers if ly.engine == "hyperframes"],
        key=lambda x: (int(x.z_order), str(x.layer_id)),
    )
    if not hf_layers:
        return []

    fps_f = clock["fps_float"]
    dur_s = max(0.4, (scene.end_frame_exclusive - scene.start_frame) / fps_f)
    seen_ids: set[str] = set()
    compiled: list[CompiledOverlay] = []

    for ly in hf_layers:
        if ly.layer_id in seen_ids:
            raise SchemaValidationError(f"Duplicate HyperFrames layer_id in {scene.scene_id}: {ly.layer_id}")
        seen_ids.add(ly.layer_id)
        params = dict(ly.params)
        params.setdefault("palette", storyboard.palette)
        html = build_adapted_hyperframes_html(
            primitive_id=ly.primitive_id,
            width=storyboard.width,
            height=storyboard.height,
            fps=fps_f,
            duration_s=dur_s,
            parameters=params,
            transparent=True,
        )
        compiled.append(CompiledOverlay(
            layer_id=ly.layer_id,
            scene_id=scene.scene_id,
            primitive_id=ly.primitive_id,
            layer_role=ly.layer_role if ly.layer_role in ("background_graphics", "foreground_graphics") else "foreground_graphics",
            z_order=int(ly.z_order),
            start_frame=int(scene.start_frame),
            end_frame_exclusive=int(scene.end_frame_exclusive),
            fps_num=int(storyboard.fps_num),
            fps_den=int(storyboard.fps_den),
            opacity=float(params.get("opacity", 1.0)),
            render_mode=str(params.get("render_mode", "combined")),
            html=html,
            html_sha256=hashlib.sha256(html.encode("utf-8")).hexdigest(),
        ))
    return compiled


def compile_scene_hyperframes_html(
    scene: SceneSpec,
    storyboard: StoryboardSpec,
    story_graph: StoryGraph,
) -> tuple[str | None, str | None, str]:
    """Backward-compatible single-layer wrapper returning the first ordered CompiledOverlay (use `compile_scene_hyperframes_layers` for multi-layer)."""
    overlays = compile_scene_hyperframes_layers(scene, storyboard, story_graph)
    if not overlays:
        return None, None, "foreground_graphics"
    first = overlays[0]
    return first.primitive_id, first.html, first.layer_role


def compile_scene_remotion_props(
    scene: SceneSpec,
    storyboard: StoryboardSpec,
    story_graph: StoryGraph,
    *,
    source_file: str | None,
    fg_alpha_file: str | None,
    prop_alpha_file: str | None,
    hf_graphic_file: str | None,
    hf_layer_role: str,
    hf_layers: list[dict[str, Any]] | None = None,
    product_asset_file: str | None = None,
    product_asset_meta: dict[str, Any] | None = None,
    task_anchor: dict[str, float] | None = None,
    face_anchor: dict[str, float] | None = None,
    continuity_tracks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compile a validated SceneSpec into exact `MotionLabGenerativeScene` Remotion props (Fixes D04, D06, D13)."""
    dur_frames = max(2, scene.end_frame_exclusive - scene.start_frame)
    clock = validate_rational_fps_clock(storyboard.fps_num, storyboard.fps_den)
    fps_int = int(clock["fps_int_remotion"])

    rem_nodes = [
        {
            "layerId": ly.layer_id,
            "primitiveId": ly.primitive_id,
            "layerRole": ly.layer_role,
            "zOrder": int(ly.z_order),
            "params": dict(ly.params),
        }
        for ly in sorted(scene.layers, key=lambda x: (int(x.z_order), str(x.layer_id)))
        if ly.engine == "remotion"
    ]

    ledger_payload = _build_persistent_ledger_payload(scene, story_graph, storyboard.palette)
    params = dict(scene.parameters)
    if scene.show_persistent_ribbon:
        params["show_persistent_ribbon"] = True

    stage_mode = scene.stage_mode
    if stage_mode == "live_depth_matte" and not fg_alpha_file:
        stage_mode = "live_clean_bridge"

    tr_out = str(params.get("transition_out", "none"))
    tr_out_frames = int(params.get("transition_out_frames", 12 if tr_out != "none" else 0))

    # Build canonical ordered hfLayers array (backward-compatible with single hfGraphicFile!)
    normalized_hf_layers: list[dict[str, Any]] = []
    if hf_layers:
        for item in sorted(hf_layers, key=lambda x: (int(x.get("zOrder", 20)), str(x.get("layerId", "")))):
            normalized_hf_layers.append({
                "layerId": str(item["layerId"]),
                "primitiveId": str(item.get("primitiveId", "")),
                "file": str(item["file"]),
                "layerRole": str(item.get("layerRole", "foreground_graphics")),
                "zOrder": int(item.get("zOrder", 20)),
                "opacity": float(item.get("opacity", 1.0)),
            })
    elif hf_graphic_file:
        first_hf = next((ly for ly in sorted(scene.layers, key=lambda x: int(x.z_order)) if ly.engine == "hyperframes"), None)
        normalized_hf_layers.append({
            "layerId": first_hf.layer_id if first_hf else f"{scene.scene_id}_hf_0",
            "primitiveId": first_hf.primitive_id if first_hf else "",
            "file": str(hf_graphic_file),
            "layerRole": str(hf_layer_role),
            "zOrder": int(first_hf.z_order) if first_hf else 25,
            "opacity": 1.0,
        })

    return {
        "sceneId": scene.scene_id,
        "chosenCandidateId": scene.chosen_candidate_id,
        "stageMode": stage_mode,
        "width": storyboard.width,
        "height": storyboard.height,
        "fps": fps_int,
        "fpsNum": int(clock["fps_num"]),
        "fpsDen": int(clock["fps_den"]),
        "fpsFloat": round(float(clock["fps_float"]), 6),
        "clockMode": str(clock["clock_mode"]),
        "durationInFrames": dur_frames,
        "entryFrames": min(10, max(4, dur_frames // 8)),
        "exitFrames": min(10, max(4, dur_frames // 8)),
        "sourceFile": source_file,
        "fgAlphaFile": fg_alpha_file,
        "propAlphaFile": prop_alpha_file,
        "hfGraphicFile": hf_graphic_file,
        "hfLayerRole": hf_layer_role,
        "hfLayers": normalized_hf_layers,
        "productAssetFile": product_asset_file,
        "productAssetMeta": product_asset_meta,
        "transitionIn": scene.transition_in.kind,
        "transitionInFrames": scene.transition_in.duration_frames,
        "transitionOut": tr_out,
        "transitionOutFrames": tr_out_frames,
        "globalStartFrame": scene.start_frame,
        "continuityTracks": list(continuity_tracks or []),
        "palette": storyboard.palette,
        "primitives": rem_nodes,
        "ledger": ledger_payload,
        "taskAnchor": task_anchor,
        "faceAnchor": face_anchor,
        "parameters": params,
        "excludeLayerRole": None,
        "excludeLayerId": None,
    }


def compile_storyboard_audio_timeline(
    media: MediaInfo,
    storyboard: StoryboardSpec,
    *,
    audio_mode: str = "mix",
    sfx_enabled: bool = True,
    music_enabled: bool = False,
    seed: int = 42,
) -> tuple[AudioTimeline, dict[str, Any]]:
    """Compile all SceneSpec SoundIntents onto a sample-exact Rational Fraction audio timeline."""
    sr = media.audio_streams[0].sample_rate if media.audio_streams else 44100
    channels = 2 if (audio_mode == "mix" and sfx_enabled) else (media.audio_streams[0].channels if media.audio_streams else 1)
    fps_num, fps_den = media.fps.num, media.fps.den
    dur_samples = frame_to_sample_exact(media.duration_frames, fps_num, fps_den, sr)
    audio_pts = [s for s in media.pts_summary if s.get("codec_type") == "audio"]
    priming_s = float(audio_pts[0].get("priming_offset_s", 0.0)) if audio_pts else 0.0

    if audio_mode == "preserve-exact" or not sfx_enabled:
        tl = AudioTimeline(
            schema_version=AUDIO_SCHEMA_VERSION,
            source_hash=media.input_hash,
            audio_mode=audio_mode,
            sfx_enabled=False,
            music_enabled=False,
            sample_rate=sr,
            channels=channels,
            duration_samples=dur_samples,
            fps_num=fps_num,
            fps_den=fps_den,
            pts_priming_offset_s=round(priming_s, 6),
            events=[],
        )
        tl.timeline_hash = tl.compute_hash()
        return tl, {"used_assets": []}

    raw_cues: list[tuple[int, str, Any]] = []
    for sc in storyboard.scenes:
        for si in sc.sound_intents:
            if si.family in SFX_FAMILIES:
                raw_cues.append((int(si.target_frame), sc.scene_id, si))
    raw_cues.sort(key=lambda x: x[0])

    events: list[SoundEvent] = []
    used_assets: dict[str, dict[str, Any]] = {}
    last_any_peak = -sr
    last_fam_peak: dict[str, int] = {}
    var_counter: dict[str, int] = {}

    for target_f, sc_id, si in raw_cues:
        clamped_f = max(1, min(media.duration_frames - 3, target_f))
        peak_samp = frame_to_sample_exact(clamped_f, fps_num, fps_den, sr)
        fam = si.family
        # Keep cues cleanly separated (min 0.35s any cue, min 0.90s same family) so audio stays crisp & uncluttered
        if peak_samp - last_any_peak < int(sr * 0.35):
            continue
        if fam in last_fam_peak and (peak_samp - last_fam_peak[fam]) < int(sr * 0.90):
            continue

        var = int(si.variant) if si.variant >= 0 else (var_counter.get(fam, 0) % 3)
        var_counter[fam] = var_counter.get(fam, 0) + 1
        wav, spec = synthesize_sfx_waveform(fam, variant=var % 3, sr=sr, seed=seed)
        used_assets[spec.asset_id] = spec.to_dict()

        trans_samp = int(round(spec.transient_offset_s * sr))
        start_samp = max(0, peak_samp - trans_samp)
        end_samp = min(dur_samples, start_samp + wav.size)
        if end_samp - start_samp < int(sr * 0.05):
            continue

        events.append(SoundEvent(
            id=f"sfx_{len(events):02d}_{fam}",
            linked_visual_event_id=sc_id,
            semantic_span_ids=[str(w) for w in si.word_ids],
            observed_action_id=f"{sc_id}:{si.cue_id}",
            asset_id=spec.asset_id,
            family=fam,
            variant=var % 3,
            asset_sha256=spec.sha256,
            start_sample=start_samp,
            peak_sample=min(end_samp - 1, start_samp + trans_samp),
            end_sample=end_samp,
            peak_frame=clamped_f,
            gain_db=float(si.gain_db),
            pan=float(si.pan),
            role=fam,
            confidence=0.95,
            reason=si.rationale or f"{sc_id}:{fam}",
        ))
        last_any_peak = peak_samp
        last_fam_peak[fam] = peak_samp
        if len(events) >= 18:
            break

    tl = AudioTimeline(
        schema_version=AUDIO_SCHEMA_VERSION,
        source_hash=media.input_hash,
        audio_mode=audio_mode,
        sfx_enabled=True,
        music_enabled=bool(music_enabled),
        sample_rate=sr,
        channels=channels,
        duration_samples=dur_samples,
        fps_num=fps_num,
        fps_den=fps_den,
        pts_priming_offset_s=round(priming_s, 6),
        events=events,
    )
    tl.timeline_hash = tl.compute_hash()
    return tl, {"used_assets": list(used_assets.values())}


def compile_storyboard_graph(
    storyboard: StoryboardSpec,
    story_graph: StoryGraph,
    media: MediaInfo,
    registry: UnifiedPrimitiveRegistry | None = None,
    require_verified_primitives: bool = True,
) -> dict[str, Any]:
    """Validate and compile a complete StoryboardSpec into an immutable deterministic render plan."""
    reg = registry or load_unified_registry()
    errs = validate_storyboard_spec(
        storyboard,
        story_graph=story_graph,
        registry=reg,
        require_verified_primitives=require_verified_primitives,
    )
    if errs:
        raise SchemaValidationError(f"StoryboardSpec validation failed ({len(errs)} error(s)): {errs[:6]}")

    jobs: list[dict[str, Any]] = []
    for sc in storyboard.scenes:
        hf_overlays = compile_scene_hyperframes_layers(sc, storyboard, story_graph)
        hf_pids = [ov.primitive_id for ov in hf_overlays]
        first_hf = hf_overlays[0] if hf_overlays else None
        rem_pids = [ly.primitive_id for ly in sc.layers if ly.engine == "remotion"]
        if sc.transition_in.kind == "frame_morph":
            rem_pids.append("rem_gtk_frame_morph")
        elif sc.transition_in.kind == "line_led":
            rem_pids.append("rem_gtk_line_led")
        elif sc.transition_in.kind == "flip_page":
            rem_pids.append("rem_rc_flip_page")
        if sc.stage_mode == "aspect_morph_editorial":
            rem_pids.append("rem_viv_aspect_morph")
        elif sc.stage_mode == "live_depth_matte":
            rem_pids.append("rem_ru_parallax")

        jobs.append({
            "scene_id": sc.scene_id,
            "chosen_candidate_id": sc.chosen_candidate_id,
            "start_frame": sc.start_frame,
            "end_frame_exclusive": sc.end_frame_exclusive,
            "stage_mode": sc.stage_mode,
            "requires_matte": sc.requires_matte,
            "intentional_cutaway": sc.intentional_cutaway,
            "hf_primitive_id": first_hf.primitive_id if first_hf else None,
            "hf_layer_role": first_hf.layer_role if first_hf else "foreground_graphics",
            "hf_html_present": bool(hf_overlays),
            "hf_layers_count": len(hf_overlays),
            "hf_layers": [ov.to_dict() for ov in hf_overlays],
            "remotion_primitives": sorted(set(rem_pids)),
            "all_primitives": sorted(set(rem_pids + hf_pids)),
        })

    return {
        "schema_version": storyboard.schema_version,
        "storyboard_hash": storyboard.storyboard_hash or storyboard.compute_hash(),
        "selected_graph_hash": storyboard.selected_graph_hash,
        "clock": validate_rational_fps_clock(storyboard.fps_num, storyboard.fps_den),
        "scene_count": len(jobs),
        "jobs": jobs,
        "distinct_primitives_planned": sorted({p for j in jobs for p in j["all_primitives"]}),
    }

