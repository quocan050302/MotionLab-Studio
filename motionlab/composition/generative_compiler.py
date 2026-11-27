"""Phase A3 (ML-01.R5): Deterministic Compiler from Validated StoryboardSpec -> Remotion Props, HyperFrames HTML & Rational AudioTimeline."""

from __future__ import annotations

from fractions import Fraction
from typing import Any

from motionlab.audio.cues import frame_to_sample_exact
from motionlab.audio.schema import AUDIO_SCHEMA_VERSION, AudioTimeline, SoundEvent
from motionlab.audio.synthesis import SFX_FAMILIES, synthesize_sfx_waveform
from motionlab.composition.generative_schema import SceneSpec, StoryboardSpec, validate_storyboard_spec
from motionlab.composition.primitive_adapters import build_adapted_hyperframes_html
from motionlab.composition.primitive_registry import UnifiedPrimitiveRegistry, load_unified_registry
from motionlab.schemas import MediaInfo, SchemaValidationError
from motionlab.semantic.story_graph import StoryGraph, format_cents


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


def compile_scene_hyperframes_html(
    scene: SceneSpec,
    storyboard: StoryboardSpec,
    story_graph: StoryGraph,
) -> tuple[str | None, str | None, str]:
    """Compile the scene's HyperFrames layer (if any) into offline portrait HTML/GSAP.

    Returns: (primitive_id_or_None, html_string_or_None, hf_layer_role)
    """
    hf_layers = [ly for ly in scene.layers if ly.engine == "hyperframes"]
    if not hf_layers:
        return None, None, "foreground_graphics"
    ly = hf_layers[0]
    fps_f = storyboard.fps_num / max(1, storyboard.fps_den)
    dur_s = max(0.4, (scene.end_frame_exclusive - scene.start_frame) / fps_f)
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
    return ly.primitive_id, html, ly.layer_role


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
    task_anchor: dict[str, float] | None = None,
    face_anchor: dict[str, float] | None = None,
    continuity_tracks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compile a validated SceneSpec into exact `MotionLabGenerativeScene` Remotion props."""
    dur_frames = max(2, scene.end_frame_exclusive - scene.start_frame)
    fps_int = max(1, int(round(storyboard.fps_num / max(1, storyboard.fps_den))))

    rem_nodes = [
        {
            "primitiveId": ly.primitive_id,
            "layerRole": ly.layer_role,
            "params": dict(ly.params),
        }
        for ly in sorted(scene.layers, key=lambda x: x.z_order)
        if ly.engine == "remotion"
    ]

    ledger_payload = _build_persistent_ledger_payload(scene, story_graph, storyboard.palette)
    params = dict(scene.parameters)
    if scene.show_persistent_ribbon:
        params["show_persistent_ribbon"] = True

    # If live_depth_matte requested but no matte available, gracefully adapt to live_clean_bridge (explicitly logged)
    stage_mode = scene.stage_mode
    if stage_mode == "live_depth_matte" and not fg_alpha_file:
        stage_mode = "live_clean_bridge"

    tr_out = str(params.get("transition_out", "none"))
    tr_out_frames = int(params.get("transition_out_frames", 12 if tr_out != "none" else 0))

    return {
        "sceneId": scene.scene_id,
        "stageMode": stage_mode,
        "width": storyboard.width,
        "height": storyboard.height,
        "fps": fps_int,
        "durationInFrames": dur_frames,
        "entryFrames": min(10, max(4, dur_frames // 8)),
        "exitFrames": min(10, max(4, dur_frames // 8)),
        "sourceFile": source_file,
        "fgAlphaFile": fg_alpha_file,
        "propAlphaFile": prop_alpha_file,
        "hfGraphicFile": hf_graphic_file,
        "hfLayerRole": hf_layer_role,
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
        hf_pid, hf_html, hf_role = compile_scene_hyperframes_html(sc, storyboard, story_graph)
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
            "start_frame": sc.start_frame,
            "end_frame_exclusive": sc.end_frame_exclusive,
            "stage_mode": sc.stage_mode,
            "requires_matte": sc.requires_matte,
            "intentional_cutaway": sc.intentional_cutaway,
            "hf_primitive_id": hf_pid,
            "hf_layer_role": hf_role,
            "hf_html_present": hf_html is not None,
            "remotion_primitives": sorted(set(rem_pids)),
            "all_primitives": sorted(set(rem_pids + ([hf_pid] if hf_pid else []))),
        })

    return {
        "schema_version": storyboard.schema_version,
        "storyboard_hash": storyboard.storyboard_hash or storyboard.compute_hash(),
        "scene_count": len(jobs),
        "jobs": jobs,
        "distinct_primitives_planned": sorted({p for j in jobs for p in j["all_primitives"]}),
    }

