"""Phase 8 (ML-01.R4): Bounded Async Semantic-VFX-Audio DAG Executor (ASR + Qwen2.5 + MODNet + PropCutout + HyperFrames + Remotion + SFX Mix)."""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from motionlab.audio import (
    build_audio_cues,
    build_sfx_asset_manifest,
    mix_preserving_master,
)
from motionlab.composition.engine import _assemble_canonical_timeline_and_mux, _render_candidate_scene_ffmpeg_fallback
from motionlab.composition.schema import CompositionGraph, CompositionScene
from motionlab.composition.vfx import build_vfx_capability_matrix, compile_cinematic_graph, get_vfx_recipe
from motionlab.config import MotionLabConfig
from motionlab.renderers.hyperframes import HyperFramesRenderer
from motionlab.renderers.remotion import RemotionRenderer
from motionlab.resource import GLOBAL_JOB_SEMAPHORE, ResourceBudget, ResourceTracker, RuntimeResourceWatchdog, run_managed_subprocess
from motionlab.schemas import AnalysisResult, EditPlan, MediaInfo, RenderResult, atomic_write_json, atomic_write_text
from motionlab.semantic import (
    build_semantic_units,
    direct_semantic_edit,
    inspect_asr_model,
    inspect_llm_model,
    transcribe_local,
)
from motionlab.semantic.asr import format_readable_transcript
from motionlab.vision.anchors import build_anchor_tracks, generate_matte_debug_sheet
from motionlab.vision.prop_mask import segment_prop_sequence
from motionlab.vision.segmentation import inspect_vision_model, segment_subject_sequence
from motionlab.vision.tracking import track_action_rois


def _assemble_r4_video_and_mux_mix(
    media: MediaInfo,
    plan: EditPlan,
    graph: CompositionGraph,
    rendered_scenes: dict[str, dict[str, Any]],
    mixed_wav_path: Path | None,
    workdir: Path,
    out_partial_path: Path,
    budget: ResourceBudget,
    config: MotionLabConfig,
    ffmpeg_bin: str,
) -> dict[str, Any]:
    """Single-pass frame-exact timeline assembly (0..duration_frames-1) + single default mixed AAC audio mux (or lossless copy)."""
    if mixed_wav_path is None or not mixed_wav_path.is_file():
        return _assemble_canonical_timeline_and_mux(
            media, plan, graph, rendered_scenes, workdir, out_partial_path, budget, config, ffmpeg_bin
        )

    t0 = time.perf_counter()
    w, h = media.display_width, media.display_height
    total_f = media.duration_frames
    fps_str = media.fps.to_ffmpeg()
    video_only = workdir / "r4_video_only.mp4"

    scene_caps: dict[str, cv2.VideoCapture] = {}
    frame_owner: list[tuple[str, int] | None] = [None] * total_f
    for sc in graph.scenes:
        rinfo = rendered_scenes.get(sc.scene_id)
        if not rinfo or not Path(rinfo["artifact_path"]).is_file():
            continue
        scene_caps[sc.scene_id] = cv2.VideoCapture(rinfo["artifact_path"])
        for f in range(sc.start_frame, min(total_f, sc.end_frame_exclusive)):
            frame_owner[f] = (sc.scene_id, f - sc.start_frame)

    dec = subprocess.Popen(
        [ffmpeg_bin, "-v", "error", "-nostdin", "-i", str(media.path), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False,
    )
    enc = subprocess.Popen(
        [
            ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error", "-nostdin",
            "-threads", str(max(1, budget.ffmpeg_threads)),
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", fps_str,
            "-i", "pipe:0",
            "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p",
            "-an", "-movflags", "+faststart", str(video_only),
        ],
        stdin=subprocess.PIPE, stderr=subprocess.PIPE, shell=False,
    )
    assert dec.stdout and enc.stdin
    wd = RuntimeResourceWatchdog(
        pids=[dec.pid, enc.pid], timeout_s=float(config.timeout_s),
        temp_dir=workdir, min_free_gb=budget.min_free_gb, max_temp_gb=budget.max_temp_gb,
    )
    wd.start()

    frame_bytes = w * h * 3
    frames_written = 0
    scene_frames_spliced = 0
    try:
        for f in range(total_f):
            buf = dec.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            owner = frame_owner[f]
            if owner is not None:
                sid, _ = owner
                ok_s, bgr_s = scene_caps[sid].read()
                if ok_s and bgr_s is not None:
                    rgb_s = cv2.cvtColor(bgr_s, cv2.COLOR_BGR2RGB)
                    if rgb_s.shape[0] != h or rgb_s.shape[1] != w:
                        rgb_s = cv2.resize(rgb_s, (w, h), interpolation=cv2.INTER_LINEAR)
                    enc.stdin.write(rgb_s.tobytes())
                    frames_written += 1
                    scene_frames_spliced += 1
                    continue
            enc.stdin.write(buf)
            frames_written += 1
    finally:
        for cap in scene_caps.values():
            cap.release()
        dec.stdout.close()
        enc.stdin.close()
        d_err = dec.stderr.read().decode("utf-8", errors="replace") if dec.stderr else ""
        e_err = enc.stderr.read().decode("utf-8", errors="replace") if enc.stderr else ""
        d_rc = dec.wait(timeout=60)
        e_rc = enc.wait(timeout=120)
        wd.stop()
        if wd.violation:
            raise RuntimeError(f"Resource watchdog terminated R4 assembly: {wd.violation}")
        if d_rc != 0 or e_rc != 0:
            raise RuntimeError(f"Canonical R4 assembly failed: dec={d_rc}:{d_err[:200]}, enc={e_rc}:{e_err[:200]}")

    t_vid = round(time.perf_counter() - t0, 3)

    # Mux video + newly mixed reference WAV -> clean 192k AAC default track (no -shortest!)
    t_mux0 = time.perf_counter()
    mux_cmd = [
        ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(video_only),
        "-i", str(mixed_wav_path),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        str(out_partial_path),
    ]
    with GLOBAL_JOB_SEMAPHORE:
        run_managed_subprocess(mux_cmd, timeout_s=min(120.0, float(config.timeout_s)), check=True)
    t_mux = round(time.perf_counter() - t_mux0, 3)

    return {
        "frames_written": frames_written,
        "scene_frames_spliced": scene_frames_spliced,
        "video_assembly_s": t_vid,
        "audio_mux_s": t_mux,
        "peak_rss_mb": round(wd.peak_rss_mb, 2),
    }


def execute_cinematic_r4_pipeline(
    media: MediaInfo,
    analysis: AnalysisResult,
    base_plan: EditPlan,
    config: MotionLabConfig,
    doctor_info: dict[str, Any],
    run_dir: Path,
    budget: ResourceBudget,
    tracker: ResourceTracker,
    existing_graph: CompositionGraph | None = None,
) -> dict[str, Any]:
    """Full ML-01.R4 Semantic Creative Director + Cinematic VFX + Local SFX Mixing Pipeline."""
    ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"
    ffprobe_bin = shutil.which("ffprobe") or "ffprobe"
    temp_dir = run_dir / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    ev_dir = run_dir / "evidence"
    ev_dir.mkdir(parents=True, exist_ok=True)

    engine_caps = doctor_info["engines"]
    vision_model_info = inspect_vision_model()
    asr_model_info = inspect_asr_model()
    llm_model_info = inspect_llm_model()

    cache_base = Path(config.cache_dir) if config.cache_dir else (Path.home() / ".cache" / "motionlab")
    vision_cache = (cache_base / "vision") if config.cache_enabled else None
    semantic_cache = (cache_base / "semantic") if config.cache_enabled else None

    if config.strict_cinematic:
        missing = []
        if not asr_model_info.get("available"):
            missing.append(f"asr:{asr_model_info.get('reason')}")
        if not llm_model_info.get("available"):
            missing.append(f"llm:{llm_model_info.get('reason')}")
        if not (engine_caps.get("remotion") or {}).get("available"):
            missing.append("remotion_not_opted_in_or_missing")
        if not (engine_caps.get("hyperframes") or {}).get("available"):
            missing.append("hyperframes_missing")
        if missing:
            raise RuntimeError(f"Strict cinematic mode blocked — missing required components: {missing}")

    # 1. Bounded parallel Vision ROI Tracking + Local Whisper ASR (2 independent stages!)
    tracker.start_stage("vision_and_asr")
    use_semantic = (config.semantic_mode != "off")
    if budget.analysis_workers >= 2 and use_semantic:
        with ThreadPoolExecutor(max_workers=2) as pool:
            f_roi = pool.submit(track_action_rois, media, analysis, config, ffmpeg_bin)
            f_asr = pool.submit(transcribe_local, media, config, semantic_cache, ffmpeg_bin)
            roi_tracks = f_roi.result()
            transcript = f_asr.result()
    else:
        roi_tracks = track_action_rois(media, analysis, config, ffmpeg_bin=ffmpeg_bin)
        transcript = transcribe_local(media, config, cache_dir=semantic_cache, ffmpeg_bin=ffmpeg_bin) if use_semantic else None
    tracker.end_stage("vision_and_asr", temp_dir=temp_dir)

    if transcript is None:
        from motionlab.semantic.schema import SEMANTIC_SCHEMA_VERSION, Transcript as _Tr
        transcript = _Tr(
            schema_version=SEMANTIC_SCHEMA_VERSION, source_hash=media.input_hash, source_audio_hash="",
            language="off", language_probability=0.0, duration_s=round(media.duration_s, 4), sample_rate_asr=16000,
            model_id="disabled", model_sha256="", asr_params={}, word_timestamps_exact=False,
            pts_priming_offset_s=0.0, pts_anomalies=[], status="DISABLED", full_text="",
        )
        transcript.transcript_hash = transcript.compute_hash()

    raw_units = build_semantic_units(transcript, analysis, roi_tracks, media)

    # 2. Local Qwen2.5-3B Semantic Creative Director
    tracker.start_stage("semantic_director")
    semantic_plan, units, niche_profile = direct_semantic_edit(
        media=media,
        analysis=analysis,
        roi_tracks=roi_tracks,
        transcript=transcript,
        raw_units=raw_units,
        matte_available=bool(vision_model_info.get("available")),
        config=config,
        run_dir=run_dir,
        cache_dir=semantic_cache,
    )
    tracker.end_stage("semantic_director", temp_dir=temp_dir)

    # Write ALL Phase 1 & Phase 2 semantic evidence files
    atomic_write_json(run_dir / "transcript.json", transcript.to_dict())
    atomic_write_text(run_dir / "transcript_readable.txt", format_readable_transcript(transcript))
    atomic_write_json(run_dir / "semantic_units.json", {"source_hash": media.input_hash, "units": [u.to_dict() for u in units]})
    atomic_write_json(run_dir / "semantic_plan.json", semantic_plan.to_dict())
    atomic_write_json(run_dir / "claim_evidence.json", {
        "source_hash": media.input_hash,
        "transcript_hash": transcript.transcript_hash,
        "accepted_count": len(semantic_plan.accepted_claims),
        "rejected_count": len(semantic_plan.rejected_claims),
        "accepted_claims": [c.to_dict() for c in semantic_plan.accepted_claims],
        "rejected_claims": [c.to_dict() for c in semantic_plan.rejected_claims],
    })
    atomic_write_json(run_dir / "niche_selection.json", {
        "niche": niche_profile.niche,
        "confidence": semantic_plan.niche_confidence,
        "reason": semantic_plan.niche_reason,
        "core_value_proposition": semantic_plan.core_value_proposition,
        "profile": niche_profile.to_dict(),
    })
    atomic_write_json(ev_dir / "script_alignment.json", {
        "source_hash": media.input_hash,
        "language": transcript.language,
        "language_probability": transcript.language_probability,
        "word_timestamps_exact": transcript.word_timestamps_exact,
        "pts_priming_offset_s": transcript.pts_priming_offset_s,
        "beats_aligned": [
            {
                "beat_id": b.beat_id,
                "recipe_id": b.recipe_id,
                "frame_range": [b.start_frame, b.end_frame_exclusive],
                "time_range_s": [round(b.start_frame / media.fps.as_float(), 3), round(b.end_frame_exclusive / media.fps.as_float(), 3)],
                "verbatim_quote": b.verbatim_quote,
                "intent": b.intent,
                "display_label": b.display_label,
                "secondary_labels": b.secondary_labels,
                "grounded_claim_ids": b.grounded_claim_ids,
            }
            for b in semantic_plan.beats
        ],
    })
    atomic_write_text(
        ev_dir / "semantic_qa.txt",
        "\n".join([
            "MOTIONLAB-AI ML-01.R4 LOCAL SEMANTIC & CLAIM PROVENANCE QA",
            f"ASR Engine     : {transcript.model_id} (sha256={transcript.model_sha256[:16]}, lang={transcript.language}, prob={transcript.language_probability}, elapsed={transcript.elapsed_s}s, cache_hit={transcript.cache_hit})",
            f"LLM Director   : {semantic_plan.llm_model_id} (sha256={semantic_plan.llm_model_sha256[:16]}, elapsed={semantic_plan.llm_elapsed_s}s, cache_hit={semantic_plan.cache_hit})",
            f"Detected Niche : {semantic_plan.niche} (conf={semantic_plan.niche_confidence}) — {semantic_plan.niche_reason}",
            f"Value Prop     : {semantic_plan.core_value_proposition}",
            f"Grounded Claims: {len(semantic_plan.accepted_claims)} accepted, {len(semantic_plan.rejected_claims)} rejected",
            f"Uncertain Words: {len(transcript.uncertain_spans)} flagged [UNCERTAIN]",
            "",
        ]),
    )

    prelim_anchors = build_anchor_tracks(media, analysis, roi_tracks, matte_result=None)

    # 3. Targeted Vision Matting (MODNet portrait spans + GrabCut+flow prop isolation spans)
    matte_recipe_ids = {"VFX01", "VFX04", "VFX07", "VFX09"}
    prop_recipe_ids = {"VFX02", "VFX10"}
    matte_spans = [(b.start_frame, b.end_frame_exclusive) for b in semantic_plan.beats if b.recipe_id in matte_recipe_ids]
    prop_spans = [(b.start_frame, b.end_frame_exclusive) for b in semantic_plan.beats if b.recipe_id in prop_recipe_ids]

    tracker.start_stage("vision_segmentation")
    if matte_spans and vision_model_info.get("available"):
        matte_res = segment_subject_sequence(
            media=media, spans=matte_spans, workdir=temp_dir,
            cache_dir=vision_cache if config.cache_enabled else (temp_dir / "nocache_vision"),
            preserve_tabletop_y=roi_tracks.get("tabletop_guard_y"),
            keyframe_stride=2, budget=budget, ffmpeg_bin=ffmpeg_bin,
        )
    else:
        matte_res = {"available": False, "status": "SKIPPED_NO_MATTE_SCENES", "model": vision_model_info, "spans": [], "quality_summary": {}}

    prop_res = (
        segment_prop_sequence(
            media=media, spans=prop_spans, roi_tracks=roi_tracks, workdir=temp_dir,
            cache_dir=vision_cache if config.cache_enabled else (temp_dir / "nocache_prop"),
            keyframe_stride=2, ffmpeg_bin=ffmpeg_bin,
        )
        if prop_spans
        else {"available": False, "status": "SKIPPED_NO_PROP_SPANS", "spans": [], "quality_summary": {}}
    )
    tracker.end_stage("vision_segmentation", temp_dir=temp_dir)

    matte_ok = bool(matte_res.get("available") and matte_res.get("quality_summary", {}).get("all_spans_quality_ok", False))
    prop_ok = bool(prop_res.get("available") and prop_res.get("quality_summary", {}).get("all_spans_quality_ok", False))
    anchors = build_anchor_tracks(media, analysis, roi_tracks, matte_res)
    debug_sheet = generate_matte_debug_sheet(Path(media.path), matte_res, anchors, ev_dir / "matte_debug_sheet.jpg")

    # 4. Compile deterministic R4 CompositionGraph
    if existing_graph is not None:
        graph = existing_graph
        graph_explain = {"replayed_from_graph": True, "graph_hash": graph.graph_hash}
    else:
        graph, graph_explain = compile_cinematic_graph(
            media=media, analysis=analysis, anchors=anchors, semantic_plan=semantic_plan,
            niche_profile=niche_profile, engine_caps=engine_caps,
            matte_available=matte_ok, prop_available=prop_ok, config=config,
        )

    cap_matrix = build_vfx_capability_matrix(engine_caps, vision_model_info, asr_model_info, llm_model_info)
    atomic_write_json(run_dir / "composition_graph.json", graph.to_dict())
    atomic_write_json(run_dir / "recipe_capability_matrix.json", cap_matrix)
    atomic_write_json(ev_dir / "vision_manifest.json", {
        "schema_version": "2.0.0",
        "source_hash": media.input_hash,
        "model": vision_model_info,
        "matte_status": matte_res.get("status"),
        "matte_available": matte_ok,
        "matte_cache_hit": bool(matte_res.get("cache_hit", False)),
        "matte_elapsed_s": float(matte_res.get("elapsed_s", 0.0)),
        "prop_status": prop_res.get("status"),
        "prop_available": prop_ok,
        "prop_spans": prop_res.get("spans", []),
        "roi_summary": roi_tracks.get("summary", {}),
        "tabletop_guard_y": roi_tracks.get("tabletop_guard_y"),
        "debug_sheet": str(Path(debug_sheet).relative_to(run_dir)) if debug_sheet else None,
        "matte_spans": matte_res.get("spans", []),
    })
    atomic_write_json(ev_dir / "roi_tracks.json", roi_tracks)
    atomic_write_json(ev_dir / "anchors.json", anchors)
    atomic_write_json(ev_dir / "mask_quality.json", {
        "matte_status": matte_res.get("status"),
        "matte_quality_summary": matte_res.get("quality_summary", {}),
        "prop_status": prop_res.get("status"),
        "prop_quality_summary": prop_res.get("quality_summary", {}),
        "matte_spans": matte_res.get("spans", []),
        "prop_spans": prop_res.get("spans", []),
    })

    # 5. Build Local Sound Design Timeline & Stems (Audio-Mode Aware!)
    eff_audio_mode = "preserve-exact" if config.audio_mode == "preserve-exact" else "mix"
    eff_sfx = (config.sfx_mode != "off") and (eff_audio_mode == "mix")
    eff_music = (config.music_mode == "local") and (eff_audio_mode == "mix")

    tracker.start_stage("sound_design_mix")
    audio_timeline, _used_sfx = build_audio_cues(
        media=media,
        semantic_plan=semantic_plan,
        niche_profile=niche_profile,
        audio_mode=eff_audio_mode,
        sfx_enabled=eff_sfx,
        music_enabled=eff_music,
        seed=config.seed,
    )
    sfx_manifest = build_sfx_asset_manifest(sr=audio_timeline.sample_rate, seed=config.seed)
    atomic_write_json(run_dir / "audio_timeline.json", audio_timeline.to_dict())
    atomic_write_json(run_dir / "sfx_asset_manifest.json", sfx_manifest)

    mix_info: dict[str, Any] | None = None
    mixed_wav_p: Path | None = None
    if eff_audio_mode == "mix" and eff_sfx and audio_timeline.events:
        mix_info = mix_preserving_master(
            media=media, timeline=audio_timeline, transcript=transcript,
            run_dir=run_dir, seed=config.seed, ffmpeg_bin=ffmpeg_bin, ffprobe_bin=ffprobe_bin,
        )
        mixed_wav_p = Path(mix_info["mixed_reference_path"])
    tracker.end_stage("sound_design_mix", temp_dir=temp_dir)

    if config.dry_run or not graph.scenes:
        return {
            "executed_cinematic": False,
            "graph": graph,
            "graph_explain": graph_explain,
            "transcript": transcript,
            "units": units,
            "semantic_plan": semantic_plan,
            "niche_profile": niche_profile,
            "audio_timeline": audio_timeline,
            "mix_info": mix_info,
            "matte_res": matte_res,
            "prop_res": prop_res,
            "anchors": anchors,
        }

    matte_by_span = {(int(s["start_frame"]), int(s["end_frame_exclusive"])): Path(s["matte_mp4"]) for s in (matte_res.get("spans") or [])}
    raw_matte_by_span = {(int(s["start_frame"]), int(s["end_frame_exclusive"])): Path(s.get("raw_matte_mp4", s["matte_mp4"])) for s in (matte_res.get("spans") or [])}
    prop_by_span = {(int(s["start_frame"]), int(s["end_frame_exclusive"])): Path(s["prop_matte_mp4"]) for s in (prop_res.get("spans") or []) if s.get("quality_ok")}

    # 6. Bounded parallel HyperFrames substantive semantic graphics
    hf = HyperFramesRenderer()
    aud_env = [p.normalized_rms for p in analysis.audio_envelope] if analysis.audio_envelope else None
    hf_jobs = [sc for sc in graph.scenes if sc.graphics_engine == "hyperframes"]
    hf_results: dict[str, dict[str, Any]] = {}
    fallback_history: list[dict[str, Any]] = []

    tracker.start_stage("hyperframes_graphics")
    if hf_jobs:
        def _run_hf(sc: CompositionScene) -> tuple[str, dict[str, Any] | None, dict[str, Any] | None]:
            spec = get_vfx_recipe(sc.recipe_id)
            if not hf.supports_recipe_graphics(spec.hyperframes_template, engine_caps):
                return sc.scene_id, None, {"scene_id": sc.scene_id, "requested_engine": "hyperframes", "fallback_engine": "none", "reason": "hyperframes_unavailable"}
            try:
                res = hf.render_graphics_scene(
                    scene_id=sc.scene_id, template_name=spec.hyperframes_template,
                    width=media.display_width, height=media.display_height, fps=media.fps.as_float(),
                    start_frame=sc.start_frame, end_frame_exclusive=sc.end_frame_exclusive,
                    anchor_bundle=sc.anchor_bundle, parameters=sc.parameters,
                    workdir=temp_dir, audio_envelope=aud_env, timeout_s=min(180.0, float(config.timeout_s)),
                )
                return sc.scene_id, res, None
            except Exception as exc:
                if config.strict or config.strict_cinematic:
                    raise RuntimeError(f"Strict mode: HyperFrames failed for {sc.scene_id}: {exc}") from exc
                return sc.scene_id, None, {"scene_id": sc.scene_id, "requested_engine": "hyperframes", "fallback_engine": "none", "reason": str(exc)[:360]}

        hf_workers = max(1, min(2, budget.analysis_workers, len(hf_jobs)))
        if hf_workers > 1:
            with ThreadPoolExecutor(max_workers=hf_workers) as pool:
                for sid, res, fb in [f.result() for f in [pool.submit(_run_hf, sc) for sc in hf_jobs]]:
                    if res:
                        hf_results[sid] = res
                    elif fb:
                        fallback_history.append(fb)
        else:
            for sc in hf_jobs:
                sid, res, fb = _run_hf(sc)
                if res:
                    hf_results[sid] = res
                elif fb:
                    fallback_history.append(fb)
    tracker.end_stage("hyperframes_graphics", temp_dir=temp_dir)

    # 7. Bounded Remotion full-frame VFX scene compositing + per-layer real ablation stills
    rem = RemotionRenderer()
    rendered_scenes: dict[str, dict[str, Any]] = {}
    ablation_probes: dict[str, dict[str, Any]] = {}

    tracker.start_stage("remotion_scenes")
    for sc in graph.scenes:
        matte_p = matte_by_span.get((sc.start_frame, sc.end_frame_exclusive))
        raw_matte_p = raw_matte_by_span.get((sc.start_frame, sc.end_frame_exclusive))
        prop_p = prop_by_span.get((sc.start_frame, sc.end_frame_exclusive)) if sc.parameters.get("requires_prop_mask") else None
        hf_p = Path(hf_results[sc.scene_id]["artifact_path"]) if sc.scene_id in hf_results else None

        if sc.primary_engine == "remotion" and rem.supports_composition(sc.recipe_id, engine_caps):
            try:
                r_res = rem.render_scene_composition(
                    scene=sc, media=media, workdir=temp_dir,
                    matte_mp4=matte_p, prop_matte_mp4=prop_p, hf_graphic_webm=hf_p,
                    timeout_s=min(240.0, float(config.timeout_s)),
                )
                rendered_scenes[sc.scene_id] = r_res

                # Per-layer real Remotion ablation stills on midpoint frame
                mid_local = max(2, (sc.end_frame_exclusive - sc.start_frame) // 2)
                slot_dir = Path(r_res["slot_dir"])
                c_out = cv2.VideoCapture(r_res["artifact_path"])
                c_out.set(cv2.CAP_PROP_POS_FRAMES, mid_local)
                ok_f, full_bgr = c_out.read()
                c_out.release()

                probe_roles: list[str] = []
                if r_res["used_hyperframes_graphics"]:
                    probe_roles.append("background_graphics" if sc.requires_matte else "foreground_graphics")
                if sc.requires_matte and r_res["used_matte_alpha"]:
                    probe_roles.append("foreground_subject_matte")
                if r_res.get("used_prop_alpha"):
                    probe_roles.append("prop_cutout_matte")
                if sc.uses_detail_crop:
                    probe_roles.append("source_detail")

                layer_ab: dict[str, Any] = {}
                if ok_f and full_bgr is not None:
                    full_f = full_bgr.astype(np.float32)
                    raw_m_gray = None
                    if raw_matte_p and raw_matte_p.is_file():
                        mc = cv2.VideoCapture(str(raw_matte_p))
                        mc.set(cv2.CAP_PROP_POS_FRAMES, mid_local)
                        ok_m, m_bgr = mc.read()
                        mc.release()
                        if ok_m and m_bgr is not None:
                            raw_m_gray = cv2.cvtColor(m_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

                    for role in probe_roles:
                        ab_png = slot_dir / f"ablate_{role}.png"
                        rem.render_scene_ablation_still(sc, media, slot_dir, mid_local, role, ab_png)
                        ab_bgr = cv2.imread(str(ab_png))
                        if ab_bgr is not None and ab_bgr.shape == full_bgr.shape:
                            diff = np.abs(full_f - ab_bgr.astype(np.float32))
                            mad = float(diff.mean())
                            max_px = float(diff.max())
                            cfrac = float((diff.max(axis=2) >= 6.0).mean())
                            entry: dict[str, Any] = {
                                "mid_local_frame": mid_local,
                                "ablation_mad": round(mad, 4),
                                "ablation_max_px": round(max_px, 2),
                                "ablation_changed_frac": round(cfrac, 5),
                                "verified": bool(mad >= 0.12 or (max_px >= 16.0 and cfrac >= 0.0008)),
                            }
                            if role == "background_graphics" and raw_m_gray is not None:
                                inside_mask = raw_m_gray >= 0.90
                                outside_mask = raw_m_gray <= 0.10
                                if int(inside_mask.sum()) > 500 and int(outside_mask.sum()) > 500:
                                    diff_ch = diff.mean(axis=2)
                                    in_mad = float(diff_ch[inside_mask].mean())
                                    out_mad = float(diff_ch[outside_mask].mean())
                                    entry["occlusion_inside_subject_mad"] = round(in_mad, 4)
                                    entry["occlusion_outside_subject_mad"] = round(out_mad, 4)
                                    entry["occlusion_z_order_verified"] = bool(out_mad > max(0.5, in_mad * 2.0))
                            layer_ab[role] = entry
                ablation_probes[sc.scene_id] = layer_ab
                continue
            except Exception as exc:
                if config.strict or config.strict_cinematic:
                    raise RuntimeError(f"Strict mode: Remotion scene failed for {sc.scene_id}: {exc}") from exc
                fallback_history.append({"scene_id": sc.scene_id, "requested_engine": "remotion", "fallback_engine": "ffmpeg", "reason": str(exc)[:400]})

        rendered_scenes[sc.scene_id] = _render_candidate_scene_ffmpeg_fallback(sc, media, temp_dir, matte_p, hf_p, ffmpeg_bin)
        ablation_probes[sc.scene_id] = {}
    tracker.end_stage("remotion_scenes", temp_dir=temp_dir)

    # 8. Canonical Single-Pass Assembly + Mixed Default Audio Mux
    partial_out = run_dir / "final.partial.mp4"
    tracker.start_stage("canonical_assembly")
    asm_stats = _assemble_r4_video_and_mux_mix(
        media=media, plan=base_plan, graph=graph, rendered_scenes=rendered_scenes,
        mixed_wav_path=mixed_wav_p, workdir=temp_dir, out_partial_path=partial_out,
        budget=budget, config=config, ffmpeg_bin=ffmpeg_bin,
    )
    tracker.end_stage("canonical_assembly", temp_dir=temp_dir)

    actual_engines: set[str] = {"ffmpeg"}
    for rinfo in rendered_scenes.values():
        actual_engines.add(rinfo["engine"])
        if rinfo.get("used_hyperframes_graphics"):
            actual_engines.add("hyperframes")

    ff_probe_proc = subprocess.run(
        [ffprobe_bin, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(partial_out)],
        capture_output=True, text=True, timeout=20, shell=False,
    )
    ffprobe_info = json.loads(ff_probe_proc.stdout) if ff_probe_proc.returncode == 0 else {}

    render_res = RenderResult(
        status="PASS",
        artifact_path=str(partial_out),
        engines_used=sorted(actual_engines),
        timings_s={"video_assembly_s": asm_stats["video_assembly_s"], "audio_mux_s": asm_stats["audio_mux_s"]},
        ffprobe_info=ffprobe_info,
        event_engine_map={sid: r["engine"] for sid, r in rendered_scenes.items()},
        fallback_history=fallback_history,
        encoding_settings={
            "mode": "ml01_r4_cinematic_vfx_sound_design",
            "encoder": "libx264",
            "crf": "17",
            "preset": "medium",
            "pix_fmt": "yuv420p",
            "frames_rendered": asm_stats["frames_written"],
            "scene_frames_spliced": asm_stats["scene_frames_spliced"],
            "audio_mode": eff_audio_mode,
            "sfx_events_mixed": len(audio_timeline.events) if mixed_wav_p else 0,
        },
        warnings=list(graph.warnings),
    )

    actual_engine_usage = {
        "edition": "ml-01.r4",
        "engines_used": sorted(actual_engines),
        "dual_engine_active": ("remotion" in actual_engines and "hyperframes" in actual_engines),
        "local_asr_used": transcript.model_id,
        "local_llm_used": semantic_plan.llm_model_id,
        "audio_mode": eff_audio_mode,
        "sfx_events_mixed": len(audio_timeline.events) if mixed_wav_p else 0,
        "scenes": [
            {
                "scene_id": sc.scene_id,
                "recipe_id": sc.recipe_id,
                "recipe_name": sc.recipe_name,
                "archetype": sc.archetype,
                "frame_range": [sc.start_frame, sc.end_frame_exclusive],
                "time_range_s": [round(sc.start_frame / media.fps.as_float(), 3), round(sc.end_frame_exclusive / media.fps.as_float(), 3)],
                "requested_primary_engine": sc.primary_engine,
                "actual_primary_engine": rendered_scenes[sc.scene_id]["engine"],
                "actual_graphics_engine": "hyperframes" if rendered_scenes[sc.scene_id].get("used_hyperframes_graphics") else "none",
                "used_matte_alpha": bool(rendered_scenes[sc.scene_id].get("used_matte_alpha")),
                "used_prop_alpha": bool(rendered_scenes[sc.scene_id].get("used_prop_alpha")),
                "scene_artifact_sha256": rendered_scenes[sc.scene_id]["artifact_sha256"],
                "hf_artifact_sha256": hf_results[sc.scene_id]["artifact_sha256"] if sc.scene_id in hf_results else None,
            }
            for sc in graph.scenes
            if sc.scene_id in rendered_scenes
        ],
        "fallback_history": fallback_history,
    }
    atomic_write_json(run_dir / "actual_engine_usage.json", actual_engine_usage)

    return {
        "executed_cinematic": True,
        "graph": graph,
        "graph_explain": graph_explain,
        "transcript": transcript,
        "units": units,
        "semantic_plan": semantic_plan,
        "niche_profile": niche_profile,
        "audio_timeline": audio_timeline,
        "mix_info": mix_info,
        "mixed_audio_active": mixed_wav_p is not None,
        "matte_res": matte_res,
        "prop_res": prop_res,
        "anchors": anchors,
        "hf_results": hf_results,
        "rendered_scenes": rendered_scenes,
        "ablation_probes": ablation_probes,
        "actual_engine_usage": actual_engine_usage,
        "render_result": render_res,
        "partial_out": partial_out,
    }
