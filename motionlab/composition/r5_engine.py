"""Phase A3/A5/A6 (ML-01.R5): End-to-End Generative Dual-Engine DAG Executor.

Coordinates:
  Preflight Capability Gate (fail-closed) -> Parallel ROI Tracking & Offline Faster-Whisper ASR ->
  Generic StoryGraph & Integer-Cent VerifiedLedger -> Local Qwen2.5-3B Generative Director +
  UnifiedPrimitiveRegistry Search -> Typed StoryboardSpec Compiler -> Targeted MODNet Matting ->
  Imported HyperFrames Apache-2.0 VP9 Alpha Blocks -> Imported Remotion MIT MotionLabGenerativeScene
  + Midpoint Layer Ablation Stills -> A4 Voice-Priority SFX Mix + Bit-Exact Audio Master ->
  Single-Pass 0..N-1 Assembly -> Technical Validation + Reuse Gate + A5 Creative Quality Proxy ->
  Atomic Promotion (`final.partial.mp4` -> `final.mp4`) -> Downscaled `preview_9x16.mp4` -> Run README.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from motionlab import SCHEMA_VERSION, __version__
from motionlab.assets.catalog import inspect_product_asset_catalog, match_product_assets_for_name
from motionlab.audio.assets import inspect_foley_library
from motionlab.audio.mixer import mix_preserving_master, validate_final_audio_mix
from motionlab.audio.semantic_sound import (
    align_sound_to_marker,
    extract_realized_animation_markers,
    extract_rendered_animation_markers,
    extract_sonic_opportunities,
    plan_sonic_score,
)
from motionlab.audio.synthesis import build_sfx_asset_manifest
from motionlab.composition.art_direction import (
    build_shared_element_tracks,
    build_style_tokens,
    evaluate_phone_legibility,
    solve_layout_with_protection,
    validate_layout_sweep,
)
from motionlab.composition.continuous_timeline import compile_continuous_timeline
from motionlab.composition.creative_quality import evaluate_creative_quality, verify_ablation_still_diff
from motionlab.composition.generative_compiler import (
    compile_scene_hyperframes_html,
    compile_scene_remotion_props,
    compile_storyboard_audio_timeline,
    compile_storyboard_graph,
)
from motionlab.composition.generative_schema import (
    StoryboardSpec,
    validate_canonical_reuse_gate,
)
from motionlab.composition.primitive_registry import load_unified_registry
from motionlab.composition.visual_critic import run_objective_rendered_critic
from motionlab.config import EditConfig, PROJECT_ROOT
from motionlab.renderers.base import verify_webm_alpha_pixels
from motionlab.renderers.hyperframes import HyperFramesRenderer
from motionlab.renderers.remotion import RemotionRenderer
from motionlab.resource import ResourceBudget, ResourceTracker, RuntimeResourceWatchdog
from motionlab.schemas import AnalysisResult, EditPlan, MediaInfo, atomic_write_json, atomic_write_text
from motionlab.semantic.asr import format_readable_transcript, inspect_asr_model, transcribe_local
from motionlab.semantic.generative_director import direct_generative_storyboard
from motionlab.semantic.local_model import inspect_llm_model
from motionlab.semantic.story_graph import StoryGraph, build_story_graph
from motionlab.vision.anchors import build_anchor_tracks, generate_matte_debug_sheet
from motionlab.vision.segmentation import build_foreground_alpha_webm, inspect_vision_model, segment_subject_sequence
from motionlab.vision.tracking import track_action_rois


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _assemble_full_timeline_and_mux(
    media: MediaInfo,
    storyboard: StoryboardSpec,
    rendered_scenes: dict[str, dict[str, Any]],
    mixed_wav_path: Path | None,
    workdir: Path,
    out_partial_path: Path,
    budget: ResourceBudget,
    config: EditConfig,
    ffmpeg_bin: str,
) -> dict[str, Any]:
    """Single-pass streaming frame-exact timeline assembly (0..duration_frames-1) + single default audio mux."""
    t0 = time.perf_counter()
    w, h = media.display_width, media.display_height
    total_f = media.duration_frames
    fps_str = media.fps.to_ffmpeg()
    video_only = workdir / "r5_video_only.mp4"

    scene_caps: dict[str, cv2.VideoCapture] = {}
    frame_owner: list[tuple[str, int] | None] = [None] * total_f
    for sc in storyboard.scenes:
        rinfo = rendered_scenes.get(sc.scene_id)
        if not rinfo or not Path(rinfo["artifact_path"]).is_file():
            continue
        scene_caps[sc.scene_id] = cv2.VideoCapture(rinfo["artifact_path"])
        for f in range(sc.start_frame, min(total_f, sc.end_frame_exclusive)):
            frame_owner[f] = (sc.scene_id, f - sc.start_frame)

    dec = subprocess.Popen(
        [ffmpeg_bin, "-v", "error", "-nostdin", "-i", str(media.source_path), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
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
    spliced_frames = 0
    try:
        for f in range(total_f):
            raw = dec.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            owner = frame_owner[f]
            if owner is not None:
                sc_id, _rel_f = owner
                cap = scene_caps.get(sc_id)
                if cap is not None:
                    ok_s, bgr = cap.read()
                    if ok_s and bgr is not None:
                        if bgr.shape[1] != w or bgr.shape[0] != h:
                            bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
                        enc.stdin.write(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).tobytes())
                        frames_written += 1
                        spliced_frames += 1
                        continue
            enc.stdin.write(raw)
            frames_written += 1
    finally:
        for cap in scene_caps.values():
            cap.release()
        enc.stdin.close()
        dec.stdout.close()
        dec.wait(timeout=30)
        enc.wait(timeout=60)
        wd.stop()

    if enc.returncode != 0 or not video_only.is_file():
        raise RuntimeError(f"R5 video assembly failed (rc={enc.returncode})")
    v_s = round(time.perf_counter() - t0, 3)

    t1 = time.perf_counter()
    if mixed_wav_path is not None and mixed_wav_path.is_file():
        subprocess.run(
            [
                ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(video_only),
                "-i", str(mixed_wav_path),
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k",
                "-movflags", "+faststart",
                "-shortest",
                str(out_partial_path),
            ],
            check=True, timeout=90, shell=False,
        )
    else:
        subprocess.run(
            [
                ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(video_only),
                "-i", str(media.source_path),
                "-map", "0:v:0", "-map", "1:a:0?",
                "-c:v", "copy", "-c:a", "copy",
                "-movflags", "+faststart",
                str(out_partial_path),
            ],
            check=True, timeout=90, shell=False,
        )
    a_s = round(time.perf_counter() - t1, 3)
    return {
        "frames_written": frames_written,
        "scene_frames_spliced": spliced_frames,
        "video_assembly_s": v_s,
        "audio_mux_s": a_s,
        "watchdog_peak_tree_rss_mb": wd.peak_rss_mb,
    }


def _downscale_preview_from_final(final_mp4: Path, preview_mp4: Path, ffmpeg_bin: str) -> dict[str, Any]:
    """Generate canonical 540x960 preview directly from promoted final.mp4 (zero replanning, exact timeline & audio)."""
    subprocess.run(
        [
            ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(final_mp4),
            "-vf", "scale=540:960:flags=lanczos",
            "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
            "-c:a", "copy",
            "-movflags", "+faststart",
            str(preview_mp4),
        ],
        check=True, timeout=90, shell=False,
    )
    cap = cv2.VideoCapture(str(preview_mp4))
    n_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pw, ph = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {
        "path": preview_mp4.name,
        "width": pw,
        "height": ph,
        "frames": n_f,
        "sha256": _sha256_file(preview_mp4),
        "audio_stream_copied_from_final": True,
    }


def render_compiled_storyboard_to_mp4(
    *,
    media: MediaInfo,
    storyboard: StoryboardSpec,
    story_graph: StoryGraph,
    roi_tracks: dict[str, Any],
    config: EditConfig,
    run_id: str,
    run_dir: Path,
    budget: ResourceBudget,
    tracker: ResourceTracker,
    output_filename: str = "final.mp4",
    render_preview: bool = True,
    require_dual_engine_reuse_gate: bool = True,
    transcript_obj: Any = None,
) -> dict[str, Any]:
    """Shared deterministic R5 renderer/mixer/validator used by BOTH canonical AUTO runs and separate QA benchmark."""
    ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"
    ffprobe_bin = shutil.which("ffprobe") or "ffprobe"
    temp_dir = run_dir / "temp"
    ev_dir = run_dir / "evidence"
    temp_dir.mkdir(parents=True, exist_ok=True)
    ev_dir.mkdir(parents=True, exist_ok=True)

    reg = load_unified_registry()
    rem = RemotionRenderer(license_allowed=config.allow_remotion)
    hf = HyperFramesRenderer(enabled=config.allow_hyperframes)

    compiled_plan = compile_storyboard_graph(storyboard, story_graph, media, registry=reg, require_verified_primitives=True)
    atomic_write_json(run_dir / "storyboard_spec.json", storyboard.to_dict())
    atomic_write_json(run_dir / "composition_graph.json", compiled_plan)

    # 0. R5.R2 Premium Art Direction, Continuous Timeline, Numeric State Trace & Product Asset Audit
    style_tokens = build_style_tokens(niche=storyboard.niche, source_mp4=Path(media.source_path))
    layout_plan, frame_prot_rows = solve_layout_with_protection(
        storyboard, roi_tracks, media.display_width, media.display_height
    )
    legibility_report = evaluate_phone_legibility(
        layout_plan, storyboard, media.display_width, media.display_height
    )
    continuity_doc = build_shared_element_tracks(storyboard, story_graph, layout_plan)
    continuous_tl = compile_continuous_timeline(storyboard, story_graph, layout_plan)
    prod_catalog = inspect_product_asset_catalog()
    prod_match = match_product_assets_for_name(story_graph.product_name, catalog=prod_catalog)
    layout_sweep_errors = validate_layout_sweep(layout_plan, legibility_report)
    atomic_write_json(run_dir / "layout.json", {**layout_plan, "style_tokens": style_tokens, "layout_sweep_errors": layout_sweep_errors})
    atomic_write_json(run_dir / "continuity.json", {**continuity_doc, "continuous_timeline_hash": continuous_tl["continuity_hash"]})
    atomic_write_json(run_dir / "continuous_timeline.json", continuous_tl)
    atomic_write_json(run_dir / "numeric_state_trace.json", continuous_tl["numeric_state_trace"])
    atomic_write_json(ev_dir / "phone_legibility.json", legibility_report)
    atomic_write_json(ev_dir / "product_asset_status.json", prod_match)
    prot_csv_lines = ["frame,scene_id,stage_mode,protect_live_face,eye_mouth_overlap_ratio,hero_top_px,hero_bottom_px,safe_ok"]
    for pr in frame_prot_rows:
        prot_csv_lines.append(
            f"{pr['frame']},{pr['scene_id']},{pr['stage_mode']},{pr['protect_live_face']},"
            f"{pr['eye_mouth_overlap_ratio']},{pr['hero_top_px']},{pr['hero_bottom_px']},{pr['safe_ok']}"
        )
    atomic_write_text(ev_dir / "frame_protection.csv", "\n".join(prot_csv_lines) + "\n")

    # 1. Targeted MODNet Portrait Segmentation ONLY for scenes requiring foreground matte
    matte_spans = [(sc.start_frame, sc.end_frame_exclusive) for sc in storyboard.scenes if sc.requires_matte]
    vision_model_info = inspect_vision_model()
    cache_base = Path(config.cache_dir) if config.cache_dir else (PROJECT_ROOT / ".cache" / "motionlab_r5")
    vision_cache = (cache_base / "vision") if config.cache_enabled else (temp_dir / "nocache_vision")
    hf_cache_dir = cache_base / "hyperframes_overlays"
    rem_scene_cache_dir = cache_base / "remotion_scenes"
    rem_bundle_cache_dir = (cache_base / "remotion_bundles") if config.cache_enabled else (temp_dir / "nocache_rem_bundles")

    tracker.start_stage("vision_segmentation")
    if matte_spans and vision_model_info.get("available"):
        matte_res = segment_subject_sequence(
            media=media,
            spans=matte_spans,
            workdir=temp_dir,
            cache_dir=vision_cache,
            preserve_tabletop_y=roi_tracks.get("tabletop_guard_y"),
            keyframe_stride=2,
            budget=budget,
            ffmpeg_bin=ffmpeg_bin,
        )
    else:
        matte_res = {"available": False, "status": "SKIPPED_NO_MATTE_SCENES", "model": vision_model_info, "spans": [], "quality_summary": {}}
    tracker.end_stage("vision_segmentation", temp_dir=temp_dir)

    matte_by_span = {(int(sp["start_frame"]), int(sp["end_frame_exclusive"])): Path(sp["matte_mp4"]) for sp in matte_res.get("spans", [])}

    eff_audio_mode = "preserve-exact" if config.audio_mode == "preserve-exact" else "mix"
    eff_sfx = (config.sfx_mode != "off") and (eff_audio_mode == "mix")
    eff_music = (config.music_mode == "local") and (eff_audio_mode == "mix")
    sr_target = media.audio_streams[0].sample_rate if media.audio_streams else 44100
    audio_pts = [s for s in media.pts_summary if s.get("codec_type") == "audio"]
    priming_s = float(audio_pts[0].get("priming_offset_s", 0.0)) if audio_pts else 0.0

    # Emit PLANNED animation markers before render (distinguished explicitly from post-render REALIZED markers!)
    planned_markers = extract_rendered_animation_markers(storyboard, story_graph)
    atomic_write_json(ev_dir / "animation_markers_planned.json", {"phase": "PLANNED_PRE_RENDER", "markers": [m.to_dict() for m in planned_markers]})

    if config.dry_run:
        return {
            "status": "DRY_RUN_PLANNED",
            "storyboard_hash": storyboard.storyboard_hash,
            "continuity_hash": continuous_tl["continuity_hash"],
            "scenes": len(storyboard.scenes),
            "compiled_plan": compiled_plan,
        }

    # 2. Render Imported HyperFrames Apache-2.0 VP9 Alpha Graphics (with Content-Addressed Overlay Cache & fast static lint!)
    tracker.start_stage("hyperframes_graphics")
    hf_results: dict[str, dict[str, Any]] = {}
    hf_cache_hits = 0
    for sc in storyboard.scenes:
        hf_pid, hf_html, hf_role = compile_scene_hyperframes_html(sc, storyboard, story_graph)
        if not hf_pid or not hf_html:
            continue
        hf_webm = temp_dir / f"hf_{sc.scene_id}_{hf_pid}.webm"
        r_hf = hf.render_html_slot(
            slot_dir=temp_dir / f"hf_slot_{sc.scene_id}",
            html=hf_html,
            out_path=hf_webm,
            fps=media.fps_float,
            fmt="webm",
            timeout_s=180.0,
            cache_dir=hf_cache_dir,
            use_cache=bool(config.cache_enabled),
            skip_cli_lint=True,
        )
        if r_hf.get("cache_hit"):
            hf_cache_hits += 1
        mid_hf_f = max(4, min(20, (sc.end_frame_exclusive - sc.start_frame) // 2))
        alpha_stats = verify_webm_alpha_pixels(
            hf_webm, width=media.display_width, height=media.display_height, sample_frame=mid_hf_f, ffmpeg_bin=ffmpeg_bin
        )
        if not alpha_stats.get("has_visible_alpha", False):
            raise RuntimeError(f"HyperFrames block {hf_pid} for {sc.scene_id} rendered empty alpha!")
        hf_results[sc.scene_id] = {
            **r_hf,
            "primitive_id": hf_pid,
            "layer_role": hf_role,
            "alpha_stats": alpha_stats,
        }
    tracker.end_stage("hyperframes_graphics", temp_dir=temp_dir)

    # 3. R5.R2 Optimized Remotion MIT Single-Bundle Persistent-Browser Batch + Continuity Handles + Active-Window Ablation!
    tracker.start_stage("remotion_scenes")
    rendered_scenes: dict[str, dict[str, Any]] = {}
    ablation_results: dict[str, dict[str, Any]] = {}
    actual_engines: set[str] = {"ffmpeg"}
    actual_primitives: list[str] = []

    roi_frames = roi_tracks.get("keyframes") or roi_tracks.get("frames") or []
    fps_str = media.fps.to_ffmpeg()
    rem_code_hash = rem.compute_code_hash()
    rem_scene_cache_dir.mkdir(parents=True, exist_ok=True)

    public_mounts: list[dict[str, str]] = []
    batch_jobs: list[dict[str, Any]] = []
    scene_prep: dict[str, dict[str, Any]] = {}
    compiled_props_by_scene: dict[str, dict[str, Any]] = {}
    ablation_plan: dict[str, list[dict[str, Any]]] = {}
    rem_scene_cache_hits = 0

    for sc in storyboard.scenes:
        slot_dir = temp_dir / f"rem_{sc.scene_id}"
        pub_dir = slot_dir / "public"
        pub_dir.mkdir(parents=True, exist_ok=True)

        # Slice exact source mezzanine for live scenes OR ANY cutaway with transition_in OR transition_out (fixes D06!)
        tr_out_kind = str(sc.parameters.get("transition_out", "none"))
        src_rel: str | None = None
        src_sha = ""
        if sc.stage_mode != "full_graphic_cutaway" or sc.transition_in.kind != "none" or tr_out_kind != "none":
            src_slice = pub_dir / "scene_source.mp4"
            rem._slice_source_mezzanine(Path(media.source_path), sc.start_frame, sc.end_frame_exclusive, fps_str, src_slice)
            src_rel = f"{sc.scene_id}/{src_slice.name}"
            src_sha = _sha256_file(src_slice)
            public_mounts.append({"src": str(src_slice), "rel": src_rel})

        # Build foreground alpha WebM if MODNet matte was produced for this span
        fg_rel: str | None = None
        fg_sha = ""
        matte_p = matte_by_span.get((sc.start_frame, sc.end_frame_exclusive))
        if sc.requires_matte and matte_p is not None and matte_p.is_file():
            fg_webm = pub_dir / "fg_alpha.webm"
            build_foreground_alpha_webm(
                Path(media.source_path), matte_p, sc.start_frame, sc.end_frame_exclusive, fps_str, fg_webm, ffmpeg_bin=ffmpeg_bin
            )
            fg_rel = f"{sc.scene_id}/{fg_webm.name}"
            fg_sha = _sha256_file(fg_webm)
            public_mounts.append({"src": str(fg_webm), "rel": fg_rel})

        hf_rel: str | None = None
        hf_sha = ""
        hf_role = "foreground_graphics"
        if sc.scene_id in hf_results:
            hf_src_p = Path(hf_results[sc.scene_id]["artifact_path"])
            hf_rel = f"{sc.scene_id}/hf_graphic.webm"
            hf_sha = str(hf_results[sc.scene_id]["artifact_sha256"])
            hf_role = hf_results[sc.scene_id]["layer_role"]
            public_mounts.append({"src": str(hf_src_p), "rel": hf_rel})
            actual_engines.add("hyperframes")
            actual_primitives.append(hf_results[sc.scene_id]["primitive_id"])

        sc_rois = [r for r in roi_frames if sc.start_frame <= int(r.get("frame", -1)) < sc.end_frame_exclusive]
        task_anchor = None
        face_anchor = None
        if sc_rois:
            t_boxes = [r.get("task") or r.get("task_box") for r in sc_rois if (r.get("task") or r.get("task_box"))]
            if t_boxes:
                t_cxs = [float(b.get("cx", b.get("x", 0.3) + b.get("w", 0.4) * 0.5)) for b in t_boxes]
                t_cys = [float(b.get("cy", b.get("y", 0.55) + b.get("h", 0.35) * 0.5)) for b in t_boxes]
                task_anchor = {"cx": round(float(np.median(t_cxs)), 4), "cy": round(float(np.median(t_cys)), 4), "w": 0.44, "h": 0.34}
            f_boxes = [r.get("face") or r.get("face_box") for r in sc_rois if (r.get("face") or r.get("face_box"))]
            if f_boxes:
                f_cxs = [float(b.get("cx", b.get("x", 0.35) + b.get("w", 0.3) * 0.5)) for b in f_boxes]
                f_cys = [float(b.get("cy", b.get("y", 0.18) + b.get("h", 0.28) * 0.5)) for b in f_boxes]
                face_anchor = {"cx": round(float(np.median(f_cxs)), 4), "cy": round(float(np.median(f_cys)), 4), "w": 0.30, "h": 0.28}

        props = compile_scene_remotion_props(
            sc,
            storyboard,
            story_graph,
            source_file=src_rel,
            fg_alpha_file=fg_rel,
            prop_alpha_file=None,
            hf_graphic_file=hf_rel,
            hf_layer_role=hf_role,
            task_anchor=task_anchor,
            face_anchor=face_anchor,
            continuity_tracks=continuous_tl.get("tracks", []),
        )
        compiled_props_by_scene[sc.scene_id] = props
        atomic_write_json(slot_dir / "props.json", props)

        # Content-addressed per-scene cache key (invalidates on code, props, continuity, source, matte, HF, seed, dims!)
        scene_cache_key = hashlib.sha256(
            json.dumps(
                {
                    "code_hash": rem_code_hash,
                    "schema_version": storyboard.schema_version,
                    "continuity_hash": continuous_tl["continuity_hash"],
                    "props": props,
                    "src_sha": src_sha,
                    "fg_sha": fg_sha,
                    "hf_sha": hf_sha,
                    "seed": int(config.seed),
                    "codec": "h264",
                    "crf": 16,
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:24]
        cached_sc_mp4 = rem_scene_cache_dir / f"rem_{scene_cache_key}.mp4"
        out_sc_mp4 = slot_dir / "scene.mp4"

        cache_hit = False
        if config.cache_enabled and cached_sc_mp4.is_file():
            cap_chk = cv2.VideoCapture(str(cached_sc_mp4))
            nf_chk = int(cap_chk.get(cv2.CAP_PROP_FRAME_COUNT))
            cw_chk = int(cap_chk.get(cv2.CAP_PROP_FRAME_WIDTH))
            ch_chk = int(cap_chk.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap_chk.release()
            if nf_chk == props["durationInFrames"] and cw_chk == media.display_width and ch_chk == media.display_height:
                shutil.copy2(cached_sc_mp4, out_sc_mp4)
                cache_hit = True
                rem_scene_cache_hits += 1

        if not cache_hit:
            batch_jobs.append({
                "kind": "media",
                "id": f"media:{sc.scene_id}",
                "compositionId": "MotionLabGenerativeScene",
                "props": props,
                "outputLocation": str(out_sc_mp4),
                "codec": "h264",
                "pixelFormat": "yuv420p",
                "crf": 16,
                "concurrency": 1,
            })

        # Active-Window Ablation Plan (entry / peak / exit-bridge!)
        dur_f = int(props["durationInFrames"])
        mid_rel = max(6, min(dur_f - 8, dur_f // 2))
        ab_dir = ev_dir / "ablations" / sc.scene_id
        ab_dir.mkdir(parents=True, exist_ok=True)

        sc_ab_checks: list[dict[str, Any]] = []
        if hf_rel is not None:
            sc_ab_checks.append({"role": hf_role, "frame": mid_rel, "window": "peak"})
        if props["primitives"]:
            rem_f = mid_rel
            for pn in props["primitives"]:
                if pn["primitiveId"] == "rem_rc_flip_page":
                    fd = int(pn["params"].get("flip_delay", max(10, round(dur_f * 0.28))))
                    fdur = int(pn["params"].get("flip_duration", max(16, round(dur_f * 0.34))))
                    rem_f = max(4, min(dur_f - 6, fd + fdur // 2))
                elif pn["primitiveId"] == "rem_viv_particle_field":
                    rem_f = max(8, min(dur_f - 8, dur_f // 2))
            only_particles = all(pn["primitiveId"] == "rem_viv_particle_field" for pn in props["primitives"])
            if only_particles and fg_rel is not None:
                sc_ab_checks.append({"role": "foreground_subject_matte", "frame": mid_rel, "window": "peak"})
            else:
                sc_ab_checks.append({"role": "remotion_primitive", "frame": rem_f, "window": "peak"})
        if sc.transition_in.kind != "none" and src_rel is not None and sc.transition_in.duration_frames >= 4:
            tr_f = max(2, min(dur_f - 2, sc.transition_in.duration_frames // 2))
            sc_ab_checks.append({"role": "transition_in", "frame": tr_f, "window": "entry"})
        elif tr_out_kind != "none" and src_rel is not None:
            tr_out_f = int(props.get("transitionOutFrames", 12))
            out_mid = max(2, min(dur_f - 2, dur_f - max(2, tr_out_f // 2)))
            sc_ab_checks.append({"role": "transition_out", "frame": out_mid, "window": "exit"})
        elif fg_rel is not None and not any(c["role"] == "foreground_subject_matte" for c in sc_ab_checks):
            sc_ab_checks.append({"role": "foreground_subject_matte", "frame": mid_rel, "window": "peak"})

        seen_full_frames: dict[int, Path] = {}
        for chk in sc_ab_checks[:3]:
            sf_rel = int(chk["frame"])
            if sf_rel not in seen_full_frames:
                full_p = ab_dir / f"full_f{sf_rel:03d}.png"
                seen_full_frames[sf_rel] = full_p
                batch_jobs.append({
                    "kind": "still",
                    "id": f"still:{sc.scene_id}:full:f{sf_rel}",
                    "compositionId": "MotionLabGenerativeScene",
                    "props": {**props, "excludeLayerRole": None},
                    "frame": sf_rel,
                    "outputLocation": str(full_p),
                })
            ex_p = ab_dir / f"without_{chk['role']}_f{sf_rel:03d}.png"
            batch_jobs.append({
                "kind": "still",
                "id": f"still:{sc.scene_id}:{chk['role']}:f{sf_rel}",
                "compositionId": "MotionLabGenerativeScene",
                "props": {**props, "excludeLayerRole": chk["role"]},
                "frame": sf_rel,
                "outputLocation": str(ex_p),
            })
            chk["full_png"] = full_p
            chk["ex_png"] = ex_p

        ablation_plan[sc.scene_id] = sc_ab_checks[:3]
        scene_prep[sc.scene_id] = {
            "props": props,
            "out_mp4": out_sc_mp4,
            "cached_mp4": cached_sc_mp4,
            "cache_hit": cache_hit,
            "cache_key": scene_cache_key,
            "used_matte_alpha": fg_rel is not None,
            "used_hyperframes_graphics": hf_rel is not None,
        }

        actual_engines.add("remotion")
        for ly in sc.layers:
            if ly.engine == "remotion":
                actual_primitives.append(ly.primitive_id)
        if sc.stage_mode == "aspect_morph_editorial":
            actual_primitives.append("rem_viv_aspect_morph")
        elif sc.stage_mode == "live_depth_matte" and fg_rel is not None:
            actual_primitives.append("rem_ru_parallax")
        if sc.transition_in.kind == "frame_morph":
            actual_primitives.append("rem_gtk_frame_morph")
        elif sc.transition_in.kind == "line_led":
            actual_primitives.append("rem_gtk_line_led")
        elif sc.transition_in.kind == "flip_page":
            actual_primitives.append("rem_rc_flip_page")

    # Execute ALL uncached scenes + ALL active-window ablation stills in ONE persistent-browser Remotion worker!
    batch_summary: dict[str, Any] = {
        "bundle_reused": True,
        "bundle_s": 0.0,
        "browser_open_s": 0.0,
        "browser_launches": 0,
        "select_composition_calls": 0,
        "total_s": 0.0,
        "jobs": [],
    }
    if batch_jobs:
        batch_summary = rem.render_batch_jobs(
            jobs=batch_jobs,
            public_mounts=public_mounts,
            work_dir=temp_dir / "rem_batch",
            bundle_cache_root=rem_bundle_cache_dir,
            timeout_s=max(300.0, float(config.timeout_s)),
        )

    # Populate validated per-scene cache atomically & collect per-scene artifacts + active-window ablations
    job_elapsed_map = {j["id"]: j["elapsed_s"] for j in batch_summary.get("jobs", [])}
    for sc in storyboard.scenes:
        sp = scene_prep[sc.scene_id]
        out_sc_mp4 = sp["out_mp4"]
        if not out_sc_mp4.is_file():
            raise RuntimeError(f"Remotion batch did not produce scene output: {out_sc_mp4}")
        if not sp["cache_hit"]:
            tmp_c = sp["cached_mp4"].with_suffix(".mp4.partial")
            try:
                shutil.copy2(out_sc_mp4, tmp_c)
                os.replace(tmp_c, sp["cached_mp4"])
            except Exception:
                tmp_c.unlink(missing_ok=True)
        rendered_scenes[sc.scene_id] = {
            "composition_id": "MotionLabGenerativeScene",
            "engine": "remotion",
            "artifact_path": str(out_sc_mp4),
            "artifact_sha256": _sha256_file(out_sc_mp4),
            "size_bytes": int(out_sc_mp4.stat().st_size),
            "elapsed_s": 0.01 if sp["cache_hit"] else job_elapsed_map.get(f"media:{sc.scene_id}", 0.0),
            "cache_hit": sp["cache_hit"],
            "cache_key": sp["cache_key"],
            "start_frame": sc.start_frame,
            "end_frame_exclusive": sc.end_frame_exclusive,
            "used_matte_alpha": sp["used_matte_alpha"],
            "used_hyperframes_graphics": sp["used_hyperframes_graphics"],
            "stage_mode": sp["props"]["stageMode"],
        }
        sc_ab: dict[str, Any] = {}
        for chk in ablation_plan.get(sc.scene_id, []):
            diff_res = verify_ablation_still_diff(chk["full_png"], chk["ex_png"])
            diff_res["sample_frame"] = chk["frame"]
            diff_res["window"] = chk["window"]
            sc_ab[chk["role"]] = diff_res
        ablation_results[sc.scene_id] = sc_ab

    optimization_stats = {
        "remotion_code_hash": rem_code_hash,
        "remotion_bundle_reused": bool(batch_summary.get("bundle_reused", False)),
        "remotion_bundle_s": float(batch_summary.get("bundle_s", 0.0)),
        "remotion_browser_open_s": float(batch_summary.get("browser_open_s", 0.0)),
        "remotion_browser_launches": int(batch_summary.get("browser_launches", 0)),
        "remotion_select_composition_calls": int(batch_summary.get("select_composition_calls", 0)),
        "remotion_scene_cache_hits": rem_scene_cache_hits,
        "remotion_scene_total": len(storyboard.scenes),
        "hf_overlay_cache_hits": hf_cache_hits,
        "hf_overlay_total": len(hf_results),
        "ablation_window_mode": "active_window_entry_peak_exit",
        "batch_worker_peak_tree_rss_mb": float(batch_summary.get("peak_tree_rss_mb", 0.0)),
        "audio_render_marker_barrier": "REALIZED_POST_RENDER",
    }
    tracker.end_stage("remotion_scenes", temp_dir=temp_dir)

    # 4. POST-RENDER DEPENDENCY BARRIER (R5.R2 Fix D12!): Realize animation markers -> Semantic Sound Director -> Exact Mix!
    tracker.start_stage("sound_design_mix")
    anim_markers = extract_realized_animation_markers(storyboard, story_graph, compiled_scene_props=compiled_props_by_scene)
    sonic_opps = extract_sonic_opportunities(
        story_graph=story_graph,
        transcript=transcript_obj,
        observed_actions=roi_tracks,
        storyboard=storyboard,
    )
    sonic_score = plan_sonic_score(
        storyboard=storyboard,
        opportunities=sonic_opps,
        story_graph=story_graph,
        markers=anim_markers,
        niche_style=storyboard.niche,
        seed=config.seed,
        sr=sr_target,
    )
    audio_tl, cue_align_rows, used_sfx_info = align_sound_to_marker(
        sonic_score,
        fps_num=media.fps.num,
        fps_den=media.fps.den,
        sample_rate=sr_target,
        total_frames=media.duration_frames,
        source_hash=media.input_hash,
        priming_offset_s=priming_s,
        audio_mode=eff_audio_mode,
        sfx_enabled=eff_sfx,
        music_enabled=eff_music,
        seed=config.seed,
    )
    atomic_write_json(ev_dir / "animation_markers.json", {"phase": "REALIZED_POST_RENDER", "markers": [m.to_dict() for m in anim_markers]})
    atomic_write_json(ev_dir / "animation_markers_realized.json", {"phase": "REALIZED_POST_RENDER", "markers": [m.to_dict() for m in anim_markers]})
    atomic_write_json(ev_dir / "sound_rationale.json", sonic_score.to_dict())
    atomic_write_json(ev_dir / "cue_decisions.json", {
        "cue_ledger_summary": used_sfx_info.get("cue_ledger_summary", {}),
        "decisions": used_sfx_info.get("cue_decisions", []),
    })
    align_csv = ["event_id,cue_id,scene_id,marker_id,marker_type,sound_role,family,variant,align_anchor,provenance,expected_frame,expected_sample,placed_peak_sample,delta_samples,delta_frames"]
    for ar in cue_align_rows:
        align_csv.append(
            f"{ar['event_id']},{ar['cue_id']},{ar['scene_id']},{ar['marker_id']},{ar['marker_type']},"
            f"{ar['sound_role']},{ar['family']},{ar['variant']},{ar.get('align_anchor','transient')},{ar['provenance']},"
            f"{ar['expected_frame']},{ar['expected_sample']},{ar['placed_peak_sample']},{ar['delta_samples']},{ar['delta_frames']}"
        )
    atomic_write_text(ev_dir / "cue_alignment.csv", "\n".join(align_csv) + "\n")
    sfx_manifest = build_sfx_asset_manifest(sr=audio_tl.sample_rate, seed=config.seed)
    atomic_write_json(run_dir / "audio_timeline.json", audio_tl.to_dict())
    atomic_write_json(run_dir / "sfx_asset_manifest.json", sfx_manifest)

    mix_info: dict[str, Any] | None = None
    mixed_wav_p: Path | None = None
    if eff_audio_mode == "mix" and eff_sfx and audio_tl.events:
        mix_info = mix_preserving_master(
            media=media,
            timeline=audio_tl,
            transcript=transcript_obj,
            run_dir=run_dir,
            seed=config.seed,
            ffmpeg_bin=ffmpeg_bin,
            ffprobe_bin=ffprobe_bin,
        )
        mixed_wav_p = Path(mix_info["mixed_reference_path"])
    tracker.end_stage("sound_design_mix", temp_dir=temp_dir)

    # 5. Single-pass canonical 0..N-1 assembly + single default AAC mixed audio mux
    partial_out = run_dir / f"{Path(output_filename).stem}.partial.mp4"
    tracker.start_stage("canonical_assembly")
    asm_stats = _assemble_full_timeline_and_mux(
        media=media,
        storyboard=storyboard,
        rendered_scenes=rendered_scenes,
        mixed_wav_path=mixed_wav_p,
        workdir=temp_dir,
        out_partial_path=partial_out,
        budget=budget,
        config=config,
        ffmpeg_bin=ffmpeg_bin,
    )
    tracker.end_stage("canonical_assembly", temp_dir=temp_dir)

    # 6. Technical Validation + Audio QA + Reuse Gate + A5 Creative Quality Proxy BEFORE atomic promotion
    tracker.start_stage("validation_and_qa")
    audio_checks, audio_qa = (
        validate_final_audio_mix(media, partial_out, audio_tl, mix_info, run_dir, ffmpeg_bin=ffmpeg_bin, ffprobe_bin=ffprobe_bin)
        if mix_info is not None
        else ([], {"status": "PRESERVE_EXACT"})
    )

    source_sha_after = _sha256_file(Path(media.source_path))
    source_immutable = (source_sha_after == media.input_hash)

    # Probe assembled MP4 geometry & exact frame count
    ff_probe = json.loads(
        subprocess.run(
            [ffprobe_bin, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(partial_out)],
            capture_output=True, text=True, timeout=20, shell=False,
        ).stdout or "{}"
    )
    v_streams = [s for s in ff_probe.get("streams", []) if s.get("codec_type") == "video"]
    out_w = int(v_streams[0].get("width", 0)) if v_streams else 0
    out_h = int(v_streams[0].get("height", 0)) if v_streams else 0
    out_frames = int(v_streams[0]["nb_frames"]) if (v_streams and str(v_streams[0].get("nb_frames", "")).isdigit()) else asm_stats["frames_written"]

    reuse_passed, reuse_report = validate_canonical_reuse_gate(
        storyboard,
        actual_rendered_engines=actual_engines,
        actual_rendered_primitives=actual_primitives,
        min_distinct_total=4 if require_dual_engine_reuse_gate else 1,
        min_per_engine=2 if require_dual_engine_reuse_gate else 0,
        registry=reg,
    )
    atomic_write_json(run_dir / "canonical_primitive_usage.json", reuse_report)

    creative_qa = evaluate_creative_quality(
        media=media,
        final_mp4=partial_out,
        storyboard=storyboard,
        story_graph=story_graph,
        ablation_results=ablation_results,
        audio_qa=audio_qa if mix_info is not None else None,
        reuse_report=reuse_report,
        run_dir=run_dir,
    )

    # Independent objective decoded-pixel critic (boundary continuity strips, face washout guard, honest VLM status)
    pixel_critic = run_objective_rendered_critic(
        source_mp4=Path(media.source_path),
        final_mp4=partial_out,
        storyboard=storyboard,
        continuous_timeline=continuous_tl,
        roi_tracks=roi_tracks,
        out_dir=ev_dir,
    )

    ablation_total = int(creative_qa.get("ablation_layers_total", 0))
    ablation_verified = int(creative_qa.get("ablation_layers_verified", 0))
    ablation_all_ok = (ablation_total == 0) or (ablation_verified == ablation_total)

    tech_ok = bool(
        source_immutable
        and out_w == media.display_width
        and out_h == media.display_height
        and out_frames == media.duration_frames
        and creative_qa["black_frames"] == 0
        and ablation_all_ok
        and continuous_tl["continuous_timeline_valid"]
        and pixel_critic["abrupt_cutaway_jumps"] == 0
        and (not require_dual_engine_reuse_gate or reuse_passed)
        and (mix_info is None or audio_qa.get("status") == "PASS")
    )

    final_path = run_dir / output_filename
    preview_info: dict[str, Any] | None = None
    if tech_ok:
        os.replace(partial_out, final_path)
        if render_preview:
            preview_info = _downscale_preview_from_final(final_path, run_dir / "preview_9x16.mp4", ffmpeg_bin)
        overall_status = "TECHNICAL_PASS"
    else:
        quarantine_p = run_dir / f"QUARANTINED_{partial_out.name}"
        os.replace(partial_out, quarantine_p)
        final_path = quarantine_p
        overall_status = "TECHNICAL_FAIL"

    tracker.end_stage("validation_and_qa", temp_dir=temp_dir)

    actual_engine_usage = {
        "edition": "ml-01.r5",
        "mode": storyboard.mode,
        "engines_used": sorted(actual_engines),
        "dual_engine_active": ("remotion" in actual_engines and "hyperframes" in actual_engines),
        "reuse_gate": reuse_report,
        "scenes": [
            {
                "scene_id": sc.scene_id,
                "role": sc.role,
                "stage_mode": rendered_scenes[sc.scene_id]["stage_mode"],
                "frame_range": [sc.start_frame, sc.end_frame_exclusive],
                "remotion_artifact_sha256": rendered_scenes[sc.scene_id]["artifact_sha256"],
                "hf_primitive_id": hf_results[sc.scene_id]["primitive_id"] if sc.scene_id in hf_results else None,
                "hf_artifact_sha256": hf_results[sc.scene_id]["artifact_sha256"] if sc.scene_id in hf_results else None,
                "used_matte_alpha": rendered_scenes[sc.scene_id]["used_matte_alpha"],
                "ablation": ablation_results.get(sc.scene_id, {}),
            }
            for sc in storyboard.scenes
            if sc.scene_id in rendered_scenes
        ],
    }
    atomic_write_json(run_dir / "actual_engine_usage.json", actual_engine_usage)
    atomic_write_json(ev_dir / "ablation_results.json", ablation_results)

    validation_doc = {
        "overall_status": overall_status,
        "aesthetic_status": "PENDING_HUMAN_REVIEW",
        "foley_library_status": inspect_foley_library()["status"],
        "source_immutable": source_immutable,
        "output_dimensions": f"{out_w}x{out_h}",
        "output_frames": out_frames,
        "expected_frames": media.duration_frames,
        "reuse_gate": reuse_report,
        "audio_qa": audio_qa,
        "sonic_score_summary": {
            "score_hash": sonic_score.score_hash,
            "cues_planned": len(sonic_score.cues),
            "cue_ledger_summary": used_sfx_info.get("cue_ledger_summary", {}),
            "quiet_spans": len(sonic_score.quiet_spans),
            "rejected_decisions": len(sonic_score.rejected_decisions),
            "recorded_cues_used": used_sfx_info.get("recorded_cues_used", 0),
            "procedural_cues_used": used_sfx_info.get("procedural_cues_used", 0),
            "grounding_errors": sonic_score.validation_errors,
        },
        "continuous_timeline_summary": {
            "continuity_hash": continuous_tl["continuity_hash"],
            "continuous_timeline_valid": continuous_tl["continuous_timeline_valid"],
            "boundary_count": len(continuous_tl["boundaries"]),
            "numeric_state_integer_lock": continuous_tl["numeric_state_trace"]["integer_dollar_lock"],
            "single_numeric_owner_verified": continuous_tl["numeric_state_trace"]["single_numeric_owner_verified"],
            "validation_errors": continuous_tl["validation_errors"],
        },
        "objective_pixel_critic": {
            "vlm_status": pixel_critic["vlm_status"],
            "objective_pixel_critic_status": pixel_critic["objective_pixel_critic_status"],
            "abrupt_cutaway_jumps": pixel_critic["abrupt_cutaway_jumps"],
            "max_face_washout_luma_delta": pixel_critic["max_face_washout_luma_delta"],
            "findings_count": len(pixel_critic["findings"]),
        },
        "product_asset_status": prod_match,
        "layout_qa": {
            "zero_eye_mouth_collision": layout_plan["zero_eye_mouth_collision"],
            "max_eye_mouth_overlap": layout_plan["max_eye_mouth_overlap_any_frame"],
            "phone_legibility_passed": legibility_report["passed"],
            "consumer_jargon_leaks": legibility_report["consumer_jargon_leaks"],
            "layout_sweep_errors": layout_sweep_errors,
        },
        "optimization_stats": optimization_stats,
        "creative_quality_proxy": creative_qa,
        "assembly_stats": asm_stats,
        "preview_9x16": preview_info,
    }
    atomic_write_json(run_dir / "validation.json", validation_doc)
    res_dict = tracker.to_dict()
    res_dict["optimization_stats"] = optimization_stats
    atomic_write_json(run_dir / "resource_stats.json", res_dict)

    # Write clean human-readable run README.md
    readme_lines = [
        f"# MotionLab Studio ML-01.R5 Run — `{run_id}`",
        "",
        f"- **Plan Mode:** `{storyboard.mode}`",
        f"- **Technical Status:** `{overall_status}`",
        f"- **Human Aesthetic Approval (G8):** `PENDING_HUMAN_REVIEW` (technical pass != owner approval)",
        f"- **Foley Library Status (G4):** `{inspect_foley_library()['status']}`",
        f"- **Source Input:** `{Path(media.source_path).name}` (SHA-256 `{media.input_hash}`, immutable=`{source_immutable}`)",
        f"- **Output Video:** `{final_path.name}` (`{out_w}x{out_h}`, `{out_frames}` frames @ `{fps_str}` fps, SHA-256 `{_sha256_file(final_path)}`)",
        f"- **Preview 9:16:** `{preview_info['path'] if preview_info else 'none'}`",
        f"- **Engines Used:** `{', '.join(sorted(actual_engines))}` | **Verified Imported Mechanisms Used:** `{reuse_report['distinct_verified_total']}` (Remotion={reuse_report['remotion_primitives']}, HyperFrames={reuse_report['hyperframes_primitives']})",
        f"- **Audio:** voice untouched unity 1.0, `original_audio_master.m4a` bit-exact (`master_exact={audio_qa.get('master_exact')}`), SFX cues=`{audio_qa.get('events_total', 0)}`, audible proxy=`{audio_qa.get('events_proxy_audible', 0)}/{audio_qa.get('events_total', 0)}`, LUFS=`{audio_qa.get('final_decoded_lufs_i')}`, TP=`{audio_qa.get('final_decoded_true_peak_dbtp')} dBTP`",
        "",
        "## Scene Breakdown",
    ]
    for sc in storyboard.scenes:
        prims = [ly.primitive_id for ly in sc.layers]
        readme_lines.append(f"- `{sc.scene_id}` (`f{sc.start_frame}..{sc.end_frame_exclusive}`, `{sc.stage_mode}`): primitives=`{prims}`, transition=`{sc.transition_in.kind}` — {sc.viewer_takeaway}")
    atomic_write_text(run_dir / "README.md", "\n".join(readme_lines) + "\n")

    # Clean ONLY owned temp/ directory after all artifacts & evidence are verified
    budget.cleanup_owned_temp(run_dir, keep_on_fail=False)

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "status": overall_status,
        "aesthetic_status": "PENDING_HUMAN_REVIEW",
        "foley_library_status": inspect_foley_library()["status"],
        "mode": storyboard.mode,
        "storyboard_hash": storyboard.storyboard_hash,
        "story_graph_hash": story_graph.story_graph_hash,
        "scenes": len(storyboard.scenes),
        "engines_used": sorted(actual_engines),
        "reuse_gate": reuse_report,
        "final_path": str(final_path) if tech_ok else None,
        "final_sha256": _sha256_file(final_path) if tech_ok else None,
        "preview_9x16_path": str(run_dir / preview_info["path"]) if preview_info else None,
        "contact_sheet_path": str(run_dir / "contact_sheet.jpg"),
        "audio_qa": audio_qa,
        "optimization_stats": optimization_stats,
        "layout_qa": validation_doc["layout_qa"],
        "sonic_score_summary": validation_doc["sonic_score_summary"],
        "creative_quality_proxy": creative_qa,
        "resource_stats": res_dict,
    }


def execute_generative_r5_pipeline(
    *,
    media: MediaInfo,
    analysis: AnalysisResult,
    base_plan: EditPlan,
    config: EditConfig,
    doctor_info: dict[str, Any],
    run_id: str,
    run_dir: Path,
    budget: ResourceBudget,
    tracker: ResourceTracker,
    manifest: dict[str, Any],
    log_lines: list[str],
) -> dict[str, Any]:
    """Canonical Automatic Generalization Entry Point (`--composition generative`).

    Reads ONLY raw MP4 + generic local models/registry. ZERO Olivia hardcoding or benchmark fixtures!
    """
    ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"
    engine_caps = doctor_info["engines"]
    asr_info = inspect_asr_model()
    llm_info = inspect_llm_model()
    vision_info = inspect_vision_model()
    reg = load_unified_registry()

    # Fail-closed capability check (NEVER silently downgrade to legacy/R4!)
    missing: list[str] = []
    if not asr_info.get("available"):
        missing.append(f"asr:{asr_info.get('reason')}")
    if not llm_info.get("available"):
        missing.append(f"llm:{llm_info.get('reason')}")
    if not (engine_caps.get("remotion") or {}).get("available"):
        missing.append("remotion_unavailable_or_not_opted_in")
    if not (engine_caps.get("hyperframes") or {}).get("available"):
        missing.append("hyperframes_unavailable")
    if len(reg.verified_primitives()) < 4:
        missing.append(f"insufficient_verified_primitives:{len(reg.verified_primitives())}<4")
    if missing:
        raise RuntimeError(f"ML-01.R5 generative mode BLOCKED — missing required capabilities: {missing}")

    cache_base = Path(config.cache_dir) if config.cache_dir else (PROJECT_ROOT / ".cache" / "motionlab_r5")
    sem_cache = (cache_base / "semantic") if config.cache_enabled else None

    # 1. Bounded Parallel ROI Action Tracking + Local Faster-Whisper Small ASR
    tracker.start_stage("vision_and_asr")
    if budget.analysis_workers >= 2:
        with ThreadPoolExecutor(max_workers=2) as pool:
            f_roi = pool.submit(track_action_rois, media, analysis, config, ffmpeg_bin)
            f_asr = pool.submit(transcribe_local, media, config, sem_cache, ffmpeg_bin)
            roi_tracks = f_roi.result()
            transcript = f_asr.result()
    else:
        roi_tracks = track_action_rois(media, analysis, config, ffmpeg_bin)
        transcript = transcribe_local(media, config, sem_cache, ffmpeg_bin)
    tracker.end_stage("vision_and_asr", temp_dir=run_dir / "temp")

    atomic_write_json(run_dir / "transcript.json", transcript.to_dict())
    atomic_write_text(run_dir / "transcript_readable.txt", format_readable_transcript(transcript))
    atomic_write_json(run_dir / "evidence" / "roi_tracks.json", roi_tracks)

    # 2. Generic StoryGraph + Integer-Cent Verified Ledger
    tracker.start_stage("story_graph")
    story_graph = build_story_graph(transcript, roi_tracks, media)
    atomic_write_json(run_dir / "story_graph.json", story_graph.to_dict())
    atomic_write_json(run_dir / "ledger_claims.json", story_graph.ledger.to_dict())
    tracker.end_stage("story_graph", temp_dir=run_dir / "temp")

    # 3. Generic Local Qwen2.5-3B Generative Director + UnifiedPrimitiveRegistry Search
    tracker.start_stage("generative_director")
    storyboard, dir_audit = direct_generative_storyboard(
        media=media,
        story_graph=story_graph,
        roi_tracks=roi_tracks,
        config=config,
        matte_available=bool(vision_info.get("available")),
        registry=reg,
        cache_dir=sem_cache,
    )
    atomic_write_json(run_dir / "raw_director_response.json", dir_audit)
    atomic_write_json(run_dir / "retrieved_candidates.json", storyboard.retrieved_candidates)
    tracker.end_stage("generative_director", temp_dir=run_dir / "temp")

    # 4. Execute Shared Deterministic Renderer / Mixer / Validator
    result = render_compiled_storyboard_to_mp4(
        media=media,
        storyboard=storyboard,
        story_graph=story_graph,
        roi_tracks=roi_tracks,
        config=config,
        run_id=run_id,
        run_dir=run_dir,
        budget=budget,
        tracker=tracker,
        output_filename="final.mp4",
        render_preview=bool(config.render_preview_9x16),
        require_dual_engine_reuse_gate=(not storyboard.niche in ("tutorial", "gameplay") and len(storyboard.scenes) >= 2),
        transcript_obj=transcript,
    )

    manifest["composition_mode"] = "generative"
    manifest["status"] = result["status"]
    manifest["aesthetic_status"] = result.get("aesthetic_status", "PENDING_HUMAN_REVIEW")
    manifest["storyboard_hash"] = result.get("storyboard_hash")
    manifest["story_graph_hash"] = result.get("story_graph_hash")
    manifest["reuse_gate"] = result.get("reuse_gate")
    manifest["resource_tracker"] = tracker.to_dict()
    atomic_write_json(run_dir / "manifest.json", manifest)
    atomic_write_text(run_dir / "logs" / "pipeline.log", "\n".join(log_lines) + "\n")
    return result
