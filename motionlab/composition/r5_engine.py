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
from motionlab.audio.mixer import build_edited_voice_bed, mix_preserving_master, validate_final_audio_mix
from motionlab.audio.semantic_sound import (
    align_sound_to_marker,
    compile_animation_markers,
    emit_render_trace_markers,
    extract_realized_animation_markers,
    extract_rendered_animation_markers,
    extract_sonic_opportunities,
    observe_pixel_animation_markers,
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
from motionlab.composition.edit_continuity import evaluate_audio_joins, evaluate_cut_continuity
from motionlab.composition.edit_time_map import (
    EditTimeMap,
    build_edited_mezzanine_video,
    build_identity_edit_time_map,
    remap_roi_tracks_to_edited_domain,
    remap_transcript_to_edited_domain,
    validate_edit_time_map,
    verify_word_preservation_on_bed,
)
from motionlab.composition.generative_compiler import (
    compile_scene_hyperframes_html,
    compile_scene_hyperframes_layers,
    compile_scene_remotion_props,
    compile_storyboard_audio_timeline,
    compile_storyboard_graph,
)
from motionlab.composition.generative_schema import (
    StoryboardSpec,
    validate_canonical_reuse_gate,
)
from motionlab.composition.primitive_registry import load_unified_registry
from motionlab.composition.visual_critic import run_objective_rendered_critic, run_rhythm_critic
from motionlab.config import EditConfig, PROJECT_ROOT
from motionlab.renderers.base import verify_webm_alpha_pixels
from motionlab.renderers.hyperframes import HyperFramesRenderer
from motionlab.renderers.remotion import RemotionRenderer
from motionlab.resource import ResourceBudget, ResourceTracker, RuntimeResourceWatchdog
from motionlab.schemas import AnalysisResult, EditPlan, MediaInfo, atomic_write_json, atomic_write_text
from motionlab.semantic.asr import format_readable_transcript, inspect_asr_model, transcribe_local
from motionlab.semantic.generative_director import direct_generative_storyboard
from motionlab.semantic.local_model import inspect_llm_model
from motionlab.semantic.narrative_brain import build_evidence_index, build_narrative_contract
from motionlab.semantic.pacing_director import choose_pacing_plan
from motionlab.semantic.speech_perception import run_speech_and_action_perception
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
    original_media: MediaInfo | None = None,
    edit_time_map: EditTimeMap | None = None,
    pacing_plan: dict[str, Any] | None = None,
    edited_voice_bed_wav: Path | None = None,
    word_preservation: dict[str, Any] | None = None,
    cut_continuity: dict[str, Any] | None = None,
    join_audio_qa: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Shared deterministic R5 renderer/mixer/validator used by BOTH canonical AUTO runs and separate QA benchmark."""
    orig_media = original_media or media
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

    # 2. Render ALL Imported HyperFrames Apache-2.0 VP9 Alpha Layers per Scene (Fixes D04 — zero dropped layers!)
    tracker.start_stage("hyperframes_graphics")
    hf_results: dict[str, list[dict[str, Any]]] = {}
    hf_cache_hits = 0
    hf_total_layers = 0
    for sc in storyboard.scenes:
        compiled_overlays = compile_scene_hyperframes_layers(sc, storyboard, story_graph)
        if not compiled_overlays:
            continue
        sc_hf_list: list[dict[str, Any]] = []
        for ov in compiled_overlays:
            hf_total_layers += 1
            safe_lid = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in ov.layer_id)
            hf_webm = temp_dir / f"hf_{sc.scene_id}_{safe_lid}_{ov.primitive_id}.webm"
            r_hf = hf.render_html_slot(
                slot_dir=temp_dir / f"hf_slot_{sc.scene_id}_{safe_lid}",
                html=ov.html,
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
                raise RuntimeError(f"HyperFrames layer {ov.layer_id} ({ov.primitive_id}) for {sc.scene_id} rendered empty alpha!")
            sc_hf_list.append({
                **r_hf,
                "layer_id": ov.layer_id,
                "primitive_id": ov.primitive_id,
                "layer_role": ov.layer_role,
                "z_order": ov.z_order,
                "opacity": ov.opacity,
                "render_mode": ov.render_mode,
                "html_sha256": ov.html_sha256,
                "alpha_stats": alpha_stats,
            })
        hf_results[sc.scene_id] = sc_hf_list
    tracker.end_stage("hyperframes_graphics", temp_dir=temp_dir)

    # 3. R5.R3 Multi-Layer Remotion MIT Single-Bundle Persistent-Browser Batch + Real Product Asset Binding + Per-Layer Ablations!
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
    product_asset_bound_scenes: list[str] = []

    for sc in storyboard.scenes:
        slot_dir = temp_dir / f"rem_{sc.scene_id}"
        pub_dir = slot_dir / "public"
        pub_dir.mkdir(parents=True, exist_ok=True)

        tr_out_kind = str(sc.parameters.get("transition_out", "none"))
        src_rel: str | None = None
        src_sha = ""
        if sc.stage_mode != "full_graphic_cutaway" or sc.transition_in.kind != "none" or tr_out_kind != "none":
            src_slice = pub_dir / "scene_source.mp4"
            rem._slice_source_mezzanine(Path(media.source_path), sc.start_frame, sc.end_frame_exclusive, fps_str, src_slice)
            src_rel = f"{sc.scene_id}/{src_slice.name}"
            src_sha = _sha256_file(src_slice)
            public_mounts.append({"src": str(src_slice), "rel": src_rel})

        fg_rel: str | None = None
        fg_sha = ""
        matte_p = matte_by_span.get((sc.start_frame, sc.end_frame_exclusive))
        if sc.requires_matte and matte_p is not None and matte_p.is_file():
            # Content-addressed foreground alpha WebM cache (fixes R5.R3 warm matte cache miss -> 9/9 warm hits!)
            matte_sha = _sha256_file(matte_p)
            fg_cache_p = vision_cache / f"fg_alpha_{media.input_hash[:12]}_{sc.start_frame}_{sc.end_frame_exclusive}_{matte_sha[:12]}.webm"
            fg_webm = pub_dir / "fg_alpha.webm"
            if config.cache_enabled and fg_cache_p.is_file() and fg_cache_p.stat().st_size > 1024:
                shutil.copy2(fg_cache_p, fg_webm)
            else:
                build_foreground_alpha_webm(
                    Path(media.source_path), matte_p, sc.start_frame, sc.end_frame_exclusive, fps_str, fg_webm, ffmpeg_bin=ffmpeg_bin
                )
                if config.cache_enabled:
                    fg_cache_p.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(fg_webm, fg_cache_p)
            fg_rel = f"{sc.scene_id}/{fg_webm.name}"
            fg_sha = _sha256_file(fg_webm)
            public_mounts.append({"src": str(fg_webm), "rel": fg_rel})

        # Mount ALL ordered HyperFrames layers for this scene (Fixes D04!)
        hf_rel: str | None = None
        hf_role = "foreground_graphics"
        hf_layer_mounts: list[dict[str, Any]] = []
        hf_shas: list[str] = []
        if sc.scene_id in hf_results:
            actual_engines.add("hyperframes")
            for h_idx, hitem in enumerate(hf_results[sc.scene_id]):
                hf_src_p = Path(hitem["artifact_path"])
                safe_lid = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in hitem["layer_id"])
                rel_p = f"{sc.scene_id}/hf_{h_idx}_{safe_lid}.webm"
                public_mounts.append({"src": str(hf_src_p), "rel": rel_p})
                hf_layer_mounts.append({
                    "layerId": hitem["layer_id"],
                    "primitiveId": hitem["primitive_id"],
                    "file": rel_p,
                    "layerRole": hitem["layer_role"],
                    "zOrder": int(hitem["z_order"]),
                    "opacity": float(hitem.get("opacity", 1.0)),
                })
                hf_shas.append(str(hitem["artifact_sha256"]))
                actual_primitives.append(hitem["primitive_id"])
            hf_rel = hf_layer_mounts[0]["file"]
            hf_role = hf_layer_mounts[0]["layerRole"]

        # D06: Mount REAL owner-authorized product asset texture on product scenes when verified in catalog!
        prod_asset_rel: str | None = None
        prod_asset_meta: dict[str, Any] | None = None
        prod_asset_sha = ""
        if sc.role in ("product_intro", "product_benefit") and prod_match.get("matched") and prod_match.get("assets"):
            pa = prod_match["assets"][0]
            pa_src = Path(str(pa.get("resolved_path", "")))
            if pa_src.is_file():
                prod_asset_rel = f"{sc.scene_id}/product_asset{pa_src.suffix.lower()}"
                prod_asset_sha = str(pa.get("sha256", ""))
                prod_asset_meta = {
                    "asset_id": pa.get("asset_id"),
                    "product_name": pa.get("product_name"),
                    "asset_type": pa.get("asset_type"),
                    "usage_constraints": pa.get("usage_constraints", ""),
                    "sha256": prod_asset_sha[:16],
                }
                public_mounts.append({"src": str(pa_src), "rel": prod_asset_rel})
                product_asset_bound_scenes.append(sc.scene_id)

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
            hf_layers=hf_layer_mounts,
            product_asset_file=prod_asset_rel,
            product_asset_meta=prod_asset_meta,
            task_anchor=task_anchor,
            face_anchor=face_anchor,
            continuity_tracks=continuous_tl.get("tracks", []),
        )
        compiled_props_by_scene[sc.scene_id] = props
        atomic_write_json(slot_dir / "props.json", props)

        # Content-addressed per-scene cache key (invalidates on code, candidate, props, continuity, source, matte, ALL HF layers, product asset, seed!)
        scene_cache_key = hashlib.sha256(
            json.dumps(
                {
                    "code_hash": rem_code_hash,
                    "schema_version": storyboard.schema_version,
                    "edit_map_hash": edit_time_map.map_hash if edit_time_map is not None else "identity",
                    "chosen_candidate_id": sc.chosen_candidate_id,
                    "continuity_hash": continuous_tl["continuity_hash"],
                    "props": props,
                    "src_sha": src_sha,
                    "fg_sha": fg_sha,
                    "hf_shas": hf_shas,
                    "prod_asset_sha": prod_asset_sha,
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

        # Active-Window Ablation Plan (entry / peak / exit-bridge + PER-LAYER HF ablation when >=2 HF layers!)
        dur_f = int(props["durationInFrames"])
        mid_rel = max(6, min(dur_f - 8, dur_f // 2))
        ab_dir = ev_dir / "ablations" / sc.scene_id
        ab_dir.mkdir(parents=True, exist_ok=True)

        sc_ab_checks: list[dict[str, Any]] = []
        if len(hf_layer_mounts) >= 2:
            for hlm in hf_layer_mounts:
                # Sample frame where paper tokens are in-flight (~f10-14) for background tokens, midpoint for foreground type!
                is_bg_tok = hlm["layerRole"] == "background_graphics"
                sample_f = max(6, min(dur_f - 8, 12 if is_bg_tok else mid_rel))
                sc_ab_checks.append({
                    "role": f"hf_layer:{hlm['layerId']}",
                    "exclude_role": None,
                    "exclude_id": hlm["layerId"],
                    "frame": sample_f,
                    "window": "peak_multilayer",
                })
        elif hf_rel is not None:
            sc_ab_checks.append({"role": hf_role, "exclude_role": hf_role, "exclude_id": None, "frame": mid_rel, "window": "peak"})

        if prod_asset_rel is not None:
            prod_f = max(8, min(dur_f - 6, int(round(dur_f * 0.68))))
            sc_ab_checks.append({"role": "product_asset", "exclude_role": "product_asset", "exclude_id": None, "frame": prod_f, "window": "product_after_flip"})

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
                sc_ab_checks.append({"role": "foreground_subject_matte", "exclude_role": "foreground_subject_matte", "exclude_id": None, "frame": mid_rel, "window": "peak"})
            else:
                sc_ab_checks.append({"role": "remotion_primitive", "exclude_role": "remotion_primitive", "exclude_id": None, "frame": rem_f, "window": "peak"})
        if sc.transition_in.kind != "none" and src_rel is not None and sc.transition_in.duration_frames >= 4:
            tr_f = max(2, min(dur_f - 2, sc.transition_in.duration_frames // 2))
            sc_ab_checks.append({"role": "transition_in", "exclude_role": "transition_in", "exclude_id": None, "frame": tr_f, "window": "entry"})
        elif tr_out_kind != "none" and src_rel is not None:
            tr_out_f = int(props.get("transitionOutFrames", 12))
            out_mid = max(2, min(dur_f - 2, dur_f - max(2, tr_out_f // 2)))
            sc_ab_checks.append({"role": "transition_out", "exclude_role": "transition_out", "exclude_id": None, "frame": out_mid, "window": "exit"})
        elif fg_rel is not None and not any(c["role"] == "foreground_subject_matte" for c in sc_ab_checks):
            sc_ab_checks.append({"role": "foreground_subject_matte", "exclude_role": "foreground_subject_matte", "exclude_id": None, "frame": mid_rel, "window": "peak"})

        seen_full_frames: dict[int, Path] = {}
        for chk in sc_ab_checks[:4]:
            sf_rel = int(chk["frame"])
            if sf_rel not in seen_full_frames:
                full_p = ab_dir / f"full_f{sf_rel:03d}.png"
                seen_full_frames[sf_rel] = full_p
                batch_jobs.append({
                    "kind": "still",
                    "id": f"still:{sc.scene_id}:full:f{sf_rel}",
                    "compositionId": "MotionLabGenerativeScene",
                    "props": {**props, "excludeLayerRole": None, "excludeLayerId": None},
                    "frame": sf_rel,
                    "outputLocation": str(full_p),
                })
            safe_r = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in chk["role"])
            ex_p = ab_dir / f"without_{safe_r}_f{sf_rel:03d}.png"
            batch_jobs.append({
                "kind": "still",
                "id": f"still:{sc.scene_id}:{safe_r}:f{sf_rel}",
                "compositionId": "MotionLabGenerativeScene",
                "props": {**props, "excludeLayerRole": chk.get("exclude_role"), "excludeLayerId": chk.get("exclude_id")},
                "frame": sf_rel,
                "outputLocation": str(ex_p),
            })
            chk["full_png"] = full_p
            chk["ex_png"] = ex_p

        ablation_plan[sc.scene_id] = sc_ab_checks[:4]
        scene_prep[sc.scene_id] = {
            "props": props,
            "out_mp4": out_sc_mp4,
            "cached_mp4": cached_sc_mp4,
            "cache_hit": cache_hit,
            "cache_key": scene_cache_key,
            "used_matte_alpha": fg_rel is not None,
            "used_hyperframes_graphics": bool(hf_layer_mounts),
            "hf_layers_count": len(hf_layer_mounts),
            "product_asset_bound": prod_asset_rel is not None,
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
        "hf_overlay_total": hf_total_layers,
        "ablation_window_mode": "active_window_entry_peak_exit_per_layer",
        "batch_worker_peak_tree_rss_mb": float(batch_summary.get("peak_tree_rss_mb", 0.0)),
        "audio_render_marker_barrier": "OBSERVED_PIXEL_POST_RENDER_FROZEN",
    }
    tracker.end_stage("remotion_scenes", temp_dir=temp_dir)

    # Upgrade product asset status ONLY when rendered pixels actually prove real product asset binding (D06!)
    if product_asset_bound_scenes:
        prod_ab_ok = all(
            (ablation_results.get(sid, {}).get("product_asset") or {}).get("verified", False)
            for sid in product_asset_bound_scenes
        )
        prod_match["bound_scenes"] = product_asset_bound_scenes
        prod_match["pixel_ablation_verified"] = prod_ab_ok
        prod_match["product_demo_status"] = "AUTHORIZED_DEMO_RENDERED_VERIFIED" if prod_ab_ok else "AUTHORIZED_ASSET_BINDING_FAILED"
        atomic_write_json(ev_dir / "product_asset_status.json", prod_match)

    # 3b. Bounded Creative Repair Loop (A9.3): Assemble visual preview -> run streaming objective critic ->
    # repair actionable HIGH findings (max 2 rounds, rerendering ONLY affected scenes!) -> freeze visual graph BEFORE audio!
    revision_history: dict[str, Any] = {
        "schema_version": "3.0.0",
        "max_rounds": 2,
        "rounds_executed": 0,
        "initial_severe_findings": 0,
        "final_severe_findings": 0,
        "critic_mode": "OBJECTIVE_STREAMING_ONLY",
        "revisions": [],
        "status": "CLEAN_PASS_NO_REPAIR_NEEDED",
    }
    draft_visual_mp4 = temp_dir / "r5_visual_repair_check.mp4"
    _assemble_full_timeline_and_mux(
        media=media,
        storyboard=storyboard,
        rendered_scenes=rendered_scenes,
        mixed_wav_path=None,
        workdir=temp_dir,
        out_partial_path=draft_visual_mp4,
        budget=budget,
        config=config,
        ffmpeg_bin=ffmpeg_bin,
    )
    pre_mix_critic = run_objective_rendered_critic(
        source_mp4=Path(media.source_path),
        final_mp4=draft_visual_mp4,
        storyboard=storyboard,
        continuous_timeline=continuous_tl,
        roi_tracks=roi_tracks,
        out_dir=ev_dir,
    )
    revision_history["initial_severe_findings"] = int(pre_mix_critic.get("severe_findings_count", 0))
    revision_history["final_severe_findings"] = int(pre_mix_critic.get("severe_findings_count", 0))
    if pre_mix_critic.get("vlm_status") == "VLM_READY":
        revision_history["critic_mode"] = "OBJECTIVE_PLUS_LOCAL_VLM"

    sc_by_id = {sc.scene_id: sc for sc in storyboard.scenes}
    for rep_round in range(1, 3):
        severe_list = [f for f in pre_mix_critic.get("findings", []) if f.get("severity") == "HIGH"]
        if not severe_list:
            break
        repair_jobs: list[dict[str, Any]] = []
        changed_scenes: list[dict[str, Any]] = []
        backup_mp4s: dict[str, Path] = {}
        for fnd in severe_list:
            aff_id = str(fnd.get("affected_scene_id", ""))
            sc_aff = sc_by_id.get(aff_id)
            if sc_aff is None or aff_id not in compiled_props_by_scene:
                continue
            props_new = json.loads(json.dumps(compiled_props_by_scene[aff_id]))
            cat = str(fnd.get("category", ""))
            patch_desc = ""
            if cat in ("transition_continuity", "boundary_window_spike"):
                if props_new.get("stageMode") == "full_graphic_cutaway":
                    props_new["transitionIn"] = "sheet_sweep"
                    props_new["transitionInFrames"] = 14
                    props_new["transitionOut"] = "paper_fold_out"
                    props_new["transitionOutFrames"] = 14
                    patch_desc = "extended_sheet_sweep_and_paper_fold_handles_to_14f"
                else:
                    props_new["entryFrames"] = max(10, int(props_new.get("entryFrames", 8)))
                    props_new["exitFrames"] = max(10, int(props_new.get("exitFrames", 8)))
                    patch_desc = "smoothed_live_scene_entry_exit_envelope"
            elif cat == "face_contrast_washout":
                for hlm in props_new.get("hfLayers", []):
                    if hlm.get("primitiveId") == "hf_organic_light_leak":
                        hlm["opacity"] = min(0.12, float(hlm.get("opacity", 0.2)))
                patch_desc = "clamped_light_leak_opacity_to_0_12"
            elif cat == "internal_flash_spike":
                props_new["entryFrames"] = max(10, int(props_new.get("entryFrames", 8)))
                props_new["exitFrames"] = max(10, int(props_new.get("exitFrames", 8)))
                patch_desc = "damped_internal_flash_envelope"
            if not patch_desc or any(cs["scene_id"] == aff_id for cs in changed_scenes):
                continue
            out_p = Path(rendered_scenes[aff_id]["artifact_path"])
            bak_p = out_p.with_suffix(f".bak_r{rep_round}.mp4")
            shutil.copy2(out_p, bak_p)
            backup_mp4s[aff_id] = bak_p
            repair_jobs.append({
                "kind": "media",
                "id": f"repair_r{rep_round}:{aff_id}",
                "compositionId": "MotionLabGenerativeScene",
                "props": props_new,
                "outputLocation": str(out_p),
                "codec": "h264",
                "pixelFormat": "yuv420p",
                "crf": 16,
                "concurrency": 1,
            })
            changed_scenes.append({
                "scene_id": aff_id,
                "cause": cat,
                "patch": patch_desc,
                "before_sha256": rendered_scenes[aff_id]["artifact_sha256"],
                "new_props": props_new,
            })

        if not repair_jobs:
            break
        revision_history["rounds_executed"] = rep_round
        rem.render_batch_jobs(
            jobs=repair_jobs,
            public_mounts=public_mounts,
            work_dir=temp_dir / f"rem_repair_r{rep_round}",
            bundle_cache_root=rem_bundle_cache_dir,
            timeout_s=max(240.0, float(config.timeout_s)),
        )
        _assemble_full_timeline_and_mux(
            media=media,
            storyboard=storyboard,
            rendered_scenes=rendered_scenes,
            mixed_wav_path=None,
            workdir=temp_dir,
            out_partial_path=draft_visual_mp4,
            budget=budget,
            config=config,
            ffmpeg_bin=ffmpeg_bin,
        )
        post_critic = run_objective_rendered_critic(
            source_mp4=Path(media.source_path),
            final_mp4=draft_visual_mp4,
            storyboard=storyboard,
            continuous_timeline=continuous_tl,
            roi_tracks=roi_tracks,
            out_dir=ev_dir,
        )
        improved = int(post_critic.get("severe_findings_count", 0)) < int(pre_mix_critic.get("severe_findings_count", 0))
        if improved:
            for cs in changed_scenes:
                sid = cs["scene_id"]
                compiled_props_by_scene[sid] = cs["new_props"]
                rendered_scenes[sid]["artifact_sha256"] = _sha256_file(Path(rendered_scenes[sid]["artifact_path"]))
                cs["after_sha256"] = rendered_scenes[sid]["artifact_sha256"]
                backup_mp4s[sid].unlink(missing_ok=True)
            pre_mix_critic = post_critic
            revision_history["final_severe_findings"] = int(post_critic.get("severe_findings_count", 0))
            revision_history["revisions"].append({
                "round": rep_round,
                "accepted": True,
                "changes": [{k: v for k, v in cs.items() if k != "new_props"} for cs in changed_scenes],
                "severe_after": int(post_critic.get("severe_findings_count", 0)),
            })
            revision_history["status"] = "REPAIRED_IMPROVED"
        else:
            # Rollback cleanly if repair did not reduce severe findings!
            for sid, bak_p in backup_mp4s.items():
                shutil.move(str(bak_p), rendered_scenes[sid]["artifact_path"])
            revision_history["revisions"].append({
                "round": rep_round,
                "accepted": False,
                "rolled_back": True,
                "changes": [{k: v for k, v in cs.items() if k != "new_props"} for cs in changed_scenes],
                "severe_after": int(post_critic.get("severe_findings_count", 0)),
            })
            revision_history["status"] = "REPAIR_ROLLED_BACK_UNRESOLVED"
            break

    draft_visual_mp4.unlink(missing_ok=True)
    atomic_write_json(ev_dir / "revision_history.json", revision_history)

    # 4. POST-RENDER FROZEN VISUAL BARRIER (R5.R3 Fix D07!): 4-Tier Markers (RENDER_EMITTED -> OBSERVED_PIXEL) -> Sound -> Mix!
    tracker.start_stage("sound_design_mix")
    anim_markers = extract_realized_animation_markers(
        storyboard,
        story_graph,
        compiled_scene_props=compiled_props_by_scene,
        rendered_scenes=rendered_scenes,
        observe_pixels=True,
    )
    tier_counts: dict[str, int] = {}
    for m in anim_markers:
        tier_counts[m.marker_tier] = tier_counts.get(m.marker_tier, 0) + 1
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
    markers_payload = {
        "phase": "POST_RENDER_FROZEN_4TIER",
        "tier_counts": tier_counts,
        "markers": [m.to_dict() for m in anim_markers],
    }
    atomic_write_json(ev_dir / "animation_markers.json", markers_payload)
    atomic_write_json(ev_dir / "animation_markers_realized.json", markers_payload)
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
            original_source_path=Path(orig_media.source_path),
            edited_voice_bed_wav=edited_voice_bed_wav,
            edit_map_hash=edit_time_map.map_hash if edit_time_map is not None else None,
        )
        mixed_wav_p = Path(mix_info["mixed_reference_path"])
    elif eff_audio_mode == "mix" and edited_voice_bed_wav is not None and edited_voice_bed_wav.is_file():
        mixed_wav_p = edited_voice_bed_wav
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

    # 6. Technical Validation + Audio QA + Reuse Gate + Rhythm Critic BEFORE atomic promotion
    tracker.start_stage("validation_and_qa")
    audio_checks, audio_qa = (
        validate_final_audio_mix(media, partial_out, audio_tl, mix_info, run_dir, ffmpeg_bin=ffmpeg_bin, ffprobe_bin=ffprobe_bin)
        if mix_info is not None
        else ([], {"status": "PRESERVE_EXACT"})
    )

    # Always verify immutability of the ORIGINAL source MP4 (Fixes D01/D05!)
    source_sha_after = _sha256_file(Path(orig_media.source_path))
    source_immutable = (source_sha_after == orig_media.input_hash)

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

    # Objective Rhythm Critic (A8.2) on actual edited voice bed (NOT SFX mix!)
    rhythm_critic = run_rhythm_critic(
        edit_map=edit_time_map,
        pacing_plan=pacing_plan,
        edited_voice_bed_wav=edited_voice_bed_wav,
        word_preservation=word_preservation,
        cut_continuity=cut_continuity,
        join_audio_qa=join_audio_qa,
        storyboard=storyboard,
        pixel_critic=pixel_critic,
        niche=storyboard.niche,
        out_dir=ev_dir,
    )
    atomic_write_json(ev_dir / "repair_history.json", revision_history)

    expected_frames = int(edit_time_map.output_total_frames) if edit_time_map is not None else int(media.duration_frames)
    edit_map_ok = (len(validate_edit_time_map(edit_time_map)) == 0) if edit_time_map is not None else True
    word_pres_ok = True if word_preservation is None else bool(word_preservation.get("passed", False))
    joins_aud_ok = True if join_audio_qa is None else (join_audio_qa.get("status") == "PASS")

    ablation_total = int(creative_qa.get("ablation_layers_total", 0))
    ablation_verified = int(creative_qa.get("ablation_layers_verified", 0))
    planned_active_layers = sum(len(sc.layers) for sc in storyboard.scenes)
    # Fixes D16: Never allow ablation_total==0 to pass when active graphics layers were planned!
    ablation_all_ok = (ablation_total > 0 and ablation_verified == ablation_total) if planned_active_layers > 0 else True
    # Fixes D16: Never allow mix_info is None to pass when audio-mode=mix + SFX cues were planned!
    mix_required = bool(eff_audio_mode == "mix" and eff_sfx and len(sonic_score.cues) > 0)
    audio_gate_ok = (
        (mix_info is not None and audio_qa.get("status") == "PASS")
        if mix_required
        else (mix_info is None or audio_qa.get("status") == "PASS")
    )
    critic_gate_ok = bool(
        pixel_critic["abrupt_cutaway_jumps"] == 0
        and int(pixel_critic.get("severe_findings_count", 0)) == 0
    )
    selected_candidates_ok = all(
        sc.scene_id in rendered_scenes and len(hf_results.get(sc.scene_id, [])) == sum(1 for ly in sc.layers if ly.engine == "hyperframes")
        for sc in storyboard.scenes
    )

    tech_ok_breakdown = {
        "source_immutable": source_immutable,
        "geometry_exact": (out_w == media.display_width and out_h == media.display_height),
        "frames_exact": (out_frames == expected_frames),
        "edit_map_valid": edit_map_ok,
        "word_preservation_ok": word_pres_ok,
        "audio_joins_ok": joins_aud_ok,
        "zero_black_frames": (creative_qa["black_frames"] == 0),
        "ablation_all_ok": ablation_all_ok,
        "continuous_timeline_valid": bool(continuous_tl["continuous_timeline_valid"]),
        "critic_gate_ok": critic_gate_ok,
        "reuse_gate_ok": (not require_dual_engine_reuse_gate or reuse_passed),
        "audio_gate_ok": audio_gate_ok,
        "selected_candidates_and_multilayer_ok": selected_candidates_ok,
    }
    tech_ok = bool(all(tech_ok_breakdown.values()))

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
        "edition": "ml-01.r5.r4",
        "mode": storyboard.mode,
        "time_domain": "edited" if (edit_time_map is not None and not edit_time_map.is_identity) else "source",
        "edit_map_hash": edit_time_map.map_hash if edit_time_map is not None else None,
        "selected_graph_hash": storyboard.selected_graph_hash,
        "engines_used": sorted(actual_engines),
        "dual_engine_active": ("remotion" in actual_engines and "hyperframes" in actual_engines),
        "reuse_gate": reuse_report,
        "scenes": [
            {
                "scene_id": sc.scene_id,
                "chosen_candidate_id": sc.chosen_candidate_id,
                "candidate_provenance": sc.candidate_provenance,
                "role": sc.role,
                "stage_mode": rendered_scenes[sc.scene_id]["stage_mode"],
                "frame_range": [sc.start_frame, sc.end_frame_exclusive],
                "remotion_artifact_sha256": rendered_scenes[sc.scene_id]["artifact_sha256"],
                "hf_primitive_id": hf_results[sc.scene_id][0]["primitive_id"] if sc.scene_id in hf_results else None,
                "hf_artifact_sha256": hf_results[sc.scene_id][0]["artifact_sha256"] if sc.scene_id in hf_results else None,
                "hf_layers_executed": [
                    {
                        "layer_id": h["layer_id"],
                        "primitive_id": h["primitive_id"],
                        "layer_role": h["layer_role"],
                        "z_order": h["z_order"],
                        "render_mode": h.get("render_mode", "combined"),
                        "artifact_sha256": h["artifact_sha256"],
                    }
                    for h in hf_results.get(sc.scene_id, [])
                ],
                "used_matte_alpha": rendered_scenes[sc.scene_id]["used_matte_alpha"],
                "product_asset_bound": scene_prep[sc.scene_id].get("product_asset_bound", False),
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
        "tech_ok_breakdown": tech_ok_breakdown,
        "aesthetic_status": "PENDING_HUMAN_REVIEW",
        "foley_library_status": inspect_foley_library()["status"],
        "source_immutable": source_immutable,
        "output_dimensions": f"{out_w}x{out_h}",
        "output_frames": out_frames,
        "expected_frames": expected_frames,
        "source_frames": orig_media.duration_frames,
        "pacing_mode": edit_time_map.pacing_mode if edit_time_map is not None else "preserve",
        "pacing_status": edit_time_map.status if edit_time_map is not None else "IDENTITY_PRESERVE",
        "edit_map_hash": edit_time_map.map_hash if edit_time_map is not None else None,
        "rhythm_critic": rhythm_critic,
        "word_preservation": word_preservation,
        "reuse_gate": reuse_report,
        "audio_qa": audio_qa,
        "marker_tier_counts": tier_counts,
        "revision_history": revision_history,
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

    # 2. Original Source StoryGraph + A1 Speech Perception (Silero VAD, Alignment, Prosody, Action Evidence)
    tracker.start_stage("story_graph")
    source_story_graph = build_story_graph(transcript, roi_tracks, media)
    atomic_write_json(run_dir / "source_probe.json", media.to_dict())
    atomic_write_json(run_dir / "story_graph.json", source_story_graph.to_dict())
    atomic_write_json(run_dir / "ledger_claims.json", source_story_graph.ledger.to_dict())
    ev_dir = run_dir / "evidence"
    ev_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = run_dir / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    perception = run_speech_and_action_perception(
        media=media,
        transcript=transcript,
        roi_tracks=roi_tracks,
        out_dir=ev_dir,
        ffmpeg_bin=ffmpeg_bin,
    )
    pre_ev_idx = build_evidence_index(media, transcript, source_story_graph, roi_tracks=roi_tracks)
    pre_niche = build_narrative_contract(pre_ev_idx, source_story_graph).inferred_niche
    tracker.end_stage("story_graph", temp_dir=temp_dir)

    # 2b. Phase A2–A4: Causal Intelligent Pacing Director + EditTimeMap + Strategy A Derived Mezzanine & Voice Bed
    tracker.start_stage("pacing_and_cutting")
    excluded_cut_ids: set[str] = set()
    pacing_plan: dict[str, Any] = {}
    edit_map = build_identity_edit_time_map(media, pacing_mode=config.pacing_mode)
    edited_media = media
    edited_transcript = transcript
    edited_roi_tracks = roi_tracks
    edited_story_graph = source_story_graph
    edited_voice_bed_wav: Path | None = None
    word_pres_doc: dict[str, Any] | None = None
    cut_cont_doc: dict[str, Any] | None = None
    join_aud_doc: dict[str, Any] | None = None
    mezzanine_info: dict[str, Any] = {"mode": "identity_passthrough"}
    pacing_rollback_log: list[dict[str, Any]] = []

    for cut_round in range(3):  # Initial + max 2 bounded upstream cut rollbacks (A8.3)!
        pacing_plan, edit_map = choose_pacing_plan(
            media=media,
            story_graph=source_story_graph,
            perception=perception,
            pacing_mode=config.pacing_mode,
            niche=pre_niche,
            excluded_cut_ids=excluded_cut_ids,
        )
        if edit_map.is_identity:
            edited_media = media
            edited_transcript = transcript
            edited_roi_tracks = roi_tracks
            edited_story_graph = source_story_graph
            edited_voice_bed_wav = None
            word_pres_doc = {
                "edit_map_hash": edit_map.map_hash,
                "is_identity": True,
                "passed": True,
                "token_recall": 1.0,
                "critical_token_recall": 1.0,
                "re_asr_status": "IDENTITY_SKIPPED",
            }
            join_aud_doc = {"status": "PASS", "joins_checked": 0, "clicks_detected": 0, "dc_jumps_detected": 0, "edit_map_hash": edit_map.map_hash}
            cut_cont_doc = {"status": "PASS_NO_CUTS", "joins_count": 0, "severe_jump_joins": 0, "edit_map_hash": edit_map.map_hash}
            break

        # Strategy A: Near-lossless CRF-10 edited video mezzanine (content-addressed cached when cache_enabled!) + crossfaded PCM bed
        mezz_mp4 = temp_dir / f"edited_mezzanine_r{cut_round}.mp4"
        mezz_cache_p = cache_base / "edited_mezzanine" / f"mezz_{media.input_hash[:12]}_{edit_map.map_hash[:16]}.mp4"
        if config.cache_enabled and mezz_cache_p.is_file() and mezz_cache_p.stat().st_size > 1024:
            shutil.copy2(mezz_cache_p, mezz_mp4)
            mezzanine_info = {
                "mode": "near_lossless_crf10_mezzanine_cached",
                "mezzanine_path": str(mezz_mp4),
                "mezzanine_sha256": _sha256_file(mezz_mp4),
                "mezzanine_bytes": mezz_mp4.stat().st_size,
                "parent_source_sha256": edit_map.source_sha256,
                "edit_map_hash": edit_map.map_hash,
                "verified_frame_count": edit_map.output_total_frames,
                "cache_hit": True,
            }
        else:
            mezzanine_info = build_edited_mezzanine_video(
                Path(media.source_path), edit_map, mezz_mp4, ffmpeg_bin=ffmpeg_bin
            )
            if config.cache_enabled:
                mezz_cache_p.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(mezz_mp4, mezz_cache_p)
        bed_wav = ev_dir / "edited_voice_bed.wav"
        build_edited_voice_bed(Path(media.source_path), edit_map, bed_wav, ffmpeg_bin=ffmpeg_bin)
        edited_voice_bed_wav = bed_wav

        # Verify audio joins & re-ASR word preservation strictly on derived voice bed (NEVER SFX mix!)
        join_aud_doc = evaluate_audio_joins(bed_wav, edit_map, out_dir=ev_dir)
        word_pres_doc = verify_word_preservation_on_bed(
            original_transcript=transcript,
            edited_voice_bed_wav=bed_wav,
            edit_map=edit_map,
            media=media,
            config=config,
            aligned_words=perception["speech_alignment"]["words"],
            ffmpeg_bin=ffmpeg_bin,
        )
        suspects: list[str] = list(word_pres_doc.get("suspect_cut_ids") or [])
        if join_aud_doc.get("status") != "PASS":
            for j_idx, j_row in enumerate(join_aud_doc.get("joins") or []):
                if not j_row.get("join_ok") and j_idx < len(edit_map.omitted_spans):
                    suspects.append(edit_map.omitted_spans[j_idx].cut_id)

        if word_pres_doc.get("passed") and join_aud_doc.get("status") == "PASS":
            cut_cont_doc = evaluate_cut_continuity(
                Path(media.source_path), mezz_mp4, edit_map, roi_tracks_source=roi_tracks, out_dir=ev_dir
            )
            edited_transcript = remap_transcript_to_edited_domain(transcript, edit_map)
            edited_roi_tracks = remap_roi_tracks_to_edited_domain(roi_tracks, edit_map)
            edited_media = MediaInfo.from_dict({
                **media.to_dict(),
                "source_path": str(mezz_mp4),
                "input_hash": str(mezzanine_info.get("mezzanine_sha256", media.input_hash)),
                "file_size_bytes": int(mezzanine_info.get("mezzanine_bytes", media.file_size_bytes)),
                "nb_frames": int(edit_map.output_total_frames),
                "duration_s": float(edit_map.output_duration_s),
            })
            edited_story_graph = build_story_graph(edited_transcript, edited_roi_tracks, edited_media)
            break

        # Roll back offending cuts and retry (bounded max 2 rounds!)
        new_excl = [s for s in suspects if s not in excluded_cut_ids]
        if not new_excl:
            new_excl = [o.cut_id for o in edit_map.omitted_spans if o.cut_id not in excluded_cut_ids]
        pacing_rollback_log.append({
            "round": cut_round + 1,
            "reason": "word_preservation_or_audio_join_failed",
            "word_preservation_passed": word_pres_doc.get("passed"),
            "join_audio_status": join_aud_doc.get("status"),
            "rolled_back_cuts": new_excl,
        })
        excluded_cut_ids.update(new_excl)

    pacing_plan["rollback_history"] = pacing_rollback_log
    pacing_plan["mezzanine_strategy"] = mezzanine_info
    atomic_write_json(run_dir / "pacing_plan.json", pacing_plan)
    atomic_write_json(run_dir / "edit_time_map.json", edit_map.to_dict())
    atomic_write_json(run_dir / "edit_decisions.json", {
        "schema_version": "1.0.0",
        "edit_map_hash": edit_map.map_hash,
        "kept_spans": [k.to_dict() for k in edit_map.kept_spans],
        "omitted_spans": [o.to_dict() for o in edit_map.omitted_spans],
    })
    atomic_write_json(run_dir / "edited_probe.json", edited_media.to_dict())
    atomic_write_json(run_dir / "word_preservation.json", word_pres_doc or {})
    atomic_write_json(run_dir / "edited_story_graph.json", edited_story_graph.to_dict())
    tracker.end_stage("pacing_and_cutting", temp_dir=temp_dir)

    # 3. Generic Causal Global Creative Director strictly in EDITED domain (Qwen2.5-3B + Hybrid Retrieval + Real Transcript!)
    tracker.start_stage("generative_director")
    storyboard, dir_audit = direct_generative_storyboard(
        media=edited_media,
        story_graph=edited_story_graph,
        roi_tracks=edited_roi_tracks,
        config=config,
        matte_available=bool(vision_info.get("available")),
        registry=reg,
        cache_dir=sem_cache,
        transcript=edited_transcript,
    )
    atomic_write_json(run_dir / "raw_director_response.json", dir_audit)
    atomic_write_json(run_dir / "retrieved_candidates.json", storyboard.retrieved_candidates)
    tracker.end_stage("generative_director", temp_dir=temp_dir)

    # 4. Execute Shared Deterministic Renderer / Mixer / Validator strictly on EDITED clock while preserving immutable original source!
    result = render_compiled_storyboard_to_mp4(
        media=edited_media,
        storyboard=storyboard,
        story_graph=edited_story_graph,
        roi_tracks=edited_roi_tracks,
        config=config,
        run_id=run_id,
        run_dir=run_dir,
        budget=budget,
        tracker=tracker,
        output_filename="final.mp4",
        render_preview=bool(config.render_preview_9x16),
        require_dual_engine_reuse_gate=(not storyboard.niche in ("tutorial", "gameplay", "silent_art") and len(storyboard.scenes) >= 2),
        transcript_obj=edited_transcript,
        original_media=media,
        edit_time_map=edit_map,
        pacing_plan=pacing_plan,
        edited_voice_bed_wav=edited_voice_bed_wav,
        word_preservation=word_pres_doc,
        cut_continuity=cut_cont_doc,
        join_audio_qa=join_aud_doc,
    )

    manifest["composition_mode"] = "generative"
    manifest["pacing_mode"] = config.pacing_mode
    manifest["edit_map_hash"] = edit_map.map_hash
    manifest["original_frames"] = media.duration_frames
    manifest["edited_frames"] = edit_map.output_total_frames
    manifest["cuts_count"] = len(edit_map.omitted_spans)
    manifest["removed_total_s"] = round(edit_map.removed_total_s, 4)
    manifest["status"] = result["status"]
    manifest["aesthetic_status"] = result.get("aesthetic_status", "PENDING_HUMAN_REVIEW")
    manifest["storyboard_hash"] = result.get("storyboard_hash")
    manifest["story_graph_hash"] = result.get("story_graph_hash")
    manifest["reuse_gate"] = result.get("reuse_gate")
    manifest["resource_tracker"] = tracker.to_dict()
    atomic_write_json(run_dir / "manifest.json", manifest)
    atomic_write_text(run_dir / "logs" / "pipeline.log", "\n".join(log_lines) + "\n")
    return result
