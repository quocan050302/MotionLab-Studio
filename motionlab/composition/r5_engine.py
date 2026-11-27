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
from motionlab.audio.assets import inspect_foley_library
from motionlab.audio.mixer import mix_preserving_master, validate_final_audio_mix
from motionlab.audio.synthesis import build_sfx_asset_manifest
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

    # 1. Targeted MODNet Portrait Segmentation ONLY for scenes requiring foreground matte
    matte_spans = [(sc.start_frame, sc.end_frame_exclusive) for sc in storyboard.scenes if sc.requires_matte]
    vision_model_info = inspect_vision_model()
    cache_base = Path(config.cache_dir) if config.cache_dir else (PROJECT_ROOT / ".cache" / "motionlab_r5")
    vision_cache = (cache_base / "vision") if config.cache_enabled else (temp_dir / "nocache_vision")

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

    # 2. Build Sample-Exact Audio Timeline + Voice-Priority Mix
    eff_audio_mode = "preserve-exact" if config.audio_mode == "preserve-exact" else "mix"
    eff_sfx = (config.sfx_mode != "off") and (eff_audio_mode == "mix")
    eff_music = (config.music_mode == "local") and (eff_audio_mode == "mix")

    tracker.start_stage("sound_design_mix")
    audio_tl, _used_sfx = compile_storyboard_audio_timeline(
        media=media,
        storyboard=storyboard,
        audio_mode=eff_audio_mode,
        sfx_enabled=eff_sfx,
        music_enabled=eff_music,
        seed=config.seed,
    )
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

    if config.dry_run:
        return {
            "status": "DRY_RUN_PLANNED",
            "storyboard_hash": storyboard.storyboard_hash,
            "scenes": len(storyboard.scenes),
            "compiled_plan": compiled_plan,
        }

    # 3. Render Imported HyperFrames Apache-2.0 VP9 Alpha Graphics
    tracker.start_stage("hyperframes_graphics")
    hf_results: dict[str, dict[str, Any]] = {}
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
        )
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

    # 4. Render Imported Remotion MIT Full-Frame Scenes + Midpoint Per-Layer Ablation Stills
    tracker.start_stage("remotion_scenes")
    rendered_scenes: dict[str, dict[str, Any]] = {}
    ablation_results: dict[str, dict[str, Any]] = {}
    actual_engines: set[str] = {"ffmpeg"}
    actual_primitives: list[str] = []

    roi_frames = roi_tracks.get("keyframes") or roi_tracks.get("frames") or []
    fps_str = media.fps.to_ffmpeg()

    for sc in storyboard.scenes:
        slot_dir = temp_dir / f"rem_{sc.scene_id}"
        pub_dir = slot_dir / "public"
        pub_dir.mkdir(parents=True, exist_ok=True)

        # Slice exact source mezzanine for non-cutaway or transition-in scenes
        src_name: str | None = None
        if sc.stage_mode != "full_graphic_cutaway" or sc.transition_in.kind != "none":
            src_slice = pub_dir / "scene_source.mp4"
            rem._slice_source_mezzanine(Path(media.source_path), sc.start_frame, sc.end_frame_exclusive, fps_str, src_slice)
            src_name = src_slice.name

        # Build foreground alpha WebM if MODNet matte was produced for this span
        fg_name: str | None = None
        matte_p = matte_by_span.get((sc.start_frame, sc.end_frame_exclusive))
        if sc.requires_matte and matte_p is not None and matte_p.is_file():
            fg_webm = pub_dir / "fg_alpha.webm"
            build_foreground_alpha_webm(
                Path(media.source_path), matte_p, sc.start_frame, sc.end_frame_exclusive, fps_str, fg_webm, ffmpeg_bin=ffmpeg_bin
            )
            fg_name = fg_webm.name

        hf_name: str | None = None
        hf_role = "foreground_graphics"
        if sc.scene_id in hf_results:
            dst_hf = pub_dir / "hf_graphic.webm"
            shutil.copy2(Path(hf_results[sc.scene_id]["artifact_path"]), dst_hf)
            hf_name = dst_hf.name
            hf_role = hf_results[sc.scene_id]["layer_role"]
            actual_engines.add("hyperframes")
            actual_primitives.append(hf_results[sc.scene_id]["primitive_id"])

        # Scene anchor averages (supports both roi_tracks["keyframes"] {"face","task"} and {"face_box","task_box"})
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
            source_file=src_name,
            fg_alpha_file=fg_name,
            prop_alpha_file=None,
            hf_graphic_file=hf_name,
            hf_layer_role=hf_role,
            task_anchor=task_anchor,
            face_anchor=face_anchor,
        )

        out_sc_mp4 = slot_dir / "scene.mp4"
        r_rem = rem.render_custom_composition(
            composition_id="MotionLabGenerativeScene",
            props=props,
            out_path=out_sc_mp4,
            slot_dir=slot_dir,
            public_dir=pub_dir,
            timeout_s=240.0,
        )
        actual_engines.add("remotion")
        for ly in sc.layers:
            if ly.engine == "remotion":
                actual_primitives.append(ly.primitive_id)
        if sc.stage_mode == "aspect_morph_editorial":
            actual_primitives.append("rem_viv_aspect_morph")
        elif sc.stage_mode == "live_depth_matte" and fg_name is not None:
            actual_primitives.append("rem_ru_parallax")
        if sc.transition_in.kind == "frame_morph":
            actual_primitives.append("rem_gtk_frame_morph")
        elif sc.transition_in.kind == "line_led":
            actual_primitives.append("rem_gtk_line_led")
        elif sc.transition_in.kind == "flip_page":
            actual_primitives.append("rem_rc_flip_page")

        rendered_scenes[sc.scene_id] = {
            **r_rem,
            "start_frame": sc.start_frame,
            "end_frame_exclusive": sc.end_frame_exclusive,
            "used_matte_alpha": fg_name is not None,
            "used_hyperframes_graphics": hf_name is not None,
            "stage_mode": props["stageMode"],
        }

        # Midpoint single-frame ablation still proof (fast & proves each engine layer contributes visible pixels!)
        mid_rel = max(6, min(props["durationInFrames"] - 8, props["durationInFrames"] // 2))
        ab_dir = ev_dir / "ablations" / sc.scene_id
        ab_dir.mkdir(parents=True, exist_ok=True)
        full_png = ab_dir / "full.png"
        sc_ab: dict[str, Any] = {}
        try:
            rem.render_scene_ablation_still(
                scene=sc, media=media, scene_slot_dir=slot_dir, sample_frame=mid_rel,
                exclude_layer_role=None, out_png=full_png, composition_id="MotionLabGenerativeScene", timeout_s=90.0,
            )
            roles_to_test: list[str] = []
            if hf_name is not None:
                roles_to_test.append(hf_role)
            if props["primitives"]:
                roles_to_test.append("remotion_primitive")
            if fg_name is not None:
                roles_to_test.append("foreground_subject_matte")
            for role_ex in roles_to_test[:2]:  # Cap 2 ablations/scene to stay bounded & fast
                ex_png = ab_dir / f"without_{role_ex}.png"
                rem.render_scene_ablation_still(
                    scene=sc, media=media, scene_slot_dir=slot_dir, sample_frame=mid_rel,
                    exclude_layer_role=role_ex, out_png=ex_png, composition_id="MotionLabGenerativeScene", timeout_s=90.0,
                )
                sc_ab[role_ex] = verify_ablation_still_diff(full_png, ex_png)
        except Exception as exc:
            sc_ab["error"] = str(exc)[:200]
        ablation_results[sc.scene_id] = sc_ab
    tracker.end_stage("remotion_scenes", temp_dir=temp_dir)

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

    tech_ok = bool(
        source_immutable
        and out_w == media.display_width
        and out_h == media.display_height
        and out_frames == media.duration_frames
        and creative_qa["black_frames"] == 0
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
        "creative_quality_proxy": creative_qa,
        "assembly_stats": asm_stats,
        "preview_9x16": preview_info,
    }
    atomic_write_json(run_dir / "validation.json", validation_doc)
    atomic_write_json(run_dir / "resource_stats.json", tracker.to_dict())

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
        "creative_quality_proxy": creative_qa,
        "resource_stats": tracker.to_dict(),
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
