"""Phase 5 (ML-01.R3): Bounded Dependency DAG Executor & Canonical Single-Pass Timeline Assembler.

Orchestrates:
  1. Phase 1 Vision (ROI tracking + targeted MODNet portrait matting on matte spans only)
  2. Phase 2 Composition Graph Solver (deterministic 9-recipe graph)
  3. Bounded parallel HyperFrames substantive GSAP+SVG graphic rendering
  4. Bounded Remotion full-frame creative scene compositing + per-layer ablation probes
  5. Single-pass FFmpeg canonical frame-exact assembly + lossless audio-master mux (0 audio filters, no -shortest)
"""

from __future__ import annotations

import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from motionlab.composition.planner import plan_composition_graph
from motionlab.composition.recipes import build_recipe_capability_matrix, get_recipe
from motionlab.composition.schema import CompositionGraph, CompositionScene
from motionlab.composition.vfx import get_vfx_recipe, is_vfx_recipe
from motionlab.config import MotionLabConfig
from motionlab.policies import ProfilePolicy
from motionlab.renderers.ffmpeg import FFmpegRenderer
from motionlab.renderers.hyperframes import HyperFramesRenderer
from motionlab.renderers.remotion import RemotionRenderer
from motionlab.resource import GLOBAL_JOB_SEMAPHORE, ResourceBudget, ResourceTracker, RuntimeResourceWatchdog, run_managed_subprocess
from motionlab.schemas import AnalysisResult, EditPlan, MediaInfo, RenderResult, atomic_write_json
from motionlab.vision.anchors import build_anchor_tracks, generate_matte_debug_sheet
from motionlab.vision.segmentation import inspect_vision_model, segment_subject_sequence
from motionlab.vision.tracking import track_action_rois


def _render_candidate_scene_ffmpeg_fallback(
    scene: CompositionScene,
    media: MediaInfo,
    workdir: Path,
    matte_mp4: Path | None,
    hf_graphic_webm: Path | None,
    ffmpeg_bin: str,
) -> dict[str, Any]:
    """Honest single-engine native compositor candidate when Remotion is NOT opted-in/licensed.

    Never claims Remotion executed; renders real multi-layer depth/detail/graphic composite so
    an owner without a Remotion license still gets an honest candidate artifact clearly tagged `ffmpeg`.
    """
    t0 = time.perf_counter()
    slot_dir = (workdir / f"ff_cand_{scene.scene_id}").resolve()
    slot_dir.mkdir(parents=True, exist_ok=True)
    out_mp4 = slot_dir / "scene.mp4"
    w, h = media.display_width, media.display_height
    n_frames = max(2, scene.end_frame_exclusive - scene.start_frame)
    fps_str = media.fps.to_ffmpeg()

    # Decode source slice
    src_cap = cv2.VideoCapture(str(media.path))
    src_cap.set(cv2.CAP_PROP_POS_FRAMES, scene.start_frame)
    m_cap = cv2.VideoCapture(str(matte_mp4)) if (matte_mp4 and matte_mp4.is_file()) else None

    # Decode HyperFrames RGBA graphic frames if present
    hf_frames: list[np.ndarray] = []
    if hf_graphic_webm and hf_graphic_webm.is_file():
        dec = subprocess.run(
            [ffmpeg_bin, "-v", "error", "-c:v", "libvpx-vp9", "-i", str(hf_graphic_webm), "-f", "rawvideo", "-pix_fmt", "rgba", "pipe:1"],
            capture_output=True, timeout=60, shell=False,
        )
        fbytes = w * h * 4
        if dec.returncode == 0 and len(dec.stdout) >= fbytes:
            cnt = len(dec.stdout) // fbytes
            for i in range(min(n_frames, cnt)):
                hf_frames.append(np.frombuffer(dec.stdout[i * fbytes : (i + 1) * fbytes], dtype=np.uint8).reshape((h, w, 4)))

    enc_cmd = [
        ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", fps_str,
        "-i", "pipe:0",
        "-c:v", "libx264", "-preset", "fast", "-crf", "16", "-pix_fmt", "yuv420p",
        "-an", "-movflags", "+faststart", str(out_mp4),
    ]
    enc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
    assert enc.stdin is not None
    anc = scene.anchor_bundle
    slot = anc.get("detail_slot") or {"cx": 0.27, "cy": 0.19, "w": 0.42, "h": 0.26}
    task = anc.get("task_mid") or {"cx": 0.5, "cy": 0.68}

    try:
        for i in range(n_frames):
            ok, bgr = src_cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            u = i / max(1, n_frames - 1)
            env = 0.0 if (i == 0 or i == n_frames - 1) else min(1.0, min(i / max(2, scene.entry_frames), (n_frames - 1 - i) / max(2, scene.exit_frames)))
            out = rgb.copy()

            if scene.requires_matte and m_cap is not None:
                ok_m, m_bgr = m_cap.read()
                if ok_m:
                    alpha = cv2.cvtColor(m_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
                    bg = cv2.GaussianBlur(rgb, (0, 0), max(0.5, 7.0 * env))
                    bg = np.clip(bg.astype(np.float32) * (1.0 - 0.10 * env), 0, 255).astype(np.uint8)
                    if i < len(hf_frames):
                        hf_rgba = hf_frames[i]
                        ha = (hf_rgba[..., 3:4].astype(np.float32) / 255.0) * env
                        bg = np.clip(hf_rgba[..., :3].astype(np.float32) * ha + bg.astype(np.float32) * (1.0 - ha), 0, 255).astype(np.uint8)
                    a3 = alpha[..., None]
                    out = np.clip(rgb.astype(np.float32) * a3 + bg.astype(np.float32) * (1.0 - a3), 0, 255).astype(np.uint8)
            else:
                if scene.uses_detail_crop and env > 0.02:
                    bw, bh = int(round(w * float(slot["w"]))), int(round(h * float(slot["h"])))
                    bx, by = max(0, int(round(w * float(slot["cx"]) - bw / 2))), max(0, int(round(h * float(slot["cy"]) - bh / 2)))
                    bx2, by2 = min(w, bx + bw), min(h, by + bh)
                    cw, ch = max(16, int((bx2 - bx) / 1.5)), max(16, int((by2 - by) / 1.5))
                    tcx, tcy = int(round(w * float(task["cx"]))), int(round(h * float(task["cy"])))
                    x0 = max(0, min(w - cw, tcx - cw // 2))
                    y0 = max(0, min(h - ch, tcy - ch // 2))
                    crop = cv2.resize(rgb[y0 : y0 + ch, x0 : x0 + cw], (bx2 - bx, by2 - by), interpolation=cv2.INTER_CUBIC)
                    out[by:by2, bx:bx2] = np.clip(crop.astype(np.float32) * env + out[by:by2, bx:bx2].astype(np.float32) * (1.0 - env), 0, 255).astype(np.uint8)
                if i < len(hf_frames):
                    hf_rgba = hf_frames[i]
                    ha = (hf_rgba[..., 3:4].astype(np.float32) / 255.0) * env
                    out = np.clip(hf_rgba[..., :3].astype(np.float32) * ha + out.astype(np.float32) * (1.0 - ha), 0, 255).astype(np.uint8)

            enc.stdin.write(out.tobytes())
    finally:
        src_cap.release()
        if m_cap is not None:
            m_cap.release()
        enc.stdin.close()
        enc.wait(timeout=60)

    import hashlib  # noqa: PLC0415

    hh = hashlib.sha256()
    with out_mp4.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            hh.update(chunk)
    return {
        "scene_id": scene.scene_id,
        "recipe_id": scene.recipe_id,
        "scene_kind": (
            get_vfx_recipe(scene.recipe_id).remotion_scene_kind
            if is_vfx_recipe(scene.recipe_id)
            else get_recipe(scene.recipe_id).remotion_scene_kind
        ),
        "engine": "ffmpeg",
        "slot_dir": str(slot_dir),
        "artifact_path": str(out_mp4),
        "artifact_sha256": hh.hexdigest(),
        "size_bytes": int(out_mp4.stat().st_size),
        "start_frame": scene.start_frame,
        "end_frame_exclusive": scene.end_frame_exclusive,
        "frame_count": n_frames,
        "dimensions": f"{w}x{h}",
        "pix_fmt": "yuv420p",
        "used_matte_alpha": bool(scene.requires_matte and matte_mp4),
        "used_hyperframes_graphics": len(hf_frames) > 0,
        "exclude_layer_role": None,
        "elapsed_s": round(time.perf_counter() - t0, 3),
    }


def _assemble_canonical_timeline_and_mux(
    media: MediaInfo,
    plan: EditPlan,
    graph: CompositionGraph,
    rendered_scenes: dict[str, dict[str, Any]],
    workdir: Path,
    out_partial_path: Path,
    budget: ResourceBudget,
    config: MotionLabConfig,
    ffmpeg_bin: str,
) -> dict[str, Any]:
    """Assemble exact 0..duration_frames-1 timeline (clean source + rendered Remotion scenes) + lossless audio mux."""
    t0 = time.perf_counter()
    w, h = media.display_width, media.display_height
    total_f = media.duration_frames
    fps_str = media.fps.to_ffmpeg()
    video_only = workdir / "r3_video_only.mp4"

    # Build frame -> (scene_id, local_frame_idx) lookup table
    scene_caps: dict[str, cv2.VideoCapture] = {}
    frame_owner: list[tuple[str, int] | None] = [None] * total_f
    for sc in graph.scenes:
        rinfo = rendered_scenes.get(sc.scene_id)
        if not rinfo or not Path(rinfo["artifact_path"]).is_file():
            continue
        scene_caps[sc.scene_id] = cv2.VideoCapture(rinfo["artifact_path"])
        for f in range(sc.start_frame, min(total_f, sc.end_frame_exclusive)):
            frame_owner[f] = (sc.scene_id, f - sc.start_frame)

    dec_cmd = [
        ffmpeg_bin, "-v", "error", "-nostdin",
        "-i", str(media.path),
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]
    enc_cmd = [
        ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error", "-nostdin",
        "-threads", str(max(1, budget.ffmpeg_threads)),
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", fps_str,
        "-i", "pipe:0",
        "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p",
        "-an", "-movflags", "+faststart", str(video_only),
    ]

    dec = subprocess.Popen(dec_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
    enc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
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
                sid, _local_i = owner
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
            raise RuntimeError(f"Resource watchdog terminated assembly: {wd.violation}")
        if d_rc != 0 or e_rc != 0:
            raise RuntimeError(f"Canonical R3 assembly failed: dec={d_rc}:{d_err[:200]}, enc={e_rc}:{e_err[:200]}")

    t_vid = round(time.perf_counter() - t0, 3)

    # Single lossless audio-master mux (identical P1-B contract: zero audio filters, zero -shortest!)
    t_mux0 = time.perf_counter()
    selected_tracks = list(plan.audio_policy.get("selected_tracks") or [])
    per_track_modes = dict(plan.audio_policy.get("per_track_modes") or {})
    mux_cmd = [ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error", "-i", str(video_only)]
    if media.has_audio and selected_tracks:
        mux_cmd += ["-i", str(media.path), "-map", "0:v:0"]
        for out_idx, src_idx in enumerate(selected_tracks):
            mux_cmd += ["-map", f"1:a:{src_idx}"]
        mux_cmd += ["-c:v", "copy"]
        for out_idx, src_idx in enumerate(selected_tracks):
            mode = per_track_modes.get(str(src_idx), "stream_copy")
            if mode == "stream_copy":
                mux_cmd += [f"-c:a:{out_idx}", "copy"]
            else:
                mux_cmd += [f"-c:a:{out_idx}", "aac", f"-b:a:{out_idx}", "256k"]
    else:
        mux_cmd += ["-map", "0:v:0", "-c:v", "copy", "-an"]
    mux_cmd += ["-movflags", "+faststart", str(out_partial_path)]
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


def execute_composition_pipeline(
    media: MediaInfo,
    analysis: AnalysisResult,
    policy: ProfilePolicy,
    plan: EditPlan,
    config: MotionLabConfig,
    doctor_info: dict[str, Any],
    run_dir: Path,
    budget: ResourceBudget,
    tracker: ResourceTracker,
    existing_graph: CompositionGraph | None = None,
) -> dict[str, Any]:
    """Full ML-01.R3 creative composition DAG: vision -> graph -> HyperFrames -> Remotion -> assembly -> ablation."""
    ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"
    temp_dir = run_dir / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    ev_dir = run_dir / "evidence"
    ev_dir.mkdir(parents=True, exist_ok=True)
    engine_caps = doctor_info["engines"]
    vision_model_info = inspect_vision_model()
    cache_root = Path(config.cache_dir) / "vision" if config.cache_dir else None

    # 1. ROI tracking (fast ~1.2s) + preliminary graph to identify EXACT spans needing MODNet mattes
    tracker.start_stage("vision_tracking")
    roi_tracks = track_action_rois(media, analysis, config, ffmpeg_bin=ffmpeg_bin)
    prelim_anchors = build_anchor_tracks(media, analysis, roi_tracks, matte_result=None)
    tracker.end_stage("vision_tracking", temp_dir=temp_dir)

    if existing_graph is not None:
        graph = existing_graph
        graph_explain: dict[str, Any] = {"replayed_from_graph": True, "graph_hash": graph.graph_hash}
    else:
        graph, graph_explain = plan_composition_graph(
            media, analysis, prelim_anchors, policy, config, engine_caps,
            matte_available_override=bool(vision_model_info.get("available")),
        )

    # 2. Targeted MODNet segmentation ONLY on spans whose selected recipe requires a subject matte!
    matte_spans = [(sc.start_frame, sc.end_frame_exclusive) for sc in graph.scenes if sc.requires_matte]
    tracker.start_stage("vision_segmentation")
    if matte_spans and vision_model_info.get("available"):
        matte_res = segment_subject_sequence(
            media=media,
            spans=matte_spans,
            workdir=temp_dir,
            cache_dir=cache_root if config.cache_enabled else (temp_dir / "nocache_vision"),
            preserve_tabletop_y=roi_tracks.get("tabletop_guard_y"),
            keyframe_stride=2,
            budget=budget,
            ffmpeg_bin=ffmpeg_bin,
        )
        # If matte quality failed on real video -> safely replan without matte recipes!
        if not matte_res.get("quality_summary", {}).get("all_spans_quality_ok", False):
            if existing_graph is None:
                graph, graph_explain = plan_composition_graph(
                    media, analysis, prelim_anchors, policy, config, engine_caps, matte_available_override=False
                )
    else:
        matte_res = {
            "available": False,
            "status": "SKIPPED_NO_MATTE_SCENES" if not matte_spans else "BLOCKED_CREATIVE_MASK",
            "model": vision_model_info,
            "spans": [],
            "quality_summary": {},
        }
    tracker.end_stage("vision_segmentation", temp_dir=temp_dir)

    anchors = build_anchor_tracks(media, analysis, roi_tracks, matte_res)
    debug_sheet = generate_matte_debug_sheet(Path(media.path), matte_res, anchors, ev_dir / "matte_debug_sheet.jpg")

    # Write Phase 1 & Phase 2 canonical manifests
    cap_matrix = build_recipe_capability_matrix(engine_caps, vision_model_info)
    atomic_write_json(run_dir / "composition_graph.json", graph.to_dict())
    atomic_write_json(run_dir / "recipe_capability_matrix.json", cap_matrix)
    atomic_write_json(ev_dir / "vision_manifest.json", {
        "schema_version": "1.0.0",
        "source_hash": media.input_hash,
        "model": vision_model_info,
        "matte_status": matte_res.get("status"),
        "matte_available": bool(anchors["matte_available"]),
        "matte_cache_hit": bool(matte_res.get("cache_hit", False)),
        "matte_elapsed_s": float(matte_res.get("elapsed_s", 0.0)),
        "roi_summary": roi_tracks.get("summary", {}),
        "tabletop_guard_y": roi_tracks.get("tabletop_guard_y"),
        "debug_sheet": str(Path(debug_sheet).relative_to(run_dir)) if debug_sheet else None,
        "matte_spans": matte_res.get("spans", []),
    })
    atomic_write_json(ev_dir / "roi_tracks.json", roi_tracks)
    atomic_write_json(ev_dir / "anchors.json", anchors)
    atomic_write_json(ev_dir / "mask_quality.json", {
        "status": matte_res.get("status"),
        "quality_summary": matte_res.get("quality_summary", {}),
        "spans": matte_res.get("spans", []),
    })

    if not graph.scenes:
        return {
            "executed_advanced": False,
            "graph": graph,
            "graph_explain": graph_explain,
            "matte_res": matte_res,
            "anchors": anchors,
        }

    # Map (sf, ef) -> matte_mp4 path
    matte_by_span = {
        (int(sp["start_frame"]), int(sp["end_frame_exclusive"])): Path(sp["matte_mp4"])
        for sp in (matte_res.get("spans") or [])
    }
    raw_matte_by_span = {
        (int(sp["start_frame"]), int(sp["end_frame_exclusive"])): Path(sp.get("raw_matte_mp4", sp["matte_mp4"]))
        for sp in (matte_res.get("spans") or [])
    }

    # 3. Bounded parallel HyperFrames substantive graphics renders
    hf = HyperFramesRenderer()
    aud_env = [p.normalized_rms for p in analysis.audio_envelope] if analysis.audio_envelope else None
    hf_jobs = [sc for sc in graph.scenes if sc.graphics_engine == "hyperframes"]
    hf_results: dict[str, dict[str, Any]] = {}
    fallback_history: list[dict[str, Any]] = []

    tracker.start_stage("hyperframes_graphics")
    if hf_jobs:
        def _run_hf(sc: CompositionScene) -> tuple[str, dict[str, Any] | None, dict[str, Any] | None]:
            spec = get_recipe(sc.recipe_id)
            if not spec.hyperframes_template or not hf.supports_recipe_graphics(spec.hyperframes_template, engine_caps):
                return sc.scene_id, None, {"scene_id": sc.scene_id, "requested_engine": "hyperframes", "fallback_engine": "none", "reason": "hyperframes_unavailable"}
            try:
                res = hf.render_graphics_scene(
                    scene_id=sc.scene_id,
                    template_name=spec.hyperframes_template,
                    width=media.display_width,
                    height=media.display_height,
                    fps=media.fps.as_float(),
                    start_frame=sc.start_frame,
                    end_frame_exclusive=sc.end_frame_exclusive,
                    anchor_bundle=sc.anchor_bundle,
                    parameters=sc.parameters,
                    workdir=temp_dir,
                    audio_envelope=aud_env,
                    timeout_s=min(180.0, float(config.timeout_s)),
                )
                return sc.scene_id, res, None
            except Exception as exc:
                if config.strict:
                    raise RuntimeError(f"Strict mode: HyperFrames graphic failed for {sc.scene_id}: {exc}") from exc
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

    # 4. Bounded Remotion full-frame scene compositing (worker=1 to guard 16GB RAM) + real per-layer ablation stills!
    rem = RemotionRenderer()
    rendered_scenes: dict[str, dict[str, Any]] = {}
    ablation_probes: dict[str, dict[str, Any]] = {}

    tracker.start_stage("remotion_scenes")
    for sc in graph.scenes:
        matte_p = matte_by_span.get((sc.start_frame, sc.end_frame_exclusive))
        raw_matte_p = raw_matte_by_span.get((sc.start_frame, sc.end_frame_exclusive))
        hf_p = Path(hf_results[sc.scene_id]["artifact_path"]) if sc.scene_id in hf_results else None

        if sc.primary_engine == "remotion" and rem.supports_composition(sc.recipe_id, engine_caps):
            try:
                r_res = rem.render_scene_composition(
                    scene=sc,
                    media=media,
                    workdir=temp_dir,
                    matte_mp4=matte_p,
                    hf_graphic_webm=hf_p,
                    timeout_s=min(240.0, float(config.timeout_s)),
                )
                rendered_scenes[sc.scene_id] = r_res

                # Run real Remotion 1-frame per-layer ablation stills on midpoint frame for G5 proof!
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
                if sc.uses_detail_crop:
                    probe_roles.append("source_detail")

                layer_ab: dict[str, Any] = {}
                if ok_f and full_bgr is not None:
                    full_f = full_bgr.astype(np.float32)
                    # Load raw matte midpoint if available to prove true inside-vs-outside occlusion z-order!
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
                                "verified": bool(mad >= 0.15 or (max_px >= 18.0 and cfrac >= 0.001)),
                            }
                            # Occlusion z-order proof: behind-subject graphics MUST appear outside matte & stay occluded inside solid matte!
                            if role == "background_graphics" and raw_m_gray is not None:
                                inside_mask = raw_m_gray >= 0.90
                                outside_mask = raw_m_gray <= 0.10
                                if int(inside_mask.sum()) > 500 and int(outside_mask.sum()) > 500:
                                    diff_ch = diff.mean(axis=2)
                                    in_mad = float(diff_ch[inside_mask].mean())
                                    out_mad = float(diff_ch[outside_mask].mean())
                                    entry["occlusion_inside_subject_mad"] = round(in_mad, 4)
                                    entry["occlusion_outside_subject_mad"] = round(out_mad, 4)
                                    entry["occlusion_z_order_verified"] = bool(out_mad > max(0.5, in_mad * 2.5))
                            layer_ab[role] = entry
                ablation_probes[sc.scene_id] = layer_ab
                continue
            except Exception as exc:
                if config.strict:
                    raise RuntimeError(f"Strict mode: Remotion scene failed for {sc.scene_id}: {exc}") from exc
                fallback_history.append({
                    "scene_id": sc.scene_id,
                    "requested_engine": "remotion",
                    "fallback_engine": "ffmpeg",
                    "reason": str(exc)[:400],
                })

        # Honest single-engine candidate fallback (never claims Remotion executed!)
        rendered_scenes[sc.scene_id] = _render_candidate_scene_ffmpeg_fallback(
            sc, media, temp_dir, matte_p, hf_p, ffmpeg_bin
        )
        ablation_probes[sc.scene_id] = {}
    tracker.end_stage("remotion_scenes", temp_dir=temp_dir)

    # 5. Canonical single-pass FFmpeg timeline assembly + lossless audio-master mux
    partial_out = run_dir / "final.partial.mp4"
    tracker.start_stage("canonical_assembly")
    asm_stats = _assemble_canonical_timeline_and_mux(
        media, plan, graph, rendered_scenes, temp_dir, partial_out, budget, config, ffmpeg_bin
    )
    tracker.end_stage("canonical_assembly", temp_dir=temp_dir)

    # Build honest engine-usage ledger
    actual_engines: set[str] = {"ffmpeg"}
    for rinfo in rendered_scenes.values():
        actual_engines.add(rinfo["engine"])
        if rinfo.get("used_hyperframes_graphics"):
            actual_engines.add("hyperframes")

    ff_probe_proc = subprocess.run(
        [shutil.which("ffprobe") or "ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(partial_out)],
        capture_output=True, text=True, timeout=20, shell=False,
    )
    ffprobe_info = {}
    if ff_probe_proc.returncode == 0:
        import json as _json  # noqa: PLC0415

        ffprobe_info = _json.loads(ff_probe_proc.stdout)

    render_res = RenderResult(
        status="PASS",
        artifact_path=str(partial_out),
        engines_used=sorted(actual_engines),
        timings_s={"video_assembly_s": asm_stats["video_assembly_s"], "audio_mux_s": asm_stats["audio_mux_s"]},
        ffprobe_info=ffprobe_info,
        event_engine_map={sid: r["engine"] for sid, r in rendered_scenes.items()},
        fallback_history=fallback_history,
        encoding_settings={
            "mode": "ml01_r3_advanced_composition",
            "encoder": "libx264",
            "crf": "17",
            "preset": "medium",
            "pix_fmt": "yuv420p",
            "frames_rendered": asm_stats["frames_written"],
            "scene_frames_spliced": asm_stats["scene_frames_spliced"],
            "audio_mux_mode": plan.audio_policy.get("mode", "stream_copy"),
        },
        warnings=list(graph.warnings),
    )

    actual_engine_usage = {
        "engines_used": sorted(actual_engines),
        "dual_engine_active": ("remotion" in actual_engines and "hyperframes" in actual_engines),
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
        "executed_advanced": True,
        "graph": graph,
        "graph_explain": graph_explain,
        "matte_res": matte_res,
        "anchors": anchors,
        "hf_results": hf_results,
        "rendered_scenes": rendered_scenes,
        "ablation_probes": ablation_probes,
        "actual_engine_usage": actual_engine_usage,
        "render_result": render_res,
        "partial_out": partial_out,
    }
