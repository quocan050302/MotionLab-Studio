"""End-to-End Pipeline Orchestrator for MotionLab AI ML-01.

Coordinates:
  Preflight -> ResourceBudget -> Async Local Analysis -> Rule-Based Planner ->
  Bounded Async Multi-Engine Overlay Render (Remotion/HyperFrames with certified FFmpeg fallback) ->
  Streaming FFmpeg Visual Pipeline -> Alpha Overlay Composite -> Hard-Locked Audio Mux ->
  Technical Validation -> Atomic Promotion (`final.partial.mp4` -> `final.mp4`) -> Run Handover Bundle.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from motionlab import ANALYZER_VERSION, PLANNER_VERSION, SCHEMA_VERSION, __version__
from motionlab.analyzer import analyze_media_async
from motionlab.config import EditConfig, PROJECT_ROOT, atomic_write_json, atomic_write_text, ensure_safe_subpath
from motionlab.planner import build_edit_plan, validate_plan
from motionlab.preflight import find_binary, probe_media
from motionlab.registry import load_effects
from motionlab.renderers.base import RendererError, probe_all_engine_capabilities
from motionlab.renderers.ffmpeg import FFmpegRenderer
from motionlab.renderers.hyperframes import HyperFramesRenderer
from motionlab.renderers.remotion import RemotionRenderer
from motionlab.resource import ResourceBudget, run_bounded_async
from motionlab.schemas import AnalysisResult, EditPlan, EffectEvent, MediaInfo, RenderResult, ValidationResult
from motionlab.validation import validate_render_output


def _current_git_sha() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if proc.returncode == 0:
            return proc.stdout.strip()
    except Exception:
        pass
    return "unknown"


def create_unique_run_dir(output_root: Path, short_hash: str) -> tuple[str, Path]:
    """Create a unique, collision-safe run directory (`YYYYMMDDTHHMMSSZ_<hash>_<seq>`)."""
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for seq in range(1, 1000):
        run_id = f"{ts}_{short_hash[:8]}_{seq:03d}"
        cand = output_root / run_id
        try:
            cand.mkdir(parents=False, exist_ok=False)
            (cand / "evidence").mkdir(exist_ok=True)
            (cand / "logs").mkdir(exist_ok=True)
            (cand / "temp").mkdir(exist_ok=True)
            return run_id, cand
        except FileExistsError:
            continue
    raise RuntimeError(f"Could not allocate unique run_id in {output_root}")


def _write_run_readme(
    run_dir: Path,
    run_id: str,
    media: MediaInfo,
    config: EditConfig,
    plan: EditPlan,
    render_res: RenderResult | None,
    val_res: ValidationResult | None,
    extra_notes: list[str] | None = None,
) -> None:
    lines = [
        f"# MotionLab AI ML-01 Run — `{run_id}`",
        "",
        "## Summary",
        f"- **Engine Version:** `{__version__}` (schema `{SCHEMA_VERSION}`)",
        f"- **Source File (local):** `{Path(media.source_path).name}` (`{media.display_width}x{media.display_height}` @ `{media.fps_num}/{media.fps_den} fps`, `{media.duration_s:.3f}s`)",
        f"- **Source SHA-256:** `{media.input_hash}`",
        f"- **Profile:** `{plan.profile}` (requested: `{plan.requested_profile}`)",
        f"- **Seed:** `{plan.seed}` | **Config Hash:** `{plan.config_hash}` | **Plan Hash:** `{plan.plan_hash()}`",
        f"- **Target Format:** `{plan.width}x{plan.height}` (`aspect={config.aspect}`, `quality={config.quality}`)",
        f"- **Audio Policy:** `HARD_LOCK` (`mode={plan.audio_policy.get('mode')}`, no retime/filters/fades/loudnorm/subtitles/music)",
        f"- **Offline Runtime:** `true` (zero API keys, zero LLM/cloud calls)",
        "",
        "## Planned Effects",
        f"- **Plan Status:** `{plan.status}` ({len(plan.events)} event(s))",
    ]
    for ev in plan.events:
        backend = (render_res.event_backends.get(ev.id, ev.engine) if render_res else ev.engine)
        lines.append(
            f"  - `{ev.id}`: `{ev.effect_id}` (`{ev.layer}`, frames `{ev.start_frame}..{ev.end_frame_exclusive}`, backend=`{backend}`)"
        )
    if render_res:
        lines.extend(
            [
                "",
                "## Render & Resource Metrics",
                f"- **Render Status:** `{render_res.status}`",
                f"- **Engines Used:** `{', '.join(render_res.engines_used) or 'none'}`",
                f"- **Audio Mode:** `{render_res.audio_mode}`",
                f"- **Peak RSS:** `{render_res.peak_rss_mb} MB`",
                f"- **Peak Temp Disk:** `{render_res.peak_temp_bytes / (1024*1024):.2f} MB`",
                f"- **Stage Timings (s):** `{json.dumps(render_res.timings_s, sort_keys=True)}`",
            ]
        )
        if render_res.fallback_history:
            lines.append(f"- **Fallbacks:** `{json.dumps(render_res.fallback_history)}`")
    if val_res:
        val_d = val_res.to_dict()
        lines.extend(
            [
                "",
                "## Technical Validation & Honest Aesthetic Status",
                f"- **Technical Gate:** `{val_res.overall_status}` (`{val_d['summary']}`)",
                f"- **Human Aesthetic Review:** `{val_res.aesthetic_status}` (structural/technical pass does NOT certify aesthetic approval; review `final.mp4` and `contact_sheet.jpg`)",
            ]
        )
        for chk in val_res.checks:
            lines.append(f"  - [{chk.status}] `{chk.name}` — expected `{chk.expected}`, actual `{chk.actual}`")
    if extra_notes:
        lines.extend(["", "## Notes & Limitations"])
        for n in extra_notes:
            lines.append(f"- {n}")
    lines.extend(
        [
            "",
            "## Artifacts",
            "- `manifest.json`, `analysis.json`, `edit_plan.json`, `plan_explain.json`",
            "- `render_manifest.json`, `validation.json`, `contact_sheet.jpg`, `evidence/`",
            "- `final.mp4` (present ONLY when technical validation passes)",
            "",
        ]
    )
    atomic_write_text(run_dir / "README.md", "\n".join(lines))


async def _render_plan_async(
    source_path: Path,
    media: MediaInfo,
    analysis: AnalysisResult,
    plan: EditPlan,
    config: EditConfig,
    run_dir: Path,
    budget: ResourceBudget,
    log_lines: list[str],
    output_filename: str = "final.mp4",
) -> tuple[RenderResult, ValidationResult, dict[str, Any]]:
    """Execute multi-engine render + hard-locked audio mux + validation + atomic promotion."""
    temp_dir = run_dir / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    ff = FFmpegRenderer()
    rem = RemotionRenderer(license_allowed=config.allow_remotion)
    hf = HyperFramesRenderer(enabled=config.allow_hyperframes)

    timings: dict[str, float] = {}
    fallback_history: list[dict[str, str]] = []
    event_backends: dict[str, str] = {}
    engines_used: list[str] = ["ffmpeg"]
    warnings: list[str] = list(plan.warnings)

    ffmpeg_events: list[EffectEvent] = []
    external_overlay_events: list[EffectEvent] = []
    for ev in plan.events:
        if ev.engine in ("remotion", "hyperframes") and ev.layer == "overlay":
            external_overlay_events.append(ev)
        else:
            ffmpeg_events.append(ev)
            event_backends[ev.id] = "ffmpeg"

    # Step 1: Render external Remotion / HyperFrames alpha overlays under bounded async concurrency
    rendered_overlays: list[tuple[EffectEvent, Path]] = []
    if external_overlay_events:
        t_ov = time.monotonic()

        def _render_one_external(ev: EffectEvent) -> tuple[EffectEvent, Path | None, str | None]:
            out_webm = temp_dir / f"overlay_{ev.engine}_{ev.id}.webm"
            try:
                if ev.engine == "remotion":
                    rem.render_overlay_event(ev, plan, analysis, out_webm, temp_dir, budget=budget)
                elif ev.engine == "hyperframes":
                    hf.render_overlay_event(ev, plan, analysis, out_webm, temp_dir, budget=budget)
                else:
                    raise RendererError(f"Unexpected external engine {ev.engine}")
                return ev, out_webm, None
            except Exception as exc:
                return ev, None, str(exc)

        tasks = [(ev.id, (lambda e=ev: _render_one_external(e))) for ev in external_overlay_events]
        # Headless browser renders strictly capped by render_workers (default 1) to protect RAM
        ov_results = await run_bounded_async(tasks, max_concurrency=max(1, budget.snapshot.render_workers if budget.snapshot else 1))
        for ev in external_overlay_events:
            _, ov_path, err = ov_results[ev.id]
            if ov_path is not None and err is None:
                rendered_overlays.append((ev, ov_path))
                event_backends[ev.id] = ev.engine
                engines_used.append(ev.engine)
                log_lines.append(f"[overlay] {ev.id} ({ev.effect_id}) rendered via {ev.engine} -> {ov_path.name}")
            else:
                if config.strict:
                    raise RendererError(
                        f"Strict mode failed: external engine '{ev.engine}' failed for event {ev.id} ({ev.effect_id}): {err}"
                    )
                fallback_history.append(
                    {
                        "event_id": ev.id,
                        "effect_id": ev.effect_id,
                        "from_engine": ev.engine,
                        "to_engine": "ffmpeg",
                        "reason": (err or "unknown_error")[:300],
                    }
                )
                warnings.append(f"fallback:{ev.id}:{ev.engine}->ffmpeg")
                log_lines.append(f"[fallback] {ev.id} ({ev.effect_id}): {ev.engine} -> ffmpeg ({err})")
                ffmpeg_events.append(ev)
                event_backends[ev.id] = "ffmpeg"
        timings["external_overlays_s"] = round(time.monotonic() - t_ov, 4)

    budget.check_runtime_disk_guard()

    # Step 2: Streaming FFmpeg visual render (zero audio)
    video_base = temp_dir / "visual_base.mp4"
    vis_stats = ff.render_visual_track(
        source_path=source_path,
        media=media,
        analysis=analysis,
        plan=plan,
        ffmpeg_events=ffmpeg_events,
        out_video_only=video_base,
        config=config,
        budget=budget,
    )
    timings["visual_stream_s"] = vis_stats["elapsed_s"]
    log_lines.append(f"[visual] rendered {vis_stats['frames_written']} frames in {vis_stats['elapsed_s']}s")

    # Step 3: Composite external alpha overlays (if any succeeded)
    video_for_mux = video_base
    if rendered_overlays:
        video_comp = temp_dir / "visual_composited.mp4"
        comp_stats = ff.composite_alpha_overlays(
            base_video=video_base,
            overlays=rendered_overlays,
            out_video=video_comp,
            plan=plan,
            config=config,
            budget=budget,
        )
        timings["alpha_composite_s"] = comp_stats["elapsed_s"]
        video_for_mux = video_comp
        log_lines.append(f"[composite] composited {len(rendered_overlays)} external overlay(s) in {comp_stats['elapsed_s']}s")

    # Step 4: Hard-locked audio mux into partial file
    partial_path = run_dir / f"{Path(output_filename).stem}.partial.mp4"
    mux_stats = ff.mux_with_source_audio(
        video_only_path=video_for_mux,
        source_path=source_path,
        media=media,
        plan=plan,
        out_partial_mp4=partial_path,
        budget=budget,
    )
    timings["audio_mux_s"] = mux_stats["elapsed_s"]
    budget.sample_metrics()

    out_probe = ff.verify_artifact(partial_path, expected_duration_s=media.duration_s)
    render_res = RenderResult(
        artifact_path=output_filename,
        engines_used=sorted(set(engines_used)),
        event_backends=event_backends,
        timings_s=timings,
        ffprobe_info=out_probe,
        status="PASS",
        audio_mode=str(mux_stats["audio_mode"]),
        warnings=sorted(set(warnings)),
        fallback_history=fallback_history,
        peak_rss_mb=budget.peak_rss_mb,
        peak_temp_bytes=budget.peak_temp_bytes,
    )

    # Step 5: Technical validation BEFORE atomic rename to final.mp4
    t_val = time.monotonic()
    val_res, sheet_info = validate_render_output(
        source_path=source_path,
        media=media,
        plan=plan,
        candidate_output_path=partial_path,
        render_result=render_res,
        run_dir=run_dir,
    )
    timings["validation_s"] = round(time.monotonic() - t_val, 4)

    final_path = run_dir / output_filename
    if val_res.overall_status == "TECHNICAL_PASS":
        os.replace(partial_path, final_path)
        log_lines.append(f"[promote] validated TECHNICAL_PASS -> atomic rename to {final_path.name}")
    else:
        render_res.status = val_res.overall_status
        quarantine = run_dir / f"QUARANTINED_{partial_path.name}"
        os.replace(partial_path, quarantine)
        render_res.artifact_path = quarantine.name
        log_lines.append(f"[quarantine] validation {val_res.overall_status} -> quarantined as {quarantine.name}")

    return render_res, val_res, sheet_info


async def run_edit_pipeline_async(
    input_path: Path,
    output_root: Path,
    config: EditConfig,
    preloaded_plan: EditPlan | None = None,
) -> dict[str, Any]:
    """Execute the complete ML-01 pipeline asynchronously with resource governance & atomic artifacts."""
    t_total = time.monotonic()
    config.validate()
    log_lines: list[str] = []

    # Step 1: Preflight probe
    t_pre = time.monotonic()
    media = probe_media(input_path, allow_hdr_tonemap=config.allow_hdr_tonemap)
    log_lines.append(
        f"[preflight] {Path(media.source_path).name}: {media.display_width}x{media.display_height} "
        f"@ {media.fps_num}/{media.fps_den}fps, {media.duration_s:.3f}s, sha256={media.input_hash[:12]}"
    )

    # Allocate unique run directory
    run_id, run_dir = create_unique_run_dir(output_root, f"{media.input_hash[:8]}")
    temp_dir = run_dir / "temp"

    # Step 2: Engine capabilities (strictly offline)
    engine_caps = probe_all_engine_capabilities(
        allow_remotion=config.allow_remotion,
        allow_hyperframes=config.allow_hyperframes,
    )
    if config.strict and config.preferred_engine != "auto":
        req_cap = engine_caps.get(config.preferred_engine) or {}
        if not req_cap.get("available"):
            raise RendererError(
                f"Strict mode rejected: requested engine '{config.preferred_engine}' unavailable ({req_cap.get('reason')})"
            )

    # Step 3: ResourceBudget preflight
    budget = ResourceBudget(config=config.resources, dest_dir=run_dir, temp_dir=temp_dir)
    target_w = media.display_width if config.aspect == "source" else (540 if config.quality == "preview" else 1080)
    target_h = media.display_height if config.aspect == "source" else (960 if config.quality == "preview" else 1920)
    res_snap = budget.preflight_budget(
        duration_s=media.duration_s,
        width=target_w,
        height=target_h,
        fps=media.fps_float,
        quality=config.quality,
        overlay_events=4,
    )
    log_lines.append(f"[budget] {json.dumps(res_snap.to_dict(), sort_keys=True)}")

    # Step 4: Local signal analysis (concurrent video + audio, cached unless disabled)
    t_ana = time.monotonic()
    analysis, cache_hit = await analyze_media_async(
        media=media,
        edit_config=config,
        work_dir=temp_dir,
        budget=budget,
    )
    ana_s = round(time.monotonic() - t_ana, 4)
    atomic_write_json(run_dir / "analysis.json", analysis.to_dict())
    log_lines.append(
        f"[analyze] cache_hit={cache_hit} cuts={len(analysis.cuts)} peaks={len(analysis.energy_peaks)} "
        f"subjects={len(analysis.subject_regions)} candidates={len(analysis.candidates)} ({ana_s}s)"
    )

    # Step 5: Planner (or replay preloaded plan)
    t_plan = time.monotonic()
    if preloaded_plan is not None:
        validate_plan(preloaded_plan)
        plan = preloaded_plan
        plan_explain: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "mode": "replay_external_plan",
            "plan_hash": plan.plan_hash(),
            "event_count": len(plan.events),
        }
    else:
        plan, plan_explain = build_edit_plan(
            media=media,
            analysis=analysis,
            config=config,
            engine_capabilities=engine_caps,
        )
    plan_s = round(time.monotonic() - t_plan, 4)
    atomic_write_json(run_dir / "edit_plan.json", plan.to_dict())
    atomic_write_json(run_dir / "plan_explain.json", plan_explain)
    log_lines.append(
        f"[plan] status={plan.status} profile={plan.profile} events={len(plan.events)} plan_hash={plan.plan_hash()} ({plan_s}s)"
    )

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "motionlab_version": __version__,
        "git_sha": _current_git_sha(),
        "source_filename": Path(media.source_path).name,
        "source_sha256": media.input_hash,
        "config": config.canonical_dict(),
        "config_hash": config.config_hash(),
        "plan_hash": plan.plan_hash(),
        "seed": config.seed,
        "analysis_cache_hit": cache_hit,
        "media_info": media.to_dict(),
        "engine_capabilities": engine_caps,
        "resource_snapshot": res_snap.to_dict(),
        "timings_s": {
            "preflight_s": round(time.monotonic() - t_pre - ana_s - plan_s, 4),
            "analyze_s": ana_s,
            "plan_s": plan_s,
        },
        "dry_run": config.dry_run,
    }

    if config.dry_run:
        manifest["status"] = "DRY_RUN_PLANNED"
        atomic_write_json(run_dir / "manifest.json", manifest)
        _write_run_readme(run_dir, run_id, media, config, plan, None, None, extra_notes=["Dry-run mode: no final video rendered."])
        atomic_write_text(run_dir / "logs" / "pipeline.log", "\n".join(log_lines) + "\n")
        budget.cleanup_owned_temp(run_dir)
        return {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "status": "DRY_RUN_PLANNED",
            "plan_hash": plan.plan_hash(),
            "events": len(plan.events),
            "final_path": None,
        }

    # Step 6: Render + Validate + Atomic Promote
    render_res, val_res, sheet_info = await _render_plan_async(
        source_path=Path(media.source_path),
        media=media,
        analysis=analysis,
        plan=plan,
        config=config,
        run_dir=run_dir,
        budget=budget,
        log_lines=log_lines,
        output_filename="final.mp4",
    )

    # Optional fresh 9:16 preview render in same run dir if requested
    preview_rel: str | None = None
    if config.render_preview_9x16 and val_res.overall_status == "TECHNICAL_PASS":
        prev_cfg = EditConfig(
            profile=plan.profile,
            seed=config.seed,
            aspect="9:16",
            quality="preview",
            offline=True,
            strict=False,
            audio_track=config.audio_track,
            preferred_engine=config.preferred_engine,
            allow_remotion=config.allow_remotion,
            allow_hyperframes=config.allow_hyperframes,
            allow_hdr_tonemap=config.allow_hdr_tonemap,
            aspect_bg_mode=config.aspect_bg_mode,
            density_scale=config.density_scale,
            intensity_scale=config.intensity_scale,
            analysis=config.analysis,
            resources=config.resources,
        )
        prev_plan, _ = build_edit_plan(media, analysis, prev_cfg, engine_caps)
        prev_res, prev_val, _ = await _render_plan_async(
            source_path=Path(media.source_path),
            media=media,
            analysis=analysis,
            plan=prev_plan,
            config=prev_cfg,
            run_dir=run_dir,
            budget=budget,
            log_lines=log_lines,
            output_filename="preview_9x16.mp4",
        )
        if prev_val.overall_status == "TECHNICAL_PASS":
            preview_rel = "preview_9x16.mp4"
            manifest["preview_9x16"] = {
                "path": preview_rel,
                "width": prev_plan.width,
                "height": prev_plan.height,
                "plan_hash": prev_plan.plan_hash(),
                "status": prev_val.overall_status,
            }

    manifest["timings_s"].update(render_res.timings_s)
    manifest["timings_s"]["total_s"] = round(time.monotonic() - t_total, 4)
    manifest["status"] = val_res.overall_status
    manifest["aesthetic_status"] = val_res.aesthetic_status
    manifest["final_artifact"] = render_res.artifact_path
    manifest["contact_sheet"] = sheet_info

    atomic_write_json(run_dir / "render_manifest.json", render_res.to_dict())
    atomic_write_json(run_dir / "validation.json", val_res.to_dict())
    atomic_write_json(run_dir / "manifest.json", manifest)
    atomic_write_text(run_dir / "logs" / "pipeline.log", "\n".join(log_lines) + "\n")
    _write_run_readme(run_dir, run_id, media, config, plan, render_res, val_res)

    # Clean up ONLY run-owned temp/ safely
    budget.cleanup_owned_temp(run_dir, keep_on_fail=False)

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "status": val_res.overall_status,
        "aesthetic_status": val_res.aesthetic_status,
        "plan_hash": plan.plan_hash(),
        "events": len(plan.events),
        "engines_used": render_res.engines_used,
        "audio_mode": render_res.audio_mode,
        "final_path": str(run_dir / "final.mp4") if val_res.overall_status == "TECHNICAL_PASS" else None,
        "preview_9x16_path": str(run_dir / preview_rel) if preview_rel else None,
        "contact_sheet_path": str(run_dir / "contact_sheet.jpg"),
        "peak_rss_mb": render_res.peak_rss_mb,
        "peak_temp_bytes": render_res.peak_temp_bytes,
    }


def run_edit_pipeline(
    input_path: Path,
    output_root: Path,
    config: EditConfig,
    preloaded_plan: EditPlan | None = None,
) -> dict[str, Any]:
    return asyncio.run(run_edit_pipeline_async(input_path, output_root, config, preloaded_plan=preloaded_plan))


def run_validate_existing_run(run_dir: Path) -> ValidationResult:
    """Re-run full technical validation on an existing run folder."""
    run_dir = run_dir.resolve()
    manifest_p = run_dir / "manifest.json"
    plan_p = run_dir / "edit_plan.json"
    rm_p = run_dir / "render_manifest.json"
    final_p = run_dir / "final.mp4"
    if not manifest_p.exists() or not plan_p.exists() or not final_p.exists():
        raise FileNotFoundError(f"Run directory missing manifest.json, edit_plan.json, or final.mp4: {run_dir}")

    manifest = json.loads(manifest_p.read_text(encoding="utf-8"))
    media = MediaInfo.from_dict(manifest["media_info"])
    plan = EditPlan.from_dict(json.loads(plan_p.read_text(encoding="utf-8")))
    rm_data = json.loads(rm_p.read_text(encoding="utf-8")) if rm_p.exists() else {}
    render_res = RenderResult(
        artifact_path="final.mp4",
        engines_used=rm_data.get("engines_used", ["ffmpeg"]),
        event_backends=rm_data.get("event_backends", {}),
        timings_s=rm_data.get("timings_s", {}),
        ffprobe_info=rm_data.get("ffprobe_info", {}),
        status=rm_data.get("status", "PASS"),
        audio_mode=rm_data.get("audio_mode", plan.audio_policy.get("mode", "none")),
    )
    source_path = Path(media.source_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Original source file referenced by manifest not found: {source_path}")

    val_res, _ = validate_render_output(
        source_path=source_path,
        media=media,
        plan=plan,
        candidate_output_path=final_p,
        render_result=render_res,
        run_dir=run_dir,
    )
    atomic_write_json(run_dir / "validation.json", val_res.to_dict())
    return val_res


def build_effects_gallery(output_root: Path) -> dict[str, Any]:
    """Phase G4 — Render real frames for ALL 40 registered effects on a legal procedural fixture + build gallery montage."""
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw
    from motionlab.effects.implementations import FrameRenderContext, apply_effect_frame

    reg = load_effects()
    specs = reg.all_specs()
    run_id, run_dir = create_unique_run_dir(output_root, "gallery40")

    # Build a crisp, rich procedural 360x640 (9:16) dual-shot test frame pair (shot A + shot B)
    w, h = 270, 480
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    shot_a = np.zeros((h, w, 3), dtype=np.uint8)
    shot_a[:, :, 0] = np.clip(45 + 140 * (xx / w), 0, 255).astype(np.uint8)
    shot_a[:, :, 1] = np.clip(65 + 120 * (yy / h), 0, 255).astype(np.uint8)
    shot_a[:, :, 2] = np.clip(110 + 90 * np.sin(xx * 0.03), 0, 255).astype(np.uint8)
    # Crisp geometric subject proxy + bright highlight zone so optical/camera/mask effects show clearly
    cv2.circle(shot_a, (w // 2, int(h * 0.42)), 58, (235, 198, 172), -1)
    cv2.rectangle(shot_a, (w // 2 - 48, int(h * 0.56)), (w // 2 + 48, int(h * 0.78)), (48, 82, 130), -1)
    cv2.circle(shot_a, (int(w * 0.22), int(h * 0.18)), 22, (250, 246, 232), -1)
    for gx in range(0, w, 30):
        cv2.line(shot_a, (gx, 0), (gx, h), (220, 220, 230), 1)

    shot_b = np.zeros((h, w, 3), dtype=np.uint8)
    shot_b[:, :, 0] = np.clip(150 - 90 * (yy / h), 0, 255).astype(np.uint8)
    shot_b[:, :, 1] = np.clip(90 + 110 * (xx / w), 0, 255).astype(np.uint8)
    shot_b[:, :, 2] = np.clip(60 + 140 * (yy / h), 0, 255).astype(np.uint8)
    cv2.rectangle(shot_b, (40, 120), (w - 40, h - 120), (240, 235, 220), 4)

    ctx = FrameRenderContext(
        width=w,
        height=h,
        fps=30.0,
        seed=42,
        audio_envelope_by_frame={f: 0.35 + 0.55 * abs(np.sin(f * 0.25)) for f in range(0, 60)},
        subject_track_by_frame={f: (0.5 + 0.04 * np.sin(f * 0.1), 0.42) for f in range(0, 60)},
        recent_frames={f: np.roll(shot_a, shift=3, axis=1) for f in range(0, 60)},
    )

    records: list[dict[str, Any]] = []
    tiles: list[tuple[str, Image.Image]] = []
    verified_count = 0

    for spec in specs:
        params = reg.validate_params(spec.id, dict(spec.default_params))
        ev = EffectEvent(
            id=f"gal_{spec.code.lower()}",
            effect_id=spec.id,
            effect_version=spec.version,
            engine="ffmpeg",
            start_frame=0,
            end_frame_exclusive=30,
            layer=spec.layer,
            region={"cx": 0.5, "cy": 0.42, "w": 1.0, "h": 1.0} if spec.family != "motion_graphics" else {"cx": 0.84, "cy": 0.86, "w": 0.24, "h": 0.18},
            parameters=params,
            easing="ease_in_out_cubic",
            priority=10,
            reason_codes=["gallery_verification"],
            evidence_refs=["procedural_fixture"],
        )
        ctx.cut_anchor_frames[ev.id] = shot_b.copy()
        mid_out = apply_effect_frame(shot_a.copy(), 15, ev, ctx)
        diff = np.abs(mid_out.astype(np.float32) - shot_a.astype(np.float32))
        mad = float(np.mean(diff))
        max_delta = float(np.max(diff))
        changed_pixels = int(np.sum(np.max(diff, axis=2) >= 6.0))

        # Zero-intensity invariant test
        zero_params = dict(params)
        zero_params["intensity"] = 0.0
        ev_zero = EffectEvent(
            **{**ev.to_dict(), "id": f"gal_{spec.code.lower()}_z", "parameters": zero_params}
        )
        zero_out = apply_effect_frame(shot_a.copy(), 15, ev_zero, ctx)
        zero_mad = float(np.mean(np.abs(zero_out.astype(np.float32) - shot_a.astype(np.float32))))

        # Scene-specific verification: global frame transform/grade OR localized crisp peripheral graphic/highlight
        visible_change = (mad >= 0.15) or (changed_pixels >= 60 and max_delta >= 12.0)
        ok = visible_change and (zero_mad == 0.0) and (mid_out.shape == shot_a.shape) and (float(np.mean(mid_out)) > 5.0)
        if ok:
            verified_count += 1
        records.append(
            {
                "code": spec.code,
                "id": spec.id,
                "family": spec.family,
                "primary_engine": spec.engine,
                "verified_frame_delta_mad": round(mad, 3),
                "changed_pixels": changed_pixels,
                "max_pixel_delta": round(max_delta, 2),
                "zero_intensity_mad": round(zero_mad, 6),
                "status": "VERIFIED" if ok else "FAIL",
            }
        )
        thumb = Image.fromarray(mid_out).resize((135, 240), Image.Resampling.LANCZOS)
        tiles.append((f"{spec.code} {spec.id[:16]}", thumb))

    # Assemble 8x5 montage sheet of all 40 effects
    cols = 8
    rows = (len(tiles) + cols - 1) // cols
    pad = 8
    tw, th = 135, 240
    label_h = 20
    header_h = 44
    sheet = Image.new("RGB", (cols * (tw + pad) + pad, header_h + rows * (th + label_h + pad) + pad), (15, 17, 22))
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (pad, 12),
        f"MotionLab AI ML-01 Effect Gallery | {verified_count}/{len(specs)} Verified Distinct Effects (Zero-Intensity No-Op Verified)",
        fill=(242, 244, 250),
    )
    for idx, (lbl, im) in enumerate(tiles):
        r, c = divmod(idx, cols)
        x = pad + c * (tw + pad)
        y = header_h + r * (th + label_h + pad)
        sheet.paste(im, (x, y))
        draw.text((x + 2, y + th + 3), lbl, fill=(210, 215, 225))

    sheet_path = run_dir / "effects_gallery_40.jpg"
    sheet.save(sheet_path, "JPEG", quality=90)
    manifest = {
        "run_id": run_id,
        "registered_count": len(specs),
        "verified_count": verified_count,
        "gallery_sheet": sheet_path.name,
        "effects": records,
    }
    atomic_write_json(run_dir / "gallery_manifest.json", manifest)
    shutil.rmtree(run_dir / "temp", ignore_errors=True)
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "registered_count": len(specs),
        "verified_count": verified_count,
        "gallery_sheet_path": str(sheet_path),
    }
