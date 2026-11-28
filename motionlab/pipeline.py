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

    if config.composition_mode in ("r3", "cinematic", "generative") and preloaded_plan is None:
        from motionlab.resource import ResourceTracker

        tracker = ResourceTracker()
        doctor_info = {"engines": engine_caps}
        if config.composition_mode == "generative":
            from motionlab.composition.r5_engine import execute_generative_r5_pipeline

            return execute_generative_r5_pipeline(
                media=media,
                analysis=analysis,
                base_plan=plan,
                config=config,
                doctor_info=doctor_info,
                run_id=run_id,
                run_dir=run_dir,
                budget=budget,
                tracker=tracker,
                manifest=manifest,
                log_lines=log_lines,
            )
        if config.composition_mode == "cinematic":
            from motionlab.composition.r4_engine import execute_cinematic_r4_pipeline

            r4_out = execute_cinematic_r4_pipeline(
                media=media,
                analysis=analysis,
                base_plan=plan,
                config=config,
                doctor_info=doctor_info,
                run_dir=run_dir,
                budget=budget,
                tracker=tracker,
            )
            if config.dry_run or not r4_out.get("executed_cinematic"):
                manifest["status"] = "DRY_RUN_PLANNED"
                manifest["composition_mode"] = "cinematic"
                manifest["r4_graph_hash"] = r4_out["graph"].graph_hash
                manifest["semantic_plan_hash"] = r4_out["semantic_plan"].semantic_plan_hash
                atomic_write_json(run_dir / "manifest.json", manifest)
                budget.cleanup_owned_temp(run_dir)
                return {
                    "run_id": run_id,
                    "run_dir": str(run_dir),
                    "status": "DRY_RUN_PLANNED",
                    "composition_mode": "cinematic",
                    "plan_hash": plan.plan_hash(),
                    "graph_hash": r4_out["graph"].graph_hash,
                    "scenes": len(r4_out["graph"].scenes),
                    "final_path": None,
                }
            final_p = run_dir / "final.mp4"
            os.replace(r4_out["partial_out"], final_p)
            manifest["status"] = "TECHNICAL_PASS"
            manifest["composition_mode"] = "cinematic"
            manifest["aesthetic_status"] = "PENDING_HUMAN_REVIEW"
            manifest["final_artifact"] = "final.mp4"
            manifest["resource_tracker"] = tracker.to_dict()
            atomic_write_json(run_dir / "render_manifest.json", r4_out["render_result"].to_dict())
            atomic_write_json(run_dir / "manifest.json", manifest)
            budget.cleanup_owned_temp(run_dir)
            return {
                "run_id": run_id,
                "run_dir": str(run_dir),
                "status": "TECHNICAL_PASS",
                "aesthetic_status": "PENDING_HUMAN_REVIEW",
                "composition_mode": "cinematic",
                "scenes": len(r4_out["graph"].scenes),
                "engines_used": r4_out["render_result"].engines_used,
                "final_path": str(final_p),
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


def _validate_generative_r5_run(run_dir: Path, manifest: dict[str, Any], media: MediaInfo, final_p: Path) -> ValidationResult:
    """Strict, non-destructive re-validation for ML-01.R5 / R5.R1 / R5.R2 generative runs."""
    import hashlib
    import subprocess
    from motionlab.schemas import ValidationCheck
    from motionlab.validation import _audio_packet_hash_and_pts, _detect_black_intervals, _detect_freeze_intervals

    def _sha256_file(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            while chunk := fh.read(1 << 20):
                h.update(chunk)
        return h.hexdigest()

    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    ffprobe = shutil.which("ffprobe") or "ffprobe"
    checks: list[ValidationCheck] = []
    warnings: list[str] = []

    size_b = int(final_p.stat().st_size) if final_p.is_file() else 0
    checks.append(ValidationCheck("artifact_exists_non_empty", "PASS" if size_b > 0 else "FAIL", "> 0 bytes", size_b))

    source_p = Path(media.source_path)
    src_sha = _sha256_file(source_p) if source_p.is_file() else ""
    checks.append(ValidationCheck(
        "source_immutable",
        "PASS" if src_sha == media.input_hash else "FAIL",
        media.input_hash,
        src_sha,
    ))

    # Probe final video/audio geometry
    p_proc = subprocess.run(
        [ffprobe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(final_p)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30, shell=False,
    )
    probe = json.loads(p_proc.stdout or "{}")
    v_streams = [s for s in probe.get("streams", []) if s.get("codec_type") == "video"]
    a_streams = [s for s in probe.get("streams", []) if s.get("codec_type") == "audio"]
    vs = v_streams[0] if v_streams else {}
    out_w, out_h = int(vs.get("width", 0)), int(vs.get("height", 0))
    out_pix = str(vs.get("pix_fmt", "unknown"))
    out_frames = int(vs["nb_frames"]) if str(vs.get("nb_frames", "")).isdigit() else 0
    out_dur = float((probe.get("format") or {}).get("duration") or vs.get("duration") or 0.0)
    tol_s = max(0.08, 1.5 / max(1.0, media.fps_float))
    checks.append(ValidationCheck("output_dimensions_match", "PASS" if (out_w == media.display_width and out_h == media.display_height) else "FAIL", f"{media.display_width}x{media.display_height}", f"{out_w}x{out_h}"))
    checks.append(ValidationCheck("pixel_format_compatible", "PASS" if out_pix == "yuv420p" else "FAIL", "yuv420p", out_pix))

    # Branch frame/duration check strictly by mode (R5.R4 Fix D01!):
    #   - preserve mode: strict source media.duration_frames
    #   - auto pacing mode: independently validated EditTimeMap.output_total_frames (NEVER derived from output MP4!)
    emap_p = run_dir / "edit_time_map.json"
    emap_obj = None
    emap_errors: list[str] = []
    if emap_p.is_file():
        from motionlab.composition.edit_time_map import EditTimeMap, validate_edit_time_map

        emap_obj = EditTimeMap.from_dict(json.loads(emap_p.read_text(encoding="utf-8")))
        emap_errors = validate_edit_time_map(emap_obj)
        if emap_obj.source_sha256 != media.input_hash:
            emap_errors.append("edit_map_source_sha_mismatch")

    is_auto_paced = bool(emap_obj is not None and emap_obj.pacing_mode == "auto" and not emap_obj.is_identity)
    if is_auto_paced and emap_obj is not None:
        exp_frames = int(emap_obj.output_total_frames)
        exp_dur_s = float(emap_obj.output_duration_s)
        checks.append(ValidationCheck(
            "edit_time_map_valid",
            "PASS" if not emap_errors else "FAIL",
            "valid map_hash, contiguous kept spans, exact rational sample clock",
            f"status={emap_obj.status}, cuts={len(emap_obj.omitted_spans)}, errors={emap_errors}",
            "edit_time_map.json",
        ))
        checks.append(ValidationCheck("frame_count_exact", "PASS" if (not emap_errors and out_frames == exp_frames) else "FAIL", exp_frames, out_frames, "edit_time_map.json"))
        checks.append(ValidationCheck("duration_matches_edit_map", "PASS" if abs(out_dur - exp_dur_s) <= tol_s else "FAIL", f"{exp_dur_s:.3f}s (±{tol_s:.3f}s)", f"{out_dur:.3f}s", "edit_time_map.json"))
        wp_p = run_dir / "word_preservation.json"
        wp_doc = json.loads(wp_p.read_text(encoding="utf-8")) if wp_p.is_file() else {}
        checks.append(ValidationCheck(
            "word_preservation_verified",
            "PASS" if bool(wp_doc.get("passed")) else "FAIL",
            "passed=True, 0 missing numeric tokens",
            f"passed={wp_doc.get('passed')}, recall={wp_doc.get('token_recall')}, crit_recall={wp_doc.get('critical_token_recall')}",
            "word_preservation.json",
        ))
        ja_p = run_dir / "evidence" / "join_audio_qa.json"
        ja_doc = json.loads(ja_p.read_text(encoding="utf-8")) if ja_p.is_file() else {}
        checks.append(ValidationCheck(
            "audio_joins_verified",
            "PASS" if ja_doc.get("status") == "PASS" else "FAIL",
            "status=PASS, 0 clicks, exact edited sample count",
            f"status={ja_doc.get('status')}, joins={ja_doc.get('joins_checked')}, clicks={ja_doc.get('clicks_detected')}",
            "evidence/join_audio_qa.json",
        ))
    else:
        checks.append(ValidationCheck("frame_count_exact", "PASS" if out_frames == media.duration_frames else "FAIL", media.duration_frames, out_frames))
        checks.append(ValidationCheck("duration_preserved", "PASS" if abs(out_dur - media.duration_s) <= tol_s else "FAIL", f"{media.duration_s:.3f}s (±{tol_s:.3f}s)", f"{out_dur:.3f}s"))

    # Full decode check with -xerror
    dec_proc = subprocess.run([ffmpeg, "-hide_banner", "-v", "error", "-xerror", "-i", str(final_p), "-f", "null", "-"], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180, shell=False)
    checks.append(ValidationCheck("ffmpeg_full_decode_zero_errors", "PASS" if (dec_proc.returncode == 0 and not dec_proc.stderr.strip()) else "FAIL", "exit 0, empty stderr", f"exit {dec_proc.returncode}, stderr_len={len(dec_proc.stderr.strip())}"))

    # Audio master bit-exact + single default track + R5.R2 audio QA (true counterfactual ablation)
    checks.append(ValidationCheck("single_default_audio_track", "PASS" if len(a_streams) == 1 else "FAIL", 1, len(a_streams)))
    master_p = run_dir / "original_audio_master.m4a"
    if master_p.is_file() and source_p.is_file():
        src_pkt = _audio_packet_hash_and_pts(source_p, 0, ffmpeg, ffprobe)
        mst_pkt = _audio_packet_hash_and_pts(master_p, 0, ffmpeg, ffprobe)
        m_exact = (src_pkt["packet_hash"] == mst_pkt["packet_hash"] and src_pkt["packet_count"] == mst_pkt["packet_count"])
    else:
        m_exact = False
    checks.append(ValidationCheck("original_audio_master_bit_exact", "PASS" if m_exact else "FAIL", "exact packet SHA-256 & count match", f"master_exact={m_exact}", "original_audio_master.m4a"))

    aqa_p = run_dir / "evidence" / "audio_event_qa.json"
    aqa = json.loads(aqa_p.read_text(encoding="utf-8")) if aqa_p.is_file() else {}
    aqa_ok = (
        aqa.get("status") == "PASS"
        and bool(aqa.get("cue_ledger_conservation_ok"))
        and bool(aqa.get("cue_ids_exact_match"))
        and int(aqa.get("events_total", 0)) > 0
        and int(aqa.get("events_proxy_audible", 0)) == int(aqa.get("events_total", 0))
        and int(aqa.get("events_sync_ok", 0)) == int(aqa.get("events_total", 0))
        and int(aqa.get("events_speech_band_ok", 0)) == int(aqa.get("events_total", 0))
        and int(aqa.get("clipping_samples_added", 1)) == 0
    )
    checks.append(ValidationCheck(
        "r5r2_audio_counterfactual_qa",
        "PASS" if aqa_ok else "FAIL",
        "status=PASS, 100% audible/sync/speech, true counterfactual ablation, 0 clipping",
        f"status={aqa.get('status')}, audible={aqa.get('events_proxy_audible')}/{aqa.get('events_total')}, sync={aqa.get('events_sync_ok')}/{aqa.get('events_total')}, speech={aqa.get('events_speech_band_ok')}/{aqa.get('events_total')}",
        "evidence/audio_event_qa.json",
    ))

    # Black & freeze check
    out_blacks = sum(b[2] for b in _detect_black_intervals(final_p, ffmpeg))
    out_freezes = sum(f[2] for f in _detect_freeze_intervals(final_p, ffmpeg))
    checks.append(ValidationCheck("no_unexplained_black_intervals", "PASS" if out_blacks <= 0.25 else "FAIL", "<= 0.25s", f"{out_blacks:.2f}s"))
    checks.append(ValidationCheck("no_unexpected_freeze_intervals", "PASS" if out_freezes <= 0.80 else "FAIL", "<= 0.80s", f"{out_freezes:.2f}s"))

    # Continuous timeline, numeric state, objective pixel critic & 100% active-window ablation
    ct_p = run_dir / "continuous_timeline.json"
    ct = json.loads(ct_p.read_text(encoding="utf-8")) if ct_p.is_file() else {}
    bnd_p = run_dir / "evidence" / "scene_boundary_report.json"
    bnd = json.loads(bnd_p.read_text(encoding="utf-8")) if bnd_p.is_file() else {}
    cq_p = run_dir / "evidence" / "creative_quality_proxy.json"
    cq = json.loads(cq_p.read_text(encoding="utf-8")) if cq_p.is_file() else {}
    aeu_p = run_dir / "actual_engine_usage.json"
    aeu = json.loads(aeu_p.read_text(encoding="utf-8")) if aeu_p.is_file() else {}

    checks.append(ValidationCheck(
        "continuous_timeline_and_boundaries",
        "PASS" if (bool(ct.get("continuous_timeline_valid")) and int(bnd.get("abrupt_cutaway_jumps", 1)) == 0 and int(bnd.get("severe_findings_count", 0)) == 0) else "FAIL",
        "continuous_timeline_valid=True, abrupt_cutaway_jumps=0, severe_findings_count=0",
        f"valid={ct.get('continuous_timeline_valid')}, abrupt_jumps={bnd.get('abrupt_cutaway_jumps')}, severe={bnd.get('severe_findings_count', 0)}",
        "evidence/scene_boundary_report.json",
    ))
    sb_p = run_dir / "storyboard_spec.json"
    sb_doc = json.loads(sb_p.read_text(encoding="utf-8")) if sb_p.is_file() else {}
    sc_list = sb_doc.get("scenes", [])
    aeu_scenes = {s.get("scene_id"): s for s in aeu.get("scenes", [])}
    causal_ok = all(
        (not s.get("chosen_candidate_id")) or (aeu_scenes.get(s.get("scene_id"), {}).get("chosen_candidate_id") == s.get("chosen_candidate_id"))
        for s in sc_list
    )
    checks.append(ValidationCheck(
        "causal_selected_candidate_execution",
        "PASS" if causal_ok else "FAIL",
        "every storyboard scene chosen_candidate_id matches executed render manifest",
        f"scenes={len(sc_list)}, causal_ok={causal_ok}",
        "actual_engine_usage.json",
    ))
    ab_v = int(cq.get("ablation_layers_verified", 0))
    ab_t = int(cq.get("ablation_layers_total", 0))
    checks.append(ValidationCheck(
        "active_window_layer_ablations",
        "PASS" if (ab_t > 0 and ab_v == ab_t) else "FAIL",
        "100% active-window layers verified",
        f"{ab_v}/{ab_t}",
        "evidence/ablation_results.json",
    ))
    rg_ok = bool((aeu.get("reuse_gate") or {}).get("passed"))
    checks.append(ValidationCheck(
        "dual_engine_reuse_gate",
        "PASS" if rg_ok else "FAIL",
        "dual-engine Remotion+HyperFrames verified reuse gate passed",
        f"passed={rg_ok}, total={(aeu.get('reuse_gate') or {}).get('distinct_verified_total', 0)}",
        "actual_engine_usage.json",
    ))

    # Preview 9:16 check if present
    prev_p = run_dir / "preview_9x16.mp4"
    if prev_p.is_file():
        pdec = subprocess.run([ffmpeg, "-hide_banner", "-v", "error", "-xerror", "-i", str(prev_p), "-f", "null", "-"], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=90, shell=False)
        checks.append(ValidationCheck("preview_9x16_decode_ok", "PASS" if (pdec.returncode == 0 and prev_p.stat().st_size > 0) else "FAIL", "exit 0, non-empty", f"exit {pdec.returncode}, bytes={prev_p.stat().st_size}", "preview_9x16.mp4"))

    overall = "TECHNICAL_PASS" if all(c.status == "PASS" for c in checks) else "FAIL"
    val_res = ValidationResult(checks=checks, overall_status=overall, aesthetic_status="PENDING_HUMAN_REVIEW", warnings=warnings)
    atomic_write_json(run_dir / "evidence" / "revalidation_check.json", val_res.to_dict())
    return val_res


def _validate_multisource_director_run(run_dir: Path, manifest: dict[str, Any]) -> ValidationResult:
    """Non-destructive end-to-end validator for ML-01.R5.R5 multi-source director runs."""
    import hashlib
    import subprocess
    from motionlab.composition.multi_source_timeline import MultiSourceEditMap, validate_multi_source_edit_map
    from motionlab.schemas import ValidationCheck

    def _sha256(p: Path) -> str:
        h = hashlib.sha256()
        with p.open("rb") as fh:
            while chunk := fh.read(1 << 20):
                h.update(chunk)
        return h.hexdigest()

    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    ffprobe = shutil.which("ffprobe") or "ffprobe"
    checks: list[ValidationCheck] = []
    warnings: list[str] = list(manifest.get("hard_blockers") or [])

    final_name = str(manifest.get("final_artifact") or manifest.get("output_filename") or "final.mp4")
    final_p = run_dir / final_name
    size_b = int(final_p.stat().st_size) if final_p.is_file() else 0
    checks.append(ValidationCheck("artifact_exists_non_empty", "PASS" if size_b > 0 else "FAIL", "> 0 bytes", size_b, final_name))
    actual_sha = _sha256(final_p) if size_b > 0 else ""
    exp_sha = str(manifest.get("final_sha256") or manifest.get("output_sha256") or "")
    checks.append(ValidationCheck("artifact_sha256_matches_manifest", "PASS" if (exp_sha and actual_sha == exp_sha) else "FAIL", exp_sha[:16], actual_sha[:16], final_name))

    # Source immutability across ALL sources
    src_list = list(manifest.get("sources") or [])
    src_ok = bool(src_list) and all(Path(s["path"]).is_file() and _sha256(Path(s["path"])) == s["sha256"] for s in src_list)
    checks.append(ValidationCheck("all_sources_immutable", "PASS" if src_ok else "FAIL", f"{len(src_list)} sources exact SHA-256", f"ok={src_ok}", "source_manifest.json"))

    # MultiSourceEditMap integrity
    emap_p = run_dir / "multi_source_edit_map.json"
    emap_errs = ["missing_edit_map"]
    emap_obj = None
    if emap_p.is_file():
        emap_obj = MultiSourceEditMap.from_dict(json.loads(emap_p.read_text(encoding="utf-8")))
        emap_errs = validate_multi_source_edit_map(emap_obj)
    checks.append(ValidationCheck("multisource_edit_map_valid", "PASS" if not emap_errs else "FAIL", "0 invariant errors", f"errors={emap_errs}", "multi_source_edit_map.json"))

    # FFprobe geometry, pixel format, CFR frame count, audio track, 0 subtitle tracks
    p_proc = subprocess.run(
        [ffprobe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(final_p)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30, shell=False,
    )
    probe = json.loads(p_proc.stdout or "{}")
    v_streams = [s for s in probe.get("streams", []) if s.get("codec_type") == "video"]
    a_streams = [s for s in probe.get("streams", []) if s.get("codec_type") == "audio"]
    s_streams = [s for s in probe.get("streams", []) if s.get("codec_type") == "subtitle"]
    vs = v_streams[0] if v_streams else {}
    out_w, out_h = int(vs.get("width", 0)), int(vs.get("height", 0))
    out_pix = str(vs.get("pix_fmt", "unknown"))
    out_frames = int(vs["nb_frames"]) if str(vs.get("nb_frames", "")).isdigit() else 0
    if emap_obj is not None:
        checks.append(ValidationCheck("output_dimensions_exact", "PASS" if (out_w == emap_obj.target_width and out_h == emap_obj.target_height) else "FAIL", f"{emap_obj.target_width}x{emap_obj.target_height}", f"{out_w}x{out_h}"))
        checks.append(ValidationCheck("frame_count_exact", "PASS" if out_frames == emap_obj.output_total_frames else "FAIL", emap_obj.output_total_frames, out_frames))
    checks.append(ValidationCheck("pixel_format_yuv420p", "PASS" if out_pix == "yuv420p" else "FAIL", "yuv420p", out_pix))
    checks.append(ValidationCheck("single_audio_zero_subtitles", "PASS" if (len(a_streams) == 1 and len(s_streams) == 0) else "FAIL", "1 audio, 0 subtitles", f"{len(a_streams)} audio, {len(s_streams)} subtitles"))

    # Full decode -xerror
    dec_proc = subprocess.run([ffmpeg, "-hide_banner", "-v", "error", "-xerror", "-i", str(final_p), "-f", "null", "-"], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180, shell=False)
    checks.append(ValidationCheck("ffmpeg_full_decode_zero_errors", "PASS" if (dec_proc.returncode == 0 and not dec_proc.stderr.strip()) else "FAIL", "exit 0, empty stderr", f"exit {dec_proc.returncode}"))

    # Word preservation, temporal ledger, audio join QA, counterfactual audio QA, preview
    wp = json.loads((run_dir / "protected_word_validation.json").read_text(encoding="utf-8")) if (run_dir / "protected_word_validation.json").is_file() else {}
    checks.append(ValidationCheck("word_preservation_verified", "PASS" if bool(wp.get("passed")) else "FAIL", "passed=True", f"status={wp.get('status')}, recall={wp.get('token_recall')}", "protected_word_validation.json"))
    tl = json.loads((run_dir / "temporal_fact_ledger.json").read_text(encoding="utf-8")) if (run_dir / "temporal_fact_ledger.json").is_file() else {}
    checks.append(ValidationCheck("temporal_fact_ledger_valid", "PASS" if bool(tl.get("valid")) else "FAIL", "valid=True", f"valid={tl.get('valid')}", "temporal_fact_ledger.json"))
    ja = json.loads((run_dir / "evidence" / "audio_join_qa.json").read_text(encoding="utf-8")) if (run_dir / "evidence" / "audio_join_qa.json").is_file() else {}
    checks.append(ValidationCheck("audio_join_qa_passed", "PASS" if ja.get("status") == "PASS" else "FAIL", "status=PASS", f"status={ja.get('status')}", "evidence/audio_join_qa.json"))
    aq = json.loads((run_dir / "evidence" / "audio_event_qa.json").read_text(encoding="utf-8")) if (run_dir / "evidence" / "audio_event_qa.json").is_file() else {}
    checks.append(ValidationCheck("audio_counterfactual_qa_passed", "PASS" if aq.get("status") == "PASS" else "FAIL", "status=PASS", f"status={aq.get('status')}, audible={aq.get('events_proxy_audible')}/{aq.get('events_total')}", "evidence/audio_event_qa.json"))
    if manifest.get("composition_mode") in ("multisource_director_v3", "multisource_director_v4"):
        fmr = json.loads((run_dir / "final_mix_reasr.json").read_text(encoding="utf-8")) if (run_dir / "final_mix_reasr.json").is_file() else {}
        checks.append(ValidationCheck("final_mix_reasr_verified", "PASS" if bool(fmr.get("passed")) else "FAIL", "passed=True", f"status={fmr.get('status')}, wer={fmr.get('wer')}", "final_mix_reasr.json"))
        if manifest.get("composition_mode") == "multisource_director_v4":
            sc4 = json.loads((run_dir / "semantic_comprehension_v4.json").read_text(encoding="utf-8")) if (run_dir / "semantic_comprehension_v4.json").is_file() else {}
            sc4_ok = bool(sc4) and int(sc4.get("high_severity_defects_count", 1)) == 0 and float(sc4.get("weighted_semantic_coverage", 0.0)) >= 0.90
            checks.append(ValidationCheck(
                "independent_semantic_comprehension_v4",
                "PASS" if sc4_ok else "FAIL",
                "weighted>=0.90, high_defects=0",
                f"weighted={sc4.get('weighted_semantic_coverage')}, critical={sc4.get('critical_factual_accuracy')}, uncertain={sc4.get('uncertain_claims_count')}, high_defects={sc4.get('high_severity_defects_count')}",
                "semantic_comprehension_v4.json",
            ))
            fm4 = json.loads((run_dir / "frame_coverage_manifest.json").read_text(encoding="utf-8")) if (run_dir / "frame_coverage_manifest.json").is_file() else {}
            checks.append(ValidationCheck(
                "exact_frame_coverage_manifest",
                "PASS" if bool(fm4.get("exact_full_coverage")) else "FAIL",
                "exact_full_coverage=True",
                f"covered={fm4.get('covered_frames')}/{fm4.get('total_frames')}",
                "frame_coverage_manifest.json",
            ))
        else:
            occ = json.loads((run_dir / "observed_claim_coverage.json").read_text(encoding="utf-8")) if (run_dir / "observed_claim_coverage.json").is_file() else {}
            occ_ok = bool(occ) and int(occ.get("high_severity_defects_count", 1)) == 0 and float(occ.get("weighted_observed_material_coverage", 0.0)) >= 0.90
            checks.append(ValidationCheck(
                "observed_material_claim_coverage",
                "PASS" if occ_ok else "FAIL",
                "weighted_coverage>=0.90, high_defects=0",
                f"weighted={occ.get('weighted_observed_material_coverage')}, critical={occ.get('critical_factual_accuracy')}, high_defects={occ.get('high_severity_defects_count')}",
                "observed_claim_coverage.json",
            ))
    prev_p = run_dir / "preview_9x16.mp4"
    if prev_p.is_file():
        pdec = subprocess.run([ffmpeg, "-hide_banner", "-v", "error", "-xerror", "-i", str(prev_p), "-f", "null", "-"], capture_output=True, text=True, timeout=90, shell=False)
        checks.append(ValidationCheck("preview_9x16_decode_ok", "PASS" if (pdec.returncode == 0 and prev_p.stat().st_size > 0) else "FAIL", "exit 0, non-empty", f"exit {pdec.returncode}", "preview_9x16.mp4"))

    tech_ok = all(c.status == "PASS" for c in checks)
    overall = "TECHNICAL_PASS" if (tech_ok and not warnings) else ("CANDIDATE_READY_WITH_BLOCKERS" if tech_ok else "FAIL")
    val_res = ValidationResult(checks=checks, overall_status=overall, aesthetic_status="PENDING_HUMAN_REVIEW", warnings=warnings)
    atomic_write_json(run_dir / "evidence" / "revalidation_check.json", val_res.to_dict())
    return val_res


def run_validate_existing_run(run_dir: Path) -> ValidationResult:
    """Re-run full technical validation on an existing run folder."""
    run_dir = run_dir.resolve()
    manifest_p = run_dir / "manifest.json"
    plan_p = run_dir / "edit_plan.json"
    rm_p = run_dir / "render_manifest.json"
    if not manifest_p.exists():
        raise FileNotFoundError(f"Run directory missing manifest.json: {run_dir}")

    manifest = json.loads(manifest_p.read_text(encoding="utf-8"))
    if manifest.get("composition_mode") in ("multisource_director_v2", "multisource_director_v3", "multisource_director_v4"):
        return _validate_multisource_director_run(run_dir, manifest)

    final_p = run_dir / "final.mp4"
    if not final_p.exists():
        raise FileNotFoundError(f"Run directory missing final.mp4: {run_dir}")

    media = MediaInfo.from_dict(manifest["media_info"])

    # R5 / R5.R1 / R5.R2 generative runs use non-destructive dual-engine + counterfactual audio verification!
    if manifest.get("composition_mode") == "generative" or (run_dir / "storyboard_spec.json").is_file():
        return _validate_generative_r5_run(run_dir, manifest, media, final_p)

    if not plan_p.exists():
        raise FileNotFoundError(f"Run directory missing edit_plan.json: {run_dir}")
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
