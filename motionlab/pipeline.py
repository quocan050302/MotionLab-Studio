"""Main ML-01 end-to-end pipeline orchestrator, run folder lifecycle, gallery, and atomic output promotion."""

from __future__ import annotations

import datetime
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from motionlab import ANALYZER_VERSION, PLANNER_VERSION, SCHEMA_VERSION, __version__
from motionlab.analyzer import analyze_media
from motionlab.cache import AnalysisCache
from motionlab.config import MotionLabConfig, REPO_ROOT, run_doctor
from motionlab.planner import build_edit_plan, validate_plan
from motionlab.preflight import probe_media
from motionlab.registry import EffectRegistry, load_effects
from motionlab.renderers.ffmpeg import FFmpegRenderer
from motionlab.renderers.hyperframes import HyperFramesRenderer
from motionlab.renderers.remotion import RemotionRenderer
from motionlab.resource import (
    ResourceTracker,
    compute_resource_budget,
    preflight_budget,
    safe_cleanup_run_temp,
)
from motionlab.schemas import (
    AnalysisResult,
    EditPlan,
    MediaInfo,
    RenderResult,
    ValidationResult,
    atomic_write_json,
    atomic_write_text,
    ensure_contained_path,
)
from motionlab.validation import validate_rendered_output


def _git_head_sha() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True, text=True, timeout=5, shell=False,
        )
        if proc.returncode == 0:
            return proc.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def allocate_run_dir(output_root: Path, source_hash: str, plan_hash: str | None = None) -> tuple[str, Path]:
    """Allocate a guaranteed unique, collision-safe UTC run folder under output_root."""
    output_root.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = (plan_hash[:8] if plan_hash else source_hash[:8])
    for seq in range(1, 1000):
        run_id = f"{ts}_{tag}_{seq:03d}"
        cand = output_root / run_id
        try:
            cand.mkdir(parents=False, exist_ok=False)
            (cand / "evidence").mkdir(exist_ok=True)
            (cand / "logs").mkdir(exist_ok=True)
            (cand / "temp").mkdir(exist_ok=True)
            return run_id, cand.resolve()
        except FileExistsError:
            continue
    raise RuntimeError(f"Could not allocate unique run_id under {output_root}")


def _write_run_readme(
    run_dir: Path,
    run_id: str,
    media: MediaInfo,
    plan: EditPlan,
    render_res: RenderResult | None,
    val_res: ValidationResult | None,
    preview_rel: str | None,
    config: MotionLabConfig,
    command_str: str,
) -> None:
    lines = [
        f"# MotionLab-AI ML-01 Run — `{run_id}`",
        "",
        "## Summary",
        f"- **Technical Gate**: `{val_res.technical_status if val_res else 'DRY_RUN'}`",
        f"- **Aesthetic Review**: `{val_res.aesthetic_status if val_res else 'NOT_RENDERED'}` (requires human visual sign-off)",
        f"- **Profile**: `{plan.profile}` (`{plan.profile_selection_reason}`)",
        f"- **Plan Status**: `{plan.status}` (`{len(plan.events)}` planned effect event(s))",
        f"- **Plan Hash**: `{plan.plan_hash}`",
        f"- **Seed**: `{plan.seed}`",
        f"- **Offline Runtime**: `true` (zero API keys, zero cloud calls)",
        "",
        "## Input & Output",
        f"- **Source Hash (SHA-256)**: `{media.input_hash}`",
        f"- **Source Geometry**: `{media.display_width}x{media.display_height}` @ `{media.fps.to_ffmpeg()}` fps (`{media.duration_s:.3f}s`, `{media.duration_frames}` frames, VFR=`{media.vfr}`)",
        f"- **Source Audio**: `{len(media.audio_streams)}` stream(s) | Policy: `{plan.audio_policy.get('mode')}` (preserve master, no fades/retime/loudnorm)",
        f"- **Target Geometry**: `{plan.width}x{plan.height}` (`aspect_policy={plan.aspect_policy}`)",
        f"- **Final Deliverable**: `final.mp4` ({'PROMOTED' if (run_dir / 'final.mp4').is_file() else 'NOT_PROMOTED'})",
    ]
    if preview_rel:
        lines.append(f"- **Fresh 9:16 Preview**: `{preview_rel}`")
    if (run_dir / "contact_sheet.jpg").is_file():
        lines.append("- **Contact Sheet**: `contact_sheet.jpg`")
    lines.extend([
        "",
        "## Planned Effects",
    ])
    if not plan.events:
        lines.append("- _None (`NO_SAFE_EFFECTS` passthrough)_")
    else:
        for ev in plan.events:
            eng = render_res.event_engine_map.get(ev.id, ev.engine) if render_res else ev.engine
            lines.append(
                f"- `{ev.id}` (`{ev.effect_id}` via `{eng}`): frames `[{ev.start_frame}, {ev.end_frame_exclusive})` "
                f"layer=`{ev.layer}` reasons={ev.reason_codes}"
            )
    lines.extend([
        "",
        "## Technical Validation Checks",
    ])
    if val_res:
        for ch in val_res.checks:
            lines.append(f"- **{ch.name}**: `{ch.status}` — expected `{ch.expected}`, actual `{ch.actual}`")
    else:
        lines.append("- _Validation not run (`--dry-run` or plan-only)._")

    lines.extend([
        "",
        "## Human Aesthetic Review Checklist (PENDING)",
        "- [ ] Taste & restraint: effects enhance footage without clutter or repetitive cards",
        "- [ ] Pacing & rhythm: movements feel natural and match shot beats",
        "- [ ] Source fidelity: original character, subject, text/UI, and colors preserved",
        "- [ ] Audio & speech sync: voice matches lip/action timing with zero pops or drift",
        "",
        "## Scope & Limitations",
        "- Rule-based local signal engine; does NOT understand spoken semantics or narrative context.",
        "- No guaranteed aesthetic perfection across every niche without human review.",
        "- No voice generation, speech cutting, auto subtitles, stock footage, or default music.",
        "",
        "## Invocation",
        "```",
        command_str,
        "```",
        "",
    ])
    atomic_write_text(run_dir / "README.md", "\n".join(lines))


def run_analyze(
    input_path: Path,
    output_root: Path,
    config: MotionLabConfig,
) -> dict[str, Any]:
    """Execute preflight + local analysis (+ cache) and write artifacts to a new run dir."""
    config.validate()
    tracker = ResourceTracker()
    doctor_info = run_doctor(config)
    budget = compute_resource_budget(config.max_workers, config.max_temp_gb, config.min_free_gb)

    tracker.start_stage("preflight")
    media = probe_media(input_path)
    tracker.end_stage("preflight")

    run_id, run_dir = allocate_run_dir(output_root, media.input_hash)
    cache = AnalysisCache(Path(config.cache_dir) if config.cache_dir else None)
    ffmpeg_ver = str((doctor_info["tools"]["ffmpeg"] or {}).get("version") or "unknown")
    cache_key = cache.make_key(media, config, ffmpeg_ver)

    tracker.start_stage("analysis")
    cached = cache.get(cache_key) if config.cache_enabled else None
    cache_hit = cached is not None
    if cached is not None:
        analysis = cached
    else:
        analysis = analyze_media(media, config, analysis_workers=budget.analysis_workers)
        if config.cache_enabled:
            cache.put(cache_key, analysis)
    tracker.end_stage("analysis")

    atomic_write_json(run_dir / "media_info.json", media.to_dict())
    atomic_write_json(run_dir / "analysis.json", analysis.to_dict())
    atomic_write_json(
        run_dir / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "motionlab_version": __version__,
            "mode": "analyze",
            "run_id": run_id,
            "git_sha": _git_head_sha(),
            "source_hash": media.input_hash,
            "config_hash": config.config_hash(),
            "cache_hit": cache_hit,
            "cache_key": cache_key,
            "resource_stats": tracker.to_dict(),
        },
    )
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "analysis_path": str(run_dir / "analysis.json"),
        "cache_hit": cache_hit,
        "cuts": len(analysis.cuts),
        "energy_peaks": len(analysis.energy_peaks),
        "subject_regions": len(analysis.subject_regions),
    }


def run_plan(
    input_path: Path,
    output_root: Path,
    config: MotionLabConfig,
    registry: EffectRegistry | None = None,
) -> dict[str, Any]:
    """Execute preflight + analysis + deterministic rule-based planner without rendering."""
    config.validate()
    tracker = ResourceTracker()
    doctor_info = run_doctor(config)
    budget = compute_resource_budget(config.max_workers, config.max_temp_gb, config.min_free_gb)

    tracker.start_stage("preflight")
    media = probe_media(input_path)
    tracker.end_stage("preflight")

    cache = AnalysisCache(Path(config.cache_dir) if config.cache_dir else None)
    ffmpeg_ver = str((doctor_info["tools"]["ffmpeg"] or {}).get("version") or "unknown")
    cache_key = cache.make_key(media, config, ffmpeg_ver)

    tracker.start_stage("analysis")
    cached = cache.get(cache_key) if config.cache_enabled else None
    cache_hit = cached is not None
    if cached is not None:
        analysis = cached
    else:
        analysis = analyze_media(media, config, analysis_workers=budget.analysis_workers)
        if config.cache_enabled:
            cache.put(cache_key, analysis)
    tracker.end_stage("analysis")

    tracker.start_stage("planning")
    plan, explain = build_edit_plan(media, analysis, config, doctor_info["engines"], registry=registry)
    tracker.end_stage("planning")

    run_id, run_dir = allocate_run_dir(output_root, media.input_hash, plan.plan_hash)
    atomic_write_json(run_dir / "media_info.json", media.to_dict())
    atomic_write_json(run_dir / "analysis.json", analysis.to_dict())
    atomic_write_json(run_dir / "edit_plan.json", plan.to_dict())
    atomic_write_json(run_dir / "plan_explain.json", explain)
    atomic_write_json(
        run_dir / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "motionlab_version": __version__,
            "mode": "plan",
            "run_id": run_id,
            "git_sha": _git_head_sha(),
            "source_hash": media.input_hash,
            "config_hash": config.config_hash(),
            "plan_hash": plan.plan_hash,
            "seed": config.seed,
            "cache_hit": cache_hit,
            "resource_stats": tracker.to_dict(),
        },
    )
    _write_run_readme(
        run_dir, run_id, media, plan, None, None, None, config,
        f"python -m motionlab plan --input \"{input_path}\" --profile {config.profile} --seed {config.seed}",
    )
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "plan_path": str(run_dir / "edit_plan.json"),
        "plan_hash": plan.plan_hash,
        "profile": plan.profile,
        "status": plan.status,
        "events": len(plan.events),
    }


def _render_external_overlays(
    plan: EditPlan,
    analysis: AnalysisResult | None,
    config: MotionLabConfig,
    doctor_info: dict[str, Any],
    temp_dir: Path,
    registry: EffectRegistry,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Execute HyperFrames/Remotion overlay renders ONLY for events routed to those engines."""
    overlays: list[dict[str, Any]] = []
    fallbacks: list[dict[str, Any]] = []
    hf = HyperFramesRenderer()
    rem = RemotionRenderer()
    aud_list = [p.normalized_rms for p in analysis.audio_envelope] if (analysis and analysis.audio_envelope) else None

    for ev in plan.events:
        if ev.engine == "ffmpeg":
            continue
        spec = registry.get(ev.effect_id)
        if ev.engine == "hyperframes":
            if hf.supports(spec, MediaInfo if False else None, doctor_info["engines"]):  # type: ignore[arg-type]
                try:
                    res = hf.render_overlay_event(ev, plan, temp_dir, timeout_s=min(180.0, float(config.timeout_s)))
                    overlays.append(res)
                    continue
                except Exception as exc:
                    if config.strict:
                        raise RuntimeError(f"Strict mode: HyperFrames failed for {ev.id}: {exc}") from exc
                    fallbacks.append({
                        "event_id": ev.id,
                        "requested_engine": "hyperframes",
                        "fallback_engine": "ffmpeg",
                        "reason": str(exc)[:300],
                    })
                    ev.engine = "ffmpeg"
            else:
                if config.strict:
                    raise RuntimeError(f"Strict mode: HyperFrames unavailable for {ev.id}")
                fallbacks.append({
                    "event_id": ev.id,
                    "requested_engine": "hyperframes",
                    "fallback_engine": "ffmpeg",
                    "reason": "hyperframes_unavailable_or_unsupported",
                })
                ev.engine = "ffmpeg"
        elif ev.engine == "remotion":
            if rem.supports(spec, MediaInfo if False else None, doctor_info["engines"]):  # type: ignore[arg-type]
                try:
                    res = rem.render_overlay_event(
                        ev, plan, temp_dir, audio_envelope=aud_list, timeout_s=min(180.0, float(config.timeout_s))
                    )
                    overlays.append(res)
                    continue
                except Exception as exc:
                    if config.strict:
                        raise RuntimeError(f"Strict mode: Remotion failed for {ev.id}: {exc}") from exc
                    fallbacks.append({
                        "event_id": ev.id,
                        "requested_engine": "remotion",
                        "fallback_engine": "ffmpeg",
                        "reason": str(exc)[:300],
                    })
                    ev.engine = "ffmpeg"
            else:
                if config.strict:
                    raise RuntimeError(f"Strict mode: Remotion unavailable or not license-opted-in for {ev.id}")
                fallbacks.append({
                    "event_id": ev.id,
                    "requested_engine": "remotion",
                    "fallback_engine": "ffmpeg",
                    "reason": "remotion_unavailable_or_license_gated",
                })
                ev.engine = "ffmpeg"
    return overlays, fallbacks


def run_edit(
    input_path: Path,
    output_root: Path,
    config: MotionLabConfig,
    existing_plan: EditPlan | None = None,
    command_str: str = "",
) -> dict[str, Any]:
    """Full ML-01 end-to-end offline edit: preflight -> analyze -> plan -> render -> validate -> promote."""
    config.validate()
    tracker = ResourceTracker()
    doctor_info = run_doctor(config)
    if doctor_info["status"] != "OK":
        raise RuntimeError(f"Environment check failed: {doctor_info}")

    reg = load_effects()
    budget = compute_resource_budget(config.max_workers, config.max_temp_gb, config.min_free_gb)

    # Strict engine availability check
    if config.strict and config.preferred_engine != "auto":
        eng_info = doctor_info["engines"].get(config.preferred_engine, {})
        if not eng_info.get("available"):
            raise RuntimeError(
                f"Strict mode blocked: requested engine '{config.preferred_engine}' is not available "
                f"(status={eng_info})."
            )

    tracker.start_stage("preflight")
    media = probe_media(input_path)
    tracker.end_stage("preflight")

    # Strict HDR/VFR policy checks
    if media.is_hdr and not config.allow_hdr_tonemap and config.strict:
        raise RuntimeError("Strict mode blocked: HDR source without explicit --allow-hdr-tonemap.")
    if media.vfr and not config.allow_vfr_cfr_normalize and config.strict:
        raise RuntimeError("Strict mode blocked: VFR source with CFR normalization disabled.")

    # Analysis (with content-addressed cache)
    cache = AnalysisCache(Path(config.cache_dir) if config.cache_dir else None)
    ffmpeg_ver = str((doctor_info["tools"]["ffmpeg"] or {}).get("version") or "unknown")
    cache_key = cache.make_key(media, config, ffmpeg_ver)

    tracker.start_stage("analysis")
    cached = cache.get(cache_key) if config.cache_enabled else None
    cache_hit = cached is not None
    if cached is not None:
        analysis = cached
    else:
        analysis = analyze_media(media, config, analysis_workers=budget.analysis_workers)
        if config.cache_enabled:
            cache.put(cache_key, analysis)
    tracker.end_stage("analysis")

    # Planning (or replay validated existing plan)
    tracker.start_stage("planning")
    if existing_plan is not None:
        validate_plan(existing_plan, media, reg)
        plan = existing_plan
        explain = {"replayed_from_plan": True, "plan_hash": plan.plan_hash}
    else:
        plan, explain = build_edit_plan(media, analysis, config, doctor_info["engines"], registry=reg)
    tracker.end_stage("planning")

    run_id, run_dir = allocate_run_dir(output_root, media.input_hash, plan.plan_hash)
    temp_dir = run_dir / "temp"

    # Disk & resource budget preflight BEFORE writing big temp artifacts
    budget_report = preflight_budget(
        dest_dir=run_dir,
        temp_dir=temp_dir,
        duration_s=media.duration_s,
        width=plan.width,
        height=plan.height,
        fps=plan.fps.as_float(),
        budget=budget,
        include_preview=config.render_preview_9x16,
    )

    atomic_write_json(run_dir / "media_info.json", media.to_dict())
    atomic_write_json(run_dir / "analysis.json", analysis.to_dict())
    atomic_write_json(run_dir / "edit_plan.json", plan.to_dict())
    atomic_write_json(run_dir / "plan_explain.json", explain)

    if config.dry_run:
        atomic_write_json(
            run_dir / "manifest.json",
            {
                "schema_version": SCHEMA_VERSION,
                "motionlab_version": __version__,
                "mode": "dry_run",
                "run_id": run_id,
                "git_sha": _git_head_sha(),
                "source_hash": media.input_hash,
                "config_hash": config.config_hash(),
                "plan_hash": plan.plan_hash,
                "seed": config.seed,
                "cache_hit": cache_hit,
                "budget": budget.to_dict(),
                "disk_preflight": budget_report,
                "resource_stats": tracker.to_dict(),
            },
        )
        _write_run_readme(run_dir, run_id, media, plan, None, None, None, config, command_str)
        return {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "status": "DRY_RUN",
            "plan_hash": plan.plan_hash,
            "final_path": None,
            "events": len(plan.events),
        }

    # Render external overlays (HyperFrames / Remotion) if any event selected them
    tracker.start_stage("external_overlays")
    ext_overlays, fallback_history = _render_external_overlays(
        plan, analysis, config, doctor_info, temp_dir, reg
    )
    tracker.end_stage("external_overlays", temp_dir=temp_dir)

    # Primary timeline render into final.partial.mp4
    ff_renderer = FFmpegRenderer()
    partial_out = run_dir / "final.partial.mp4"

    tracker.start_stage("render")
    render_res = ff_renderer.render_timeline(
        media=media,
        plan=plan,
        config=config,
        analysis=analysis,
        workdir=temp_dir,
        out_partial_path=partial_out,
        budget=budget,
        external_overlays=ext_overlays,
        fallback_history=fallback_history,
        tracker=tracker,
    )
    tracker.end_stage("render", temp_dir=temp_dir)

    # Render baseline control encode (when aspect transforms applied or for strict effect diff verification)
    control_path: Path | None = None
    if plan.events and plan.aspect_policy != "preserve":
        tracker.start_stage("control_encode")
        control_path = temp_dir / "control_baseline.mp4"
        ff_renderer.render_control_encode(media, plan, config, analysis, control_path, budget)
        tracker.end_stage("control_encode", temp_dir=temp_dir)

    # Validate final.partial.mp4 BEFORE atomic promotion to final.mp4
    tracker.start_stage("validation")
    val_res = validate_rendered_output(
        source_media=media,
        plan=plan,
        rendered_path=partial_out,
        run_dir=run_dir,
        control_path=control_path,
    )
    tracker.end_stage("validation", temp_dir=temp_dir)

    final_path = run_dir / "final.mp4"
    promoted = False
    if val_res.overall_status == "PASS":
        os.replace(partial_out, final_path)
        promoted = True
        render_res.artifact_path = str(final_path)
    else:
        # Quarantine partial artifact (never promote to final.mp4!)
        quarantine_path = run_dir / "evidence" / "quarantined_final.partial.mp4"
        if partial_out.is_file():
            os.replace(partial_out, quarantine_path)
        render_res.status = "FAIL"
        render_res.artifact_path = str(quarantine_path)

    # Optional fresh 9:16 preview separate from final
    preview_rel: str | None = None
    if config.render_preview_9x16 and promoted:
        tracker.start_stage("preview_9x16")
        prev_cfg = MotionLabConfig(**{**config.to_dict(), "aspect": "9:16", "quality": "preview", "render_preview_9x16": False})
        prev_plan, _ = build_edit_plan(media, analysis, prev_cfg, doctor_info["engines"], registry=reg)
        prev_partial = run_dir / "preview_9x16.partial.mp4"
        prev_work = temp_dir / "preview_9x16"
        ff_renderer.render_timeline(
            media=media,
            plan=prev_plan,
            config=prev_cfg,
            analysis=analysis,
            workdir=prev_work,
            out_partial_path=prev_partial,
            budget=budget,
            tracker=tracker,
        )
        prev_final = run_dir / "preview_9x16.mp4"
        os.replace(prev_partial, prev_final)
        preview_rel = "preview_9x16.mp4"
        tracker.end_stage("preview_9x16", temp_dir=temp_dir)

    # Safely clean owned temp directory after sampling peak temp usage
    tracker.sample_self(temp_dir=temp_dir)
    cleaned_files = safe_cleanup_run_temp(temp_dir, run_dir)

    atomic_write_json(run_dir / "render_manifest.json", render_res.to_dict())
    atomic_write_json(run_dir / "validation.json", val_res.to_dict())
    atomic_write_json(
        run_dir / "logs" / "resource_stats.json",
        {
            "budget": budget.to_dict(),
            "disk_preflight": budget_report,
            "runtime": tracker.to_dict(),
            "temp_files_cleaned": len(cleaned_files),
        },
    )
    atomic_write_json(
        run_dir / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "motionlab_version": __version__,
            "analyzer_version": ANALYZER_VERSION,
            "planner_version": PLANNER_VERSION,
            "mode": "edit",
            "run_id": run_id,
            "git_sha": _git_head_sha(),
            "source_hash": media.input_hash,
            "config_hash": config.config_hash(),
            "plan_hash": plan.plan_hash,
            "seed": config.seed,
            "cache_hit": cache_hit,
            "engines_used": render_res.engines_used,
            "engine_capabilities": plan.engine_capabilities,
            "technical_status": val_res.technical_status,
            "aesthetic_status": val_res.aesthetic_status,
            "final_promoted": promoted,
            "preview_9x16": preview_rel,
            "resource_stats": tracker.to_dict(),
        },
    )
    _write_run_readme(
        run_dir, run_id, media, plan, render_res, val_res, preview_rel, config,
        command_str or f"python -m motionlab edit --input \"{input_path}\" --profile {config.profile} --seed {config.seed}",
    )

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "final_path": str(final_path) if promoted else None,
        "preview_9x16_path": str(run_dir / preview_rel) if preview_rel else None,
        "contact_sheet_path": str(run_dir / "contact_sheet.jpg") if (run_dir / "contact_sheet.jpg").is_file() else None,
        "plan_hash": plan.plan_hash,
        "profile": plan.profile,
        "plan_status": plan.status,
        "events": len(plan.events),
        "engines_used": render_res.engines_used,
        "overall_status": val_res.overall_status,
        "technical_status": val_res.technical_status,
        "aesthetic_status": val_res.aesthetic_status,
        "resource_stats": tracker.to_dict(),
    }


def run_validate_existing(run_dir: Path) -> dict[str, Any]:
    """Re-run technical validation on an existing run folder."""
    run_path = run_dir.resolve()
    media_json = run_path / "media_info.json"
    plan_json = run_path / "edit_plan.json"
    target_mp4 = run_path / "final.mp4"
    if not media_json.is_file() or not plan_json.is_file():
        raise FileNotFoundError(f"Run folder missing media_info.json or edit_plan.json: {run_path}")
    if not target_mp4.is_file():
        raise FileNotFoundError(f"Run folder missing final.mp4: {run_path}")

    media = MediaInfo.from_dict(json.loads(media_json.read_text(encoding="utf-8")))
    plan = EditPlan.from_dict(json.loads(plan_json.read_text(encoding="utf-8")))
    val_res = validate_rendered_output(media, plan, target_mp4, run_path)
    atomic_write_json(run_path / "validation.json", val_res.to_dict())
    return {
        "run_dir": str(run_path),
        "overall_status": val_res.overall_status,
        "technical_status": val_res.technical_status,
        "aesthetic_status": val_res.aesthetic_status,
        "checks": [c.to_dict() for c in val_res.checks],
    }


def run_effects_gallery(output_root: Path, seed: int = 42) -> dict[str, Any]:
    """Render a real procedural before/after visual gallery contact sheet + sample verification for EVERY registered effect."""
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw
    from motionlab.effects.engine_frames import apply_single_effect
    from motionlab.schemas import EffectEvent
    from motionlab.validation import _load_qa_font

    reg = load_effects()
    specs = reg.all_specs()
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    gal_dir = (output_root / f"gallery_{ts}").resolve()
    gal_dir.mkdir(parents=True, exist_ok=True)

    # Generate rich two-shot synthetic 360x480 procedural frames + audio envelope
    w, h = 240, 320
    shot_a = np.zeros((h, w, 3), dtype=np.uint8)
    shot_b = np.zeros((h, w, 3), dtype=np.uint8)
    ys = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    xs = np.linspace(0, 1, w, dtype=np.float32)[None, :]
    shot_a[..., 0] = np.clip((45 + 140 * xs + 40 * ys), 0, 255).astype(np.uint8)
    shot_a[..., 1] = np.clip((65 + 110 * (1 - ys)), 0, 255).astype(np.uint8)
    shot_a[..., 2] = np.clip((120 + 90 * (1 - xs)), 0, 255).astype(np.uint8)
    cv2.circle(shot_a, (w // 2, int(h * 0.42)), 48, (235, 195, 165), -1)
    cv2.rectangle(shot_a, (36, 190), (204, 285), (42, 66, 98), -1)
    cv2.circle(shot_a, (w // 2 - 14, int(h * 0.39)), 5, (40, 40, 50), -1)
    cv2.circle(shot_a, (w // 2 + 14, int(h * 0.39)), 5, (40, 40, 50), -1)

    shot_b[..., 0] = np.clip((160 + 60 * ys), 0, 255).astype(np.uint8)
    shot_b[..., 1] = np.clip((90 + 120 * xs), 0, 255).astype(np.uint8)
    shot_b[..., 2] = np.clip((60 + 130 * (1 - ys)), 0, 255).astype(np.uint8)
    cv2.rectangle(shot_b, (40, 55), (200, 240), (245, 240, 225), 4)
    cv2.circle(shot_b, (w // 2, h // 2), 40, (255, 120, 70), -1)

    # 30-frame ring buffer with cut at frame 15
    ring = {i: (shot_a.copy() if i < 15 else shot_b.copy()) for i in range(30)}
    audio_env = np.array([0.2 + 0.75 * abs(np.sin(i * 0.45)) for i in range(30)], dtype=np.float32)

    records: list[dict[str, Any]] = []
    tiles: list[tuple[str, np.ndarray]] = []
    for spec in specs:
        params = spec.validate_params({})
        params["intensity"] = min(spec.max_intensity, 0.95)
        ev = EffectEvent(
            id=f"gal_{spec.id}",
            effect_id=spec.id,
            effect_version=spec.version,
            start_frame=0,
            end_frame_exclusive=30,
            layer=spec.layer,
            region={"x": 0.3, "y": 0.25, "w": 0.4, "h": 0.35},
            parameters=params,
            easing="ease_in_out_cubic",
            priority=5,
            engine="ffmpeg",
            reason_codes=["gallery_verification"],
            evidence_refs=["procedural_fixture"],
            fallback_id=spec.fallback_id,
            uses_neighbor_frames=spec.uses_neighbor_frames,
        )
        # Test midpoint frame AND intensity=0 control equivalence
        mid_f = 15
        src_mid = ring[mid_f]
        out_mid = apply_single_effect(src_mid, ev, mid_f, ring, audio_env, seed)
        diff_arr = np.abs(out_mid.astype(np.float32) - src_mid.astype(np.float32))
        mad = float(diff_arr.mean())
        max_px_diff = float(diff_arr.max())
        changed_frac = float((diff_arr.max(axis=2) >= 4.0).mean())

        zero_ev = EffectEvent(**{**ev.to_dict(), "parameters": {**params, "intensity": 0.0}})
        out_zero = apply_single_effect(src_mid, zero_ev, mid_f, ring, audio_env, seed)
        zero_mad = float(np.abs(out_zero.astype(np.float32) - src_mid.astype(np.float32)).mean())

        non_black = float(out_mid.mean()) > 10.0
        # Peripheral graphic overlays touch a small safe-margin fraction with crisp contrast,
        # while spatial/color/mask effects shift broader image regions. Both must clearly differ from no-op.
        visible_change = (mad >= 0.35) or (max_px_diff >= 20.0 and changed_frac >= 0.0012)
        verified = visible_change and (zero_mad == 0.0) and non_black
        records.append({
            "id": spec.id,
            "name": spec.name,
            "family": spec.family,
            "milestone": spec.milestone,
            "engine": spec.engine,
            "midpoint_mad": round(mad, 3),
            "max_pixel_diff": round(max_px_diff, 2),
            "changed_pixels_frac": round(changed_frac, 5),
            "zero_intensity_mad": round(zero_mad, 4),
            "verified": verified,
        })
        # Compose small side-by-side before/after tile
        pair = np.concatenate([
            cv2.resize(src_mid, (120, 160), interpolation=cv2.INTER_AREA),
            cv2.resize(out_mid, (120, 160), interpolation=cv2.INTER_AREA),
        ], axis=1)
        tiles.append((f"{spec.id} {spec.name} (d={mad:.1f})", pair))

    # Render montage contact sheet
    cols = 4
    rows = (len(tiles) + cols - 1) // cols
    tw, th = 240, 160
    pad, lh, hh = 8, 22, 42
    sheet = Image.new("RGB", (cols * (tw + pad) + pad, hh + rows * (th + lh + pad) + pad), (16, 18, 24))
    draw = ImageDraw.Draw(sheet)
    font = _load_qa_font(12)
    verified_n = sum(1 for r in records if r["verified"])
    draw.text(
        (12, 12),
        f"MotionLab-AI ML-01 Effect Gallery (Before | After) — Verified {verified_n}/{len(records)} effects",
        fill=(240, 244, 250),
        font=_load_qa_font(16),
    )
    for idx, (label, arr) in enumerate(tiles):
        r, c = divmod(idx, cols)
        x = pad + c * (tw + pad)
        y = hh + pad + r * (th + lh + pad)
        sheet.paste(Image.fromarray(arr), (x, y))
        draw.text((x + 2, y + th + 3), label[:34], fill=(185, 220, 255), font=font)

    sheet_path = gal_dir / "effects_gallery.jpg"
    sheet.save(sheet_path, "JPEG", quality=90)
    manifest_path = gal_dir / "gallery_manifest.json"
    atomic_write_json(manifest_path, {"total_registered": len(records), "total_verified": verified_n, "effects": records})
    return {
        "gallery_dir": str(gal_dir),
        "gallery_image": str(sheet_path),
        "manifest": str(manifest_path),
        "total_registered": len(records),
        "total_verified": verified_n,
    }
