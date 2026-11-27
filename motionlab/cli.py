"""Production CLI for MotionLab AI ML-01 (`python -m motionlab ...`).

Commands:
  doctor
  analyze  --input <mp4> --output <runs_dir>
  plan     --input <mp4> --profile <name> --seed <int>
  edit     --input <mp4> --profile <name> --seed <int> --output <runs_dir>
  render   --plan <edit_plan.json> --input <mp4> --output <runs_dir>
  validate --run <runs/run_id>
  effects  list | gallery --output <runs_dir>
  cache    prune [--cache-dir <dir>]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

import psutil

from motionlab import __version__
from motionlab.analyzer import analyze_media
from motionlab.cache import LocalAnalysisCache
from motionlab.config import (
    AnalysisConfig,
    EditConfig,
    PROJECT_ROOT,
    ResourceConfig,
    VALID_ASPECTS,
    VALID_ENGINES,
    VALID_PROFILES,
    VALID_QUALITIES,
    atomic_write_json,
)
from motionlab.pipeline import (
    build_effects_gallery,
    create_unique_run_dir,
    run_edit_pipeline,
    run_validate_existing_run,
)
from motionlab.planner import build_edit_plan
from motionlab.preflight import PreflightError, probe_media
from motionlab.registry import load_effects
from motionlab.renderers.base import RendererError, probe_all_engine_capabilities
from motionlab.resource import GIB, ResourceBudget, ResourceBudgetError
from motionlab.schemas import EditPlan


def _cmd_doctor(args: argparse.Namespace) -> int:
    caps = probe_all_engine_capabilities(
        allow_remotion=not args.disable_remotion,
        allow_hyperframes=not args.disable_hyperframes,
    )
    reg = load_effects()
    vm = psutil.virtual_memory()
    disk = shutil.disk_usage(PROJECT_ROOT)
    report: dict[str, Any] = {
        "motionlab_version": __version__,
        "offline_runtime": True,
        "api_keys_required": False,
        "system": {
            "os": platform.platform(),
            "python": sys.version.split()[0],
            "logical_cpus": os.cpu_count(),
            "total_ram_gb": round(vm.total / GIB, 2),
            "available_ram_gb": round(vm.available / GIB, 2),
            "project_disk_free_gb": round(disk.free / GIB, 2),
        },
        "effects": {
            "registered_total": len(reg.all_specs()),
            "core_count": len(reg.core_specs()),
        },
        "engines": caps,
    }
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"MotionLab AI ML-01 Doctor (v{__version__})")
        print(f"  OS/Python : {report['system']['os']} | Python {report['system']['python']}")
        print(
            f"  Resources : {report['system']['logical_cpus']} CPUs | "
            f"RAM {report['system']['available_ram_gb']}/{report['system']['total_ram_gb']} GiB free | "
            f"Disk {report['system']['project_disk_free_gb']} GiB free"
        )
        print(f"  Effects   : {report['effects']['registered_total']} registered ({report['effects']['core_count']} core)")
        print("  Runtime   : 100% OFFLINE / ZERO-API (no API keys required)")
        for ename, info in caps.items():
            status = "AVAILABLE" if info.get("available") else f"UNAVAILABLE ({info.get('reason')})"
            ver = info.get("version", "-")
            print(f"  Engine[{ename:11s}]: {status} | version={ver}")
            if info.get("license"):
                print(f"    license: {info['license']}")
    return 0 if (caps.get("ffmpeg") or {}).get("available") else 1


def _build_edit_config_from_args(args: argparse.Namespace) -> EditConfig:
    if not getattr(args, "offline", True):
        raise ValueError("Offline mode is mandatory in ML-01; cloud editing is not permitted.")
    res_cfg = ResourceConfig(
        max_workers=getattr(args, "max_workers", None),
        max_render_workers=getattr(args, "max_render_workers", 1),
        max_temp_gb=getattr(args, "max_temp_gb", 4.0),
        min_free_gb=getattr(args, "min_free_gb", None),
        subprocess_timeout_s=getattr(args, "timeout", 600),
    )
    ana_cfg = AnalysisConfig(
        use_cache=not getattr(args, "no_cache", False),
    )
    return EditConfig(
        profile=getattr(args, "profile", "auto"),
        seed=getattr(args, "seed", 42),
        aspect=getattr(args, "aspect", "source"),
        quality=getattr(args, "quality", "final"),
        offline=True,
        strict=getattr(args, "strict", False),
        dry_run=getattr(args, "dry_run", False),
        overwrite=getattr(args, "overwrite", False),
        audio_track=getattr(args, "audio_track", None),
        preferred_engine=getattr(args, "engine", "auto"),
        allow_remotion=not getattr(args, "disable_remotion", False),
        allow_hyperframes=not getattr(args, "disable_hyperframes", False),
        allow_hdr_tonemap=getattr(args, "allow_hdr_tonemap", False),
        aspect_bg_mode=getattr(args, "aspect_bg", "blur"),
        density_scale=getattr(args, "density_scale", 1.0),
        intensity_scale=getattr(args, "intensity_scale", 1.0),
        analysis=ana_cfg,
        resources=res_cfg,
        render_preview_9x16=getattr(args, "preview_9x16", False),
    )


def _cmd_analyze(args: argparse.Namespace) -> int:
    cfg = _build_edit_config_from_args(args)
    media = probe_media(Path(args.input), allow_hdr_tonemap=cfg.allow_hdr_tonemap)
    run_id, run_dir = create_unique_run_dir(Path(args.output), f"ana{media.input_hash[:6]}")
    budget = ResourceBudget(cfg.resources, run_dir, run_dir / "temp")
    budget.preflight_budget(media.duration_s, media.display_width, media.display_height, media.fps_float)
    analysis, cache_hit = analyze_media(media, cfg, run_dir / "temp", budget=budget)
    atomic_write_json(run_dir / "media_info.json", media.to_dict())
    atomic_write_json(run_dir / "analysis.json", analysis.to_dict())
    budget.cleanup_owned_temp(run_dir)
    out = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "input": str(Path(args.input).resolve()),
        "source_hash": media.input_hash,
        "cache_hit": cache_hit,
        "cuts": len(analysis.cuts),
        "energy_peaks": len(analysis.energy_peaks),
        "subject_regions": len(analysis.subject_regions),
        "candidates": len(analysis.candidates),
        "capabilities": analysis.capabilities,
        "analysis_path": str(run_dir / "analysis.json"),
    }
    print(json.dumps(out, indent=2))
    return 0


def _cmd_plan(args: argparse.Namespace) -> int:
    cfg = _build_edit_config_from_args(args)
    media = probe_media(Path(args.input), allow_hdr_tonemap=cfg.allow_hdr_tonemap)
    caps = probe_all_engine_capabilities(allow_remotion=cfg.allow_remotion, allow_hyperframes=cfg.allow_hyperframes)
    run_id, run_dir = create_unique_run_dir(Path(args.output), f"pln{media.input_hash[:6]}")
    budget = ResourceBudget(cfg.resources, run_dir, run_dir / "temp")
    budget.preflight_budget(media.duration_s, media.display_width, media.display_height, media.fps_float)
    analysis, cache_hit = analyze_media(media, cfg, run_dir / "temp", budget=budget)
    plan, explain = build_edit_plan(media, analysis, cfg, caps)
    atomic_write_json(run_dir / "analysis.json", analysis.to_dict())
    atomic_write_json(run_dir / "edit_plan.json", plan.to_dict())
    atomic_write_json(run_dir / "plan_explain.json", explain)
    budget.cleanup_owned_temp(run_dir)
    out = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "profile": plan.profile,
        "status": plan.status,
        "plan_hash": plan.plan_hash(),
        "events": len(plan.events),
        "cache_hit": cache_hit,
        "plan_path": str(run_dir / "edit_plan.json"),
        "explain_path": str(run_dir / "plan_explain.json"),
    }
    print(json.dumps(out, indent=2))
    return 0


def _cmd_edit(args: argparse.Namespace) -> int:
    cfg = _build_edit_config_from_args(args)
    res = run_edit_pipeline(Path(args.input), Path(args.output), cfg)
    print(json.dumps(res, indent=2))
    if res["status"] in ("TECHNICAL_PASS", "DRY_RUN_PLANNED"):
        return 0
    return 1


def _cmd_render(args: argparse.Namespace) -> int:
    cfg = _build_edit_config_from_args(args)
    plan_path = Path(args.plan).resolve()
    if not plan_path.exists():
        raise FileNotFoundError(f"Plan file not found: {plan_path}")
    plan = EditPlan.from_dict(json.loads(plan_path.read_text(encoding="utf-8")))
    res = run_edit_pipeline(Path(args.input), Path(args.output), cfg, preloaded_plan=plan)
    print(json.dumps(res, indent=2))
    return 0 if res["status"] == "TECHNICAL_PASS" else 1


def _cmd_validate(args: argparse.Namespace) -> int:
    val = run_validate_existing_run(Path(args.run))
    print(json.dumps(val.to_dict(), indent=2))
    return 0 if val.overall_status == "TECHNICAL_PASS" else 1


def _cmd_effects_list(args: argparse.Namespace) -> int:
    reg = load_effects()
    specs = reg.all_specs()
    if args.json:
        print(json.dumps([s.to_dict() for s in specs], indent=2))
        return 0
    print(f"MotionLab AI ML-01 Effect Registry — {len(specs)} registered effects ({len(reg.core_specs())} core):")
    for s in specs:
        core_tag = "CORE" if s.is_core else "FULL"
        print(
            f"  {s.code}  {s.id:26s} [{core_tag}] family={s.family:15s} layer={s.layer:10s} "
            f"engines={','.join(s.supported_engines)}"
        )
    return 0


def _cmd_effects_gallery(args: argparse.Namespace) -> int:
    res = build_effects_gallery(Path(args.output))
    print(json.dumps(res, indent=2))
    return 0 if res["verified_count"] == res["registered_count"] else 1


def _cmd_cache_prune(args: argparse.Namespace) -> int:
    cache_dir = Path(args.cache_dir).resolve()
    cache = LocalAnalysisCache(cache_dir)
    removed = cache.prune_all()
    print(json.dumps({"cache_dir": str(cache_dir), "removed_files": removed}, indent=2))
    return 0


def _add_common_edit_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--profile", choices=VALID_PROFILES, default="auto", help="Effect style policy profile (default: auto)")
    p.add_argument("--seed", type=int, default=42, help="Deterministic planner & procedural seed (default: 42)")
    p.add_argument("--aspect", choices=VALID_ASPECTS, default="source", help="Output aspect ratio (default: preserve source)")
    p.add_argument("--aspect-bg", choices=("blur", "solid", "smart_crop"), default="blur", help="Aspect adaptation mode (default: blur)")
    p.add_argument("--quality", choices=VALID_QUALITIES, default="final", help="Render quality: preview or final (default: final)")
    p.add_argument("--offline", action="store_true", default=True, help="Enforce 100% offline execution (default: true; cloud forbidden)")
    p.add_argument("--strict", action="store_true", help="Fail closed if required engine/check is unavailable")
    p.add_argument("--dry-run", action="store_true", help="Run preflight, analysis, and planner without rendering final video")
    p.add_argument("--overwrite", action="store_true", help="Allowed flag (runs always use unique run_ids and never touch source)")
    p.add_argument("--audio-track", type=int, default=None, help="Explicit 0-based audio track index (default: preserve all compatible tracks)")
    p.add_argument("--engine", choices=VALID_ENGINES, default="auto", help="Preferred renderer engine (default: auto)")
    p.add_argument("--disable-remotion", action="store_true", help="Disable Remotion engine (e.g. for orgs >=4 without Remotion license)")
    p.add_argument("--disable-hyperframes", action="store_true", help="Disable HyperFrames engine")
    p.add_argument("--allow-hdr-tonemap", action="store_true", help="Explicitly opt into HDR->SDR BT.709 tone-mapping")
    p.add_argument("--density-scale", type=float, default=1.0, help="Scale effect density in [0.0, 2.0] (default: 1.0)")
    p.add_argument("--intensity-scale", type=float, default=1.0, help="Scale effect intensity in [0.0, 1.5] (default: 1.0)")
    p.add_argument("--preview-9x16", action="store_true", help="Also render a fresh 9:16 preview_9x16.mp4 in the run folder")
    p.add_argument("--no-cache", action="store_true", help="Disable local analysis cache for cold deterministic runs")
    p.add_argument("--max-workers", type=int, default=None, help="Upper bound for async analysis workers")
    p.add_argument("--max-render-workers", type=int, default=1, help="Upper bound for external overlay render workers (default: 1)")
    p.add_argument("--max-temp-gb", type=float, default=4.0, help="Max temporary disk budget in GiB (default: 4.0)")
    p.add_argument("--min-free-gb", type=float, default=None, help="Minimum free disk reserve in GiB (default: max(5 GiB, 10%% vol))")
    p.add_argument("--timeout", type=int, default=600, help="Per-subprocess timeout in seconds (default: 600)")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m motionlab",
        description="MotionLab AI (ML-01) — Zero-API Local Automatic Video Effects Engine",
    )
    ap.add_argument("--version", action="version", version=f"motionlab {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)

    # doctor
    p_doc = sub.add_parser("doctor", help="Check local FFmpeg, Remotion, HyperFrames, CPU/RAM/disk, and license status")
    p_doc.add_argument("--json", action="store_true", help="Emit JSON output")
    p_doc.add_argument("--disable-remotion", action="store_true")
    p_doc.add_argument("--disable-hyperframes", action="store_true")
    p_doc.set_defaults(func=_cmd_doctor)

    # analyze
    p_ana = sub.add_parser("analyze", help="Run local zero-API video/audio signal analysis")
    p_ana.add_argument("--input", required=True, help="Path to source MP4")
    p_ana.add_argument("--output", default="runs", help="Output runs directory (default: runs)")
    _add_common_edit_flags(p_ana)
    p_ana.set_defaults(func=_cmd_analyze)

    # plan
    p_pln = sub.add_parser("plan", help="Analyze and build deterministic EditPlan JSON without rendering")
    p_pln.add_argument("--input", required=True, help="Path to source MP4")
    p_pln.add_argument("--output", default="runs", help="Output runs directory (default: runs)")
    _add_common_edit_flags(p_pln)
    p_pln.set_defaults(func=_cmd_plan)

    # edit
    p_ed = sub.add_parser("edit", help="One-command offline auto-edit: analyze -> plan -> render -> validate -> final.mp4")
    p_ed.add_argument("--input", required=True, help="Path to source MP4")
    p_ed.add_argument("--output", default="runs", help="Output runs directory (default: runs)")
    _add_common_edit_flags(p_ed)
    p_ed.set_defaults(func=_cmd_edit)

    # render (replay)
    p_ren = sub.add_parser("render", help="Render directly from an existing validated edit_plan.json")
    p_ren.add_argument("--plan", required=True, help="Path to edit_plan.json")
    p_ren.add_argument("--input", required=True, help="Path to source MP4")
    p_ren.add_argument("--output", default="runs", help="Output runs directory (default: runs)")
    _add_common_edit_flags(p_ren)
    p_ren.set_defaults(func=_cmd_render)

    # validate
    p_val = sub.add_parser("validate", help="Re-validate an existing run folder (ffprobe, full decode, audio integrity)")
    p_val.add_argument("--run", required=True, help="Path to runs/<run_id>")
    p_val.set_defaults(func=_cmd_validate)

    # effects
    p_eff = sub.add_parser("effects", help="Inspect registered effects or render the 40-effect verification gallery")
    eff_sub = p_eff.add_subparsers(dest="effects_cmd", required=True)
    p_el = eff_sub.add_parser("list", help="List all registered effects")
    p_el.add_argument("--json", action="store_true")
    p_el.set_defaults(func=_cmd_effects_list)
    p_eg = eff_sub.add_parser("gallery", help="Render & verify all 40 effects into a gallery contact sheet")
    p_eg.add_argument("--output", default="runs", help="Output runs directory (default: runs)")
    p_eg.set_defaults(func=_cmd_effects_gallery)

    # cache
    p_cache = sub.add_parser("cache", help="Manage local analysis cache")
    cache_sub = p_cache.add_subparsers(dest="cache_cmd", required=True)
    p_cp = cache_sub.add_parser("prune", help="Safely prune MotionLab's own analysis cache entries")
    p_cp.add_argument("--cache-dir", default=str(PROJECT_ROOT / "runs" / ".cache"), help="Cache directory to prune")
    p_cp.set_defaults(func=_cmd_cache_prune)

    return ap


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        return int(args.func(args))
    except (PreflightError, ResourceBudgetError, RendererError, ValueError, FileNotFoundError, KeyError) as exc:
        print(f"ERROR [{type(exc).__name__}]: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
