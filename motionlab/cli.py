"""ML-01 Command-Line Interface (`python -m motionlab <command>`)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from motionlab import __version__
from motionlab.cache import AnalysisCache
from motionlab.config import (
    MotionLabConfig,
    VALID_ASPECTS,
    VALID_ENGINES,
    VALID_PROFILES,
    VALID_QUALITIES,
    run_doctor,
)
from motionlab.pipeline import (
    run_analyze,
    run_edit,
    run_effects_gallery,
    run_plan,
    run_validate_existing,
)
from motionlab.registry import load_effects
from motionlab.resource import BudgetExceededError
from motionlab.schemas import EditPlan, SchemaValidationError


def _add_common_edit_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--input", "-i", type=str, required=True, help="Path to local input MP4/video file (never modified).")
    p.add_argument("--output", "-o", type=str, default="runs", help="Output runs root directory (default: runs).")
    p.add_argument("--profile", type=str, default="auto", choices=VALID_PROFILES, help="Editing policy profile (default: auto).")
    p.add_argument("--aspect", type=str, default="source", choices=VALID_ASPECTS, help="Aspect policy: 'source' (default) or explicit '9:16'.")
    p.add_argument("--quality", type=str, default="final", choices=VALID_QUALITIES, help="Render quality: 'final' (default) or 'preview'.")
    p.add_argument("--seed", type=int, default=42, help="Deterministic planner & grain seed (default: 42).")
    p.add_argument("--offline", action="store_true", default=True, help="Offline zero-API execution (always enforced).")
    p.add_argument("--strict", action="store_true", default=False, help="Fail closed if any requested capability/engine/check is missing.")
    p.add_argument("--dry-run", action="store_true", default=False, help="Run preflight/analysis/plan only; do not render final MP4.")
    p.add_argument("--overwrite", action="store_true", default=False, help="Explicit opt-in flag (runs still allocate unique run_id).")
    p.add_argument("--audio-track", type=int, default=None, help="Explicit 0-based audio track index (default: preserve all tracks).")
    p.add_argument("--max-workers", type=int, default=None, help="Cap bounded worker concurrency.")
    p.add_argument("--max-temp-gb", type=float, default=4.0, help="Max temporary workspace budget in GiB (default: 4.0).")
    p.add_argument("--min-free-gb", type=float, default=5.0, help="Minimum free disk reserve in GiB (default: 5.0).")
    p.add_argument("--no-cache", action="store_true", default=False, help="Disable local content-addressed analysis cache.")
    p.add_argument("--cache-dir", type=str, default=None, help="Override analysis cache directory.")
    p.add_argument("--no-face-tracking", action="store_true", default=False, help="Disable optional local OpenCV Haar face tracking.")
    p.add_argument("--engine", type=str, default="auto", choices=VALID_ENGINES, help="Preferred renderer engine (default: auto).")
    p.add_argument("--enable-remotion", action="store_true", default=False, help="Opt in to Remotion renderer after confirming license eligibility.")
    p.add_argument("--disable-hyperframes", action="store_true", default=False, help="Disable HyperFrames renderer backend.")
    p.add_argument("--allow-hdr-tonemap", action="store_true", default=False, help="Explicitly opt in to HDR->SDR Hable tone-mapping.")
    p.add_argument("--allow-split-layout", action="store_true", default=False, help="Explicitly opt in to L05 split composition.")
    p.add_argument("--pad-background", type=str, default="blur", choices=("blur", "solid"), help="9:16 pad background style.")
    p.add_argument("--preview-9x16", action="store_true", default=False, help="Also render a fresh 9:16 preview MP4 in the run folder.")
    p.add_argument("--timeout", type=int, default=600, help="Per-stage render timeout in seconds (default: 600).")


def _config_from_args(args: argparse.Namespace) -> MotionLabConfig:
    return MotionLabConfig(
        profile=getattr(args, "profile", "auto"),
        aspect=getattr(args, "aspect", "source"),
        quality=getattr(args, "quality", "final"),
        seed=int(getattr(args, "seed", 42)),
        offline=True,
        strict=bool(getattr(args, "strict", False)),
        dry_run=bool(getattr(args, "dry_run", False)),
        overwrite=bool(getattr(args, "overwrite", False)),
        audio_track=getattr(args, "audio_track", None),
        max_workers=getattr(args, "max_workers", None),
        max_temp_gb=float(getattr(args, "max_temp_gb", 4.0)),
        min_free_gb=float(getattr(args, "min_free_gb", 5.0)),
        cache_enabled=not bool(getattr(args, "no_cache", False)),
        cache_dir=getattr(args, "cache_dir", None),
        enable_face_tracking=not bool(getattr(args, "no_face_tracking", False)),
        enable_hyperframes=not bool(getattr(args, "disable_hyperframes", False)),
        enable_remotion=bool(getattr(args, "enable_remotion", False)),
        preferred_engine=getattr(args, "engine", "auto"),
        allow_hdr_tonemap=bool(getattr(args, "allow_hdr_tonemap", False)),
        allow_split_layout=bool(getattr(args, "allow_split_layout", False)),
        pad_background=getattr(args, "pad_background", "blur"),
        render_preview_9x16=bool(getattr(args, "preview_9x16", False)),
        timeout_s=int(getattr(args, "timeout", 600)),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m motionlab",
        description="MotionLab-AI ML-01 — Zero-API Local Automatic Video Effects Engine",
    )
    parser.add_argument("--version", action="version", version=f"motionlab {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    # doctor
    p_doc = sub.add_parser("doctor", help="Inspect local tools, Python/Node renderers, licenses, and disk space.")
    p_doc.add_argument("--enable-remotion", action="store_true", default=False, help="Evaluate Remotion with license opt-in enabled.")
    p_doc.add_argument("--json", action="store_true", default=False, help="Emit raw JSON report.")

    # analyze
    p_ana = sub.add_parser("analyze", help="Run local signal analysis (scene cuts, motion, audio envelope, safe zones).")
    p_ana.add_argument("--input", "-i", type=str, required=True, help="Input local video file.")
    p_ana.add_argument("--output", "-o", type=str, default="runs", help="Output runs root directory.")
    p_ana.add_argument("--audio-track", type=int, default=None, help="Explicit 0-based audio track index.")
    p_ana.add_argument("--max-workers", type=int, default=None, help="Max bounded workers.")
    p_ana.add_argument("--no-cache", action="store_true", default=False, help="Disable analysis cache.")
    p_ana.add_argument("--cache-dir", type=str, default=None, help="Override cache dir.")
    p_ana.add_argument("--no-face-tracking", action="store_true", default=False, help="Disable local OpenCV Haar face tracking.")

    # plan
    p_plan = sub.add_parser("plan", help="Analyze and build deterministic edit_plan.json without rendering.")
    _add_common_edit_flags(p_plan)

    # edit
    p_edit = sub.add_parser("edit", help="One-command offline automatic video effects edit.")
    _add_common_edit_flags(p_edit)

    # render (replay existing edit_plan.json)
    p_ren = sub.add_parser("render", help="Replay an existing edit_plan.json against a local input video.")
    p_ren.add_argument("--plan", type=str, required=True, help="Path to edit_plan.json.")
    _add_common_edit_flags(p_ren)

    # validate
    p_val = sub.add_parser("validate", help="Run technical validation checks on an existing run directory.")
    p_val.add_argument("--run", type=str, required=True, help="Path to runs/<run_id> directory.")

    # effects
    p_eff = sub.add_parser("effects", help="List or render visual gallery of registered ML-01 effects.")
    eff_sub = p_eff.add_subparsers(dest="effects_cmd", required=True)
    p_elist = eff_sub.add_parser("list", help="List all registered effects with engines, families, and milestones.")
    p_elist.add_argument("--json", action="store_true", default=False, help="Output JSON catalog.")
    p_egal = eff_sub.add_parser("gallery", help="Render before/after procedural gallery sheet for all effects.")
    p_egal.add_argument("--output", "-o", type=str, default="runs", help="Output directory for gallery.")
    p_egal.add_argument("--seed", type=int, default=42, help="Deterministic seed.")

    # cache
    p_cache = sub.add_parser("cache", help="Manage local MotionLab analysis cache.")
    cache_sub = p_cache.add_subparsers(dest="cache_cmd", required=True)
    p_cprune = cache_sub.add_parser("prune", help="Safely remove cached analysis JSON files owned by MotionLab.")
    p_cprune.add_argument("--cache-dir", type=str, default=None, help="Override cache dir.")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    # Explicitly reject unknown flags (parse_args fails closed with exit code 2!)
    args = parser.parse_args(list(argv) if argv is not None else None)
    cmd_repr = "python -m motionlab " + " ".join(argv if argv is not None else sys.argv[1:])

    try:
        if args.command == "doctor":
            cfg = MotionLabConfig(enable_remotion=bool(args.enable_remotion))
            info = run_doctor(cfg)
            print(json.dumps(info, indent=2, sort_keys=True))
            return 0 if info["status"] == "OK" else 2

        if args.command == "analyze":
            cfg = _config_from_args(args)
            res = run_analyze(Path(args.input), Path(args.output), cfg)
            print(json.dumps(res, indent=2, sort_keys=True))
            return 0

        if args.command == "plan":
            cfg = _config_from_args(args)
            res = run_plan(Path(args.input), Path(args.output), cfg)
            print(json.dumps(res, indent=2, sort_keys=True))
            return 0

        if args.command == "edit":
            cfg = _config_from_args(args)
            res = run_edit(Path(args.input), Path(args.output), cfg, command_str=cmd_repr)
            print(json.dumps(res, indent=2, sort_keys=True))
            if res.get("status") == "DRY_RUN":
                return 0
            return 0 if res.get("overall_status") == "PASS" else 1

        if args.command == "render":
            cfg = _config_from_args(args)
            plan_path = Path(args.plan).resolve()
            if not plan_path.is_file():
                raise FileNotFoundError(f"Plan file not found: {plan_path}")
            plan = EditPlan.from_dict(json.loads(plan_path.read_text(encoding="utf-8")))
            res = run_edit(Path(args.input), Path(args.output), cfg, existing_plan=plan, command_str=cmd_repr)
            print(json.dumps(res, indent=2, sort_keys=True))
            return 0 if res.get("overall_status") == "PASS" else 1

        if args.command == "validate":
            res = run_validate_existing(Path(args.run))
            print(json.dumps(res, indent=2, sort_keys=True))
            return 0 if res.get("overall_status") == "PASS" else 1

        if args.command == "effects":
            if args.effects_cmd == "list":
                reg = load_effects()
                specs = reg.all_specs()
                if args.json:
                    print(json.dumps({
                        "total_registered": len(specs),
                        "by_family": reg.count_by_family(),
                        "effects": [s.to_dict() for s in specs],
                    }, indent=2, sort_keys=True))
                else:
                    print(f"MotionLab-AI ML-01 Effect Registry: {len(specs)} registered effects")
                    for s in specs:
                        print(
                            f"  {s.id:4s} {s.name:28s} [{s.family:10s}] milestone={s.milestone:8s} "
                            f"engines={','.join(s.supported_engines)} layer={s.layer}"
                        )
                return 0
            if args.effects_cmd == "gallery":
                res = run_effects_gallery(Path(args.output), seed=int(args.seed))
                print(json.dumps(res, indent=2, sort_keys=True))
                return 0 if res["total_verified"] == res["total_registered"] else 1

        if args.command == "cache":
            if args.cache_cmd == "prune":
                cache = AnalysisCache(Path(args.cache_dir) if args.cache_dir else None)
                removed = cache.prune()
                print(json.dumps({"cache_dir": str(cache.cache_dir), "removed_files": removed}, indent=2))
                return 0

        parser.error(f"Unknown command: {args.command}")
        return 2
    except (SchemaValidationError, BudgetExceededError, FileNotFoundError, ValueError, RuntimeError) as exc:
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
