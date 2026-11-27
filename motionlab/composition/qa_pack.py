"""Phase QA-01 (ML-01.R5.R1): Lightweight, Redacted, Git-Safe QA Evidence Pack Builder.

Produces a compact (<= 2 MiB target) repository-trackable evidence pack under:
  `qa_evidence/ML-01.R5.R1/<tested_code_sha_short>/<run_id>/`
containing ONLY relative paths, non-sensitive metadata, redacted summaries, sound rationale,
layout/continuity reports, mix/visual QA JSONs, and an optional compact phone-scale contact sheet
(only when explicitly authorized). NEVER commits MP4/WAV/private raw transcript/secrets into Git!
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from PIL import Image

from motionlab.schemas import atomic_write_json, atomic_write_text

MAX_QA_PACK_BYTES = 2 * 1024 * 1024  # 2 MiB budget


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _redact_paths(obj: Any) -> Any:
    """Strip absolute host paths / Windows drive prefixes so tracked QA manifests contain relative references only."""
    if isinstance(obj, str):
        norm = obj.replace("\\", "/")
        if "MotionLab-AI-ML01/" in norm:
            return norm.split("MotionLab-AI-ML01/", 1)[1]
        if len(norm) >= 3 and norm[1:3] == ":/":
            return Path(norm).name
        return obj
    if isinstance(obj, list):
        return [_redact_paths(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _redact_paths(v) for k, v in obj.items()}
    return obj


def build_qa01_evidence_pack(
    *,
    run_dir: Path,
    pack_dir: Path,
    tested_code_sha: str,
    repo_url: str,
    branch: str,
    label: str,
    render_mode: str,
    final_mp4_name: str,
    preview_mp4_name: str = "preview_9x16.mp4",
    include_contact_sheet: bool = False,
    include_raw_transcript: bool = False,
    test_summary: dict[str, Any] | None = None,
    benchmark_summary: dict[str, Any] | None = None,
    blockers: list[str] | None = None,
) -> dict[str, Any]:
    """Build a strictly <=2 MiB redacted QA-01 pack from an actual verified local run directory."""
    run_dir = run_dir.resolve()
    pack_dir = pack_dir.resolve()
    if pack_dir.exists():
        shutil.rmtree(pack_dir)
    pack_dir.mkdir(parents=True, exist_ok=True)

    def _load(rel: str) -> dict[str, Any]:
        p = run_dir / rel
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))
        return {}

    manifest_src = _load("manifest.json")
    val_doc = _load("validation.json")
    aeu_doc = _load("actual_engine_usage.json")
    res_doc = _load("resource_stats.json")
    sound_rat = _load("evidence/sound_rationale.json")
    audio_tl = _load("audio_timeline.json")
    mix_metrics = _load("evidence/audio_mix_metrics.json")
    audio_qa = _load("evidence/audio_event_qa.json")
    cq_doc = _load("evidence/creative_quality_proxy.json")
    layout_doc = _load("layout.json")
    cont_doc = _load("continuity.json")
    leg_doc = _load("evidence/phone_legibility.json")
    sb_doc = _load("storyboard_spec.json")
    sg_doc = _load("story_graph.json")

    final_p = run_dir / final_mp4_name
    if not final_p.is_file():
        final_p = run_dir / "final.mp4"
    prev_p = run_dir / preview_mp4_name

    # Write clean, path-redacted JSON artifacts into pack_dir
    artifacts_written: dict[str, str] = {}

    def _emit_json(fname: str, data: dict[str, Any]) -> None:
        clean = _redact_paths(data)
        out_p = pack_dir / fname
        atomic_write_json(out_p, clean)
        artifacts_written[fname] = _sha256_file(out_p)

    # Redact verbatim spoken quotes when public raw transcript publication is NOT explicitly authorized
    def _redact_quotes(doc: Any) -> Any:
        if include_raw_transcript:
            return doc
        if isinstance(doc, dict):
            out = {}
            for k, v in doc.items():
                if k in ("verbatim_quote", "full_text", "quote", "raw_text") and isinstance(v, str):
                    out[k] = f"<redacted_sha256:{hashlib.sha256(v.encode('utf-8')).hexdigest()[:16]} len={len(v)}>"
                else:
                    out[k] = _redact_quotes(v)
            return out
        if isinstance(doc, list):
            return [_redact_quotes(x) for x in doc]
        return doc

    _emit_json("storyboard.json", _redact_quotes(sb_doc))
    _emit_json("story_graph_redacted.json", _redact_quotes(sg_doc))
    _emit_json("effect_execution.json", aeu_doc)
    _emit_json("sound_rationale.json", _redact_quotes(sound_rat))
    _emit_json("audio_timeline.json", _redact_quotes(audio_tl))
    _emit_json("mix_metrics.json", {
        "master_exact": mix_metrics.get("master_exact"),
        "source_true_peak_dbtp": mix_metrics.get("source_true_peak_dbtp"),
        "mix_preencode_true_peak_dbtp": mix_metrics.get("mix_preencode_true_peak_dbtp"),
        "metrics": mix_metrics.get("metrics"),
        "audio_event_qa": audio_qa,
    })
    _emit_json("visual_quality.json", {
        "creative_quality_proxy": cq_doc,
        "layout_summary": {k: v for k, v in layout_doc.items() if k != "scenes"},
        "continuity_summary": {
            "persistent_element_ids": cont_doc.get("persistent_element_ids", []),
            "track_count": cont_doc.get("track_count", 0),
        },
        "phone_legibility": leg_doc,
    })
    _emit_json("benchmark.json", {
        "resource_stats": res_doc,
        "optimization_summary": benchmark_summary or {},
    })
    if test_summary is not None:
        _emit_json("test_summary.json", test_summary)

    # Optional compact contact sheet ONLY when explicitly authorized for repo visibility
    if include_contact_sheet and (run_dir / "contact_sheet.jpg").is_file():
        im = Image.open(run_dir / "contact_sheet.jpg")
        im.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
        cs_out = pack_dir / "contact_sheet_compact.jpg"
        im.save(cs_out, "JPEG", quality=78, optimize=True)
        artifacts_written["contact_sheet_compact.jpg"] = _sha256_file(cs_out)

    reuse_gate = val_doc.get("reuse_gate") or aeu_doc.get("reuse_gate") or {}
    pack_manifest = {
        "schema_version": "1.0.0",
        "run_id": run_dir.name,
        "label": label,
        "repo": repo_url,
        "branch": branch,
        "tested_code_sha": tested_code_sha,
        "source_sha256": sb_doc.get("source_hash") or manifest_src.get("input_hash", ""),
        "render_mode": render_mode,
        "input_geometry": f"{sb_doc.get('width', 720)}x{sb_doc.get('height', 1280)}",
        "input_fps": f"{sb_doc.get('fps_num', 24)}/{sb_doc.get('fps_den', 1)}",
        "input_frames": sb_doc.get("total_frames", 720),
        "final_artifact": {
            "filename": final_p.name if final_p.is_file() else final_mp4_name,
            "sha256": _sha256_file(final_p) if final_p.is_file() else None,
            "bytes": int(final_p.stat().st_size) if final_p.is_file() else 0,
            "access": "ARTIFACT_LOCAL_ONLY",
            "url": None,
        },
        "preview_artifact": {
            "filename": prev_p.name if prev_p.is_file() else preview_mp4_name,
            "sha256": _sha256_file(prev_p) if prev_p.is_file() else None,
            "bytes": int(prev_p.stat().st_size) if prev_p.is_file() else 0,
            "access": "ARTIFACT_LOCAL_ONLY",
            "url": None,
        },
        "engines_requested": ["ffmpeg", "hyperframes", "remotion"],
        "engines_executed": aeu_doc.get("engines_used", []),
        "fallbacks": [],
        "plan_hash": sb_doc.get("storyboard_hash", ""),
        "story_graph_hash": sb_doc.get("story_graph_hash", ""),
        "sonic_score_hash": sound_rat.get("score_hash", ""),
        "imported_primitives_used": (
            list(reuse_gate.get("remotion_primitives", []))
            + list(reuse_gate.get("hyperframes_primitives", []))
        ),
        "foley_provenance": {
            "library_status": val_doc.get("foley_library_status", "PARTIAL_FOLEY_ASSETS"),
            "recorded_cues_used": int(audio_qa.get("recorded_foley_events_used", 0)),
            "procedural_cues_used": int(audio_qa.get("procedural_dsp_events_used", 0)),
        },
        "technical_status": val_doc.get("overall_status", "UNKNOWN"),
        "creative_status": val_doc.get("aesthetic_status", "PENDING_HUMAN_REVIEW"),
        "listening_status": audio_qa.get("listening_review_status", "LISTENING_NOT_RUN_AUTOMATED_PROXY_ONLY"),
        "critic_status": cq_doc.get("vision_critic_status", "VISION_CRITIC_NOT_RUN"),
        "heldout_status": "REAL_HELDOUT_NOT_RUN",
        "asset_status": "ASSET_MISSING_CONCEPTUAL",
        "offline_proof_scope": "api_keys_unset_static_ast_zero_remote_runtime_local_models_only",
        "test_results": test_summary or {},
        "benchmark_timings": res_doc,
        "blockers": blockers or [
            "G8_HUMAN_VISUAL_AUDIO_REVIEW_PENDING",
            "REAL_HELDOUT_FOOTAGE_NOT_PROVISIONED",
            "LOCAL_VLM_NOT_PROVISIONED",
            "OWNER_PRODUCT_SCREENSHOT_ASSET_MISSING_CONCEPTUAL_USED",
        ],
        "reviewer_approval": None,
        "artifact_hashes": artifacts_written,
    }
    atomic_write_json(pack_dir / "manifest.json", pack_manifest)

    readme_txt = "\n".join([
        f"# QA-01 Evidence Pack — `{label}` (`{run_dir.name}`)",
        "",
        f"- **Tested Code SHA:** `{tested_code_sha}`",
        f"- **Render Mode:** `{render_mode}`",
        f"- **Technical Status:** `{pack_manifest['technical_status']}` | **Creative Status:** `{pack_manifest['creative_status']}`",
        f"- **Foley Status:** `{pack_manifest['foley_provenance']['library_status']}` (recorded=`{pack_manifest['foley_provenance']['recorded_cues_used']}`, procedural=`{pack_manifest['foley_provenance']['procedural_cues_used']}`)",
        f"- **Listening:** `{pack_manifest['listening_status']}` | **Vision Critic:** `{pack_manifest['critic_status']}` | **Held-Out:** `{pack_manifest['heldout_status']}`",
        f"- **Final MP4 (Local Only):** `{pack_manifest['final_artifact']['filename']}` (`SHA-256: {pack_manifest['final_artifact']['sha256']}`, `{pack_manifest['final_artifact']['bytes']}` bytes)",
        f"- **Preview MP4 (Local Only):** `{pack_manifest['preview_artifact']['filename']}` (`SHA-256: {pack_manifest['preview_artifact']['sha256']}`, `{pack_manifest['preview_artifact']['bytes']}` bytes)",
        "",
        "> Privacy notice: Raw user footage, audio WAV/M4A stems, and verbatim transcripts are intentionally excluded/redacted from this Git-tracked QA pack.",
    ]) + "\n"
    atomic_write_text(pack_dir / "README.md", readme_txt)

    total_bytes = sum(p.stat().st_size for p in pack_dir.rglob("*") if p.is_file())
    if total_bytes > MAX_QA_PACK_BYTES:
        raise RuntimeError(f"QA-01 pack exceeded 2 MiB budget: {total_bytes} bytes > {MAX_QA_PACK_BYTES}")
    return {
        "pack_dir": str(pack_dir),
        "total_bytes": total_bytes,
        "within_2mib_budget": total_bytes <= MAX_QA_PACK_BYTES,
        "files_written": sorted(p.name for p in pack_dir.iterdir() if p.is_file()),
    }
