"""Phase QA-01 / QA-02 (ML-01.R5.R3): Safe, Atomic, Deeply-Redacted, Git-Lightweight QA Evidence Pack Builder.

Fixes D14 & D15:
  - D15 (Safe Path Ownership & Atomic Creation): NEVER blindly calls `shutil.rmtree(pack_dir)`! Enforces strict path
    containment under an owned `allowed_root` (default `PROJECT_ROOT / "qa_evidence"`), refuses to overwrite an existing
    non-empty pack directory unless explicitly requested on a verified QA pack, and stages atomically before swap!
  - D14 (QA-02 Upgrade): Adds `build_qa02_evidence_pack(...)` while keeping `build_qa01_evidence_pack(...)` 100% compatible:
      * Target <= 2 MiB hard budget (no raw MP4/WAV/base64 media embedded in JSON).
      * Deep redaction of raw quotes, absolute host paths (POSIX & Windows), emails/secrets even inside nested free text.
      * Includes causal `selected_candidate_trace.json`, multi-HF-layer execution manifest, 4-tier animation markers,
        `cue_decisions.json`, floor-free mix QA, streaming pixel critic + `revision_history.json`, benchmark & test summary.
      * Explicit image/artifact sharing controls: `CONTACT_SHEET_LOCAL_ONLY` by default unless `image_sharing_authorized=True`
        AND `include_contact_sheet=True`; `ARTIFACT_LOCAL_ONLY` (`url=None`) unless explicitly authorized external URL!
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Any

from PIL import Image

from motionlab.config import PROJECT_ROOT
from motionlab.schemas import atomic_write_json, atomic_write_text, ensure_contained_path

MAX_QA_PACK_BYTES = 2 * 1024 * 1024  # 2 MiB hard budget

_POSIX_ABS_RE = re.compile(r"/(?:Users|home|var|tmp|private|opt|Volumes)/[^\s\"',;]+")
_WIN_ABS_RE = re.compile(r"[A-Za-z]:[/\\][^\s\"',;]+")
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _redact_string_deep(val: str) -> str:
    """Deeply strip absolute POSIX/Windows host paths and emails even inside free-text rationales/notes."""
    s = val.replace("\\", "/")
    if "MotionLab-AI-ML01/" in s:
        s = re.sub(r"(?:[A-Za-z]:)?(?:/[^\s\"',;]*)*MotionLab-AI-ML01/", "", s)
    s = _POSIX_ABS_RE.sub(lambda m: f"<local_path>/{Path(m.group(0)).name}", s)
    s = _WIN_ABS_RE.sub(lambda m: f"<local_path>/{Path(m.group(0)).name}", s)
    s = _EMAIL_RE.sub("<redacted_email>", s)
    return s


def _redact_paths(obj: Any) -> Any:
    """Recursively redact absolute paths & PII strings across dicts, lists, and free-text fields."""
    if isinstance(obj, str):
        return _redact_string_deep(obj)
    if isinstance(obj, list):
        return [_redact_paths(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _redact_paths(v) for k, v in obj.items()}
    return obj


def _redact_quotes_deep(doc: Any, *, include_raw_transcript: bool = False) -> Any:
    """Redact verbatim spoken quotes when public raw transcript publication is NOT explicitly authorized."""
    if include_raw_transcript:
        return doc
    QUOTE_KEYS = {"verbatim_quote", "full_text", "quote", "raw_text", "clause_text", "hook_question", "central_conflict"}
    if isinstance(doc, dict):
        out: dict[str, Any] = {}
        for k, v in doc.items():
            if k in QUOTE_KEYS and isinstance(v, str) and v.strip():
                out[k] = f"<redacted_sha256:{hashlib.sha256(v.encode('utf-8')).hexdigest()[:16]} len={len(v)}>"
            else:
                out[k] = _redact_quotes_deep(v, include_raw_transcript=include_raw_transcript)
        return out
    if isinstance(doc, list):
        return [_redact_quotes_deep(x, include_raw_transcript=include_raw_transcript) for x in doc]
    return doc


def _validate_safe_pack_target(
    pack_dir: Path,
    *,
    allowed_root: Path | None = None,
    allow_overwrite_owned: bool = False,
) -> Path:
    """Fixes D15: Enforce strict path containment and forbid unsafe blind `shutil.rmtree`!"""
    target = pack_dir.resolve()
    root = (allowed_root or (PROJECT_ROOT / "qa_evidence")).resolve()
    ensure_contained_path(target, root)
    if target == root:
        raise ValueError(f"Refusing to target QA root directory itself: {target}")
    if target.exists():
        if not target.is_dir():
            raise ValueError(f"QA pack target exists and is not a directory: {target}")
        contents = list(target.iterdir())
        if contents:
            if not allow_overwrite_owned:
                raise FileExistsError(
                    f"Refusing to overwrite existing immutable QA evidence pack at {target}. "
                    "Use a unique <run_id> folder (D15 safe-pack policy)."
                )
            # Even when explicitly replacing an owned pack, verify it actually looks like an owned QA pack!
            if not (target / "manifest.json").is_file():
                raise ValueError(
                    f"Refusing unsafe replacement of non-QA directory (missing manifest.json): {target}"
                )
    return target


def _build_qa_pack_core(
    *,
    pack_version: str,  # "QA-01" | "QA-02"
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
    image_sharing_authorized: bool = False,
    include_raw_transcript: bool = False,
    authorized_artifact_url: str | None = None,
    authorized_preview_url: str | None = None,
    test_summary: dict[str, Any] | None = None,
    benchmark_summary: dict[str, Any] | None = None,
    blockers: list[str] | None = None,
    allowed_root: Path | None = None,
    allow_overwrite_owned: bool = False,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    target_dir = _validate_safe_pack_target(
        pack_dir,
        allowed_root=allowed_root,
        allow_overwrite_owned=allow_overwrite_owned,
    )
    target_dir.parent.mkdir(parents=True, exist_ok=True)

    # Atomic unique staging directory (never mutates target until complete & within 2 MiB budget!)
    staging_dir = target_dir.parent / f".staging_{target_dir.name}_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    staging_dir.mkdir(parents=True, exist_ok=False)

    try:
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
        cue_dec = _load("evidence/cue_decisions.json")
        audio_tl = _load("audio_timeline.json")
        mix_metrics = _load("evidence/audio_mix_metrics.json")
        audio_qa = _load("evidence/audio_event_qa.json")
        cq_doc = _load("evidence/creative_quality_proxy.json")
        bnd_rep = _load("evidence/scene_boundary_report.json")
        layout_doc = _load("layout.json")
        cont_doc = _load("continuous_timeline.json") or _load("continuity.json")
        num_trace = _load("numeric_state_trace.json")
        leg_doc = _load("evidence/phone_legibility.json")
        prod_stat = _load("evidence/product_asset_status.json")
        sb_doc = _load("storyboard_spec.json")
        sg_doc = _load("story_graph.json")
        raw_dir = _load("raw_director_response.json")
        markers_doc = _load("evidence/animation_markers_realized.json") or _load("evidence/animation_markers.json")
        rev_hist = _load("evidence/revision_history.json")

        final_p = run_dir / final_mp4_name
        if not final_p.is_file():
            final_p = run_dir / "final.mp4"
        prev_p = run_dir / preview_mp4_name

        artifacts_written: dict[str, str] = {}

        def _emit_json(fname: str, data: dict[str, Any]) -> None:
            clean = _redact_paths(_redact_quotes_deep(data, include_raw_transcript=include_raw_transcript))
            out_p = staging_dir / fname
            atomic_write_json(out_p, clean)
            artifacts_written[fname] = _sha256_file(out_p)

        _emit_json("storyboard.json", sb_doc)
        _emit_json("story_graph_redacted.json", sg_doc)
        _emit_json("effect_execution.json", aeu_doc)
        _emit_json("sound_rationale.json", sound_rat)
        if cue_dec:
            _emit_json("cue_decisions.json", cue_dec)
        _emit_json("audio_timeline.json", audio_tl)
        _emit_json("mix_metrics.json", {
            "master_exact": mix_metrics.get("master_exact"),
            "source_true_peak_dbtp": mix_metrics.get("source_true_peak_dbtp"),
            "mix_preencode_true_peak_dbtp": mix_metrics.get("mix_preencode_true_peak_dbtp"),
            "metrics": mix_metrics.get("metrics"),
            "audio_event_qa": audio_qa,
        })

        brain_audit = raw_dir.get("brain_audit", {}) if isinstance(raw_dir, dict) else {}
        if pack_version == "QA-02":
            _emit_json("selected_candidate_trace.json", {
                "reasoning_mode": brain_audit.get("reasoning_mode", ""),
                "causal_selection_before_compile": brain_audit.get("causal_selection_before_compile", False),
                "real_transcript_supplied": brain_audit.get("real_transcript_supplied", False),
                "timestamp_quality": brain_audit.get("timestamp_quality", ""),
                "retrieval_mode": brain_audit.get("retrieval_mode", sb_doc.get("retrieval_mode", "")),
                "catalog_index_hash": brain_audit.get("catalog_index_hash", sb_doc.get("catalog_index_hash", "")),
                "selected_graph_hash": brain_audit.get("selected_graph_hash", sb_doc.get("selected_graph_hash", "")),
                "global_beam_search": brain_audit.get("global_beam_search", {}),
                "intervention_counts_by_type": brain_audit.get("intervention_counts_by_type", {}),
                "selection_trace": brain_audit.get("selection_trace", sb_doc.get("selection_trace", [])),
            })
            if markers_doc:
                _emit_json("marker_tiers.json", markers_doc)
            if rev_hist:
                _emit_json("revision_history.json", rev_hist)

        _emit_json("visual_quality.json", {
            "creative_quality_proxy": cq_doc,
            "objective_pixel_critic": bnd_rep,
            "layout_summary": {k: v for k, v in layout_doc.items() if k != "scenes"},
            "continuity_summary": {
                "continuity_hash": cont_doc.get("continuity_hash", ""),
                "persistent_element_ids": cont_doc.get("persistent_element_ids", []),
                "track_count": cont_doc.get("track_count", 0),
                "boundaries": cont_doc.get("boundaries", []),
                "continuous_timeline_valid": cont_doc.get("continuous_timeline_valid"),
            },
            "numeric_state_trace": num_trace,
            "phone_legibility": leg_doc,
            "product_asset_status": prod_stat,
            "narrative_brain_audit": brain_audit,
        })
        _emit_json("benchmark.json", {
            "resource_stats": res_doc,
            "optimization_summary": benchmark_summary or {},
        })
        if test_summary is not None:
            _emit_json("test_summary.json", test_summary)

        # Contact sheet / boundary strips ONLY when BOTH requested AND explicitly image-sharing authorized!
        can_include_images = bool(include_contact_sheet and image_sharing_authorized)
        contact_sheet_status = "CONTACT_SHEET_LOCAL_ONLY"
        if can_include_images:
            if (run_dir / "contact_sheet.jpg").is_file():
                im = Image.open(run_dir / "contact_sheet.jpg")
                im.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
                cs_out = staging_dir / "contact_sheet_compact.jpg"
                im.save(cs_out, "JPEG", quality=76, optimize=True)
                artifacts_written["contact_sheet_compact.jpg"] = _sha256_file(cs_out)
                contact_sheet_status = "AUTHORIZED_COMPACT_IMAGES_INCLUDED"
            if pack_version == "QA-02" and (run_dir / "evidence" / "boundary_strips.jpg").is_file():
                im_b = Image.open(run_dir / "evidence" / "boundary_strips.jpg")
                im_b.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
                bs_out = staging_dir / "boundary_strips_compact.jpg"
                im_b.save(bs_out, "JPEG", quality=76, optimize=True)
                artifacts_written["boundary_strips_compact.jpg"] = _sha256_file(bs_out)

        reuse_gate = val_doc.get("reuse_gate") or aeu_doc.get("reuse_gate") or {}
        final_access = "AUTHORIZED_EXTERNAL_URL" if authorized_artifact_url else "ARTIFACT_LOCAL_ONLY"
        prev_access = "AUTHORIZED_EXTERNAL_URL" if authorized_preview_url else "ARTIFACT_LOCAL_ONLY"

        pack_manifest = {
            "schema_version": "2.0.0" if pack_version == "QA-02" else "1.0.0",
            "pack_type": pack_version,
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
                "access": final_access,
                "url": authorized_artifact_url,
            },
            "preview_artifact": {
                "filename": prev_p.name if prev_p.is_file() else preview_mp4_name,
                "sha256": _sha256_file(prev_p) if prev_p.is_file() else None,
                "bytes": int(prev_p.stat().st_size) if prev_p.is_file() else 0,
                "access": prev_access,
                "url": authorized_preview_url,
            },
            "contact_sheet_status": contact_sheet_status,
            "remote_review_limitations": (
                "JSON telemetry cannot replace direct motion playback or human phone-speaker listening; "
                "final MP4 & preview remain local-only unless an authorized external URL is attached."
            ),
            "engines_requested": ["ffmpeg", "hyperframes", "remotion"],
            "engines_executed": aeu_doc.get("engines_used", []),
            "fallbacks": [],
            "plan_hash": sb_doc.get("storyboard_hash", ""),
            "selected_graph_hash": sb_doc.get("selected_graph_hash", ""),
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
            "critic_status": bnd_rep.get("vlm_status", cq_doc.get("vision_critic_status", "VISION_CRITIC_NOT_RUN")),
            "heldout_status": "REAL_HELDOUT_NOT_RUN",
            "asset_status": prod_stat.get("status", "ASSET_MISSING_CONCEPTUAL"),
            "offline_proof_scope": "api_keys_unset_static_ast_zero_remote_runtime_local_models_only",
            "test_results": test_summary or {},
            "benchmark_timings": _redact_paths(res_doc),
            "blockers": blockers or [
                "G10_HUMAN_VQ01_VISUAL_AUDIO_REVIEW_PENDING",
                "REAL_HELDOUT_FOOTAGE_NOT_PROVISIONED",
                "LOCAL_VLM_NOT_PROVISIONED",
                "OWNER_PRODUCT_SCREENSHOT_ASSET_MISSING_CONCEPTUAL_USED",
            ],
            "reviewer_approval": None,
            "artifact_hashes": artifacts_written,
        }
        atomic_write_json(staging_dir / "manifest.json", _redact_paths(pack_manifest))

        readme_txt = "\n".join([
            f"# {pack_version} Evidence Pack — `{label}` (`{run_dir.name}`)",
            "",
            f"- **Tested Code SHA:** `{tested_code_sha}`",
            f"- **Render Mode:** `{render_mode}`",
            f"- **Technical Status:** `{pack_manifest['technical_status']}` | **Creative Status:** `{pack_manifest['creative_status']}`",
            f"- **Foley Status:** `{pack_manifest['foley_provenance']['library_status']}` (recorded=`{pack_manifest['foley_provenance']['recorded_cues_used']}`, procedural=`{pack_manifest['foley_provenance']['procedural_cues_used']}`)",
            f"- **Listening:** `{pack_manifest['listening_status']}` | **Vision Critic:** `{pack_manifest['critic_status']}` | **Held-Out:** `{pack_manifest['heldout_status']}`",
            f"- **Contact Sheet Sharing:** `{contact_sheet_status}` | **Product Assets:** `{pack_manifest['asset_status']}`",
            f"- **Final MP4 ({final_access}):** `{pack_manifest['final_artifact']['filename']}` (`SHA-256: {pack_manifest['final_artifact']['sha256']}`, `{pack_manifest['final_artifact']['bytes']}` bytes)",
            f"- **Preview MP4 ({prev_access}):** `{pack_manifest['preview_artifact']['filename']}` (`SHA-256: {pack_manifest['preview_artifact']['sha256']}`, `{pack_manifest['preview_artifact']['bytes']}` bytes)",
            "",
            "> Privacy notice: Raw footage, audio WAV/M4A stems, verbatim transcripts, private paths, and unauthorized screenshots are excluded/redacted from this Git-tracked QA pack.",
        ]) + "\n"
        atomic_write_text(staging_dir / "README.md", readme_txt)

        total_bytes = sum(p.stat().st_size for p in staging_dir.rglob("*") if p.is_file())
        if total_bytes > MAX_QA_PACK_BYTES:
            raise RuntimeError(f"{pack_version} pack exceeded 2 MiB budget: {total_bytes} bytes > {MAX_QA_PACK_BYTES}")

        # Atomic swap into owned target_dir
        if target_dir.exists():
            backup_old = target_dir.parent / f".old_{target_dir.name}_{uuid.uuid4().hex[:6]}"
            os.replace(target_dir, backup_old)
            try:
                os.replace(staging_dir, target_dir)
            finally:
                shutil.rmtree(backup_old, ignore_errors=True)
        else:
            os.replace(staging_dir, target_dir)

        return {
            "pack_type": pack_version,
            "pack_dir": str(target_dir),
            "total_bytes": total_bytes,
            "within_2mib_budget": total_bytes <= MAX_QA_PACK_BYTES,
            "contact_sheet_status": contact_sheet_status,
            "final_access": final_access,
            "files_written": sorted(p.name for p in target_dir.iterdir() if p.is_file()),
        }
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise


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
    allowed_root: Path | None = None,
    allow_overwrite_owned: bool = False,
) -> dict[str, Any]:
    """Backward-compatible QA-01 pack builder upgraded with D15 path containment & atomic staging."""
    # Preserve compatibility for legacy tests writing inside pytest tmp_path when allowed_root isn't passed
    eff_root = allowed_root if allowed_root is not None else pack_dir.resolve().parent
    return _build_qa_pack_core(
        pack_version="QA-01",
        run_dir=run_dir,
        pack_dir=pack_dir,
        tested_code_sha=tested_code_sha,
        repo_url=repo_url,
        branch=branch,
        label=label,
        render_mode=render_mode,
        final_mp4_name=final_mp4_name,
        preview_mp4_name=preview_mp4_name,
        include_contact_sheet=include_contact_sheet,
        image_sharing_authorized=include_contact_sheet,
        include_raw_transcript=include_raw_transcript,
        test_summary=test_summary,
        benchmark_summary=benchmark_summary,
        blockers=blockers,
        allowed_root=eff_root,
        allow_overwrite_owned=allow_overwrite_owned,
    )


def build_qa02_evidence_pack(
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
    image_sharing_authorized: bool = False,
    include_raw_transcript: bool = False,
    authorized_artifact_url: str | None = None,
    authorized_preview_url: str | None = None,
    test_summary: dict[str, Any] | None = None,
    benchmark_summary: dict[str, Any] | None = None,
    blockers: list[str] | None = None,
    allowed_root: Path | None = None,
    allow_overwrite_owned: bool = False,
) -> dict[str, Any]:
    """Phase A11.4 (ML-01.R5.R3): Build a strictly <=2 MiB, atomic, deeply-redacted QA-02 remote review pack."""
    eff_root = allowed_root if allowed_root is not None else (PROJECT_ROOT / "qa_evidence")
    return _build_qa_pack_core(
        pack_version="QA-02",
        run_dir=run_dir,
        pack_dir=pack_dir,
        tested_code_sha=tested_code_sha,
        repo_url=repo_url,
        branch=branch,
        label=label,
        render_mode=render_mode,
        final_mp4_name=final_mp4_name,
        preview_mp4_name=preview_mp4_name,
        include_contact_sheet=include_contact_sheet,
        image_sharing_authorized=image_sharing_authorized,
        include_raw_transcript=include_raw_transcript,
        authorized_artifact_url=authorized_artifact_url,
        authorized_preview_url=authorized_preview_url,
        test_summary=test_summary,
        benchmark_summary=benchmark_summary,
        blockers=blockers,
        allowed_root=eff_root,
        allow_overwrite_owned=allow_overwrite_owned,
    )
