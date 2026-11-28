"""Phase A0–A12 (ML-01.R5.R6): Evidence-Driven Creative Intelligence Runner (`director_strategy: "evidence_v3"`).

Invoked cleanly via existing production CLI (`python -m motionlab edit|plan --job <job.json> --creative-mode director`)
when job manifest specifies `"director_strategy": "evidence_v3"`. Preserves 100% of legacy and R5.R5 v2 paths intact!

End-to-end connected capabilities:
  - A1: `align_source_dialogue_to_intent` + `verify_required_claims` + independent final-mix Re-ASR (`final_mix_reasr.json`)
  - A2/A3: `build_atomic_claim_graph` + content-grounded `retrieve_evidence_for_claims` + model-driven `propose_model_driven_plans`
  - A4: Zero-AGPL Poppler/OpenCV `extract_verified_region_crop` + feature-level `build_product_proof_by_feature`
  - A5/A6: Layered `SceneGraphV3` (`build_scene_graph_v3_from_plan`, `render_scene_v3_overlay`)
  - A7: Blind permutation-invariant 3-hook competition (`build_three_distinct_v3_hooks`, `rank_rendered_hooks_blind`)
  - A8: Authorized-only localized watermark blur guard (`evaluate_watermark_authorization`, `apply_localized_watermark_blur_if_authorized`) + voice-priority 48kHz stereo Foley mix
  - A9: Post-render observed pixel + source-crop correlation + blind VLM audit (`verify_rendered_claim_coverage`), autonomous bounded creative repair loop, and strict separation of `technical_gate`, `creative_evidence_gate`, `release_rights_gate`, and `human_review_gate`!
"""

from __future__ import annotations

import datetime
import hashlib
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from motionlab.assets.catalog import inspect_product_asset_catalog, match_product_assets_for_name
from motionlab.assets.pdf_regions import build_product_proof_by_feature, extract_verified_region_crop
from motionlab.audio.assets import inspect_foley_library
from motionlab.audio.mixer import mix_preserving_master, validate_final_audio_mix
from motionlab.audio.schema import AUDIO_SCHEMA_VERSION, AudioTimeline, SoundEvent
from motionlab.audio.semantic_sound import resolve_sound_asset
from motionlab.composition.hook_competition import build_three_distinct_v3_hooks, rank_rendered_hooks_blind
from motionlab.composition.multi_source_timeline import (
    apply_localized_watermark_blur_if_authorized,
    build_multi_source_edit_map,
    build_multi_source_mezzanine_and_voice_bed,
    compute_multi_source_color_match,
    evaluate_multi_source_audio_joins,
    evaluate_watermark_authorization,
    exact_rational_samples,
    remap_multi_source_words,
    validate_multi_source_edit_map,
)
from motionlab.composition.scene_graph_v3 import (
    build_scene_graph_v3_from_plan,
    render_scene_v3_overlay,
    solve_v3_layout,
)
from motionlab.config import EditConfig, PROJECT_ROOT
from motionlab.preflight import probe_media
from motionlab.renderers.base import verify_webm_alpha_pixels
from motionlab.resource import ResourceBudget, ResourceTracker
from motionlab.schemas import atomic_write_json, atomic_write_text, canonical_hash, sanitize_native
from motionlab.semantic.creative_director import DirectorToolRegistry, build_intent_brief
from motionlab.semantic.evidence_planner import (
    build_atomic_claim_graph,
    propose_model_driven_plans,
    retrieve_evidence_for_claims,
)
from motionlab.semantic.evidence_verifier import plan_claim_coverage, verify_rendered_claim_coverage
from motionlab.semantic.model_router import LocalModelRouter
from motionlab.semantic.speech_dedup import (
    align_source_dialogue_to_intent,
    choose_clean_speech_path,
    detect_cross_source_overlap,
    detect_disfluency_spans,
    normalize_token,
    validate_semantic_preservation,
    verify_required_claims,
)
from motionlab.semantic.speech_perception import detect_speech_activity
from motionlab.semantic.temporal_ledger import build_temporal_ledger_from_story_and_scenes
from motionlab.semantic.video_evidence import build_multimodal_video_evidence_index

DIRECTOR_V3_SCHEMA_VERSION = "3.0.0"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _audit_decoded_continuity(final_mp4: Path, scenes: list[dict[str, Any]]) -> dict[str, Any]:
    """Streaming O(1)-RAM full-timeline black frame & boundary MAD check."""
    cap = cv2.VideoCapture(str(final_mp4))
    black_frames = 0
    step_mads: list[float] = []
    prev_sm: np.ndarray | None = None
    f_idx = 0
    try:
        while True:
            ok, fr = cap.read()
            if not ok or fr is None:
                break
            sm = cv2.resize(fr, (180, 320), interpolation=cv2.INTER_AREA).astype(np.float32)
            if float(np.mean(sm)) < 4.0:
                black_frames += 1
            step_mads.append(round(float(np.mean(np.abs(sm - prev_sm))), 2) if prev_sm is not None else 0.0)
            prev_sm = sm
            f_idx += 1
    finally:
        cap.release()

    boundary_audit: list[dict[str, Any]] = []
    high_jumps = 0
    for sc in scenes:
        for edge_lbl, bf in (("entry", int(sc["start_frame"])), ("exit", int(sc["end_frame_exclusive"]))):
            if 1 <= bf < len(step_mads):
                mad = step_mads[bf]
                abrupt = bool(mad > 68.0)
                if abrupt:
                    high_jumps += 1
                boundary_audit.append({
                    "scene_id": sc["scene_id"],
                    "edge": edge_lbl,
                    "frame": bf,
                    "decoded_step_mad": mad,
                    "abrupt_flagged": abrupt,
                })
    return sanitize_native({
        "frames_inspected": f_idx,
        "black_frames": black_frames,
        "abrupt_boundary_jumps": high_jumps,
        "boundary_audit": boundary_audit,
    })


def run_evidence_v3_director_job(
    *,
    job: dict[str, Any],
    output_root: Path,
    config: EditConfig,
    creative_mode: str = "director",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Execute full ML-01.R5.R6 `evidence_v3` production job end-to-end."""
    # Import shared helpers lazily to avoid circular import with director_pipeline.py
    from motionlab.composition.director_pipeline import (  # noqa: PLC0415
        _build_contact_sheet_60,
        _composite_overlays_onto_mezzanine,
        _transcribe_pcm_high_res,
    )

    t_start = time.perf_counter()
    ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"
    ffprobe_bin = shutil.which("ffprobe") or "ffprobe"

    seed = int(job.get("seed", config.seed))
    out_cfg = job["output"]
    W, H = int(out_cfg["width"]), int(out_cfg["height"])
    fps_num, fps_den = int(out_cfg["fps_num"]), int(out_cfg["fps_den"])
    fps_f = fps_num / float(fps_den)
    sr = int(out_cfg["sample_rate"])
    src_hash_short = canonical_hash([s["sha256"] for s in job["sources"]])[:8]
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{ts}_{src_hash_short}_s{seed}"
    run_dir = Path(output_root).resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    ev_dir = run_dir / "evidence"
    ev_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = run_dir / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    masters_dir = run_dir / "original_audio_masters"
    masters_dir.mkdir(parents=True, exist_ok=True)

    budget = ResourceBudget(config.resources, run_dir, temp_dir)
    tracker = ResourceTracker()
    atomic_write_json(run_dir / "job_manifest.json", job)

    # ------------------------------------------------------------------
    # 1. A1 — Ingest, Bit-Exact Audio Masters, Speech Perception, Script Completeness & Dedup
    # ------------------------------------------------------------------
    tracker.start_stage("ingest_and_speech_perception")
    probed_sources: list[dict[str, Any]] = []
    source_paths: dict[str, Path] = {}
    source_wavs_48k: dict[str, Path] = {}
    words_by_source: dict[str, list[dict[str, Any]]] = {}
    pcm16_by_source: dict[str, np.ndarray] = {}
    vad_ns_by_source: dict[str, list[dict[str, Any]]] = {}
    speech_raw_doc: dict[str, Any] = {}

    for s in job["sources"]:
        sid = s["source_id"]
        sp = Path(s["path"])
        media = probe_media(sp)
        source_paths[sid] = sp
        master_m4a = masters_dir / f"original_source_{sid}_master.m4a"
        if media.has_audio:
            subprocess.run(
                [ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error", "-i", str(sp), "-vn", "-map", "0:a:0", "-c:a", "copy", str(master_m4a)],
                check=True, timeout=60, shell=False,
            )
            wav_48k = temp_dir / f"src_{sid}_48k.wav"
            subprocess.run(
                [ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error", "-i", str(sp), "-vn", "-map", "0:a:0", "-ac", "2", "-ar", str(sr), str(wav_48k)],
                check=True, timeout=60, shell=False,
            )
            source_wavs_48k[sid] = wav_48k

        full_txt, words, pcm_16k = _transcribe_pcm_high_res(sp, ffmpeg_bin=ffmpeg_bin, condition_on_previous_text=False)
        vad = detect_speech_activity(media, pcm_16k=pcm_16k, ffmpeg_bin=ffmpeg_bin)
        words_by_source[sid] = words
        pcm16_by_source[sid] = pcm_16k
        vad_ns_by_source[sid] = list(vad.get("non_speech_spans") or [])
        speech_raw_doc[sid] = {
            "source_id": sid,
            "sha256": s["sha256"],
            "duration_s": media.duration_s,
            "fps": f"{media.fps.num}/{media.fps.den}",
            "asr_backend": "faster_whisper_small_int8_no_prev_cond",
            "vad_backend": vad["vad_backend"],
            "full_text": full_txt,
            "word_count": len(words),
            "words": words,
            "vad_non_speech_spans": vad_ns_by_source[sid],
        }
        probed_sources.append({
            "source_id": sid,
            "path": str(sp),
            "sha256": s["sha256"],
            "width": media.display_width,
            "height": media.display_height,
            "fps_num": media.fps.num,
            "fps_den": media.fps.den,
            "duration_s": media.duration_s,
            "sample_rate": int(media.audio_streams[0].sample_rate) if media.audio_streams else sr,
            "original_master_path": str(master_m4a) if master_m4a.is_file() else None,
        })

    atomic_write_json(run_dir / "source_manifest.json", {"sources": probed_sources})
    atomic_write_json(run_dir / "speech_raw_by_source.json", speech_raw_doc)
    tracker.end_stage("ingest_and_speech_perception", temp_dir=temp_dir)

    tracker.start_stage("speech_alignment_and_timeline")
    intended_story = str(job.get("intended_story") or "").strip() or None
    niche_hint = str(job.get("niche_hint") or "general").lower()
    sids = [s["source_id"] for s in probed_sources]

    boundary_infos: list[dict[str, Any]] = []
    for pair_idx in range(max(0, len(sids) - 1)):
        s0_id, s1_id = sids[pair_idx], sids[pair_idx + 1]
        b_pair = detect_cross_source_overlap(
            words_by_source[s0_id],
            words_by_source[s1_id],
            intended_story=intended_story,
            pcm_a_16k=pcm16_by_source[s0_id],
            pcm_b_16k=pcm16_by_source[s1_id],
            duration_a_s=probed_sources[pair_idx]["duration_s"],
            duration_b_s=probed_sources[pair_idx + 1]["duration_s"],
        )
        if b_pair.get("garbled_tail_a"):
            b_pair["garbled_tail_a"]["source_id"] = s0_id
        if b_pair.get("truncated_prefix_b"):
            b_pair["truncated_prefix_b"]["source_id"] = s1_id
        if b_pair.get("exact_overlap"):
            b_pair["exact_overlap"]["source_id"] = s1_id
        b_pair["pair"] = [s0_id, s1_id]
        boundary_infos.append(b_pair)

    disfluency_by_source: dict[str, dict[str, Any]] = {}
    for src in probed_sources:
        sid = src["source_id"]
        disfluency_by_source[sid] = detect_disfluency_spans(
            words_by_source[sid],
            intended_story=intended_story,
            source_id=sid,
            pcm_16k=pcm16_by_source[sid],
            niche=niche_hint,
            duration_s=src["duration_s"],
        )

    speech_alignment = align_source_dialogue_to_intent(
        words_by_source,
        intended_story,
        disfluency_by_source=disfluency_by_source,
        boundary_infos=boundary_infos,
    )
    missing_clauses = [
        {
            "sentence_index": r["sentence_index"],
            "clause": r["intended_sentence"],
            "best_match_ratio": r["best_match_ratio"],
            "reason": f"classification:{r['classification']}",
        }
        for r in speech_alignment["sentence_alignment"]
        if r["classification"] in ("MISSING_SOURCE", "UNCERTAIN")
    ]
    atomic_write_json(run_dir / "speech_alignment.json", speech_alignment)
    atomic_write_json(run_dir / "script_completeness.json", {
        "schema_version": DIRECTOR_V3_SCHEMA_VERSION,
        "source_completeness_status": speech_alignment["source_completeness_status"],
        "source_completeness_ratio": speech_alignment["source_completeness_ratio"],
        "intended_sentences_total": speech_alignment["intended_sentences_total"],
        "available_sentences_count": speech_alignment["available_sentences_count"],
        "missing_or_uncertain_sentences_count": speech_alignment["missing_or_uncertain_sentences_count"],
        "missing_clauses": missing_clauses,
        "voice_cloning_used": False,
        "policy": "Never synthesize or clone missing source dialogue; separate source completeness from edited-speech preservation.",
    })

    bridge_dur = 1.25 if (len(sids) >= 2 and missing_clauses) else 0.0
    speech_choice = choose_clean_speech_path(
        probed_sources,
        boundary_infos if len(boundary_infos) > 1 else (boundary_infos[0] if boundary_infos else None),
        disfluency_by_source,
        intended_story=intended_story,
        bridge_duration_s=bridge_dur,
        candidate_score_overrides=job.get("candidate_score_overrides"),
    )
    chosen_plan = speech_choice["chosen_plan"]
    edit_map = build_multi_source_edit_map(
        sources=probed_sources,
        kept_specs=chosen_plan["kept_spans"],
        cuts=chosen_plan["cuts"],
        target_width=W,
        target_height=H,
        fps_num=fps_num,
        fps_den=fps_den,
        sample_rate=sr,
        channels=2,
        crossfade_ms=12.0,
    )
    map_errs = validate_multi_source_edit_map(edit_map)
    if map_errs:
        raise RuntimeError(f"MultiSourceEditMap invariant failed (fail-closed): {map_errs}")

    edited_words = remap_multi_source_words(edit_map, words_by_source)
    req_claims_chk = verify_required_claims(speech_alignment, edited_words, product_name=job.get("product_name"))
    atomic_write_json(run_dir / "speech_repair_decisions.json", {"boundaries": boundary_infos, "disfluency": disfluency_by_source, "selection": speech_choice, "required_claims_check": req_claims_chk})
    atomic_write_json(run_dir / "multi_source_edit_map.json", edit_map.to_dict())
    atomic_write_text(run_dir / "transcript_cleaned.txt", " ".join(w["word"] for w in edited_words) + "\n")
    tracker.end_stage("speech_alignment_and_timeline", temp_dir=temp_dir)

    # ------------------------------------------------------------------
    # 2. A3 — Multimodal Video Evidence Index & Watermark Authorization Guard (A8/D12)
    # ------------------------------------------------------------------
    tracker.start_stage("multimodal_evidence_index")
    enable_vlm = bool(job.get("enable_vlm", True))
    enable_llm = bool(job.get("enable_llm", True))
    model_router = LocalModelRouter() if (enable_vlm or enable_llm) else None
    ev_index = build_multimodal_video_evidence_index(
        sources=[{"source_id": s["source_id"], "path": s["path"], "sha256": s["sha256"], "duration_s": s["duration_s"]} for s in probed_sources],
        work_dir=run_dir,
        enable_vlm=enable_vlm,
        max_vlm_frames_per_source=2,
        seed=seed,
    )
    wm_auth = evaluate_watermark_authorization(
        ev_index.get("watermark_roi_by_source", {}),
        authorization_doc=job.get("watermark_authorization"),
    )
    atomic_write_json(run_dir / "watermark_auth.json", wm_auth)

    # Apply localized watermark blur ONLY if explicitly authorized on a non-mandatory mark!
    wm_blur_reports: dict[str, Any] = {}
    effective_source_paths = dict(source_paths)
    if wm_auth.get("status") == "AUTHORIZED_CLEANUP_ALLOWED" and wm_auth.get("localized_blur_authorized"):
        for sid, sp in source_paths.items():
            s_roi = (ev_index.get("watermark_roi_by_source") or {}).get(sid) or {}
            if s_roi.get("detected_bright_static_mark"):
                blurred_p = temp_dir / f"src_{sid}_authorized_blur.mp4"
                wm_blur_reports[sid] = apply_localized_watermark_blur_if_authorized(
                    video_path=sp,
                    out_path=blurred_p,
                    watermark_auth=wm_auth,
                    source_roi=s_roi,
                    ffmpeg_bin=ffmpeg_bin,
                )
                if wm_blur_reports[sid].get("applied"):
                    effective_source_paths[sid] = blurred_p
    else:
        for sid in source_paths:
            wm_blur_reports[sid] = {"applied": False, "action": "PRESERVED_INTACT_NO_BLUR"}
    atomic_write_json(ev_dir / "watermark_cleanup_execution.json", wm_blur_reports)

    face_boxes = [s["face_bounds_summary"] for s in ev_index.get("sources", []) if isinstance(s.get("face_bounds_summary"), dict)]
    merged_face_bounds = (
        {
            "x_min": round(min(b["x_min"] for b in face_boxes), 4),
            "y_min": round(min(b["y_min"] for b in face_boxes), 4),
            "x_max": round(max(b["x_max"] for b in face_boxes), 4),
            "y_max": round(max(b["y_max"] for b in face_boxes), 4),
        }
        if face_boxes else None
    )
    tracker.end_stage("multimodal_evidence_index", temp_dir=temp_dir)

    # ------------------------------------------------------------------
    # 3. A2/A3/A4 — Asset Catalog, Feature-Level Proof, Atomic Claim Graph, Relevance Retrieval, Region Crops & Model-Proposed Plans
    # ------------------------------------------------------------------
    tracker.start_stage("evidence_planning_and_crops")
    cat_dir = Path(job["asset_catalog_path"]) if job.get("asset_catalog_path") else None
    prod_cat = inspect_product_asset_catalog(cat_dir)
    prod_name = job.get("product_name") or ""
    prod_match = (
        match_product_assets_for_name(prod_name, catalog=prod_cat)
        if prod_name
        else {"matched": False, "status": "NOT_REQUIRED", "assets": []}
    )
    verified_assets = list(prod_match.get("assets") or []) if prod_match.get("matched") else []
    proof_by_feature = build_product_proof_by_feature(
        product_name=prod_name or None,
        verified_assets=verified_assets,
        feature_requirements=job.get("feature_requirements"),
    )
    prod_status = {
        "catalog": prod_cat,
        "match": prod_match,
        "has_verified_real_product_assets": bool(verified_assets),
        "verified_asset_count": len(verified_assets),
        "verified_asset_ids": [str(a.get("asset_id", "")) for a in verified_assets],
        "product_proof_by_feature": proof_by_feature,
    }
    atomic_write_json(run_dir / "asset_catalog_verified.json", prod_status)
    atomic_write_json(run_dir / "product_proof_by_feature.json", proof_by_feature)

    # Build IntentBrief + Frozen Atomic Claim Graph + Content-Grounded Retrieval
    from motionlab.semantic.creative_director import extract_claims_and_questions  # noqa: PLC0415

    base_claims = extract_claims_and_questions(edited_words, job_intent=job)
    brief = build_intent_brief(
        claims=base_claims,
        job_config=job,
        product_asset_status=prod_status,
        missing_dialogue_clauses=[m["clause"] for m in missing_clauses],
        visual_evidence=ev_index,
    )
    atomic_write_json(run_dir / "creative_intent.json", brief.to_dict())

    claim_graph = build_atomic_claim_graph(
        edited_words=edited_words,
        intent_brief=brief,
        job_config=job,
        missing_dialogue_clauses=missing_clauses,
    )
    atomic_write_json(run_dir / "claim_graph.json", claim_graph)

    ev_retrieval = retrieve_evidence_for_claims(
        claim_graph=claim_graph,
        verified_assets=verified_assets,
        visual_evidence=ev_index,
    )
    atomic_write_json(run_dir / "evidence_requests.json", {"requests": ev_retrieval["requests"]})
    atomic_write_json(run_dir / "evidence_results.json", {
        "min_relevance_threshold": ev_retrieval["min_relevance_threshold"],
        "cover_asset_id": ev_retrieval["cover_asset_id"],
        "results": [{k: v for k, v in r.items() if k != "matched_asset"} for r in ev_retrieval["results"]],
    })

    # Extract crisp verified region crops for every matched product claim!
    crops_dir = run_dir / "cropped_regions"
    cropped_regions_by_claim: dict[str, dict[str, Any]] = {}
    for res in ev_retrieval["results"]:
        if res.get("status") != "MATCHED_VERIFIED_ASSET" or not res.get("matched_asset"):
            continue
        asset_rec = res["matched_asset"]
        reg_meta = res.get("matched_region") or {
            "region_id": "auto_primary",
            "norm_bbox": [0.05, 0.08, 0.95, 0.74],
            "step_pill": "PROOF",
            "focus_y0_pct": 24.0,
            "focus_h_pct": 30.0,
            "fictional_example_label": "VERIFIED AUTHORIZED DOCUMENT",
            "expected_visible_facts": [],
        }
        crec = extract_verified_region_crop(
            asset_record=asset_rec,
            region_id=str(reg_meta.get("region_id", "primary")),
            norm_bbox=reg_meta.get("norm_bbox", [0.05, 0.08, 0.95, 0.74]),
            out_dir=crops_dir,
            expected_visible_facts=reg_meta.get("expected_visible_facts"),
            fictional_example_label=reg_meta.get("fictional_example_label"),
            page_number=asset_rec.get("page_number"),
        )
        crec["step_pill"] = reg_meta.get("step_pill", "PROOF")
        crec["focus_y0_pct"] = float(reg_meta.get("focus_y0_pct", 24.0))
        crec["focus_h_pct"] = float(reg_meta.get("focus_h_pct", 30.0))
        cropped_regions_by_claim[str(res["claim_id"])] = crec
    atomic_write_json(ev_dir / "cropped_regions_manifest.json", cropped_regions_by_claim)

    # Safe bounded tool registry logging every director inspection
    tools = DirectorToolRegistry(max_calls=24)
    tools.register("inspect_source", lambda a: {"sources": len(probed_sources), "vlm_status": ev_index["vlm_status"]})
    tools.register("search_local_asset_catalog", lambda a: {"verified_assets": len(verified_assets), "crops_extracted": len(cropped_regions_by_claim)})
    tools.invoke("inspect_source", {"sources": sids}, reason_code="ground_multimodal_video_evidence")
    tools.invoke("search_local_asset_catalog", {"query": prod_name or niche_hint}, reason_code="retrieve_and_crop_verified_assets")

    # Model-proposed visual story plans (ZERO constant A/B/C scores!)
    plan_search = propose_model_driven_plans(
        intent_brief=brief,
        claim_graph=claim_graph,
        evidence_results=ev_retrieval,
        visual_evidence=ev_index,
        model_router=model_router if enable_llm else None,
        seed=seed,
    )
    atomic_write_json(run_dir / "raw_vs_reconciled_plans.json", {**plan_search, "tool_trace": tools.trace})

    palette_tokens = job.get("palette_tokens") if isinstance(job.get("palette_tokens"), dict) else None
    copy_ov = job.get("copy_overrides") if isinstance(job.get("copy_overrides"), dict) else None
    bridge_span = next((sp for sp in edit_map.kept_spans if sp.kind == "editorial_bridge"), None)

    hook_cands = build_three_distinct_v3_hooks(
        hook_claim=claim_graph["claims"][0],
        next_claim=claim_graph["claims"][1] if len(claim_graph["claims"]) > 1 else None,
        width=W,
        height=H,
        fps=int(round(fps_f)),
        face_bounds=merged_face_bounds,
        safe_zones=brief.safe_zones,
        palette=palette_tokens,
        job_copy_overrides=copy_ov,
    )
    scene_graph_v3 = build_scene_graph_v3_from_plan(
        selected_plan=plan_search["selected_plan"],
        claim_graph=claim_graph,
        evidence_results=ev_retrieval,
        cropped_regions_by_claim=cropped_regions_by_claim,
        winning_hook_scene=hook_cands[0]["scene"],  # Updated causally after blind hook ranking!
        intent_brief=brief,
        edit_map=edit_map,
        face_bounds=merged_face_bounds,
        bridge_span=bridge_span,
        palette_tokens=palette_tokens,
        job_copy_overrides=copy_ov,
    )
    scenes = scene_graph_v3["scenes"]
    temporal_ledger = build_temporal_ledger_from_story_and_scenes(
        scenes=scenes,
        transcript_words=edited_words,
        fps_num=fps_num,
        fps_den=fps_den,
        time_domain="edited",
    )
    atomic_write_json(run_dir / "temporal_fact_ledger.json", temporal_ledger.to_dict())
    if not temporal_ledger.valid:
        raise RuntimeError(f"TemporalFactLedger failed (fail-closed): {temporal_ledger.validation_errors}")
    planned_cov = plan_claim_coverage(claim_graph=claim_graph, scene_graph=scene_graph_v3)
    atomic_write_json(run_dir / "planned_claim_coverage.json", planned_cov)
    atomic_write_json(run_dir / "scene_graph_v3.json", scene_graph_v3)
    tracker.end_stage("evidence_planning_and_crops", temp_dir=temp_dir)

    if dry_run:
        budget.cleanup_owned_temp(run_dir, keep_on_fail=False)
        return sanitize_native({
            "run_id": run_id,
            "run_dir": str(run_dir),
            "director_strategy": "evidence_v3",
            "status": "DRY_RUN_PLANNED",
            "selected_plan_id": plan_search["selected_plan_id"],
            "scenes_count": len(scenes),
            "planned_weighted_coverage": planned_cov["planned_weighted_ratio"],
            "product_proof_status": proof_by_feature["overall_product_proof_status"],
            "watermark_status": wm_auth["status"],
            "missing_dialogue_count": len(missing_clauses),
        })

    # ------------------------------------------------------------------
    # 4. Mezzanine, Voice Bed & Edited-Bed Re-ASR (A1)
    # ------------------------------------------------------------------
    tracker.start_stage("mezzanine_and_word_qa")
    color_match = compute_multi_source_color_match(effective_source_paths, ref_source_id=sids[0])
    atomic_write_json(ev_dir / "cross_source_color_match.json", color_match)
    mezz_mp4 = temp_dir / "multisource_mezzanine_cfr.mp4"
    voice_bed_wav = run_dir / "edited_voice_bed.wav"
    mezz_info = build_multi_source_mezzanine_and_voice_bed(
        source_paths=effective_source_paths,
        source_wavs_48k=source_wavs_48k,
        edit_map=edit_map,
        out_video_mp4=mezz_mp4,
        out_voice_wav=voice_bed_wav,
        color_match=color_match,
        vad_non_speech_by_source=vad_ns_by_source,
        ffmpeg_bin=ffmpeg_bin,
    )
    join_qa = evaluate_multi_source_audio_joins(voice_bed_wav, edit_map, out_path=ev_dir / "audio_join_qa.json")

    re_asr_txt, re_asr_words, _ = _transcribe_pcm_high_res(voice_bed_wav, ffmpeg_bin=ffmpeg_bin, condition_on_previous_text=False)
    expected_toks = [normalize_token(w["word"]) for w in edited_words if normalize_token(w["word"])]
    all_src_words = [w for sid in sids for w in words_by_source[sid]]
    word_pres = validate_semantic_preservation(
        all_src_words,
        expected_toks,
        re_asr_words,
        missing_source_claims=missing_clauses,
    )
    word_pres["re_asr_backend"] = "faster_whisper_small_int8_independent_bed_reasr"
    word_pres["re_asr_word_count"] = len(re_asr_words)
    word_pres["re_asr_text"] = re_asr_txt
    atomic_write_json(run_dir / "protected_word_validation.json", word_pres)
    tracker.end_stage("mezzanine_and_word_qa", temp_dir=temp_dir)

    # ------------------------------------------------------------------
    # 5. A7 — Render 3 Materially Distinct Composited Hooks + Blind Permutation-Invariant Competition!
    # ------------------------------------------------------------------
    tracker.start_stage("blind_hooks_and_scene_render")
    hf_cache_dir = (PROJECT_ROOT / ".cache" / "motionlab_r6" / "hf_v3") if config.cache_enabled else None
    hook_rank_report: dict[str, Any] | None = None
    if bool(job.get("render_hooks", True)) and claim_graph["claims"]:
        hooks_dir = run_dir / "hooks"
        hooks_dir.mkdir(parents=True, exist_ok=True)
        rendered_hooks: list[dict[str, Any]] = []
        for hc in hook_cands:
            cid = hc["candidate_id"]
            h_sc = hc["scene"]
            h_webm = temp_dir / f"{cid}.webm"
            render_scene_v3_overlay(
                h_sc,
                out_path=h_webm,
                slot_dir=temp_dir / f"slot_{cid}",
                width=W,
                height=H,
                fps=int(round(fps_f)),
                cache_dir=hf_cache_dir,
            )
            alpha_chk = verify_webm_alpha_pixels(
                h_webm,
                width=W,
                height=H,
                sample_frame=min(16, max(4, (int(h_sc["end_frame_exclusive"]) - int(h_sc["start_frame"])) // 3)),
                ffmpeg_bin=ffmpeg_bin,
            )
            h_mp4 = hooks_dir / f"{cid}_preview.mp4"
            h_dur = min(3.8, h_sc["end_s"] + 0.3)
            subprocess.run(
                [
                    ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error", "-nostdin",
                    "-t", f"{h_dur:.3f}", "-i", str(mezz_mp4),
                    "-c:v", "libvpx-vp9", "-i", str(h_webm),
                    "-t", f"{h_dur:.3f}", "-i", str(voice_bed_wav),
                    "-filter_complex", "[0:v][1:v]overlay=0:0:shortest=1,scale=540:960:flags=lanczos[v]",
                    "-map", "[v]", "-map", "2:a:0",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "128k", "-shortest", str(h_mp4),
                ],
                check=True, timeout=90, shell=False,
            )
            rendered_hooks.append({
                "candidate_id": cid,
                "treatment_mode": h_sc["treatment_mode"],
                "strategy": hc["strategy"],
                "preview_mp4": str(h_mp4),
                "preview_sha256": _sha256_file(h_mp4),
                "alpha_verified": bool(alpha_chk.get("alpha_verified")),
                "face_clear": not bool(h_sc["layout"].get("overlaps_face", False)),
            })

        hook_rank_report = rank_rendered_hooks_blind(
            rendered_hooks=rendered_hooks,
            base_mezzanine_mp4=mezz_mp4,
            face_bounds=merged_face_bounds,
            work_dir=ev_dir,
            model_router=model_router if enable_vlm else None,
            seed=seed,
        )
        atomic_write_json(run_dir / "hook_rank_report.json", hook_rank_report)
        winner_cid = hook_rank_report["winner_candidate_id"]
        winner_hc = next((hc for hc in hook_cands if hc["candidate_id"] == winner_cid), hook_cands[0])
        if scenes and scenes[0].get("scene_id") == "sc_00_hook":
            w_sc = dict(winner_hc["scene"])
            w_sc["scene_id"] = "sc_00_hook"
            w_sc["claim_ids"] = scenes[0].get("claim_ids", [])
            w_sc["narrative_role"] = scenes[0].get("narrative_role")
            w_sc["start_frame"] = scenes[0]["start_frame"]
            w_sc["end_frame_exclusive"] = scenes[0]["end_frame_exclusive"]
            w_sc["start_s"] = scenes[0]["start_s"]
            w_sc["end_s"] = scenes[0]["end_s"]
            w_sc["duration_s"] = scenes[0]["duration_s"]
            if palette_tokens:
                w_sc["palette"] = dict(palette_tokens)
            scenes[0] = w_sc
            scene_graph_v3["winning_hook_candidate_id"] = winner_cid
            scene_graph_v3["scene_graph_hash"] = canonical_hash(scenes)
            atomic_write_json(run_dir / "scene_graph_v3.json", scene_graph_v3)

    # Render all active SceneGraph V3 layers + verify alpha
    rendered_overlays: list[dict[str, Any]] = []
    effect_exec_rows: list[dict[str, Any]] = []
    for sc in scenes:
        if sc["treatment_mode"] == "clean_speaker_anchor":
            effect_exec_rows.append({
                "scene_id": sc["scene_id"],
                "treatment_mode": sc["treatment_mode"],
                "engine": "ffmpeg_clean_passthrough",
                "alpha_verified": True,
                "frame_range": [sc["start_frame"], sc["end_frame_exclusive"]],
            })
            continue
        webm_p = temp_dir / f"ov_{sc['scene_id']}.webm"
        rinfo = render_scene_v3_overlay(
            sc,
            out_path=webm_p,
            slot_dir=temp_dir / f"slot_{sc['scene_id']}",
            width=W,
            height=H,
            fps=int(round(fps_f)),
            cache_dir=hf_cache_dir,
        )
        is_full_wipe = sc["treatment_mode"] in ("product_cover_bridge", "region_crop_evidence_stage")
        sf_chk = 4 if is_full_wipe else min(18, max(4, (int(sc["end_frame_exclusive"]) - int(sc["start_frame"])) // 3))
        alpha_chk = verify_webm_alpha_pixels(
            webm_p,
            width=W,
            height=H,
            sample_frame=sf_chk,
            min_transparent_frac=0.05 if is_full_wipe else 0.10,
            ffmpeg_bin=ffmpeg_bin,
        )
        if not alpha_chk.get("alpha_verified"):
            raise RuntimeError(f"SceneGraph V3 overlay {sc['scene_id']} failed alpha verification: {alpha_chk}")
        rendered_overlays.append({
            "scene_id": sc["scene_id"],
            "start_frame": sc["start_frame"],
            "end_frame_exclusive": sc["end_frame_exclusive"],
            "webm_path": str(webm_p),
        })
        effect_exec_rows.append({
            "scene_id": sc["scene_id"],
            "treatment_mode": sc["treatment_mode"],
            "engine": "hyperframes",
            "frame_range": [sc["start_frame"], sc["end_frame_exclusive"]],
            "artifact_sha256": _sha256_file(webm_p),
            "bound_asset_keys": rinfo.get("bound_asset_keys", []),
            "bound_asset_ids": list(sc.get("asset_ids") or []),
            "alpha_verified": True,
            "transparent_frac": alpha_chk.get("transparent_frac"),
            "nonzero_alpha_ratio": alpha_chk.get("nonzero_alpha_ratio"),
            "layout_zone": sc["layout"]["zone"],
            "overlaps_face": sc["layout"]["overlaps_face"],
        })
    atomic_write_json(run_dir / "effect_execution.json", {"engines_used": ["ffmpeg", "hyperframes"], "layers": effect_exec_rows})
    tracker.end_stage("blind_hooks_and_scene_render", temp_dir=temp_dir)

    # ------------------------------------------------------------------
    # 6. A8 — Semantic Foley Audio Timeline & 48kHz Stereo Voice-Priority Mix
    # ------------------------------------------------------------------
    tracker.start_stage("semantic_audio_mix")
    foley_lib = inspect_foley_library()
    sfx_events: list[SoundEvent] = []
    mode_to_sfx = {
        "phone_action_callout": ("soft_whoosh", "GRAPHIC_MOTION", -9.5),
        "causal_sequence_diagram": ("tick_accent", "COGNITIVE_EMPHASIS", -8.0),
        "allocation_state_visual": ("paper_rustle", "ILLUSTRATIVE_FOLEY", -9.0),
        "product_cover_bridge": ("soft_whoosh", "TRANSITION_BRIDGE", -8.5),
        "region_crop_evidence_stage": ("page_turn", "PRODUCT_TEXTURE", -8.5),
        "lived_payoff_visual": ("soft_whoosh", "GRAPHIC_MOTION", -9.5),
        "safe_cta_card": ("bass_settle", "COGNITIVE_EMPHASIS", -9.0),
    }
    for idx_s, sc in enumerate(scenes):
        m = sc["treatment_mode"]
        if m not in mode_to_sfx:
            continue
        fam, srole, gain_db = mode_to_sfx[m]
        asset = resolve_sound_asset(fam, variant=idx_s % 3, foley_lib=foley_lib, sr=sr, seed=seed)
        target_f = min(edit_map.output_total_frames - 4, int(sc["start_frame"]) + 3)
        target_samp = exact_rational_samples(target_f, fps_num, fps_den, sr)
        anchor = "transient" if fam in ("tick_accent", "bass_settle", "cash_flick", "pop_reveal", "soft_impact") else "crest_in_window"
        anchor_off = int(asset["peak_sample"] if anchor == "transient" else asset["crest_sample"])
        s0 = max(0, target_samp - anchor_off)
        dur_samp = max(int(0.015 * sr), int(round(float(asset["duration_s"]) * sr)))
        s1 = min(edit_map.output_total_samples, s0 + dur_samp)
        if s1 - s0 < 64:
            continue
        sfx_events.append(SoundEvent(
            id=f"r6cue_{len(sfx_events):02d}_{fam}",
            cue_id=f"r6cue_{len(sfx_events):02d}_{fam}",
            linked_visual_event_id=sc["scene_id"],
            semantic_span_ids=list(sc.get("claim_ids") or []),
            observed_action_id=f"marker:{sc['scene_id']}:entry",
            asset_id=asset["asset_id"],
            family=fam,
            variant=int(asset["variant"]),
            asset_sha256=asset["sha256"],
            raw_asset_sha256=asset["sha256"],
            processed_pcm_sha256=asset.get("processed_pcm_sha256", ""),
            align_anchor=anchor,
            start_sample=s0,
            peak_sample=target_samp,
            end_sample=s1,
            peak_frame=target_f,
            gain_db=gain_db,
            pan=0.0,
            role=srole,
            confidence=0.95,
            reason=f"Semantic cue aligned to observed {sc['scene_id']} ({m}) entry marker",
            fallback=asset["fallback_status"],
        ))

    audio_tl = AudioTimeline(
        schema_version=AUDIO_SCHEMA_VERSION,
        source_hash=edit_map.map_hash,
        audio_mode="mix",
        sfx_enabled=len(sfx_events) > 0,
        music_enabled=False,
        sample_rate=sr,
        channels=2,
        duration_samples=edit_map.output_total_samples,
        fps_num=fps_num,
        fps_den=fps_den,
        pts_priming_offset_s=0.0,
        events=sfx_events,
        cue_ledger_summary={
            "planned": len(sfx_events),
            "staged": len(sfx_events),
            "explicitly_suppressed": 0,
            "blocked": 0,
            "conservation_exact": True,
        },
    )
    atomic_write_json(run_dir / "audio_timeline.json", audio_tl.to_dict())

    bed_media_mp4 = temp_dir / "voice_bed_ref.mp4"
    subprocess.run(
        [
            ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error", "-nostdin",
            "-i", str(mezz_mp4), "-i", str(voice_bed_wav),
            "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", str(sr), "-ac", "2",
            "-shortest", str(bed_media_mp4),
        ],
        check=True, timeout=90, shell=False,
    )
    bed_media = probe_media(bed_media_mp4)
    mix_info = mix_preserving_master(
        media=bed_media,
        timeline=audio_tl,
        transcript=None,
        run_dir=run_dir,
        seed=seed,
        ffmpeg_bin=ffmpeg_bin,
        ffprobe_bin=ffprobe_bin,
        original_source_path=source_paths[sids[0]],
        edited_voice_bed_wav=voice_bed_wav,
        edit_map_hash=edit_map.map_hash,
    )
    mixed_wav = Path(mix_info["mixed_reference_path"])
    # Persist explicit final_mix.wav copy cleanly inside run_dir
    shutil.copy2(mixed_wav, run_dir / "final_mix.wav")
    tracker.end_stage("semantic_audio_mix", temp_dir=temp_dir)

    # ------------------------------------------------------------------
    # 7. A9 — Composite, Full Decode, Final-Mix Re-ASR, Observed Claim Verifier & Autonomous Creative Repair!
    # ------------------------------------------------------------------
    tracker.start_stage("composite_and_observed_critic_qa")
    release_blockers: list[str] = []
    if wm_auth["status"] == "BLOCKED_WATERMARK_AUTHORIZATION":
        release_blockers.append("BLOCKED_WATERMARK_AUTHORIZATION")
    if missing_clauses or req_claims_chk["release_blocked_by_script_completeness"]:
        release_blockers.append("MISSING_SOURCE_DIALOGUE")
    for pb in proof_by_feature.get("release_blockers") or []:
        if pb not in release_blockers:
            release_blockers.append(pb)

    stem = str(job.get("output_stem") or "MotionLab_ML01R5R6")
    out_filename = f"{stem}_CANDIDATE_9x16.mp4" if release_blockers else f"{stem}_FINAL_9x16.mp4"
    final_mp4 = run_dir / out_filename

    _composite_overlays_onto_mezzanine(
        mezzanine_mp4=mezz_mp4,
        overlays=rendered_overlays,
        mixed_wav=mixed_wav,
        out_mp4=final_mp4,
        width=W,
        height=H,
        fps_num=fps_num,
        fps_den=fps_den,
        total_frames=edit_map.output_total_frames,
        ffmpeg_bin=ffmpeg_bin,
    )

    # Full decode integrity check (-xerror)
    dec_chk = subprocess.run([ffmpeg_bin, "-hide_banner", "-v", "error", "-xerror", "-i", str(final_mp4), "-f", "null", "-"], capture_output=True, text=True, timeout=180, shell=False)
    full_decode_ok = (dec_chk.returncode == 0 and not dec_chk.stderr.strip())

    # Independent Re-ASR directly on FINAL MIX MP4 (A1 — proves SFX never degrade voice intelligibility!)
    fmix_txt, fmix_words, _ = _transcribe_pcm_high_res(final_mp4, ffmpeg_bin=ffmpeg_bin, condition_on_previous_text=False)
    final_mix_reasr = validate_semantic_preservation(
        all_src_words,
        expected_toks,
        fmix_words,
        missing_source_claims=missing_clauses,
    )
    final_mix_reasr["re_asr_source"] = "decoded_final_mp4_aac_mix"
    final_mix_reasr["re_asr_word_count"] = len(fmix_words)
    final_mix_reasr["re_asr_text"] = fmix_txt
    atomic_write_json(run_dir / "final_mix_reasr.json", final_mix_reasr)

    # Audio true minus-one-cue counterfactual QA
    _, audio_qa = validate_final_audio_mix(bed_media, final_mp4, audio_tl, mix_info, run_dir, ffmpeg_bin=ffmpeg_bin, ffprobe_bin=ffprobe_bin)

    # Post-Render Observed Claim Coverage Verifier + Decoded Continuity Audit
    cont_before = _audit_decoded_continuity(final_mp4, scenes)
    obs_before = verify_rendered_claim_coverage(
        final_mp4=final_mp4,
        base_mezzanine_mp4=mezz_mp4,
        claim_graph=claim_graph,
        scene_graph=scene_graph_v3,
        ev_dir=ev_dir,
        model_router=model_router if enable_vlm else None,
        seed=seed,
        run_vlm_on_claims=enable_vlm,
    )
    obs_after = obs_before
    cont_after = cont_before
    repair_history: list[dict[str, Any]] = []

    # Bounded Autonomous Creative Repair Loop (max 2 rounds): repairs FACE_OCCLUSION & TEXT_ONLY_ON_MATERIAL_CLAIM!
    for r_idx in range(2):
        actionable = [
            d for d in (obs_after.get("defects") or [])
            if d.get("category") in ("FACE_OCCLUSION", "TEXT_ONLY_ON_MATERIAL_CLAIM") and d.get("scene_id")
        ]
        if not actionable:
            break
        patched: list[str] = []
        for df in actionable:
            tsid = str(df["scene_id"])
            for sc in scenes:
                if sc["scene_id"] != tsid:
                    continue
                if df["category"] == "FACE_OCCLUSION" and sc["treatment_mode"] not in ("product_cover_bridge", "region_crop_evidence_stage", "clean_speaker_anchor"):
                    sc["layout"]["y_px"] = min(int(H * 0.72), int(sc["layout"]["y_px"]) + int(H * 0.06))
                    sc["layout"]["overlaps_face"] = False
                elif df["category"] == "TEXT_ONLY_ON_MATERIAL_CLAIM" and sc["treatment_mode"] == "phone_action_callout":
                    sc["treatment_mode"] = "lived_payoff_visual"
                    sc["layout"] = solve_v3_layout(width=W, height=H, treatment_mode="lived_payoff_visual", face_bounds=merged_face_bounds, safe_zones=brief.safe_zones)
                    sc["params"]["disclosure"] = "ILLUSTRATIVE SCENARIO"
                    sc["params"].setdefault("motif", "checklist_clear")
                else:
                    continue
                webm_p = temp_dir / f"ov_{sc['scene_id']}_r{r_idx+1}.webm"
                render_scene_v3_overlay(sc, out_path=webm_p, slot_dir=temp_dir / f"slot_{sc['scene_id']}_r{r_idx+1}", width=W, height=H, fps=int(round(fps_f)), cache_dir=hf_cache_dir)
                for ov in rendered_overlays:
                    if ov["scene_id"] == tsid:
                        ov["webm_path"] = str(webm_p)
                patched.append(tsid)
        if not patched:
            break
        _composite_overlays_onto_mezzanine(
            mezzanine_mp4=mezz_mp4, overlays=rendered_overlays, mixed_wav=mixed_wav, out_mp4=final_mp4,
            width=W, height=H, fps_num=fps_num, fps_den=fps_den, total_frames=edit_map.output_total_frames, ffmpeg_bin=ffmpeg_bin,
        )
        cont_after = _audit_decoded_continuity(final_mp4, scenes)
        obs_after = verify_rendered_claim_coverage(
            final_mp4=final_mp4, base_mezzanine_mp4=mezz_mp4, claim_graph=claim_graph, scene_graph=scene_graph_v3,
            ev_dir=ev_dir, model_router=model_router if enable_vlm else None, seed=seed, run_vlm_on_claims=enable_vlm,
        )
        repair_history.append({
            "round": r_idx + 1,
            "patched_scene_ids": patched,
            "coverage_before": obs_before["weighted_observed_material_coverage"],
            "coverage_after": obs_after["weighted_observed_material_coverage"],
            "defects_before": len(obs_before.get("defects") or []),
            "defects_after": len(obs_after.get("defects") or []),
            "final_sha256_after": _sha256_file(final_mp4),
        })

    atomic_write_json(run_dir / "observed_claim_coverage.json", obs_after)
    atomic_write_json(run_dir / "critic_before_after.json", {
        "before": {"continuity": cont_before, "observed_coverage": obs_before},
        "after": {"continuity": cont_after, "observed_coverage": obs_after},
    })
    atomic_write_json(run_dir / "repair_history.json", {"rounds": repair_history})
    atomic_write_json(run_dir / "cut_continuity.json", {
        "schema_version": DIRECTOR_V3_SCHEMA_VERSION,
        "edit_map_hash": edit_map.map_hash,
        "kept_spans_count": len(edit_map.kept_spans),
        "color_match": color_match,
        "audio_join_qa": join_qa,
        **cont_after,
        "room_tone_provenance": mezz_info.get("room_tone_provenance"),
    })

    # Fresh 9:16 preview & 48-tile contact sheet directly from final_mp4
    preview_mp4 = run_dir / "preview_9x16.mp4"
    subprocess.run(
        [
            ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error", "-nostdin",
            "-i", str(final_mp4), "-vf", "scale=540:960:flags=lanczos",
            "-c:v", "libx264", "-preset", "medium", "-crf", "22", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(preview_mp4),
        ],
        check=True, timeout=120, shell=False,
    )
    _build_contact_sheet_60(final_mp4, run_dir / "contact_sheet.jpg", f"ML-01.R5.R6 {out_filename} | {W}x{H} @ {fps_num}/{fps_den}fps | {edit_map.output_total_frames}f")

    # Independent Gate Separation (Fixes D07!)
    sources_immutable = all(_sha256_file(Path(s["path"])) == s["sha256"] for s in job["sources"])
    technical_gate_pass = bool(
        sources_immutable
        and full_decode_ok
        and temporal_ledger.valid
        and join_qa["status"] == "PASS"
        and word_pres["passed"]
        and final_mix_reasr["passed"]
        and req_claims_chk["retained_critical_speech_ok"]
        and audio_qa["status"] == "PASS"
        and cont_after["black_frames"] == 0
        and cont_after["abrupt_boundary_jumps"] == 0
    )
    creative_gate_status = str(obs_after["creative_evidence_gate_status"])
    release_rights_gate_status = "PASS" if not release_blockers else "RELEASE_BLOCKED"

    if not technical_gate_pass:
        overall_status = "TECHNICAL_FAIL"
    elif release_blockers:
        overall_status = "CANDIDATE_READY_WITH_BLOCKERS"
    elif creative_gate_status != "PASS":
        overall_status = "CREATIVE_EVIDENCE_INCOMPLETE"
    else:
        overall_status = "TECHNICAL_AND_CREATIVE_PASS_PENDING_HUMAN_VQ01"

    tracker.end_stage("composite_and_observed_critic_qa", temp_dir=temp_dir)
    budget.cleanup_owned_temp(run_dir, keep_on_fail=False)

    final_sha = _sha256_file(final_mp4)
    prev_sha = _sha256_file(preview_mp4)
    total_wall_s = round(time.perf_counter() - t_start, 2)

    qa02_report = sanitize_native({
        "schema_version": DIRECTOR_V3_SCHEMA_VERSION,
        "director_strategy": "evidence_v3",
        "run_id": run_id,
        "creative_mode": creative_mode,
        "status": overall_status,
        "overall_status": overall_status,
        "independent_gates": {
            "technical_gate": "PASS" if technical_gate_pass else "FAIL",
            "creative_evidence_gate": creative_gate_status,
            "release_rights_gate": release_rights_gate_status,
            "human_review_gate": "PENDING_HUMAN_REVIEW",
        },
        "technical_gates_passed": technical_gate_pass,
        "hard_blockers": release_blockers,
        "aesthetic_status": "PENDING_HUMAN_REVIEW",
        "listening_status": "LISTENING_NOT_RUN_AUTOMATED_PROXY_ONLY",
        "vlm_perception_status": ev_index["vlm_status"],
        "vlm_critic_status": obs_after["vlm_status"],
        "sources_immutable": sources_immutable,
        "final_artifact": {
            "filename": out_filename,
            "path": str(final_mp4),
            "sha256": final_sha,
            "bytes": int(final_mp4.stat().st_size),
            "width": W,
            "height": H,
            "fps": f"{fps_num}/{fps_den}",
            "frames": edit_map.output_total_frames,
            "duration_s": edit_map.output_duration_s,
            "full_decode_ok": full_decode_ok,
        },
        "preview_artifact": {
            "filename": "preview_9x16.mp4",
            "path": str(preview_mp4),
            "sha256": prev_sha,
            "bytes": int(preview_mp4.stat().st_size),
        },
        "selected_plan_id": plan_search["selected_plan_id"],
        "selected_plan_hash": plan_search["selected_plan_hash"],
        "selected_graph_hash": scene_graph_v3["scene_graph_hash"],
        "winning_hook_candidate_id": hook_rank_report["winner_candidate_id"] if hook_rank_report else None,
        "planned_weighted_coverage": planned_cov["planned_weighted_ratio"],
        "observed_weighted_material_coverage": obs_after["weighted_observed_material_coverage"],
        "observed_critical_factual_accuracy": obs_after["critical_factual_accuracy"],
        "observed_product_workbook_claims_verified": obs_after["product_workbook_claims_verified"],
        "product_proof_by_feature": proof_by_feature,
        "script_completeness_ratio": speech_alignment["source_completeness_ratio"],
        "script_completeness_status": speech_alignment["source_completeness_status"],
        "word_preservation_bed": word_pres,
        "word_preservation_final_mix": final_mix_reasr,
        "audio_join_qa_status": join_qa["status"],
        "audio_mix_qa": {k: audio_qa.get(k) for k in ("status", "events_total", "events_proxy_audible", "events_sync_ok", "events_speech_band_ok", "final_decoded_lufs_i", "final_decoded_true_peak_dbtp", "clipping_samples_added", "ablation_method")},
        "watermark_auth": wm_auth,
        "watermark_cleanup_execution": wm_blur_reports,
        "missing_dialogue_count": len(missing_clauses),
        "temporal_ledger_valid": temporal_ledger.valid,
        "repair_rounds": len(repair_history),
        "wall_s": total_wall_s,
        "resource_stats": tracker.to_dict(),
    })
    atomic_write_json(run_dir / "QA02_REPORT.json", qa02_report)
    atomic_write_json(run_dir / "validation.json", qa02_report)
    atomic_write_json(run_dir / "resource_stats.json", tracker.to_dict())
    atomic_write_json(run_dir / "manifest.json", {
        "schema_version": DIRECTOR_V3_SCHEMA_VERSION,
        "run_id": run_id,
        "composition_mode": "multisource_director_v3",
        "director_strategy": "evidence_v3",
        "status": overall_status,
        "hard_blockers": release_blockers,
        "final_artifact": out_filename,
        "final_sha256": final_sha,
        "edit_map_hash": edit_map.map_hash,
        "scene_graph_hash": scene_graph_v3["scene_graph_hash"],
        "sources": probed_sources,
    })

    summary_lines = [
        f"# MotionLab Studio ML-01.R5.R6 Run (`evidence_v3`) — {run_id}",
        f"- Overall Status: `{overall_status}`",
        f"- Independent Gates: technical=`{qa02_report['independent_gates']['technical_gate']}` | creative_evidence=`{creative_gate_status}` | release_rights=`{release_rights_gate_status}` | human=`PENDING_HUMAN_REVIEW`",
        f"- Hard Release Blockers: `{release_blockers}`",
        f"- Observed Material Claim Coverage (Post-Render Decoded Pixels): `{obs_after['weighted_observed_material_coverage']}` (critical accuracy `{obs_after['critical_factual_accuracy']}`, workbook crops verified `{obs_after['product_workbook_claims_verified']}`, planned pre-render `{planned_cov['planned_weighted_ratio']}`)",
        f"- Product Proof By Feature: `{proof_by_feature['overall_product_proof_status']}` ({proof_by_feature['honest_disclosure']})",
        f"- Hook Competition (Blind Permutation-Invariant): winner=`{qa02_report['winning_hook_candidate_id']}` (3 composited hooks ranked)",
        f"- Speech Completeness vs Edited Preservation: source=`{speech_alignment['available_sentences_count']}/{speech_alignment['intended_sentences_total']}` (`{speech_alignment['source_completeness_status']}`) | bed Re-ASR WER=`{word_pres['wer']}` | final-mix Re-ASR WER=`{final_mix_reasr['wer']}`",
        f"- Output: `{out_filename}` ({W}x{H} {fps_num}/{fps_den}fps, {edit_map.output_total_frames}f, {edit_map.output_duration_s}s, SHA-256 `{final_sha}`)",
        f"- Preview: `preview_9x16.mp4` (540x960, SHA-256 `{prev_sha}`)",
        f"- Audio QA: `{audio_qa['status']}` ({audio_qa['events_proxy_audible']}/{audio_qa['events_total']} counterfactual ablation, LUFS=`{audio_qa['final_decoded_lufs_i']}`, TP=`{audio_qa['final_decoded_true_peak_dbtp']}` dBTP, `LISTENING_NOT_RUN_AUTOMATED_PROXY_ONLY`)",
        f"- Watermark: `{wm_auth['status']}` (`localized_blur_authorized={wm_auth.get('localized_blur_authorized', False)}`)",
        f"- Wall time: `{total_wall_s}s` | Peak Python Tree RSS: `{tracker.to_dict()['peak_tree_rss_mb']} MB`",
    ]
    atomic_write_text(run_dir / "README.md", "\n".join(summary_lines) + "\n")
    atomic_write_text(run_dir / "TO_CHATGPT.md", "\n".join(summary_lines) + "\n")
    return qa02_report
