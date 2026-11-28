"""Phase A0-A15 (ML-01.R5.R7): Generative Visual Reasoning & Cinematic Composition Runner (`director_strategy: "generative_v4"`).

Opt-in production runner invoked via existing CLI (`python -m motionlab edit --job <job.json> --creative-mode director`)
when job manifest specifies `"director_strategy": "generative_v4"`. Preserves 100% of legacy, R5.R5, and R5.R6 behavior!

Connects all ML-01.R5.R7 subsystems end-to-end:
  - A1: Multi-model LocalModelRouter (supports Qwen3-4B-Instruct-2507 + Qwen3-VL-4B-Instruct + incumbents)
  - A2: Global StoryBible (`story_bible.py`), canonical sentence segmentation (fixing 8/10 vs 9/11 mismatch),
        natural chapter segmentation, persistent entity snapshots, zero claims[:10] truncation
  - A3: Generative Visual Planner (`generative_visual_planner.py`) authoring structured SceneProgram V4 graphs
  - A4: MIT pdfminer.six text bbox extraction, semantic region crops, generic verified worked-example state machine
  - A5/A6: Dual-engine SceneGraph V4 (Remotion `SceneGraphV4Composition` + HyperFrames GSAP) + global MotionEventTrack
           + exact [0, N) frame coverage manifest + multi-phase illustrated action sequences (fixing D04!)
  - A7: Chunked DAG manifest + `TotalProcessTreeMonitor` (fixing D12 total child-process RSS tracking!)
  - A8: Continuous event-driven global sonic score (`continuous_score.py`), voice-ducked phase-continuous ambient bed,
        exact sample alignment & stratified audio QA (fixing D05/D13!)
  - A9: Protected speech & independent final-mix AAC Re-ASR
  - A10: Multi-frame sequence blind hook competition (`rank_hook_sequences_blind`, fixing D07), independent multi-frame
         semantic comprehension critic (`semantic_comprehension_critic.py`, per-chapter VLM budget, fixing D01/D02),
         and dependency-closure creative repair with rollback (fixing D06!)
"""

from __future__ import annotations

import datetime
import hashlib
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from motionlab.assets.catalog import inspect_product_asset_catalog, match_product_assets_for_name
from motionlab.assets.pdf_regions import (
    build_product_proof_by_feature,
    build_verified_example_state_machine,
    extract_pdf_text_bboxes,
    extract_verified_region_crop,
    index_evidence_assets,
    retrieve_claim_evidence,
)
from motionlab.audio.continuous_score import (
    build_global_sonic_score,
    render_streaming_stems,
    validate_actual_audio_event_alignment,
)
from motionlab.audio.mixer import mix_preserving_master, validate_final_audio_mix
from motionlab.composition.hook_competition import rank_hook_sequences_blind
from motionlab.composition.longform_scheduler import TotalProcessTreeMonitor, build_render_dag
from motionlab.composition.multi_source_timeline import (
    apply_localized_watermark_blur_if_authorized,
    build_multi_source_edit_map,
    build_multi_source_mezzanine_and_voice_bed,
    compute_multi_source_color_match,
    evaluate_multi_source_audio_joins,
    evaluate_watermark_authorization,
    remap_multi_source_words,
    validate_multi_source_edit_map,
)
from motionlab.composition.scene_graph_v4 import (
    build_dependency_index,
    build_motion_event_track_and_frame_manifest,
    compile_scene_program,
    render_scene_v4_overlay,
)
from motionlab.config import EditConfig, PROJECT_ROOT
from motionlab.preflight import probe_media
from motionlab.renderers.base import verify_webm_alpha_pixels
from motionlab.resource import ResourceBudget, ResourceTracker
from motionlab.schemas import atomic_write_json, atomic_write_text, canonical_hash, sanitize_native
from motionlab.semantic.creative_director import build_intent_brief
from motionlab.semantic.generative_visual_planner import ReadOnlyEvidenceToolRegistry, propose_beat_programs
from motionlab.semantic.model_router import LocalModelRouter, discover_compatible_models
from motionlab.semantic.semantic_comprehension_critic import (
    plan_dependency_closure_repair,
    verify_rendered_semantic_comprehension,
)
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
from motionlab.semantic.story_bible import (
    build_chapter_context,
    build_global_story_bible,
    canonical_split_sentences,
    validate_story_coverage,
)
from motionlab.semantic.temporal_ledger import build_temporal_ledger_from_story_and_scenes
from motionlab.semantic.video_evidence import build_multimodal_video_evidence_index

DIRECTOR_V4_SCHEMA_VERSION = "4.0.0"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _audit_decoded_continuity(final_mp4: Path, scenes: list[dict[str, Any]]) -> dict[str, Any]:
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
                boundary_audit.append({"scene_id": sc["scene_id"], "edge": edge_lbl, "frame": bf, "decoded_step_mad": mad, "abrupt_flagged": abrupt})
    return sanitize_native({
        "frames_inspected": f_idx,
        "black_frames": black_frames,
        "abrupt_boundary_jumps": high_jumps,
        "boundary_audit": boundary_audit,
    })


def _blend_ambient_into_wav(mixed_wav: Path, ambient_wav: Path, out_wav: Path) -> None:
    import wave  # noqa: PLC0415

    with wave.open(str(mixed_wav), "rb") as wm, wave.open(str(ambient_wav), "rb") as wa:
        sr = wm.getframerate()
        ch = wm.getnchannels()
        m_i16 = np.frombuffer(wm.readframes(wm.getnframes()), dtype=np.int16).astype(np.float32)
        a_i16 = np.frombuffer(wa.readframes(wa.getnframes()), dtype=np.int16).astype(np.float32)
    n = min(len(m_i16), len(a_i16))
    comb = m_i16.copy()
    comb[:n] = np.clip(m_i16[:n] + a_i16[:n], -32400.0, 32400.0)
    with wave.open(str(out_wav), "wb") as wo:
        wo.setnchannels(ch)
        wo.setsampwidth(2)
        wo.setframerate(sr)
        wo.writeframes(comb.astype(np.int16).tobytes())


def run_generative_v4_director_job(
    *,
    job: dict[str, Any],
    output_root: Path,
    config: EditConfig,
    creative_mode: str = "director",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Execute full ML-01.R5.R7 `generative_v4` end-to-end production job."""
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
    tree_mon = TotalProcessTreeMonitor(run_dir)
    tree_mon.sample("job_start")
    atomic_write_json(run_dir / "job_manifest.json", job)
    atomic_write_json(run_dir / "models_installed_manifest.json", discover_compatible_models())

    text_model_id = str(job.get("text_model_id") or "qwen3_4b_instruct_2507_q4_k_m")
    vlm_model_id = str(job.get("vlm_model_id") or "qwen3_vl_4b_instruct_q4_k_m")
    prefer_remotion = bool(job.get("prefer_remotion", True))
    enable_ambient_bed = bool(job.get("enable_ambient_bed", True))
    max_render_workers = max(1, min(2, int(job.get("max_render_workers", 2))))

    # ------------------------------------------------------------------
    # 1. A9 — Ingest, Speech Perception, Canonical Sentence Segmentation & Rational EditMap
    # ------------------------------------------------------------------
    tracker.start_stage("ingest_and_speech")
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
        else:
            full_txt, words, pcm_16k = "", [], np.zeros(max(1600, int(media.duration_s * 16000)), dtype=np.int16)
            vad = {"vad_backend": "silent_source_no_audio", "non_speech_spans": []}

        words_by_source[sid] = words
        pcm16_by_source[sid] = pcm_16k
        vad_ns_by_source[sid] = list(vad.get("non_speech_spans") or [])
        speech_raw_doc[sid] = {
            "source_id": sid,
            "sha256": s["sha256"],
            "duration_s": media.duration_s,
            "fps": f"{media.fps.num}/{media.fps.den}",
            "full_text": full_txt,
            "word_count": len(words),
            "words": words,
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
    tree_mon.sample("after_ingest")
    tracker.end_stage("ingest_and_speech", temp_dir=temp_dir)

    tracker.start_stage("speech_alignment_and_story_bible")
    intended_story = str(job.get("intended_story") or "").strip() or None
    canonical_seg = canonical_split_sentences(intended_story)
    niche_hint = str(job.get("niche_hint") or "general").lower()
    sids = [s["source_id"] for s in probed_sources]

    boundary_infos: list[dict[str, Any]] = []
    for pair_idx in range(max(0, len(sids) - 1)):
        s0_id, s1_id = sids[pair_idx], sids[pair_idx + 1]
        b_pair = detect_cross_source_overlap(
            words_by_source[s0_id], words_by_source[s1_id], intended_story=intended_story,
            pcm_a_16k=pcm16_by_source[s0_id], pcm_b_16k=pcm16_by_source[s1_id],
            duration_a_s=probed_sources[pair_idx]["duration_s"], duration_b_s=probed_sources[pair_idx + 1]["duration_s"],
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
            words_by_source[sid], intended_story=intended_story, source_id=sid,
            pcm_16k=pcm16_by_source[sid], niche=niche_hint, duration_s=src["duration_s"],
        )

    # Canonical single-segmentation completeness (fixes 8/10 vs 9/11 report inconsistency!)
    canonical_joined = " ".join(canonical_seg["sentences"]) if canonical_seg["sentences"] else intended_story
    speech_alignment = align_source_dialogue_to_intent(
        words_by_source, canonical_joined, disfluency_by_source=disfluency_by_source, boundary_infos=boundary_infos,
    )
    missing_clauses = [
        {"sentence_index": r["sentence_index"], "clause": r["intended_sentence"], "best_match_ratio": r["best_match_ratio"], "reason": f"classification:{r['classification']}"}
        for r in speech_alignment["sentence_alignment"]
        if r["classification"] in ("MISSING_SOURCE", "UNCERTAIN")
    ]
    atomic_write_json(run_dir / "speech_alignment.json", {**speech_alignment, "canonical_segmentation_hash": canonical_seg["segmentation_hash"]})
    atomic_write_json(run_dir / "script_completeness.json", {
        "schema_version": DIRECTOR_V4_SCHEMA_VERSION,
        "canonical_segmentation_hash": canonical_seg["segmentation_hash"],
        "canonical_sentence_count": canonical_seg["sentence_count"],
        "source_completeness_status": speech_alignment["source_completeness_status"],
        "source_completeness_ratio": speech_alignment["source_completeness_ratio"],
        "intended_sentences_total": speech_alignment["intended_sentences_total"],
        "available_sentences_count": speech_alignment["available_sentences_count"],
        "missing_or_uncertain_sentences_count": speech_alignment["missing_or_uncertain_sentences_count"],
        "missing_clauses": missing_clauses,
        "voice_cloning_used": False,
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
        sources=probed_sources, kept_specs=chosen_plan["kept_spans"], cuts=chosen_plan["cuts"],
        target_width=W, target_height=H, fps_num=fps_num, fps_den=fps_den, sample_rate=sr, channels=2, crossfade_ms=12.0,
    )
    map_errs = validate_multi_source_edit_map(edit_map)
    if map_errs:
        raise RuntimeError(f"MultiSourceEditMap invariant failed: {map_errs}")

    edited_words = remap_multi_source_words(edit_map, words_by_source)
    req_claims_chk = verify_required_claims(speech_alignment, edited_words, product_name=job.get("product_name"))
    atomic_write_json(run_dir / "multi_source_edit_map.json", edit_map.to_dict())
    atomic_write_text(run_dir / "transcript_cleaned.txt", " ".join(w["word"] for w in edited_words) + "\n")

    # ------------------------------------------------------------------
    # 2. A1/A4 — Multimodal Evidence Index, Watermark Guard, PDF BBoxes & Verified Worked Example
    # ------------------------------------------------------------------
    enable_vlm = bool(job.get("enable_vlm", True))
    enable_llm = bool(job.get("enable_llm", True))
    model_router = LocalModelRouter() if (enable_vlm or enable_llm) else None

    ev_index = build_multimodal_video_evidence_index(
        sources=[{"source_id": s["source_id"], "path": s["path"], "sha256": s["sha256"], "duration_s": s["duration_s"]} for s in probed_sources],
        work_dir=run_dir, enable_vlm=enable_vlm, max_vlm_frames_per_source=2, seed=seed,
    )
    wm_auth = evaluate_watermark_authorization(ev_index.get("watermark_roi_by_source", {}), authorization_doc=job.get("watermark_authorization"))
    atomic_write_json(run_dir / "watermark_auth.json", wm_auth)

    wm_blur_reports: dict[str, Any] = {}
    effective_source_paths = dict(source_paths)
    if wm_auth.get("status") == "AUTHORIZED_CLEANUP_ALLOWED" and wm_auth.get("localized_blur_authorized"):
        for sid, sp in source_paths.items():
            s_roi = (ev_index.get("watermark_roi_by_source") or {}).get(sid) or {}
            if s_roi.get("detected_bright_static_mark"):
                blurred_p = temp_dir / f"src_{sid}_authorized_blur.mp4"
                wm_blur_reports[sid] = apply_localized_watermark_blur_if_authorized(
                    video_path=sp, out_path=blurred_p, watermark_auth=wm_auth, source_roi=s_roi, ffmpeg_bin=ffmpeg_bin,
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

    cat_dir = Path(job["asset_catalog_path"]) if job.get("asset_catalog_path") else None
    prod_cat = inspect_product_asset_catalog(cat_dir)
    prod_name = job.get("product_name") or ""
    prod_match = match_product_assets_for_name(prod_name, catalog=prod_cat) if prod_name else {"matched": False, "status": "NOT_REQUIRED", "assets": []}
    verified_assets = list(prod_match.get("assets") or []) if prod_match.get("matched") else []
    proof_by_feature = build_product_proof_by_feature(product_name=prod_name or None, verified_assets=verified_assets, feature_requirements=job.get("feature_requirements"))
    atomic_write_json(run_dir / "product_proof_by_feature.json", proof_by_feature)

    # Optional MIT pdfminer.six text bbox extraction when owner PDF is explicitly supplied
    pdf_bboxes_doc: dict[str, Any] = {"status": "NO_SOURCE_PDF_CONFIGURED"}
    src_pdf = Path(str(job["source_pdf_path"])) if job.get("source_pdf_path") else None
    if src_pdf is not None and src_pdf.is_file():
        try:
            pb = extract_pdf_text_bboxes(src_pdf, [1, 7, 10, 12])
            pdf_bboxes_doc = {"status": "EXTRACTED_MIT_PDFMINER", "pages": {str(k): v for k, v in pb.items()}}
        except Exception as exc:
            pdf_bboxes_doc = {"status": "PDF_EXTRACT_ERROR", "error": str(exc)[:140]}
    atomic_write_json(ev_dir / "pdf_text_bboxes.json", pdf_bboxes_doc)

    # Deterministic verified worked-example state machine (NEVER asks LLM to invent numbers!)
    worked_example: dict[str, Any] | None = None
    verified_ent_states: dict[str, Any] = {}
    if isinstance(job.get("worked_example"), dict):
        we_cfg = job["worked_example"]
        worked_example = build_verified_example_state_machine(
            example_id=str(we_cfg["example_id"]),
            domain=str(we_cfg.get("domain", niche_hint)),
            unit=str(we_cfg.get("unit", "$")),
            initial_label=str(we_cfg.get("initial_label", "Starting Value")),
            initial_value=float(we_cfg["initial_value"]),
            steps=list(we_cfg.get("steps") or []),
            source_proof=str(we_cfg.get("source_proof", "verified_workbook_pdf")),
            is_hypothetical=bool(we_cfg.get("is_hypothetical", True)),
            expected_final_value=we_cfg.get("expected_final_value"),
        )
        if not worked_example["valid"]:
            raise RuntimeError(f"Worked example arithmetic invariant failed (fail-closed): {worked_example['errors']}")
        verified_ent_states[worked_example["example_id"]] = {
            "label": worked_example["example_id"],
            "unit": worked_example["unit"],
            "initial_value": worked_example["initial_value"],
            "final_value": worked_example["final_value"],
            "source_proof": worked_example["source_proof"],
            "fact_vs_hypothetical": "hypothetical_labeled" if worked_example["is_hypothetical"] else "source_grounded",
        }
        atomic_write_json(run_dir / "worked_example_state.json", worked_example)

    # ------------------------------------------------------------------
    # 3. A2 & A3 — Global StoryBible, Evidence Indexing, Region Crops & Generative Beat Programs
    # ------------------------------------------------------------------
    story_bible = build_global_story_bible(
        job=job,
        source_registry=probed_sources,
        edited_words=edited_words,
        duration_s=edit_map.output_duration_s,
        fps_num=fps_num,
        fps_den=fps_den,
        width=W,
        height=H,
        product_proof_by_feature=proof_by_feature,
        verified_example_states=verified_ent_states,
    )
    atomic_write_json(run_dir / "story_bible.json", story_bible)
    atomic_write_json(run_dir / "chapter_map.json", {"chapters": story_bible["chapters"], "story_bible_hash": story_bible["story_bible_hash"]})

    asset_ev_index = index_evidence_assets(verified_assets=verified_assets, blocked_features=proof_by_feature.get("release_blockers"))
    atomic_write_json(run_dir / "evidence_index.json", asset_ev_index)

    # Content-grounded per-claim evidence matching & verified region cropping (zero hardcoded finance buckets!)
    claim_matches: dict[str, list[dict[str, Any]]] = {}
    crops_by_claim: dict[str, dict[str, Any]] = {}
    crops_dir = run_dir / "cropped_regions"
    asset_lookup = {str(a["asset_id"]): a for a in verified_assets}
    cover_asset = next((a for a in verified_assets if str(a.get("asset_type", "")).lower() == "workbook_cover"), None)
    used_asset_ids: set[str] = set()

    for cl in story_bible["atomic_claims"]:
        cid = str(cl["claim_id"])
        hits = retrieve_claim_evidence(claim=cl, evidence_index=asset_ev_index, top_k=4)
        claim_matches[cid] = hits
        if cl["role"] == "product_or_mechanism_proof" and hits:
            # Prefer distinct relevant workbook page per atomic benefit claim!
            picked = next((h for h in hits if h["asset_id"] not in used_asset_ids and h["asset_type"] != "workbook_cover"), hits[0])
            used_asset_ids.add(picked["asset_id"])
            arec = asset_lookup.get(picked["asset_id"])
            if arec:
                regs = list(arec.get("regions") or [])
                reg0 = regs[0] if regs else {"region_id": "primary", "norm_bbox": [0.05, 0.14, 0.95, 0.78], "focus_y0_pct": 22.0, "focus_h_pct": 28.0}
                crec = extract_verified_region_crop(
                    asset_record=arec,
                    region_id=str(reg0.get("region_id", "primary")),
                    norm_bbox=reg0.get("norm_bbox", [0.05, 0.14, 0.95, 0.78]),
                    out_dir=crops_dir,
                    expected_visible_facts=list(reg0.get("expected_visible_facts") or arec.get("expected_visible_facts") or []),
                    fictional_example_label=arec.get("fictional_example_label"),
                    page_number=arec.get("page_number"),
                )
                crec["focus_y0_pct"] = float(reg0.get("focus_y0_pct", 22.0))
                crec["focus_h_pct"] = float(reg0.get("focus_h_pct", 28.0))
                crec["step_pill"] = f"PROOF {len(crops_by_claim) + 1:02d}"
                crops_by_claim[cid] = crec

    tool_registry = ReadOnlyEvidenceToolRegistry(
        evidence_index=asset_ev_index, claim_matches=claim_matches, story_bible=story_bible, max_calls=max(64, len(story_bible["atomic_claims"]) * 6),
    )

    # Plan EVERY chapter via GenerativeVisualPlanner (ZERO claims[:10] truncation!)
    chapter_plans: list[dict[str, Any]] = []
    all_selected_entries: list[dict[str, Any]] = []
    for ch in story_bible["chapters"]:
        ch_ctx = build_chapter_context(story_bible, ch["chapter_id"])
        ch_plan = propose_beat_programs(
            story_bible=story_bible,
            chapter_context=ch_ctx,
            tool_registry=tool_registry,
            model_router=model_router if enable_llm else None,
            text_model_id=text_model_id,
            seed=seed,
        )
        chapter_plans.append(ch_plan)
        all_selected_entries.extend(ch_plan["selected_programs"])

    atomic_write_json(run_dir / "generative_plan_trace.json", {
        "schema_version": DIRECTOR_V4_SCHEMA_VERSION,
        "text_model_id": text_model_id,
        "chapters_planned": len(chapter_plans),
        "chapter_plans": chapter_plans,
    })

    # Compile SceneGraph V4 scenes across all chapters without dropping any claim!
    safe_zones = (
        {"top": 0.08, "bottom": 0.12, "left": 0.05, "right": 0.05}
        if W > H
        else {"top": 0.11, "bottom": 0.21, "left": 0.06, "right": 0.14}
    )
    palette_tokens = job.get("palette_tokens") or {}
    copy_overrides = job.get("copy_overrides") or {}
    claims_by_id = {str(c["claim_id"]): c for c in story_bible["atomic_claims"]}

    scenes: list[dict[str, Any]] = []
    last_ef = 0
    # Optional authentic cover bridge if edit_map inserted a transition bridge between multi-source clips!
    bridge_span = next((sp for sp in edit_map.kept_spans if getattr(sp, "kind", "") == "editorial_bridge"), None)

    for entry in all_selected_entries:
        cid = str(entry["claim_id"])
        cl = claims_by_id[cid]
        sf = max(last_ef, int(cl["start_frame"]))
        ef = min(edit_map.output_total_frames, max(sf + int(round(fps_f * 0.9)), int(cl["end_frame_exclusive"]) + int(round(fps_f * 0.14))))
        if bridge_span is not None and cover_asset is not None:
            bsf = int(bridge_span.output_start_frame)
            bef = int(bridge_span.output_end_frame_exclusive)
            if last_ef <= bsf < sf and (bef - bsf) >= int(round(fps_f * 0.6)):
                bridge_cl = {
                    "claim_id": "bridge_product_intro",
                    "role": "product_cover_intro",
                    "text": str(prod_name or "Product Introduction"),
                    "start_frame": bsf,
                    "end_frame_exclusive": bef,
                    "question_answered": f"Introduce authentic {prod_name}",
                }
                b_entry = {
                    "claim_id": "bridge_product_intro",
                    "selected_program": {
                        "program_id": "bridge_cover_v4",
                        "authored_by": "deterministic_bridge",
                        "node_types": ["source_clip", "product_cover_stage"],
                        "camera_move": "push_to_evidence",
                        "backend_preference": "hyperframes",
                        "evidence_id_refs": [str(cover_asset["asset_id"])],
                        "audio_cue_role": "TRANSITION_BRIDGE",
                        "min_reading_hold_s": round((bef - bsf) / fps_f, 2),
                    },
                }
                b_sc = compile_scene_program(
                    selected_entry=b_entry, claim=bridge_cl, chapter_id=str(entry["chapter_id"]),
                    width=W, height=H, fps_num=fps_num, fps_den=fps_den, total_frames=edit_map.output_total_frames,
                    face_bounds=merged_face_bounds, safe_zones=safe_zones, palette=palette_tokens,
                    cover_asset=cover_asset, copy_overrides=copy_overrides, start_frame_override=bsf, end_frame_override=bef,
                )
                scenes.append(b_sc)
                last_ef = bef
                sf = max(last_ef, sf)

        if cl["role"] == "cta_direction" and (edit_map.output_total_frames - ef) <= int(round(fps_f * 0.45)):
            ef = edit_map.output_total_frames
        if ef <= sf + 6:
            continue
        sc = compile_scene_program(
            selected_entry=entry,
            claim=cl,
            chapter_id=str(entry["chapter_id"]),
            width=W, height=H, fps_num=fps_num, fps_den=fps_den, total_frames=edit_map.output_total_frames,
            face_bounds=merged_face_bounds, safe_zones=safe_zones, palette=palette_tokens,
            crop_record=crops_by_claim.get(cid),
            cover_asset=cover_asset,
            worked_example=worked_example if cl["role"] == "core_conflict_or_comparison" else None,
            copy_overrides=copy_overrides,
            start_frame_override=sf,
            end_frame_override=ef,
        )
        scenes.append(sc)
        last_ef = ef

    cov_chk = validate_story_coverage(story_bible=story_bible, planned_scene_claim_ids={cid for sc in scenes for cid in sc.get("claim_ids", [])})
    dep_index = build_dependency_index(scenes)
    motion_track, frame_manifest = build_motion_event_track_and_frame_manifest(
        scenes=scenes, total_frames=edit_map.output_total_frames, fps_num=fps_num, fps_den=fps_den,
    )
    scene_graph_v4 = sanitize_native({
        "schema_version": DIRECTOR_V4_SCHEMA_VERSION,
        "story_bible_hash": story_bible["story_bible_hash"],
        "edit_map_hash": edit_map.map_hash,
        "scenes_count": len(scenes),
        "story_coverage": cov_chk,
        "scenes": scenes,
        "dependency_index": dep_index,
        "scene_graph_hash": canonical_hash(scenes),
    })
    atomic_write_json(run_dir / "scene_graph_v4.json", scene_graph_v4)
    atomic_write_json(run_dir / "motion_event_track.json", motion_track)
    atomic_write_json(run_dir / "frame_coverage_manifest.json", frame_manifest)

    # Build temporal fact ledger for continuity validation
    temporal_ledger = build_temporal_ledger_from_story_and_scenes(
        scenes=[{"scene_id": s["scene_id"], "narrative_role": s["story_function"], "start_frame": s["start_frame"], "end_frame_exclusive": s["end_frame_exclusive"], "params": (s["nodes"][-1].get("content_tokens") or {})} for s in scenes],
        transcript_words=edited_words,
        fps_num=fps_num, fps_den=fps_den,
    )
    atomic_write_json(run_dir / "temporal_fact_ledger.json", temporal_ledger.to_dict())
    tracker.end_stage("speech_alignment_and_story_bible", temp_dir=temp_dir)

    if dry_run:
        return sanitize_native({"status": "DRY_RUN_PLAN_OK", "run_id": run_id, "story_bible_hash": story_bible["story_bible_hash"], "scenes_count": len(scenes), "story_coverage": cov_chk})

    # ------------------------------------------------------------------
    # 4. Mezzanine, Color Match, Word QA & Multi-Frame Sequence Blind Hook Competition (D07)
    # ------------------------------------------------------------------
    tracker.start_stage("mezzanine_and_blind_hooks")
    color_match = compute_multi_source_color_match(effective_source_paths, ref_source_id=sids[0])
    mezz_mp4 = temp_dir / "base_mezzanine.mp4"
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

    all_src_words = [w for sid in sids for w in words_by_source[sid]]
    expected_toks = [normalize_token(w["word"]) for w in edited_words if normalize_token(w["word"])]
    re_txt, re_words, _ = _transcribe_pcm_high_res(voice_bed_wav, ffmpeg_bin=ffmpeg_bin, condition_on_previous_text=False)
    word_pres = validate_semantic_preservation(all_src_words, expected_toks, re_words, missing_source_claims=missing_clauses)
    atomic_write_json(run_dir / "protected_word_validation.json", word_pres)

    hf_cache_dir = (PROJECT_ROOT / ".cache" / "motionlab_r5r7_hf_v4").resolve()
    hf_cache_dir.mkdir(parents=True, exist_ok=True)

    # Render 3 materially distinct hook variants & evaluate ordered sequences blindly (D07)!
    hook_rank_report: dict[str, Any] = {"status": "HOOK_COMPETITION_SKIPPED"}
    if bool(job.get("render_hooks", True)) and scenes:
        hooks_dir = run_dir / "hooks"
        hooks_dir.mkdir(parents=True, exist_ok=True)
        hook_cl = story_bible["atomic_claims"][0]
        next_cl = story_bible["atomic_claims"][1] if len(story_bible["atomic_claims"]) > 1 else None
        h_sf = max(2, int(hook_cl["start_frame"]))
        h_ef = min(edit_map.output_total_frames, max(h_sf + int(round(fps_f * 2.2)), min(int(hook_cl["end_frame_exclusive"]), h_sf + int(round(fps_f * 3.8)))))
        hook_specs = [
            ("hook_A_live_action_callout", ["source_clip", "text_label"], "static_focus", "hyperframes"),
            ("hook_B_illustrated_action_seq", ["source_clip", "illustrated_action_sequence", "text_label"], "depth_parallax", "hyperframes"),
            ("hook_C_causal_timeline_preview", ["source_clip", "timeline_axis", "arrow_connector", "text_label"], "pan_sequence", "hyperframes"),
        ]
        rendered_hooks: list[dict[str, Any]] = []
        hook_scenes_by_id: dict[str, dict[str, Any]] = {}
        for hcid, h_nodes, h_cam, h_be in hook_specs:
            h_claim_use = {**hook_cl, "weekdays_mentioned": (next_cl or {}).get("weekdays_mentioned", [])} if hcid.endswith("preview") else hook_cl
            h_sc = compile_scene_program(
                selected_entry={"claim_id": hook_cl["claim_id"], "selected_program": {"program_id": hcid, "authored_by": "hook_competition", "node_types": h_nodes, "camera_move": h_cam, "backend_preference": h_be, "evidence_id_refs": hook_cl["source_evidence_ids"][:3], "audio_cue_role": "GRAPHIC_MOTION", "min_reading_hold_s": 1.2}},
                claim=h_claim_use, chapter_id="ch_00", width=W, height=H, fps_num=fps_num, fps_den=fps_den,
                total_frames=edit_map.output_total_frames, face_bounds=merged_face_bounds, safe_zones=safe_zones,
                palette=palette_tokens, copy_overrides=copy_overrides, start_frame_override=h_sf, end_frame_override=h_ef,
            )
            h_sc["scene_id"] = hcid
            hook_scenes_by_id[hcid] = h_sc
            h_webm = temp_dir / f"{hcid}.webm"
            render_scene_v4_overlay(h_sc, out_path=h_webm, slot_dir=temp_dir / f"slot_{hcid}", width=W, height=H, fps=int(round(fps_f)), cache_dir=hf_cache_dir, prefer_remotion=False)
            alpha_chk = verify_webm_alpha_pixels(h_webm, width=W, height=H, sample_frame=min(14, max(4, (h_ef - h_sf) // 3)), min_transparent_frac=0.10, ffmpeg_bin=ffmpeg_bin)
            h_prev = hooks_dir / f"{hcid}_preview.mp4"
            h_dur = min(4.2, max(2.2, h_ef / fps_f))
            subprocess.run(
                [
                    ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error", "-nostdin",
                    "-t", f"{h_dur:.3f}", "-i", str(mezz_mp4),
                    "-t", f"{h_dur:.3f}", "-i", str(voice_bed_wav),
                    "-c:v", "libvpx-vp9", "-itsoffset", f"{h_sf / fps_f:.4f}", "-i", str(h_webm),
                    "-filter_complex", "[0:v][2:v]overlay=0:0:eof_action=pass,scale=540:960:flags=lanczos,format=yuv420p[v]",
                    "-map", "[v]", "-map", "1:a:0", "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-c:a", "aac", "-b:a", "128k", "-shortest", str(h_prev),
                ],
                check=True, timeout=120, shell=False,
            )
            rendered_hooks.append({
                "candidate_id": hcid,
                "primary_kind": h_sc["primary_kind"],
                "preview_mp4": str(h_prev),
                "preview_sha256": _sha256_file(h_prev),
                "alpha_verified": bool(alpha_chk.get("alpha_verified")),
                "face_clear": not bool(h_sc["composition"]["layout"].get("overlaps_face", False)),
            })

        hook_rank_report = rank_hook_sequences_blind(
            rendered_hooks=rendered_hooks,
            hook_speech_transcript=str(hook_cl.get("text", "")),
            base_mezzanine_mp4=mezz_mp4,
            face_bounds=merged_face_bounds,
            work_dir=ev_dir,
            model_router=model_router if enable_vlm else None,
            vlm_model_id=vlm_model_id,
            seed=seed,
        )
        atomic_write_json(run_dir / "hook_rank_report.json", hook_rank_report)
        # Causally bind winning hook into scene 0!
        win_id = hook_rank_report["winner_candidate_id"]
        if win_id in hook_scenes_by_id and scenes:
            w_sc = dict(hook_scenes_by_id[win_id])
            w_sc["scene_id"] = scenes[0]["scene_id"]
            w_sc["claim_ids"] = scenes[0]["claim_ids"]
            w_sc["start_frame"] = scenes[0]["start_frame"]
            w_sc["end_frame_exclusive"] = scenes[0]["end_frame_exclusive"]
            w_sc["start_s"] = scenes[0]["start_s"]
            w_sc["end_s"] = scenes[0]["end_s"]
            w_sc["duration_s"] = scenes[0]["duration_s"]
            scenes[0] = w_sc
            scene_graph_v4["winning_hook_candidate_id"] = win_id
            scene_graph_v4["scene_graph_hash"] = canonical_hash(scenes)
            atomic_write_json(run_dir / "scene_graph_v4.json", scene_graph_v4)
    tracker.end_stage("mezzanine_and_blind_hooks", temp_dir=temp_dir)

    # ------------------------------------------------------------------
    # 5. A5/A6/A7 — Bounded Parallel Dual-Engine SceneGraph V4 Rendering (Remotion + HyperFrames)
    # ------------------------------------------------------------------
    tracker.start_stage("dual_engine_v4_render")
    dag_manifest = build_render_dag(
        source_hash=edit_map.map_hash, graph_hash=scene_graph_v4["scene_graph_hash"],
        total_frames=edit_map.output_total_frames, fps_num=fps_num, fps_den=fps_den,
        scenes=scenes, story_bible=story_bible,
    )
    atomic_write_json(run_dir / "chunk_manifest.json", dag_manifest)

    # Render active scenes with bounded worker pool (preventing nested CPU/RAM oversubscription!)
    rendered_overlays_map: dict[str, dict[str, Any]] = {}
    effect_exec_rows: list[dict[str, Any]] = []
    engines_used: set[str] = {"ffmpeg"}
    remotion_budget_used = 0

    def _render_one_scene(sc: dict[str, Any], use_rem: bool) -> dict[str, Any]:
        sid = sc["scene_id"]
        if sc["primary_kind"] == "clean_hold":
            return {"scene_id": sid, "clean": True, "engine": "ffmpeg_clean_passthrough"}
        webm_p = temp_dir / f"ov_{sid}.webm"
        rinfo = render_scene_v4_overlay(
            sc, out_path=webm_p, slot_dir=temp_dir / f"slot_{sid}",
            width=W, height=H, fps=int(round(fps_f)), cache_dir=hf_cache_dir, prefer_remotion=use_rem,
        )
        is_full = sc["primary_kind"] in ("region_crop", "product_cover_stage")
        sf_chk = min(10, max(6, (int(sc["end_frame_exclusive"]) - int(sc["start_frame"])) // 3)) if is_full else min(16, max(4, (int(sc["end_frame_exclusive"]) - int(sc["start_frame"])) // 3))
        alpha_chk = verify_webm_alpha_pixels(webm_p, width=W, height=H, sample_frame=sf_chk, min_transparent_frac=0.0 if is_full else 0.10, ffmpeg_bin=ffmpeg_bin)
        ok_alpha = (float(alpha_chk.get("nonzero_alpha_ratio", 0.0)) >= 0.85) if is_full else bool(alpha_chk.get("alpha_verified"))
        if not ok_alpha:
            raise RuntimeError(f"SceneGraph V4 overlay {sid} failed alpha verification: {alpha_chk}")
        tree_mon.sample(f"render_{sid}")
        return {
            "scene_id": sid,
            "clean": False,
            "engine": rinfo.get("engine", "hyperframes"),
            "webm_path": str(webm_p),
            "sha256": _sha256_file(webm_p),
            "alpha_chk": alpha_chk,
            "rinfo": rinfo,
            "start_frame": sc["start_frame"],
            "end_frame_exclusive": sc["end_frame_exclusive"],
            "primary_kind": sc["primary_kind"],
        }

    # Route up to 2 high-impact Remotion-preferred scenes (e.g. timeline_axis + allocation_bar with MIT Counter)
    # through real local Remotion CLI, and remaining scenes through fast cached HyperFrames!
    render_jobs: list[tuple[dict[str, Any], bool]] = []
    for sc in scenes:
        use_rem = prefer_remotion and sc["composition"]["backend_preference"] == "remotion" and sc["primary_kind"] in ("timeline_axis", "allocation_bar") and remotion_budget_used < 2
        if use_rem:
            remotion_budget_used += 1
        render_jobs.append((sc, use_rem))

    with ThreadPoolExecutor(max_workers=max_render_workers) as pool:
        futs = {pool.submit(_render_one_scene, sc, ur): sc for sc, ur in render_jobs}
        for fut in as_completed(futs):
            res = fut.result()
            engines_used.add(res["engine"])
            if res["clean"]:
                effect_exec_rows.append({"scene_id": res["scene_id"], "engine": res["engine"], "alpha_verified": True})
            else:
                rendered_overlays_map[res["scene_id"]] = {
                    "scene_id": res["scene_id"],
                    "start_frame": res["start_frame"],
                    "end_frame_exclusive": res["end_frame_exclusive"],
                    "webm_path": res["webm_path"],
                }
                effect_exec_rows.append({
                    "scene_id": res["scene_id"],
                    "primary_kind": res["primary_kind"],
                    "engine": res["engine"],
                    "artifact_sha256": res["sha256"],
                    "alpha_verified": True,
                    "transparent_frac": res["alpha_chk"].get("transparent_frac"),
                })

    rendered_overlays = [rendered_overlays_map[sc["scene_id"]] for sc in scenes if sc["scene_id"] in rendered_overlays_map]
    atomic_write_json(run_dir / "effect_execution.json", {"engines_used": sorted(engines_used), "layers": effect_exec_rows})
    tracker.end_stage("dual_engine_v4_render", temp_dir=temp_dir)

    # ------------------------------------------------------------------
    # 6. A8 — Continuous Event-Driven Global Sonic Score & Voice-Priority Mix
    # ------------------------------------------------------------------
    tracker.start_stage("continuous_sonic_score")
    audio_tl, score_meta = build_global_sonic_score(
        source_hash=edit_map.map_hash,
        motion_event_track=motion_track,
        chapters=story_bible["chapters"],
        total_samples=edit_map.output_total_samples,
        fps_num=fps_num, fps_den=fps_den, sample_rate=sr,
        niche=niche_hint, enable_ambient_bed=enable_ambient_bed, seed=seed,
    )
    atomic_write_json(run_dir / "audio_timeline.json", audio_tl.to_dict())
    atomic_write_json(run_dir / "sonic_score.json", score_meta)

    bed_media_mp4 = temp_dir / "voice_bed_ref.mp4"
    subprocess.run(
        [
            ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error", "-nostdin",
            "-i", str(mezz_mp4), "-i", str(voice_bed_wav),
            "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", str(sr), "-ac", "2", "-shortest", str(bed_media_mp4),
        ],
        check=True, timeout=90, shell=False,
    )
    bed_media = probe_media(bed_media_mp4)
    mix_info = mix_preserving_master(
        media=bed_media, timeline=audio_tl, transcript=None, run_dir=run_dir, seed=seed,
        ffmpeg_bin=ffmpeg_bin, ffprobe_bin=ffprobe_bin, original_source_path=source_paths[sids[0]],
        edited_voice_bed_wav=voice_bed_wav, edit_map_hash=edit_map.map_hash,
    )
    mixed_wav = Path(mix_info["mixed_reference_path"])

    ambient_meta: dict[str, Any] = {"enabled": False}
    if enable_ambient_bed and niche_hint != "silent_art":
        amb_wav = run_dir / "ambient_bed_stem.wav"
        ambient_meta = render_streaming_stems(voice_wav=voice_bed_wav, out_ambient_bed_wav=amb_wav, sample_rate=sr, bed_gain_below_voice_db=-26.0)
        final_wav = run_dir / "final_mix.wav"
        _blend_ambient_into_wav(mixed_wav, amb_wav, final_wav)
    else:
        final_wav = run_dir / "final_mix.wav"
        shutil.copy2(mixed_wav, final_wav)

    onset_qa = validate_actual_audio_event_alignment(
        audio_timeline=audio_tl, mixed_wav=mixed_wav, voice_bed_wav=voice_bed_wav, fps_num=fps_num, fps_den=fps_den,
    )
    atomic_write_json(ev_dir / "audio_onset_qa.json", {**onset_qa, "ambient_stem": ambient_meta})
    tracker.end_stage("continuous_sonic_score", temp_dir=temp_dir)

    # ------------------------------------------------------------------
    # 7. A10 — Composite, Final-Mix Re-ASR, Independent Multi-Frame Critic & Dependency-Closure Repair
    # ------------------------------------------------------------------
    tracker.start_stage("composite_and_semantic_critic")
    release_blockers: list[str] = []
    if wm_auth["status"] == "BLOCKED_WATERMARK_AUTHORIZATION":
        release_blockers.append("BLOCKED_WATERMARK_AUTHORIZATION")
    if missing_clauses or req_claims_chk["release_blocked_by_script_completeness"]:
        release_blockers.append("MISSING_SOURCE_DIALOGUE")
    for pb in proof_by_feature.get("release_blockers") or []:
        if pb not in release_blockers:
            release_blockers.append(pb)

    stem = str(job.get("output_stem") or "MotionLab_ML01R5R7")
    aspect_tag = "16x9" if W > H else "9x16"
    # Always CANDIDATE unless human approval is explicitly granted (never self-approved!)
    out_filename = f"{stem}_CANDIDATE_{aspect_tag}.mp4"
    final_mp4 = run_dir / out_filename

    _composite_overlays_onto_mezzanine(
        mezzanine_mp4=mezz_mp4, overlays=rendered_overlays, mixed_wav=final_wav, out_mp4=final_mp4,
        width=W, height=H, fps_num=fps_num, fps_den=fps_den, total_frames=edit_map.output_total_frames, ffmpeg_bin=ffmpeg_bin,
    )
    dec_chk = subprocess.run([ffmpeg_bin, "-hide_banner", "-v", "error", "-xerror", "-i", str(final_mp4), "-f", "null", "-"], capture_output=True, text=True, timeout=180, shell=False)
    full_decode_ok = (dec_chk.returncode == 0 and not dec_chk.stderr.strip())

    # Independent Re-ASR directly on FINAL MIX MP4 AAC audio
    fmix_txt, fmix_words, _ = _transcribe_pcm_high_res(final_mp4, ffmpeg_bin=ffmpeg_bin, condition_on_previous_text=False)
    final_mix_reasr = validate_semantic_preservation(all_src_words, expected_toks, fmix_words, missing_source_claims=missing_clauses)
    final_mix_reasr["re_asr_source"] = "decoded_final_mp4_aac_mix"
    final_mix_reasr["re_asr_word_count"] = len(fmix_words)
    final_mix_reasr["re_asr_text"] = fmix_txt
    atomic_write_json(run_dir / "final_mix_reasr.json", final_mix_reasr)

    _, audio_qa = validate_final_audio_mix(bed_media, final_mp4, audio_tl, mix_info, run_dir, ffmpeg_bin=ffmpeg_bin, ffprobe_bin=ffprobe_bin)
    cont_before = _audit_decoded_continuity(final_mp4, scenes)

    sem_before = verify_rendered_semantic_comprehension(
        final_mp4=final_mp4, base_mezzanine_mp4=mezz_mp4, story_bible=story_bible, scenes=scenes,
        ev_dir=ev_dir, model_router=model_router if enable_vlm else None, vlm_model_id=vlm_model_id,
        seed=seed, run_vlm=enable_vlm,
    )
    sem_after = sem_before
    cont_after = cont_before
    repair_history: list[dict[str, Any]] = []

    # Bounded dependency-closure repair loop (max 2 rounds) with automatic rollback on regression!
    for r_idx in range(2):
        actionable_defs = [d for d in (sem_after.get("defects") or []) if d.get("severity") in ("HIGH", "MEDIUM") and d.get("scene_id")]
        if not actionable_defs:
            break
        backup_mp4 = temp_dir / f"baseline_before_repair_r{r_idx + 1}.mp4"
        shutil.copy2(final_mp4, backup_mp4)
        patched_ids: list[str] = []
        for df in actionable_defs[:2]:
            rep_plan = plan_dependency_closure_repair(defect=df, scenes=scenes, dependency_index=dep_index)
            if not rep_plan.get("actionable"):
                continue
            new_sc = rep_plan["repaired_scene"]
            for idx_sc, sc in enumerate(scenes):
                if sc["scene_id"] == new_sc["scene_id"]:
                    scenes[idx_sc] = new_sc
                    webm_p = temp_dir / f"ov_{new_sc['scene_id']}_rep{r_idx + 1}.webm"
                    render_scene_v4_overlay(new_sc, out_path=webm_p, slot_dir=temp_dir / f"slot_{new_sc['scene_id']}_rep{r_idx + 1}", width=W, height=H, fps=int(round(fps_f)), cache_dir=hf_cache_dir, prefer_remotion=False)
                    for ov in rendered_overlays:
                        if ov["scene_id"] == new_sc["scene_id"]:
                            ov["webm_path"] = str(webm_p)
                    patched_ids.append(new_sc["scene_id"])
                    break
        if not patched_ids:
            break
        prev_cov = float(sem_after["weighted_semantic_coverage"])
        _composite_overlays_onto_mezzanine(
            mezzanine_mp4=mezz_mp4, overlays=rendered_overlays, mixed_wav=final_wav, out_mp4=final_mp4,
            width=W, height=H, fps_num=fps_num, fps_den=fps_den, total_frames=edit_map.output_total_frames, ffmpeg_bin=ffmpeg_bin,
        )
        cand_sem = verify_rendered_semantic_comprehension(
            final_mp4=final_mp4, base_mezzanine_mp4=mezz_mp4, story_bible=story_bible, scenes=scenes,
            ev_dir=ev_dir, model_router=model_router if enable_vlm else None, vlm_model_id=vlm_model_id,
            seed=seed, run_vlm=enable_vlm,
        )
        rolled_back = float(cand_sem["weighted_semantic_coverage"]) < prev_cov
        if rolled_back:
            shutil.copy2(backup_mp4, final_mp4)
        else:
            sem_after = cand_sem
            cont_after = _audit_decoded_continuity(final_mp4, scenes)
        repair_history.append({
            "round": r_idx + 1,
            "patched_scene_ids": patched_ids,
            "coverage_before": prev_cov,
            "coverage_after": cand_sem["weighted_semantic_coverage"],
            "rolled_back": rolled_back,
            "final_sha256_after": _sha256_file(final_mp4),
        })
        if rolled_back or float(cand_sem["weighted_semantic_coverage"]) <= prev_cov:
            break

    atomic_write_json(run_dir / "semantic_comprehension_v4.json", sem_after)
    atomic_write_json(run_dir / "repair_history.json", {"rounds": repair_history})
    atomic_write_json(run_dir / "cut_continuity.json", {
        "schema_version": DIRECTOR_V4_SCHEMA_VERSION, "edit_map_hash": edit_map.map_hash,
        "kept_spans_count": len(edit_map.kept_spans), "color_match": color_match,
        "audio_join_qa": join_qa, **cont_after, "room_tone_provenance": mezz_info.get("room_tone_provenance"),
    })

    # Fresh 9:16 (or 16:9) preview & 60-frame contact sheet
    prev_w, prev_h = (960, 540) if W > H else (540, 960)
    preview_mp4 = run_dir / f"preview_{aspect_tag}.mp4"
    subprocess.run(
        [
            ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error", "-nostdin",
            "-i", str(final_mp4), "-vf", f"scale={prev_w}:{prev_h}:flags=lanczos",
            "-c:v", "libx264", "-preset", "medium", "-crf", "22", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(preview_mp4),
        ],
        check=True, timeout=180, shell=False,
    )
    _build_contact_sheet_60(final_mp4, run_dir / "contact_sheet.jpg", f"ML-01.R5.R7 {out_filename} | {W}x{H} @ {fps_num}/{fps_den}fps | {edit_map.output_total_frames}f")

    sources_immutable = all(_sha256_file(Path(s["path"])) == s["sha256"] for s in job["sources"])
    technical_gate_pass = bool(
        sources_immutable and full_decode_ok and temporal_ledger.valid and join_qa["status"] == "PASS"
        and word_pres["passed"] and final_mix_reasr["passed"] and req_claims_chk["retained_critical_speech_ok"]
        and audio_qa["status"] == "PASS" and onset_qa["passed"]
        and frame_manifest["exact_full_coverage"] and cov_chk["valid"]
        and cont_after["black_frames"] == 0 and cont_after["abrupt_boundary_jumps"] == 0
    )
    creative_gate = str(sem_after["creative_evidence_gate"])
    release_gate = "PASS" if not release_blockers else "RELEASE_BLOCKED"

    # Compute generative capability metrics transparently
    avg_model_frac = round(sum(float(p["model_involved_fraction"]) for p in chapter_plans) / max(1, len(chapter_plans)), 4)
    avg_raw_frac = round(sum(float(p["raw_authored_fraction"]) for p in chapter_plans) / max(1, len(chapter_plans)), 4)
    gen_cap_status = "PROVEN" if (avg_model_frac >= 0.4 and len(engines_used) >= 2) else ("PARTIAL" if avg_model_frac > 0 else "DETERMINISTIC_FALLBACK_ONLY")

    overall_status = (
        "TECHNICAL_FAIL" if not technical_gate_pass
        else ("CANDIDATE_READY_WITH_BLOCKERS" if release_blockers else ("CREATIVE_EVIDENCE_INCOMPLETE" if creative_gate != "PASS" else "TECHNICAL_AND_CREATIVE_PASS_PENDING_HUMAN_VQ01"))
    )

    tracker.end_stage("composite_and_semantic_critic", temp_dir=temp_dir)
    tree_mon.sample("job_end")
    budget.cleanup_owned_temp(run_dir, keep_on_fail=False)

    final_sha = _sha256_file(final_mp4)
    prev_sha = _sha256_file(preview_mp4)
    total_wall_s = round(time.perf_counter() - t_start, 2)
    res_stats = {**tracker.to_dict(), "total_process_tree_monitor": tree_mon.summary()}
    atomic_write_json(run_dir / "resource_stats.json", res_stats)

    qa02 = sanitize_native({
        "schema_version": DIRECTOR_V4_SCHEMA_VERSION,
        "director_strategy": "generative_v4",
        "run_id": run_id,
        "status": overall_status,
        "overall_status": overall_status,
        "independent_statuses": {
            "engineering_technical": "PASS" if technical_gate_pass else "FAIL",
            "generative_capability": gen_cap_status,
            "creative_evidence": creative_gate,
            "longform_status": "SHORT_FORM_JOB" if edit_map.output_duration_s < 120.0 else "LONG_FORM_JOB_EXECUTED",
            "product_scope": proof_by_feature.get("overall_product_proof_status"),
            "release_rights": release_gate,
            "human_review": "PENDING_HUMAN_REVIEW",
        },
        "hard_blockers": release_blockers,
        "aesthetic_status": "PENDING_HUMAN_REVIEW",
        "listening_status": "LISTENING_NOT_RUN_AUTOMATED_PROXY_ONLY",
        "human_comprehension_status": "HUMAN_COMPREHENSION_NOT_RUN",
        "engines_used": sorted(engines_used),
        "text_model_id": text_model_id,
        "vlm_model_id": vlm_model_id,
        "generative_authored_raw_fraction": avg_raw_frac,
        "generative_model_involved_fraction": avg_model_frac,
        "final_artifact": {
            "filename": out_filename, "path": str(final_mp4), "sha256": final_sha,
            "bytes": int(final_mp4.stat().st_size), "width": W, "height": H,
            "fps": f"{fps_num}/{fps_den}", "frames": edit_map.output_total_frames,
            "duration_s": round(edit_map.output_duration_s, 4), "full_decode_ok": full_decode_ok,
        },
        "preview_artifact": {"filename": preview_mp4.name, "path": str(preview_mp4), "sha256": prev_sha, "bytes": int(preview_mp4.stat().st_size)},
        "winning_hook_candidate_id": hook_rank_report.get("winner_candidate_id"),
        "hook_tie_detected": hook_rank_report.get("tie_detected"),
        "weighted_semantic_coverage": sem_after["weighted_semantic_coverage"],
        "critical_factual_accuracy": sem_after["critical_factual_accuracy"],
        "uncertain_claims_count": sem_after["uncertain_claims_count"],
        "vlm_total_calls": sem_after["vlm_total_calls"],
        "canonical_script_sentences": canonical_seg["sentence_count"],
        "missing_dialogue_count": len(missing_clauses),
        "final_mix_reasr_wer": final_mix_reasr["wer"],
        "audio_onset_qa_passed": onset_qa["passed"],
        "exact_frame_coverage": frame_manifest["exact_full_coverage"],
        "story_coverage_valid": cov_chk["valid"],
        "wall_s": total_wall_s,
        "resource_stats": res_stats,
    })
    atomic_write_json(run_dir / "QA02_REPORT.json", qa02)
    atomic_write_json(run_dir / "validation.json", qa02)
    atomic_write_json(run_dir / "manifest.json", {
        "schema_version": DIRECTOR_V4_SCHEMA_VERSION,
        "composition_mode": "multisource_director_v4",
        "director_strategy": "generative_v4",
        "run_id": run_id,
        "final_artifact": out_filename,
        "final_sha256": final_sha,
        "output_filename": out_filename,
        "output_sha256": final_sha,
        "preview_filename": preview_mp4.name,
        "preview_sha256": prev_sha,
        "overall_status": overall_status,
        "hard_blockers": release_blockers,
        "sources": [{"source_id": s["source_id"], "path": s["path"], "sha256": s["sha256"]} for s in probed_sources],
        "output": {"width": W, "height": H, "fps_num": fps_num, "fps_den": fps_den, "frames": edit_map.output_total_frames, "duration_s": round(edit_map.output_duration_s, 4)},
    })
    atomic_write_text(
        run_dir / "README.md",
        f"# ML-01.R5.R7 Run {run_id}\n- Output: `{out_filename}` (sha256=`{final_sha}`)\n- Status: `{overall_status}`\n- Blockers: `{release_blockers}`\n- Engines: `{sorted(engines_used)}`\n- Semantic Coverage: `{sem_after['weighted_semantic_coverage']}` | Critical Accuracy: `{sem_after['critical_factual_accuracy']}`\n",
    )
    return qa02
