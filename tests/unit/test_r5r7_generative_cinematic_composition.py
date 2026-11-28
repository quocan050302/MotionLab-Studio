"""Unit & Integration Tests for ML-01.R5.R7 — Generative Visual Reasoning & Cinematic Composition (D01..D14)."""

from __future__ import annotations

import shutil
import subprocess
import wave
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from motionlab.assets.pdf_regions import (
    build_verified_example_state_machine,
    index_evidence_assets,
    retrieve_claim_evidence,
)
from motionlab.audio.continuous_score import (
    build_global_sonic_score,
    render_streaming_stems,
    validate_actual_audio_event_alignment,
)
from motionlab.composition.hook_competition import rank_hook_sequences_blind
from motionlab.composition.longform_scheduler import (
    TotalProcessTreeMonitor,
    assemble_verified_chunks,
    build_render_dag,
    partition_render_ranges,
    resolve_chunk_frame,
    resume_run,
)
from motionlab.composition.scene_graph_v4 import (
    build_dependency_index,
    build_motion_event_track_and_frame_manifest,
    compile_scene_program,
    resolve_frame_state,
    solve_layout_and_occlusion,
    validate_node_graph,
)
from motionlab.semantic.generative_visual_planner import (
    ReadOnlyEvidenceToolRegistry,
    propose_beat_programs,
    validate_semantic_graph,
)
from motionlab.semantic.model_router import (
    BENCHMARK_48_CREATIVE_TASKS,
    discover_compatible_models,
    estimate_model_budget,
    select_role_backend,
)
from motionlab.semantic.semantic_comprehension_critic import (
    plan_dependency_closure_repair,
    verify_rendered_semantic_comprehension,
)
from motionlab.semantic.story_bible import (
    build_chapter_context,
    build_global_story_bible,
    canonical_split_sentences,
    reconcile_chapter_boundary,
    validate_story_coverage,
)


def _write_synthetic_mp4(path: Path, *, frames: int = 30, w: int = 360, h: int = 640, fps: int = 30, moving: bool = True, black: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(path), fourcc, float(fps), (w, h))
    for i in range(frames):
        if black:
            img = np.zeros((h, w, 3), dtype=np.uint8)
        else:
            img = np.full((h, w, 3), (38, 56, 22), dtype=np.uint8)
            dx = (i * 8) % max(40, w - 120) if moving else 20
            cv2.rectangle(img, (20 + dx, h // 2), (120 + dx, h // 2 + 80), (82, 154, 196), -1)
            cv2.putText(img, f"F{i}", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (245, 242, 234), 2)
        vw.write(img)
    vw.release()
    return path


def test_a1_model_discovery_budget_guard_and_upgrade_verdict() -> None:
    disc = discover_compatible_models()
    assert len(BENCHMARK_48_CREATIVE_TASKS) == 48
    assert sum(1 for c in BENCHMARK_48_CREATIVE_TASKS if c["split"] == "heldout") == 16
    assert len({c["niche"] for c in BENCHMARK_48_CREATIVE_TASKS}) >= 14

    # Budget guard must fail closed when available RAM headroom < 3 GiB
    tight_budget = estimate_model_budget(
        {"model_id": "huge_8b", "available": True, "expected_ram_mb": 6500},
        host_budget={"ram_available_gb": 7.0},
        min_ram_reserve_gb=3.0,
    )
    assert tight_budget["fits_single_residency"] is False

    # Honest upgrade verdict separation
    v_up = select_role_backend({
        "text_models": {"qwen2_5_3b_instruct_q4_k_m": {"raw_accuracy": 0.30}, "qwen3_4b_instruct_2507_q4_k_m": {"raw_accuracy": 0.65}},
        "vision_models": {"qwen3_vl_2b_instruct_q4_k_m": {"raw_accuracy": 0.60}, "qwen3_vl_4b_instruct_q4_k_m": {"raw_accuracy": 0.85}},
    })
    assert v_up["upgrade_verdict"] == "MODEL_UPGRADE_PROVEN"
    assert v_up["selected_text_model_id"] == "qwen3_4b_instruct_2507_q4_k_m"
    assert v_up["selected_vision_model_id"] == "qwen3_vl_4b_instruct_q4_k_m"

    v_no = select_role_backend({
        "text_models": {"qwen2_5_3b_instruct_q4_k_m": {"raw_accuracy": 0.50}, "qwen3_4b_instruct_2507_q4_k_m": {"raw_accuracy": 0.45}},
        "vision_models": {"qwen3_vl_2b_instruct_q4_k_m": {"raw_accuracy": 0.70}, "qwen3_vl_4b_instruct_q4_k_m": {"raw_accuracy": 0.70}},
    })
    assert v_no["upgrade_verdict"] == "MODEL_UPGRADE_NOT_PROVEN"
    assert isinstance(disc["text_models"], list)


def test_a2_story_bible_canonical_sentences_longform_65_claims_and_ch20_continuity() -> None:
    # Single authoritative segmentation handles ellipses cleanly without inflating count!
    s = "Okay, first sentence. Like, I have money... but rent is due Tuesday. Third sentence? Fourth!"
    seg1 = canonical_split_sentences(s)
    seg2 = canonical_split_sentences(s)
    assert seg1["sentence_count"] == 4
    assert seg1["segmentation_hash"] == seg2["segmentation_hash"]

    # Build a 320-second (5m20s) long-form video with 65 spoken sentences -> ZERO claims[:10] truncation!
    words: list[dict[str, Any]] = []
    t = 0.2
    for i in range(65):
        sent = f"Chapter beat {i + 1} explains step {i + 1} clearly."
        for tok in sent.split():
            words.append({"word": tok, "start": round(t, 3), "end": round(t + 0.35, 3)})
            t += 0.42
        t += 1.4  # natural pause

    bible = build_global_story_bible(
        job={"job_id": "longform_320s_test", "niche_hint": "science", "intended_story": "Long science explainer."},
        source_registry=[{"source_id": "S1", "sha256": "abc", "duration_s": 320.0, "fps_num": 30, "fps_den": 1}],
        edited_words=words,
        duration_s=320.0,
        fps_num=30,
        fps_den=1,
        width=1920,
        height=1080,
    )
    assert bible["aspect_ratio"] == "16:9"
    assert len(bible["atomic_claims"]) == 65
    assert len(bible["chapters"]) >= 5

    # Every single claim including final chapter must be planned cleanly!
    ev_idx = index_evidence_assets(verified_assets=[])
    reg = ReadOnlyEvidenceToolRegistry(evidence_index=ev_idx, claim_matches={}, story_bible=bible, max_calls=400)
    planned_cids: set[str] = set()
    for ch in bible["chapters"]:
        ctx = build_chapter_context(bible, ch["chapter_id"])
        ch_plan = propose_beat_programs(story_bible=bible, chapter_context=ctx, tool_registry=reg, model_router=None)
        for sp in ch_plan["selected_programs"]:
            planned_cids.add(sp["claim_id"])

    cov = validate_story_coverage(story_bible=bible, planned_scene_claim_ids=planned_cids)
    assert cov["valid"] is True
    assert cov["orphan_claim_ids"] == []
    assert cov["planned_count"] == 65

    # Chapter 20 silent state rollback MUST fail closed unless explicit recap!
    bad_rec = reconcile_chapter_boundary(
        prev_chapter_exit_state={"battery_pct": {"value": 18}},
        next_chapter_entry_state={"battery_pct": {"value": 100}},
        explicit_recap=False,
    )
    assert bad_rec["valid"] is False
    assert any("silent_cross_chapter_rollback" in e for e in bad_rec["errors"])

    ok_rec = reconcile_chapter_boundary(
        prev_chapter_exit_state={"battery_pct": {"value": 18}},
        next_chapter_entry_state={"battery_pct": {"value": 100}},
        explicit_recap=True,
    )
    assert ok_rec["valid"] is True


def test_a3_generative_planner_security_and_14_niche_cold_catalog_transfer() -> None:
    # Security: disallow arbitrary tools, hallucinated IDs, and XSS script tokens!
    ev_idx = index_evidence_assets(verified_assets=[])
    reg = ReadOnlyEvidenceToolRegistry(evidence_index=ev_idx, claim_matches={}, story_bible={}, max_calls=2)
    assert reg.invoke("exec_shell", {"cmd": "whoami"})["status"] == "REJECTED_NOT_IN_ALLOWLIST"
    reg.invoke("list_verified_evidence_ids", {})
    assert reg.invoke("list_verified_evidence_ids", {})["status"] == "CIRCUIT_BREAKER_MAX_CALLS"

    claim = {"claim_id": "c1", "role": "product_or_mechanism_proof", "required_evidence_type": "ACTUAL_PRODUCT_OR_REAL_DEMO", "start_s": 0.0, "end_s": 2.5}
    ok_bad, errs_bad = validate_semantic_graph(
        {"node_types": ["raw_js_eval"], "evidence_id_refs": ["https://evil.com/x.png"], "min_reading_hold_s": 0.02},
        claim=claim,
        valid_evidence_ids={"asset_real_01"},
        claim_duration_s=2.5,
    )
    assert ok_bad is False
    assert any("unknown_primitive_types" in e for e in errs_bad)
    assert any("unsafe_evidence_ref" in e for e in errs_bad)

    # 14 distinct cold-catalog niches + zero Olivia/finance leakage!
    niches = [
        "science", "history", "cooking", "gameplay", "product_saas", "comedy", "silent_art",
        "education", "screen_tutorial", "travel", "sports", "documentary", "storytelling", "diy_craft",
    ]
    for niche in niches:
        w = [] if niche == "silent_art" else [
            {"word": "First", "start": 0.2, "end": 0.5},
            {"word": "step", "start": 0.5, "end": 0.8},
            {"word": "explains", "start": 0.8, "end": 1.3},
            {"word": "mechanism.", "start": 1.3, "end": 1.9},
        ]
        bible = build_global_story_bible(
            job={"job_id": f"cold_{niche}", "niche_hint": niche},
            source_registry=[{"source_id": "S1", "sha256": "h", "duration_s": 8.0, "fps_num": 30, "fps_den": 1}],
            edited_words=w,
            duration_s=8.0,
            width=1920,
            height=1080,
        )
        r = ReadOnlyEvidenceToolRegistry(evidence_index=ev_idx, claim_matches={}, story_bible=bible)
        plan = propose_beat_programs(story_bible=bible, chapter_context=build_chapter_context(bible, "ch_00"), tool_registry=r)
        blob = str(plan).lower()
        assert "olivia" not in blob
        assert "lamborghini" not in blob
        assert "paycheck" not in blob
        if niche == "silent_art":
            assert plan["selected_programs"][0]["selected_program"]["backend_preference"] == "ffmpeg_clean"


def test_a4_worked_example_state_machine_and_evidence_hard_negatives(tmp_path: Path) -> None:
    # Valid exact Jordan workbook arithmetic
    good = build_verified_example_state_machine(
        example_id="jordan_p07",
        domain="finance",
        unit="$",
        initial_label="Paycheck",
        initial_value=1600,
        expected_final_value=120,
        source_proof="verified_pdf_p07_p12",
        steps=[
            {"step_id": "bills", "label": "Bills", "op": "sub", "amount": 920},
            {"step_id": "ess", "label": "Essentials", "op": "sub", "amount": 340},
            {"step_id": "debt", "label": "Debt", "op": "sub", "amount": 160},
            {"step_id": "buf", "label": "Buffer", "op": "sub", "amount": 60},
        ],
    )
    assert good["valid"] is True
    assert good["final_value"] == 120.0

    # Negative: wrong sign, duplicate step, unit mismatch, arithmetic mismatch -> FAIL CLOSED!
    bad = build_verified_example_state_machine(
        example_id="bad_ex",
        domain="finance",
        unit="$",
        initial_label="Start",
        initial_value=1000,
        expected_final_value=200,
        source_proof="test",
        steps=[
            {"step_id": "s1", "label": "A", "op": "sub", "amount": -50, "unit": "EUR"},
            {"step_id": "s1", "label": "Dup", "op": "sub", "amount": 100},
        ],
    )
    assert bad["valid"] is False
    assert any("wrong_sign" in e for e in bad["errors"])
    assert any("unit_mismatch" in e for e in bad["errors"])
    assert any("duplicate_step_id" in e for e in bad["errors"])
    assert any("final_value_mismatch" in e for e in bad["errors"])

    # Private prelaunch asset MUST be rejected fail-closed!
    p_ok = tmp_path / "brake_diagram.png"
    p_priv = tmp_path / "prelaunch_secret.png"
    p_ok.write_bytes(b"png")
    p_priv.write_bytes(b"png")
    idx = index_evidence_assets(verified_assets=[
        {"asset_id": "sci_brake_01", "asset_type": "diagram", "resolved_path": str(p_ok), "sha256": "111", "owner_authorized": True, "keywords": ["hydraulic", "brake", "rotor"]},
        {"asset_id": "prelaunch_private_link", "asset_type": "workbook_page", "resolved_path": str(p_priv), "sha256": "222", "owner_authorized": True, "notes": "PRE-LAUNCH PROOF"},
    ])
    assert idx["verified_count"] == 1
    hits_rel = retrieve_claim_evidence(claim={"text": "Hydraulic brake clamps the rotor", "question_answered": ""}, evidence_index=idx)
    assert len(hits_rel) == 1 and hits_rel[0]["asset_id"] == "sci_brake_01"
    hits_irrel = retrieve_claim_evidence(claim={"text": "Bake sourdough bread at 230C", "question_answered": ""}, evidence_index=idx)
    assert hits_irrel == []


def test_a5_scene_graph_v4_validation_seek_parity_and_16x9_layout() -> None:
    lay16 = solve_layout_and_occlusion(width=1920, height=1080, primary_kind="timeline_axis")
    assert lay16["aspect_class"] == "landscape"
    assert lay16["x_px"] + lay16["width_px"] <= 1920

    cl = {
        "claim_id": "c_t0",
        "role": "causal_sequence_or_timing",
        "text": "Rent due Tuesday before Friday payday",
        "start_frame": 30,
        "end_frame_exclusive": 120,
        "question_answered": "Which day arrives first?",
        "weekdays_mentioned": ["tuesday", "friday"],
        "materiality": "critical",
    }
    sc = compile_scene_program(
        selected_entry={"claim_id": "c_t0", "selected_program": {"program_id": "p1", "authored_by": "local_llm_raw", "node_types": ["source_clip", "timeline_axis", "arrow_connector"], "camera_move": "pan_sequence", "backend_preference": "remotion", "evidence_id_refs": ["word:0"], "audio_cue_role": "COGNITIVE_EMPHASIS", "min_reading_hold_s": 1.2}},
        claim=cl,
        chapter_id="ch_00",
        width=1080,
        height=1920,
        fps_num=30,
        fps_den=1,
        total_frames=180,
        face_bounds={"x_min": 0.2, "y_min": 0.15, "x_max": 0.8, "y_max": 0.48},
        safe_zones=None,
    )
    # Random / reverse / forward seek parity MUST be bit-exact!
    s_fwd = [resolve_frame_state(sc, f) for f in (30, 60, 90, 119)]
    s_rev = [resolve_frame_state(sc, f) for f in (119, 90, 60, 30)][::-1]
    assert s_fwd == s_rev

    # Full [0, 180) frame coverage manifest invariant
    mtrack, fman = build_motion_event_track_and_frame_manifest(scenes=[sc], total_frames=180, fps_num=30, fps_den=1)
    assert fman["exact_full_coverage"] is True
    assert fman["covered_frames"] == 180
    assert mtrack["events_count"] >= 2

    # Negative: cyclic parent_id and NaN transform fail closed!
    bad_sc = {
        "start_frame": 0,
        "end_frame_exclusive": 30,
        "nodes": [
            {"node_id": "a", "type": "text_label", "parent_id": "b", "geometry": {"scale": float("nan")}},
            {"node_id": "b", "type": "text_label", "parent_id": "a", "geometry": {"scale": 1.0}},
        ],
    }
    ok_b, errs_b = validate_node_graph(bad_sc)
    assert ok_b is False
    assert any("parent_cycle_detected" in e for e in errs_b)
    assert any("non_finite_transform" in e for e in errs_b)


def test_a7_longform_chunk_scheduler_exact_coverage_crash_resume_and_invalidation(tmp_path: Path) -> None:
    # Stress 300s (9,000 frames @ 30fps) partitioning
    chunks_300 = partition_render_ranges(total_frames=9000, fps_num=30, fps_den=1, target_chunk_s=20.0)
    assert chunks_300[0]["core_start_frame"] == 0
    assert chunks_300[-1]["core_end_frame_exclusive"] == 9000
    assert sum(c["core_frames"] for c in chunks_300) == 9000
    assert resolve_chunk_frame(chunks_300[2], 5) == chunks_300[2]["core_start_frame"] + 5

    # Real FFmpeg multi-chunk render + simulated crash + verified resume + cache invalidation!
    dag = build_render_dag(
        source_hash="src_hash_1",
        graph_hash="graph_hash_1",
        total_frames=90,
        fps_num=30,
        fps_den=1,
        scenes=[],
        target_chunk_s=1.0,
        handle_frames=3,
    )
    assert dag["chunk_count"] >= 2

    fail_once = {"triggered": False}

    def _render_chunk(task: dict[str, Any], out_p: Path, threads: int) -> None:
        if task["chunk_index"] == 1 and not fail_once["triggered"]:
            fail_once["triggered"] = True
            raise RuntimeError("Simulated worker crash mid-chunk")
        _write_synthetic_mp4(out_p, frames=int(task["core_frames"]), w=320, h=180, fps=30)

    mon = TotalProcessTreeMonitor(tmp_path)
    r1 = resume_run(dag=dag, work_dir=tmp_path, render_chunk_fn=_render_chunk, max_workers=1, monitor=mon)
    assert r1["manifest_valid"] is False
    assert len(r1["failures"]) == 1

    # Resume reuses already-verified chunks and completes the crashed chunk!
    r2 = resume_run(dag=dag, work_dir=tmp_path, render_chunk_fn=_render_chunk, max_workers=2, monitor=mon)
    assert r2["manifest_valid"] is True
    assert r2["cache_hits"] >= 1
    assert r2["rendered_new"] == 1
    assert r2["resource_summary"]["peak_total_process_tree_rss_mb"] > 0.0

    # Assemble verified chunks cleanly into exact 90-frame MP4
    ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"
    asm = assemble_verified_chunks(dag=dag, run_report=r2, audio_wav=None, out_mp4=tmp_path / "assembled.mp4", ffmpeg_bin=ffmpeg_bin, preset="ultrafast")
    assert asm["decoded_frames"] == 90

    # Mutating graph_hash MUST invalidate cache completely!
    dag_mut = build_render_dag(source_hash="src_hash_1", graph_hash="graph_hash_2_CHANGED", total_frames=90, fps_num=30, fps_den=1, scenes=[], target_chunk_s=1.0)
    r3 = resume_run(dag=dag_mut, work_dir=tmp_path, render_chunk_fn=_render_chunk, max_workers=1)
    assert r3["cache_hits"] == 0
    assert r3["rendered_new"] == dag_mut["chunk_count"]


def test_a8_continuous_sonic_score_late_chapters_diversity_and_alignment(tmp_path: Path) -> None:
    # Multi-chapter 180s timeline with events all the way to final chapter!
    chapters = [
        {"chapter_id": "ch_00", "start_frame": 0, "end_frame_exclusive": 1800},
        {"chapter_id": "ch_01", "start_frame": 1800, "end_frame_exclusive": 3600},
        {"chapter_id": "ch_02", "start_frame": 3600, "end_frame_exclusive": 5400},
    ]
    events = [
        {"event_id": f"ev_{i:02d}", "scene_id": f"sc_{i:02d}", "chapter_id": chapters[i // 4]["chapter_id"], "global_frame": 60 + i * 420, "action": "obligation_token_lock", "audio_marker_role": "COGNITIVE_EMPHASIS"}
        for i in range(12)
    ]
    tl, meta = build_global_sonic_score(
        source_hash="h_long",
        motion_event_track={"events": events},
        chapters=chapters,
        total_samples=180 * 48000,
        fps_num=30,
        fps_den=1,
        niche="science",
        enable_ambient_bed=True,
    )
    assert meta["cross_chapter_continuity"]["late_chapter_cues"] >= 3
    assert meta["cross_chapter_continuity"]["continuous_phase_ok"] is True
    # Anti-repetition: never 3 identical consecutive cue families!
    fams = [e.family for e in tl.events]
    for i in range(len(fams) - 2):
        assert not (fams[i] == fams[i + 1] == fams[i + 2])

    # Silent art preserves natural ambience (zero SFX!)
    tl_sil, _ = build_global_sonic_score(
        source_hash="h_sil", motion_event_track={"events": events}, chapters=chapters, total_samples=180 * 48000, fps_num=30, fps_den=1, niche="silent_art",
    )
    assert len(tl_sil.events) == 0

    # Real streaming ambient bed + exact sample conservation QA
    voice_wav = tmp_path / "voice.wav"
    sr = 48000
    n_s = sr * 2
    with wave.open(str(voice_wav), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(np.zeros((n_s, 2), dtype=np.int16).tobytes())
    amb = render_streaming_stems(voice_wav=voice_wav, out_ambient_bed_wav=tmp_path / "amb.wav", sample_rate=sr)
    assert amb["duration_samples"] == n_s
    assert amb["continuous_phase_guaranteed"] is True
    qa = validate_actual_audio_event_alignment(audio_timeline=tl_sil, mixed_wav=tmp_path / "amb.wav", voice_bed_wav=voice_wav, fps_num=30, fps_den=1)
    assert qa["exact_sample_conservation"] is True
    assert qa["clipping_free"] is True


def test_a10_independent_semantic_critic_negatives_repair_and_sequence_hooks(tmp_path: Path) -> None:
    # 1. Counterexample: black video MUST fail closed as FAIL / UNREADABLE_REGION!
    black_mp4 = _write_synthetic_mp4(tmp_path / "black.mp4", frames=45, black=True)
    static_mp4 = _write_synthetic_mp4(tmp_path / "static.mp4", frames=45, moving=False)
    moving_mp4 = _write_synthetic_mp4(tmp_path / "moving.mp4", frames=45, moving=True)

    bible = {
        "total_material_weight": 1.1,
        "chapters": [{"chapter_id": "ch_00", "claim_ids": ["claim_00"]}],
        "atomic_claims": [{
            "claim_id": "claim_00",
            "role": "lived_resolution_payoff",
            "required_evidence_type": "ILLUSTRATIVE_ACTION_SEQUENCE",
            "materiality": "critical",
            "is_material": True,
            "weight": 1.1,
            "start_s": 0.0,
            "end_s": 1.5,
            "text": "Imagine shopping calmly",
            "question_answered": "What concrete action shows calm shopping?",
            "weekdays_mentioned": [],
        }],
    }
    res_black = verify_rendered_semantic_comprehension(
        final_mp4=black_mp4, base_mezzanine_mp4=None, story_bible=bible, scenes=[], ev_dir=tmp_path / "ev_b", run_vlm=False,
    )
    assert res_black["creative_evidence_gate"] == "FAIL"
    assert any(d["category"] == "UNREADABLE_REGION" for d in res_black["defects"])

    # 2. Counterexample: static non-moving video on ILLUSTRATIVE_ACTION_SEQUENCE MUST catch ACTION_NOT_SHOWN!
    res_static = verify_rendered_semantic_comprehension(
        final_mp4=static_mp4, base_mezzanine_mp4=None, story_bible=bible, scenes=[], ev_dir=tmp_path / "ev_s", run_vlm=False,
    )
    assert res_static["rows"][0]["semantic_status"] == "FAIL"
    assert any(d["category"] == "ACTION_NOT_SHOWN" for d in res_static["defects"])

    # 3. VLM requested (`run_vlm=True`) with missing router MUST return UNCERTAIN, NEVER default PASS!
    res_unc = verify_rendered_semantic_comprehension(
        final_mp4=moving_mp4, base_mezzanine_mp4=None, story_bible=bible, scenes=[], ev_dir=tmp_path / "ev_u", model_router=None, run_vlm=True,
    )
    assert res_unc["uncertain_claims_count"] == 1
    assert res_unc["creative_evidence_gate"] == "UNCERTAIN"

    # 4. Dependency-closure repair fixes reversed timeline order cleanly!
    sc_rev = {
        "scene_id": "sc_t",
        "chapter_id": "ch_00",
        "start_frame": 0,
        "end_frame_exclusive": 60,
        "primary_kind": "timeline_axis",
        "composition": {"canvas": {"width": 1080, "height": 1920}, "layout": {"y_px": 1100}},
        "nodes": [{"node_id": "ax", "type": "timeline_axis", "content_tokens": {"start_label": "FRI", "end_label": "TUE", "start_tag": "PAYDAY", "end_tag": "RENT DUE"}}],
        "events": [],
    }
    dep = build_dependency_index([sc_rev])
    rep = plan_dependency_closure_repair(defect={"category": "WRONG_ORDER", "scene_id": "sc_t"}, scenes=[sc_rev], dependency_index=dep)
    assert rep["actionable"] is True
    assert rep["repaired_scene"]["nodes"][0]["content_tokens"]["start_label"] == "TUE"

    # 5. Sequence hook ranking: permutation invariance & honest tie detection!
    h1 = _write_synthetic_mp4(tmp_path / "hook1.mp4", frames=60, moving=True)
    h2 = _write_synthetic_mp4(tmp_path / "hook2.mp4", frames=60, moving=False)
    r_a = rank_hook_sequences_blind(
        rendered_hooks=[{"candidate_id": "cand_A", "preview_mp4": str(h1)}, {"candidate_id": "cand_B", "preview_mp4": str(h2)}],
        hook_speech_transcript="Opening question",
        base_mezzanine_mp4=None,
        face_bounds=None,
        work_dir=tmp_path / "hk_a",
        model_router=None,
    )
    r_b = rank_hook_sequences_blind(
        rendered_hooks=[{"candidate_id": "cand_B", "preview_mp4": str(h2)}, {"candidate_id": "cand_A", "preview_mp4": str(h1)}],
        hook_speech_transcript="Opening question",
        base_mezzanine_mp4=None,
        face_bounds=None,
        work_dir=tmp_path / "hk_b",
        model_router=None,
    )
    assert r_a["winner_preview_sha256"] == r_b["winner_preview_sha256"]
    assert r_a["winner_candidate_id"] == "cand_A"
