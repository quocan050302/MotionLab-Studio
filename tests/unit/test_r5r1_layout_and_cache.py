"""R1-A8 Unit & Negative Tests: Premium Art Direction, All-Frame Protection, Continuity, Cache & QA-01 (ML-01.R5.R1)."""

from __future__ import annotations

import json
from pathlib import Path

from motionlab.composition.art_direction import (
    R5R1_EDITORIAL_PALETTE,
    build_shared_element_tracks,
    build_style_tokens,
    compute_safe_margins,
    evaluate_phone_legibility,
    solve_layout_with_protection,
    validate_layout_sweep,
)
from motionlab.composition.generative_schema import (
    GENERATIVE_SCHEMA_VERSION,
    LayerSpec,
    SceneSpec,
    StoryboardSpec,
    validate_storyboard_spec,
)
from motionlab.composition.qa_pack import MAX_QA_PACK_BYTES, build_qa01_evidence_pack
from motionlab.renderers.remotion import RemotionRenderer
from motionlab.schemas import ColorMetadata, MediaInfo
from motionlab.semantic.schema import SEMANTIC_SCHEMA_VERSION, Transcript, TranscriptSegment, WordSpan
from motionlab.semantic.story_graph import build_story_graph


def _sb_with_jargon(jargon: bool = False) -> StoryboardSpec:
    cap = "Balance left: $400 (derived)" if jargon else "$400 left"
    sc1 = SceneSpec(
        scene_id="sc_01_hook", start_frame=0, end_frame_exclusive=60, role="hook_contrast",
        stage_mode="live_depth_matte", viewer_takeaway="Balance hook",
        layers=[
            LayerSpec(
                layer_id="l1", primitive_id="hf_mk_progress_stat", engine="hyperframes",
                layer_role="foreground_graphics", z_order=30,
                params={"amount_cents": 80000, "denom_cents": 120000, "caption": cap, "hero_px": 124},
            )
        ],
        persistent_ledger_step_id="ledger_step_00",
    )
    sc2 = SceneSpec(
        scene_id="sc_02_payoff", start_frame=60, end_frame_exclusive=120, role="payoff_reveal",
        stage_mode="full_graphic_cutaway", viewer_takeaway="Payoff",
        layers=[
            LayerSpec(
                layer_id="l2", primitive_id="hf_apple_money_count", engine="hyperframes",
                layer_role="foreground_graphics", z_order=30,
                params={"from_cents": 120000, "to_cents": 10000, "caption": "after these expenses", "hero_px": 148},
            )
        ],
        persistent_ledger_step_id="ledger_step_04",
    )
    return StoryboardSpec(
        schema_version=GENERATIVE_SCHEMA_VERSION, mode="AUTO_GENERATIVE", source_hash="a" * 64,
        transcript_hash="b" * 64, story_graph_hash="c" * 64, niche="finance", niche_confidence=0.95,
        core_value_proposition="Money clarity", width=720, height=1280, fps_num=24, fps_den=1,
        total_frames=120, seed=42, palette=dict(R5R1_EDITORIAL_PALETTE), scenes=[sc1, sc2],
    )


def test_safe_margins_portrait_reels_vs_landscape() -> None:
    m_port = compute_safe_margins(720, 1280)
    assert m_port.left >= 48 and m_port.right >= 105 and m_port.top >= 110 and m_port.bottom >= 230
    m_land = compute_safe_margins(1920, 1080)
    assert m_land.right < 150 and m_land.bottom < 150


def test_all_frame_face_protection_and_continuity_tracks() -> None:
    sb = _sb_with_jargon(jargon=False)
    # Place face comfortably below upper header across all 120 frames
    roi = {
        "keyframes": [
            {
                "frame": f,
                "face": {"x": 0.34, "y": 0.24, "w": 0.30, "h": 0.28, "conf": 0.95},
                "task": {"x": 0.28, "y": 0.62, "w": 0.44, "h": 0.30, "conf": 0.85},
            }
            for f in range(0, 120, 4)
        ]
    }
    layout_plan, rows = solve_layout_with_protection(sb, roi, 720, 1280)
    assert len(rows) == 120
    assert layout_plan["zero_eye_mouth_collision"] is True
    leg = evaluate_phone_legibility(layout_plan, sb, 720, 1280)
    assert leg["passed"] is True
    assert validate_layout_sweep(layout_plan, leg) == []

    # Continuity tracks link consecutive ledger scenes
    tok = WordSpan(word="Test", start_s=0.0, end_s=1.0, probability=0.99)
    tr = Transcript(
        schema_version=SEMANTIC_SCHEMA_VERSION, source_hash="a" * 64, source_audio_hash="a" * 64,
        language="en", language_probability=0.99, duration_s=5.0, sample_rate_asr=16000,
        model_id="m", model_sha256="0" * 64, asr_params={}, word_timestamps_exact=True,
        pts_priming_offset_s=0.0, pts_anomalies=[], status="OK", full_text="Test",
        segments=[TranscriptSegment("seg_00", 0.0, 1.0, "Test", -0.1, 0.99, [tok])],
    )
    tr.transcript_hash = tr.compute_hash()
    mi = MediaInfo(
        source_path="x.mp4", input_hash="a" * 64, file_size_bytes=1024, duration_pts=120000,
        duration_s=5.0, time_base_num=1, time_base_den=24000, width=720, height=1280,
        display_width=720, display_height=1280, rotation=0, sar_num=1, sar_den=1,
        fps_num=24, fps_den=1, vfr=False, nb_frames=120, stream_start_pts=0,
        stream_start_time_s=0.0, video_codec="h264", pixel_format="yuv420p",
        field_order="progressive", is_interlaced=False, has_alpha=False, color_metadata=ColorMetadata(),
    )
    sg = build_story_graph(tr, roi, mi)
    cont = build_shared_element_tracks(sb, sg, layout_plan)
    assert "persistent_ledger_balance" in cont["persistent_element_ids"]
    assert cont["tracks"][0]["continuity_to_scene"] == "sc_02_payoff"
    assert cont["tracks"][1]["continuity_from_scene"] == "sc_01_hook"


def test_consumer_qa_jargon_leak_rejected_by_both_legibility_and_schema() -> None:
    sb_bad = _sb_with_jargon(jargon=True)
    layout_plan, _ = solve_layout_with_protection(sb_bad, {"keyframes": []}, 720, 1280)
    leg = evaluate_phone_legibility(layout_plan, sb_bad, 720, 1280)
    assert leg["passed"] is False
    assert any("(derived)" in x for x in leg["consumer_jargon_leaks"])
    sweep_errs = validate_layout_sweep(layout_plan, leg)
    assert any("consumer_jargon_leak" in e for e in sweep_errs)
    schema_errs = validate_storyboard_spec(sb_bad, require_verified_primitives=True)
    assert any("consumer_qa_jargon_leak" in e for e in schema_errs)


def test_style_tokens_and_remotion_code_hash_deterministic() -> None:
    st = build_style_tokens("finance")
    assert st["palette"]["ivory"] == "#F5F0E6"
    assert st["palette"]["forest"] == "#243B32"
    rem = RemotionRenderer(license_allowed=True)
    h1 = rem.compute_code_hash()
    h2 = rem.compute_code_hash()
    assert len(h1) == 20 and h1 == h2


def test_qa01_pack_redacts_paths_and_transcript_within_2mib(tmp_path: Path) -> None:
    run_d = tmp_path / "run_demo"
    (run_d / "evidence").mkdir(parents=True)
    (run_d / "storyboard_spec.json").write_text(json.dumps({
        "source_hash": "a" * 64, "width": 720, "height": 1280, "fps_num": 24, "fps_den": 1,
        "total_frames": 720, "storyboard_hash": "sb1", "story_graph_hash": "sg1",
        "scenes": [{"verbatim_quote": "Private spoken transcript line"}],
    }), encoding="utf-8")
    (run_d / "validation.json").write_text(json.dumps({"overall_status": "TECHNICAL_PASS", "aesthetic_status": "PENDING_HUMAN_REVIEW", "foley_library_status": "RECORDED_FOLEY_VERIFIED"}), encoding="utf-8")
    (run_d / "actual_engine_usage.json").write_text(json.dumps({"engines_used": ["ffmpeg", "hyperframes", "remotion"]}), encoding="utf-8")
    (run_d / "evidence" / "audio_event_qa.json").write_text(json.dumps({"recorded_foley_events_used": 10, "procedural_dsp_events_used": 0, "listening_review_status": "LISTENING_NOT_RUN_AUTOMATED_PROXY_ONLY"}), encoding="utf-8")
    (run_d / "evidence" / "sound_rationale.json").write_text(json.dumps({"score_hash": "sc1", "cues": [{"quote": "Private line"}]}), encoding="utf-8")
    (run_d / "evidence" / "creative_quality_proxy.json").write_text(json.dumps({"vision_critic_status": "VISION_CRITIC_NOT_RUN", "path": "C:/Users/Secret/file.jpg"}), encoding="utf-8")

    pack_d = tmp_path / "qa_pack"
    res = build_qa01_evidence_pack(
        run_dir=run_d, pack_dir=pack_d, tested_code_sha="deadbeef1234",
        repo_url="https://github.com/quocan050302/MotionLab-Studio",
        branch="feat/ml-01-zero-api-effects", label="test_pack",
        render_mode="AUTO_GENERATIVE", final_mp4_name="final.mp4",
        include_contact_sheet=False, include_raw_transcript=False,
    )
    assert res["within_2mib_budget"] is True
    assert res["total_bytes"] < MAX_QA_PACK_BYTES
    sb_red = (pack_d / "storyboard.json").read_text(encoding="utf-8")
    assert "Private spoken transcript line" not in sb_red
    assert "redacted_sha256" in sb_red
    vq_txt = (pack_d / "visual_quality.json").read_text(encoding="utf-8")
    assert "C:/Users/Secret" not in vq_txt
