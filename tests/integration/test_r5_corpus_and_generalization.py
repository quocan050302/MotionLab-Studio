"""A7 Integration Tests: 5-Niche Generalization Corpus, Filename Independence, UI Protection & Offline Audit (ML-01.R5)."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from motionlab.composition.creative_quality import detect_template_repetition
from motionlab.composition.generative_schema import (
    GENERATIVE_SCHEMA_VERSION,
    LayerSpec,
    SceneSpec,
    StoryboardSpec,
)
from motionlab.composition.primitive_registry import load_unified_registry
from motionlab.config import AnalysisConfig, EditConfig
from motionlab.schemas import ColorMetadata, MediaInfo
from motionlab.semantic.generative_director import direct_generative_storyboard
from motionlab.semantic.schema import Transcript, TranscriptSegment, WordSpan
from motionlab.semantic.story_graph import build_story_graph


def _make_corpus_media(filename: str, dur_s: float = 16.0) -> MediaInfo:
    n_f = int(round(dur_s * 24))
    return MediaInfo(
        source_path=filename, input_hash="feedface" * 8, file_size_bytes=4096,
        duration_pts=n_f * 1000, duration_s=dur_s, time_base_num=1, time_base_den=24000,
        width=720, height=1280, display_width=720, display_height=1280, rotation=0,
        sar_num=1, sar_den=1, fps_num=24, fps_den=1, vfr=False, nb_frames=n_f,
        stream_start_pts=0, stream_start_time_s=0.0, video_codec="h264", pixel_format="yuv420p",
        field_order="progressive", is_interlaced=False, has_alpha=False, color_metadata=ColorMetadata(),
    )


def _make_transcript(text: str) -> Transcript:
    words = []
    t = 0.2
    for tok in text.split():
        words.append(WordSpan(word=tok, start_s=round(t, 3), end_s=round(t + 0.28, 3), probability=0.95, uncertain=False))
        t += 0.34
    seg = TranscriptSegment(segment_id="seg_000", start_s=0.2, end_s=round(t, 3), text=text, avg_logprob=-0.1, mean_word_prob=0.95, words=words)
    return Transcript(
        schema_version="1.0.0", source_hash="feedface" * 8, source_audio_hash="cafebabe" * 8,
        model_id="corpus_fixture", model_sha256="fixture", language="en", language_probability=0.99,
        duration_s=max(16.0, t + 1.0), sample_rate_asr=16000, asr_params={},
        pts_priming_offset_s=0.0, pts_anomalies=[], word_timestamps_exact=True, status="OK",
        full_text=text, segments=[seg], uncertain_spans=[],
    )


def test_filename_independence_identical_story_graph() -> None:
    """Renaming input clip to a neutral filename MUST produce identical StoryGraph hash & ledger."""
    txt = "You have $1,500 in your bank account. 900 is for rent and 300 for food. You actually have $300 left."
    tr = _make_transcript(txt)
    sg_named = build_story_graph(tr, None, _make_corpus_media("Olivia_Paycheck_V5_Kitchen_30s_Combined (1).mp4"))
    sg_neutral = build_story_graph(tr, None, _make_corpus_media("clip_neutral_001.mp4"))
    assert sg_named.story_graph_hash == sg_neutral.story_graph_hash
    assert sg_named.ledger.to_dict() == sg_neutral.ledger.to_dict()


@pytest.mark.parametrize(
    "niche,text,expect_ui_protected,expect_ledger",
    [
        ("finance", "You have $2,400 in your bank account. 1,400 is for rent and 400 for groceries. You actually have $600 left.", False, True),
        ("product", "We created Studio Desk Pro, a modular tray and a wireless dock to keep cables clean.", False, False),
        ("storytelling", "In 1969, three engineers stood on a quiet runway waiting for the signal.", False, False),
        ("tutorial", "Open the terminal window, click settings, and run the build command in your editor.", True, False),
        ("gameplay", "Watch the stamina bar in the top corner while dodging the boss combo.", True, False),
    ],
)
def test_five_niche_corpus_policy_and_ui_protection(niche: str, text: str, expect_ui_protected: bool, expect_ledger: bool) -> None:
    """Verify 5 distinct niches: finance uses verified ledger; tutorial/gameplay strictly protect UI with zero finance clutter."""
    media = _make_corpus_media(f"heldout_{niche}.mp4", dur_s=18.0)
    sg = build_story_graph(_make_transcript(text), None, media)
    assert sg.ledger.active is expect_ledger

    reg = load_unified_registry()
    if not reg.verified_primitives():
        pytest.skip("A2 primitive verification manifest not yet populated")

    # Stub LLM decision matching the niche to verify deterministic policy/UI-protection gates across all 5 niches cleanly
    fake_llm = {
        "niche": niche,
        "niche_confidence": 0.92,
        "core_value_proposition": f"Synthetic {niche} fixture",
        "beat_plans": [
            {
                "beat_id": b.beat_id,
                "stage_mode": "full_graphic_cutaway" if i % 2 == 0 else "live_clean_bridge",
                "remotion_primitive_id": "rem_ru_waterfall" if expect_ledger else "rem_ru_masked_slide",
                "hyperframes_primitive_id": "hf_apple_money_count" if expect_ledger else "hf_mk_callout_highlight",
                "transition_in": "none",
                "viewer_takeaway": b.narrative_role,
                "headline_text": b.verbatim_quote[:32],
                "sfx_family": "cash_flick" if expect_ledger else "soft_whoosh",
                "use_persistent_ledger": expect_ledger,
                "rationale": f"niche_{niche}",
            }
            for i, b in enumerate(sg.beats)
        ],
    }
    cfg = EditConfig(offline=True, analysis=AnalysisConfig(use_cache=False), seed=42)
    with patch("motionlab.semantic.generative_director.inspect_llm_model", return_value={"available": True, "sha256_actual": "stub", "model_id": "stub"}), \
         patch("motionlab.semantic.generative_director.generate_structured_json", return_value=(fake_llm, {"model_id": "stub"})):
        sb, _ = direct_generative_storyboard(
            media=media, story_graph=sg, roi_tracks={"summary": {"face_present_ratio": 0.0 if expect_ui_protected else 0.8}},
            config=cfg, matte_available=False, registry=reg,
        )

    if expect_ui_protected:
        assert sb.scenes == []
        assert any("niche_ui_protection_active" in n for n in sb.validation_notes)
    else:
        assert len(sb.scenes) >= 1
        # Non-finance niches MUST NOT emit finance waterfall/money primitives!
        if not expect_ledger:
            used_pids = {ly.primitive_id for sc in sb.scenes for ly in sc.layers}
            assert "rem_ru_waterfall" not in used_pids
            assert "hf_apple_money_count" not in used_pids


def test_template_repetition_detector_flags_three_identical_scenes() -> None:
    rep_scenes = [
        SceneSpec(
            scene_id=f"s{i}", start_frame=i * 48, end_frame_exclusive=(i + 1) * 48,
            role="step", stage_mode="live_clean_bridge", viewer_takeaway="x",
            layers=[LayerSpec("l1", "rem_ru_counter", "remotion", "foreground_graphics", 10)],
        )
        for i in range(3)
    ]
    sb = StoryboardSpec(
        schema_version=GENERATIVE_SCHEMA_VERSION, mode="AUTO_GENERATIVE", source_hash="a" * 64,
        transcript_hash="b" * 64, story_graph_hash="c" * 64, niche="finance", niche_confidence=0.9,
        core_value_proposition="x", width=720, height=1280, fps_num=24, fps_den=1, total_frames=240,
        seed=42, palette={}, scenes=rep_scenes,
    )
    audit = detect_template_repetition(sb)
    assert audit["passed"] is False
    assert audit["max_consecutive_identical_templates"] == 3
