"""R1-A8 Unit & Negative Tests: Two-Pass Semantic Sound Director, Marker Alignment & CC0 24-bit Foley (ML-01.R5.R1)."""

from __future__ import annotations

from pathlib import Path

from motionlab.audio.assets import inspect_foley_library, load_recorded_foley_pcm
from motionlab.audio.cues import frame_to_sample_exact
from motionlab.audio.semantic_sound import (
    AnimationMarker,
    PlannedSonicCue,
    SonicScore,
    align_sound_to_marker,
    extract_rendered_animation_markers,
    extract_sonic_opportunities,
    plan_sonic_score,
    validate_semantic_sound_grounding,
)
from motionlab.composition.generative_schema import (
    GENERATIVE_SCHEMA_VERSION,
    LayerSpec,
    SceneSpec,
    StoryboardSpec,
    TransitionSpec,
)
from motionlab.schemas import ColorMetadata, MediaInfo
from motionlab.semantic.schema import SEMANTIC_SCHEMA_VERSION, Transcript, TranscriptSegment, WordSpan
from motionlab.semantic.story_graph import build_story_graph


def _fake_media(duration_s: float = 30.0) -> MediaInfo:
    n_f = int(round(duration_s * 24))
    return MediaInfo(
        source_path="fake_clip.mp4",
        input_hash="a" * 64,
        file_size_bytes=1024,
        duration_pts=n_f * 1000,
        duration_s=duration_s,
        time_base_num=1,
        time_base_den=24000,
        width=720,
        height=1280,
        display_width=720,
        display_height=1280,
        rotation=0,
        sar_num=1,
        sar_den=1,
        fps_num=24,
        fps_den=1,
        vfr=False,
        nb_frames=n_f,
        stream_start_pts=0,
        stream_start_time_s=0.0,
        video_codec="h264",
        pixel_format="yuv420p",
        field_order="progressive",
        is_interlaced=False,
        has_alpha=False,
        color_metadata=ColorMetadata(),
    )


def _make_transcript(sentences: list[tuple[float, float, str]]) -> Transcript:
    segs: list[TranscriptSegment] = []
    for si, (t0, t1, text) in enumerate(sentences):
        toks = text.split()
        dt = (t1 - t0) / max(1, len(toks))
        seg_w: list[WordSpan] = []
        for k, tok in enumerate(toks):
            wt = WordSpan(word=tok, start_s=round(t0 + k * dt, 3), end_s=round(t0 + (k + 1) * dt, 3), probability=0.98)
            seg_w.append(wt)
        segs.append(TranscriptSegment(segment_id=f"seg_{si:02d}", start_s=t0, end_s=t1, text=text, avg_logprob=-0.1, mean_word_prob=0.98, words=seg_w))
    tr = Transcript(
        schema_version=SEMANTIC_SCHEMA_VERSION,
        source_hash="a" * 64,
        source_audio_hash="a" * 64,
        language="en",
        language_probability=0.99,
        duration_s=sentences[-1][1],
        sample_rate_asr=16000,
        model_id="test_whisper",
        model_sha256="0" * 64,
        asr_params={},
        word_timestamps_exact=True,
        pts_priming_offset_s=0.0,
        pts_anomalies=[],
        status="OK",
        full_text=" ".join(s[2] for s in sentences),
        segments=segs,
    )
    tr.transcript_hash = tr.compute_hash()
    return tr


def _sample_finance_storyboard_and_graph():
    media = _fake_media(30.0)
    tr = _make_transcript([
        (0.0, 2.4, "Just because you have $1,200 in your bank account"),
        (2.5, 5.2, "doesn't mean you have $1,200 to spend."),
        (5.8, 8.2, "Let's say 800 is for bills."),
        (9.7, 11.2, "200 for groceries"),
        (11.8, 14.0, "and 100 for gas."),
        (14.3, 17.2, "You actually have just $100 left."),
        (20.1, 22.1, "That's why I created The Next Paycheck Plan,"),
        (22.2, 26.2, "a Notion dashboard and a 56-page workbook to help you organize your money."),
        (26.5, 28.4, "Before your next payday, give every dollar a purpose."),
        (28.5, 29.8, "Link in my profile."),
    ])
    sg = build_story_graph(tr, {"summary": {"face_track_conf_mean": 0.9}}, media)
    steps = {st.category.lower(): st.step_id for st in sg.ledger.steps}
    scenes = [
        SceneSpec(scene_id="sc_01_hook", start_frame=0, end_frame_exclusive=60, role="hook_contrast", stage_mode="live_depth_matte", viewer_takeaway="Opening balance", persistent_ledger_step_id=sg.ledger.steps[0].step_id),
        SceneSpec(scene_id="sc_02_neg", start_frame=60, end_frame_exclusive=138, role="negation_tension", stage_mode="live_clean_bridge", viewer_takeaway="Contradiction"),
        SceneSpec(scene_id="sc_03_bills", start_frame=138, end_frame_exclusive=207, role="allocation_step", stage_mode="tracked_macro_split", viewer_takeaway="Bills", persistent_ledger_step_id=steps.get("bills")),
        SceneSpec(scene_id="sc_04_groc", start_frame=234, end_frame_exclusive=269, role="allocation_step", stage_mode="aspect_morph_editorial", viewer_takeaway="Groceries", persistent_ledger_step_id=steps.get("groceries")),
        SceneSpec(scene_id="sc_05_gas", start_frame=269, end_frame_exclusive=344, role="allocation_step", stage_mode="live_clean_bridge", viewer_takeaway="Gas", persistent_ledger_step_id=steps.get("gas")),
        SceneSpec(scene_id="sc_06_payoff", start_frame=344, end_frame_exclusive=417, role="payoff_reveal", stage_mode="full_graphic_cutaway", viewer_takeaway="Payoff", transition_in=TransitionSpec(kind="frame_morph", duration_frames=12)),
        SceneSpec(scene_id="sc_07_prod", start_frame=481, end_frame_exclusive=533, role="product_intro", stage_mode="live_depth_matte", viewer_takeaway="Product"),
        SceneSpec(scene_id="sc_08_benefit", start_frame=533, end_frame_exclusive=635, role="product_benefit", stage_mode="full_graphic_cutaway", viewer_takeaway="Workbook", layers=[LayerSpec(layer_id="l1", primitive_id="rem_rc_flip_page", engine="remotion", layer_role="background_graphics", z_order=10, params={"flip_delay": 24})]),
        SceneSpec(scene_id="sc_09_close", start_frame=635, end_frame_exclusive=720, role="closing_takeaway", stage_mode="live_clean_bridge", viewer_takeaway="Close"),
    ]
    sb = StoryboardSpec(
        schema_version=GENERATIVE_SCHEMA_VERSION, mode="AUTO_GENERATIVE", source_hash="a" * 64,
        transcript_hash=sg.transcript_hash, story_graph_hash=sg.story_graph_hash,
        niche="finance", niche_confidence=0.95, core_value_proposition="Organize money",
        width=720, height=1280, fps_num=24, fps_den=1, total_frames=720, seed=42, palette={}, scenes=scenes,
    )
    return media, tr, sg, sb


def test_semantic_sound_director_roles_quiet_spans_and_rejected_cliches() -> None:
    media, tr, sg, sb = _sample_finance_storyboard_and_graph()
    markers = extract_rendered_animation_markers(sb, sg)
    opps = extract_sonic_opportunities(sg, tr, {"keyframes": []}, storyboard=sb)
    score = plan_sonic_score(sb, opps, sg, markers, niche_style="finance", seed=42)

    assert score.validation_errors == []
    assert len(score.cues) >= 6
    # Breathing pause (f417..481) and CTA ("Link in my profile") must create explicit QuietSpans
    assert len(score.quiet_spans) >= 2
    assert any(q.reason == "natural_breathing_pause" for q in score.quiet_spans)
    assert any(q.reason == "avoid_cta_masking" for q in score.quiet_spans)
    # Negated beat must reject cash_flick, gas beat must reject engine_rev, 56-page must reject cash_flick
    rej_fams = {r["family"] for r in score.rejected_decisions}
    assert "cash_flick" in rej_fams
    assert "engine_rev" in rej_fams
    assert "notification_ding" in rej_fams
    # No cue on negated beat uses cash_flick
    neg_cues = [c for c in score.cues if c.scene_id == "sc_02_neg"]
    assert all(c.family != "cash_flick" for c in neg_cues)
    # No cue on 56-page workbook uses cash_flick
    prod_cues = [c for c in score.cues if c.scene_id == "sc_08_benefit"]
    assert all(c.family != "cash_flick" for c in prod_cues)


def test_semantic_sound_grounding_rejects_ungrounded_diegetic_and_cliche_negatives() -> None:
    _media, _tr, sg, sb = _sample_finance_storyboard_and_graph()
    markers = extract_rendered_animation_markers(sb, sg)
    neg_beat = next(b for b in sg.beats if b.negated or b.speech_act == "negation_tension")
    bad_score = SonicScore(
        schema_version="2.0.0", mode="AUTO_GENERATIVE", story_graph_hash=sg.story_graph_hash,
        storyboard_hash="h", foley_library_status="PARTIAL_FOLEY_ASSETS",
        recorded_cues_used=0, procedural_cues_used=2,
        cues=[
            # 1) Diegetic foley with low confidence + drifted frame
            PlannedSonicCue(
                cue_id="bad_diegetic", beat_id=sg.beats[0].beat_id, scene_id="sc_01_hook",
                semantic_event_id="e1", word_ids=[0], quote="", story_intent="hook", emotional_role="hook",
                visual_element_id="el", action_ref="fake", action_confidence=0.20,
                marker_id=markers[0].marker_id, marker_type=markers[0].marker_type,
                sound_role="DIEGETIC_FOLEY", material="paper", texture="snap", family="cash_flick", variant=0,
                candidate_families=["cash_flick"], rejected_candidates=[], reason="bad test",
                target_frame=markers[0].global_frame + 9, pre_roll_ms=0.0, gain_db=-5.0, pan=0.0, priority=1,
            ),
            # 2) Cash flick placed illegally on negated beat
            PlannedSonicCue(
                cue_id="bad_neg_cash", beat_id=neg_beat.beat_id, scene_id="sc_02_neg",
                semantic_event_id="e2", word_ids=neg_beat.word_ids, quote=neg_beat.verbatim_quote,
                story_intent="neg", emotional_role="neg", visual_element_id="el", action_ref="m",
                action_confidence=0.95, marker_id=markers[2].marker_id, marker_type=markers[2].marker_type,
                sound_role="ILLUSTRATIVE_FOLEY", material="paper", texture="snap", family="cash_flick", variant=0,
                candidate_families=["cash_flick"], rejected_candidates=[], reason="bad neg cash",
                target_frame=markers[2].global_frame, pre_roll_ms=0.0, gain_db=-5.0, pan=0.0, priority=1,
            ),
        ],
        markers=markers,
    )
    errs = validate_semantic_sound_grounding(bad_score, sg, None, markers)
    assert any("unsupported_diegetic_foley" in e for e in errs)
    assert any("marker_alignment_drift" in e for e in errs)
    assert any("negated_beat_money_cue_forbidden" in e for e in errs)


def test_moving_animation_marker_shifts_aligned_sample_exact() -> None:
    media, tr, sg, sb = _sample_finance_storyboard_and_graph()
    markers1 = extract_rendered_animation_markers(sb, sg)
    opps = extract_sonic_opportunities(sg, tr, {"keyframes": []}, storyboard=sb)
    score1 = plan_sonic_score(sb, opps, sg, markers1, seed=42)
    tl1, _, _ = align_sound_to_marker(
        score1, fps_num=24, fps_den=1, sample_rate=44100, total_frames=720, source_hash=media.input_hash, seed=42
    )

    # Shift page_contact marker by +10 frames and verify aligned audio cue shifts by exact +10 frames in samples!
    markers2 = [
        AnimationMarker(**{**m.to_dict(), "global_frame": m.global_frame + (10 if m.marker_type == "page_contact" else 0)})
        for m in markers1
    ]
    score2 = plan_sonic_score(sb, opps, sg, markers2, seed=42)
    tl2, _, _ = align_sound_to_marker(
        score2, fps_num=24, fps_den=1, sample_rate=44100, total_frames=720, source_hash=media.input_hash, seed=42
    )
    ev1 = next(e for e in tl1.events if e.linked_visual_event_id == "sc_08_benefit")
    ev2 = next(e for e in tl2.events if e.linked_visual_event_id == "sc_08_benefit")
    assert ev2.peak_frame - ev1.peak_frame == 10
    expected_delta_samp = frame_to_sample_exact(10, 24, 1, 44100)
    assert abs((ev2.peak_sample - ev1.peak_sample) - expected_delta_samp) <= 4


def test_cc0_24bit_foley_decode_and_empty_dir_honest_fallback(tmp_path: Path) -> None:
    # Empty directory -> honestly PARTIAL_FOLEY_ASSETS
    empty_rep = inspect_foley_library(tmp_path)
    assert empty_rep["status"] == "PARTIAL_FOLEY_ASSETS"
    assert empty_rep["recorded_assets_verified"] == 0

    # If local CC0 wyzowl/sfx-kit WAVs are provisioned, verify 24-bit WAV decodes cleanly to stereo float32 @ 44.1kHz
    repo_rep = inspect_foley_library()
    if repo_rep["recorded_assets_verified"] > 0:
        assert repo_rep["status"] == "RECORDED_FOLEY_VERIFIED"
        pcm, t_off = load_recorded_foley_pcm(repo_rep["assets"][0], target_sr=44100)
        assert pcm.ndim == 2 and pcm.shape[1] == 2 and pcm.shape[0] > 100
        assert 0.0 <= t_off <= 2.0
