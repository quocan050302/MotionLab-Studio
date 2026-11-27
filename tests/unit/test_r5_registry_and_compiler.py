"""A7 Unit Tests: Unified Primitive Registry, Intent Search, Generative DSL Security & Reuse Gate (ML-01.R5)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from motionlab.composition.catalog_search import search_primitives_by_intent
from motionlab.composition.generative_compiler import compile_storyboard_graph
from motionlab.composition.generative_schema import (
    GENERATIVE_SCHEMA_VERSION,
    KeyframePoint,
    LayerSpec,
    MotionTrack,
    SceneSpec,
    StoryboardSpec,
    TransitionSpec,
    validate_canonical_reuse_gate,
    validate_storyboard_spec,
)
from motionlab.composition.primitive_adapters import build_adapted_hyperframes_html
from motionlab.composition.primitive_registry import LOCK_PATH, load_unified_registry
from motionlab.config import PROJECT_ROOT
from motionlab.schemas import MediaInfo
from motionlab.semantic.schema import Transcript, TranscriptSegment, WordSpan
from motionlab.semantic.story_graph import build_story_graph


def _dummy_media() -> MediaInfo:
    return MediaInfo(
        source_path="generic_clip.mp4", input_hash="abcd1234" * 8, size_bytes=2048,
        container_format="mov,mp4", duration_s=10.0, display_width=720, display_height=1280,
        coded_width=720, coded_height=1280, rotation=0, fps_num=24, fps_den=1,
        video_codec="h264", pix_fmt="yuv420p", color_transfer="bt709", color_primaries="bt709",
        color_space="bt709", is_hdr=False, is_vfr=False, duration_frames=240,
    )


def _dummy_storyboard(scenes: list[SceneSpec]) -> StoryboardSpec:
    sb = StoryboardSpec(
        schema_version=GENERATIVE_SCHEMA_VERSION,
        mode="AUTO_GENERATIVE",
        source_hash="abcd1234" * 8,
        transcript_hash="1111" * 16,
        story_graph_hash="2222" * 16,
        niche="finance",
        niche_confidence=0.95,
        core_value_proposition="Test storyboard",
        width=720, height=1280, fps_num=24, fps_den=1, total_frames=240, seed=42,
        palette={"ivory": "#F5F0E6", "espresso": "#292822", "forest": "#365B49", "sage": "#A4B39A", "brass": "#C6A46A", "clay": "#B85D43"},
        scenes=scenes,
    )
    sb.storyboard_hash = sb.compute_hash()
    return sb


def test_lockfile_and_registry_provenance_integrity() -> None:
    assert LOCK_PATH.is_file()
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    sources = lock["sources"]
    assert len(sources) >= 16  # We imported 20 distinct mechanisms across 5 repos!
    rem_cnt = sum(1 for s in sources if s["engine"] == "remotion")
    hf_cnt = sum(1 for s in sources if s["engine"] == "hyperframes")
    assert rem_cnt >= 6 and hf_cnt >= 6
    distinct_repos = {s["repo"] for s in sources}
    assert len(distinct_repos) >= 3
    for s in sources:
        assert s["license"] in ("MIT", "Apache-2.0")
        assert len(s["commit_sha"]) >= 10
        imp_p = PROJECT_ROOT / s["imported_path"]
        assert imp_p.is_file(), f"Missing imported source file: {imp_p}"


def test_intent_search_retrieves_cross_engine_primitives() -> None:
    reg = load_unified_registry()
    fin_hits = search_primitives_by_intent(reg, "waterfall ledger allocation bills money", has_ledger=True, verified_only=False, top_k=6)
    assert fin_hits
    engines_found = {h["engine"] for h in fin_hits}
    assert "remotion" in engines_found and "hyperframes" in engines_found

    prod_hits = search_primitives_by_intent(reg, "workbook page flip product specs dashboard", verified_only=False, top_k=5)
    assert any(h["primitive_id"] == "rem_rc_flip_page" for h in prod_hits)


def test_generative_schema_rejects_security_and_structural_negatives() -> None:
    reg = load_unified_registry()
    # 1. Unknown primitive
    sb_unk = _dummy_storyboard([
        SceneSpec(scene_id="s1", start_frame=0, end_frame_exclusive=60, role="hook", stage_mode="full_graphic_cutaway",
                  viewer_takeaway="x", layers=[LayerSpec("l1", "fake_nonexistent_prim", "remotion", "foreground_graphics", 10)])
    ])
    assert any("unknown_primitive" in e for e in validate_storyboard_spec(sb_unk, registry=reg, require_verified_primitives=False))

    # 2. XSS / Remote URL / Path traversal / eval()
    sb_xss = _dummy_storyboard([
        SceneSpec(scene_id="s1", start_frame=0, end_frame_exclusive=60, role="hook", stage_mode="full_graphic_cutaway",
                  viewer_takeaway="<script>alert(1)</script>",
                  layers=[LayerSpec("l1", "rem_ru_masked_slide", "remotion", "foreground_graphics", 10,
                                    params={"lines": ["https://evil.com/font.woff2", "../../etc/passwd", "eval(1)"]})])
    ])
    errs = validate_storyboard_spec(sb_xss, registry=reg, require_verified_primitives=False)
    assert sum("forbidden_token" in e for e in errs) >= 3

    # 3. Overlapping scenes + out of bounds + particle cap exceeded
    sb_bad = _dummy_storyboard([
        SceneSpec(scene_id="s1", start_frame=0, end_frame_exclusive=80, role="a", stage_mode="full_graphic_cutaway", viewer_takeaway="a",
                  layers=[LayerSpec("l1", "rem_viv_particle_field", "remotion", "background_graphics", 10, params={"count": 999})]),
        SceneSpec(scene_id="s2", start_frame=60, end_frame_exclusive=300, role="b", stage_mode="full_graphic_cutaway", viewer_takeaway="b"),
    ])
    errs_bad = validate_storyboard_spec(sb_bad, registry=reg, require_verified_primitives=False)
    assert any("particle_cap_exceeded" in e for e in errs_bad)
    assert any("overlapping_scene" in e for e in errs_bad)
    assert any("out_of_bounds_scene" in e for e in errs_bad)

    # 4. NaN keyframe & invalid easing
    sb_nan = StoryboardSpec(
        schema_version=GENERATIVE_SCHEMA_VERSION, mode="AUTO_GENERATIVE", source_hash="a" * 64,
        transcript_hash="b" * 64, story_graph_hash="c" * 64, niche="finance", niche_confidence=0.9,
        core_value_proposition="x", width=720, height=1280, fps_num=24, fps_den=1, total_frames=240,
        seed=42, palette={},
        scenes=[
            SceneSpec(scene_id="s1", start_frame=0, end_frame_exclusive=60, role="a", stage_mode="full_graphic_cutaway", viewer_takeaway="a",
                      layers=[LayerSpec("l1", "rem_ru_counter", "remotion", "foreground_graphics", 10,
                                        motion_tracks=[MotionTrack("t1", "l1", [KeyframePoint(10, "opacity", float("nan"), "bogus_ease")])])])
        ],
    )
    errs_nan = validate_storyboard_spec(sb_nan, registry=reg, require_verified_primitives=False)
    assert any("non_finite_float" in e for e in errs_nan)
    assert any("invalid_easing" in e for e in errs_nan)


def test_noop_reuse_gate_fails_single_engine_or_unused_imports() -> None:
    reg = load_unified_registry()
    sb = _dummy_storyboard([])
    # Single-engine only MUST fail dual-engine canonical reuse gate!
    ok_single, rep_single = validate_canonical_reuse_gate(
        sb, actual_rendered_engines={"ffmpeg", "remotion"}, actual_rendered_primitives=["rem_ru_counter", "rem_ru_waterfall"],
        min_distinct_total=4, min_per_engine=2, registry=reg,
    )
    assert ok_single is False
    # Empty rendered primitives MUST fail!
    ok_empty, _ = validate_canonical_reuse_gate(
        sb, actual_rendered_engines={"ffmpeg", "remotion", "hyperframes"}, actual_rendered_primitives=[],
        min_distinct_total=4, min_per_engine=2, registry=reg,
    )
    assert ok_empty is False


def test_hyperframes_proportional_bar_widths_never_index_based() -> None:
    html_a = build_adapted_hyperframes_html(
        "hf_data_chart", 720, 1280, 24.0, 3.0,
        {"denom_cents": 120000, "items": [{"label": "Bills", "cents": 80000}, {"label": "Gas", "cents": 10000}]},
    )
    # 80000/120000 = 66.67%, 10000/120000 = 8.33% — strictly proportional!
    assert "width:66.67%" in html_a
    assert "width:8.33%" in html_a
    assert "http://" not in html_a and "https://" not in html_a
    # Zero denominator guard (must never crash with ZeroDivisionError!)
    html_zero = build_adapted_hyperframes_html("hf_mk_progress_stat", 720, 1280, 24.0, 2.0, {"amount_cents": 0, "denom_cents": 0})
    assert "gsap.min.js" in html_zero


def test_compositional_novelty_distinct_hashes_without_aliases() -> None:
    s_a = SceneSpec(
        scene_id="s1", start_frame=0, end_frame_exclusive=72, role="a", stage_mode="full_graphic_cutaway",
        viewer_takeaway="A", layers=[LayerSpec("l1", "rem_ru_counter", "remotion", "foreground_graphics", 10, params={"to_value": 1200})],
    )
    s_b = SceneSpec(
        scene_id="s1", start_frame=0, end_frame_exclusive=72, role="a", stage_mode="aspect_morph_editorial",
        viewer_takeaway="B", layers=[
            LayerSpec("l1", "rem_ru_counter", "remotion", "foreground_graphics", 10, params={"to_value": 100}),
            LayerSpec("l2", "hf_data_chart", "hyperframes", "foreground_graphics", 20, params={"denom_cents": 120000}),
        ],
    )
    sb_a = _dummy_storyboard([s_a])
    sb_b = _dummy_storyboard([s_b])
    assert sb_a.storyboard_hash != sb_b.storyboard_hash
    # Deterministic compile hash identical on repeat
    assert sb_a.compute_hash() == sb_a.storyboard_hash
