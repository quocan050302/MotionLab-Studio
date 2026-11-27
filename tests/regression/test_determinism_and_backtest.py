"""Phase G7 — Regression & Policy Backtest Suite (Fixed-Seed Replay + Parallel vs Sequential Equivalence)."""

from __future__ import annotations

import time
from pathlib import Path

from motionlab.analyzer import analyze_media
from motionlab.config import AnalysisConfig, EditConfig, ResourceConfig
from motionlab.planner import build_edit_plan
from motionlab.preflight import probe_media
from motionlab.resource import ResourceBudget
from tests.fixtures.procedural import make_multishot_demo


def test_sequential_vs_bounded_parallel_analysis_equivalence(tmp_path: Path) -> None:
    """Phase F/G7: Compare max_workers=1 (sequential) vs max_workers=2 (bounded async parallel) on identical input."""
    src = make_multishot_demo(tmp_path / "bench_6s.mp4", duration_s=6.0, width=540, height=960, fps=24, with_audio=True)
    media = probe_media(src)
    caps = {"ffmpeg": {"available": True}, "remotion": {"available": False}, "hyperframes": {"available": False}}

    # Sequential (max_workers=1, cache disabled)
    cfg_seq = EditConfig(
        profile="auto",
        seed=42,
        analysis=AnalysisConfig(use_cache=False),
        resources=ResourceConfig(max_workers=1, min_free_gb=1.0),
    )
    b_seq = ResourceBudget(cfg_seq.resources, tmp_path / "seq", tmp_path / "seq_tmp")
    b_seq.preflight_budget(media.duration_s, media.display_width, media.display_height, media.fps_float)
    t0 = time.monotonic()
    ana_seq, _ = analyze_media(media, cfg_seq, tmp_path / "seq_tmp", budget=b_seq)
    seq_s = time.monotonic() - t0
    plan_seq, _ = build_edit_plan(media, ana_seq, cfg_seq, caps)

    # Bounded async parallel (max_workers=2, cache disabled)
    cfg_par = EditConfig(
        profile="auto",
        seed=42,
        analysis=AnalysisConfig(use_cache=False),
        resources=ResourceConfig(max_workers=2, min_free_gb=1.0),
    )
    b_par = ResourceBudget(cfg_par.resources, tmp_path / "par", tmp_path / "par_tmp")
    b_par.preflight_budget(media.duration_s, media.display_width, media.display_height, media.fps_float)
    t1 = time.monotonic()
    ana_par, _ = analyze_media(media, cfg_par, tmp_path / "par_tmp", budget=b_par)
    par_s = time.monotonic() - t1
    plan_par, _ = build_edit_plan(media, ana_par, cfg_par, caps)

    # Exact deterministic equivalence between sequential and bounded parallel!
    assert ana_seq.to_dict() == ana_par.to_dict()
    assert plan_seq.plan_hash() == plan_par.plan_hash()
    assert seq_s > 0.0 and par_s > 0.0
