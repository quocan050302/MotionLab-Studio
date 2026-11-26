"""Phase G5: End-to-End CLI tests (offline, zero API keys, fresh 9:16 preview, tutorial safety, plan replay)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from tests.fixtures.generator import (
    generate_demo_9x16_15s,
    generate_screen_tutorial_fixture,
    generate_silent_fixture,
)


def _clean_offline_env() -> dict[str, str]:
    return {
        k: v
        for k, v in os.environ.items()
        if k not in ("ELEVENLABS_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY")
    }


def test_e2e_cli_15s_9x16_demo_audio_integrity_and_replay(tmp_path: Path):
    src = generate_demo_9x16_15s(tmp_path / "demo_15s.mp4", width=540, height=960, fps=30, duration_s=15.0)
    runs_dir = tmp_path / "runs"

    cmd = [
        sys.executable, "-m", "motionlab", "edit",
        "--input", str(src),
        "--profile", "auto",
        "--seed", "42",
        "--offline",
        "--no-cache",
        "--preview-9x16",
        "--output", str(runs_dir),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=_clean_offline_env(), check=True)
    res = json.loads(proc.stdout)
    assert res["overall_status"] == "PASS"
    assert res["technical_status"] == "TECHNICAL_PASS"
    assert res["aesthetic_status"] == "PENDING_HUMAN_REVIEW"
    assert res["events"] >= 3

    run_dir = Path(res["run_dir"])
    assert (run_dir / "final.mp4").is_file()
    assert (run_dir / "preview_9x16.mp4").is_file()
    assert (run_dir / "contact_sheet.jpg").is_file()
    assert (run_dir / "README.md").is_file()
    assert (run_dir / "manifest.json").is_file()
    assert (run_dir / "edit_plan.json").is_file()
    assert (run_dir / "plan_explain.json").is_file()
    assert (run_dir / "validation.json").is_file()

    # Audio packet payload hash exact match
    aud_ev = json.loads((run_dir / "evidence" / "audio_integrity.json").read_text(encoding="utf-8"))
    assert aud_ev["exact_packet_match"] is True
    assert aud_ev["source_packet_hash"] == aud_ev["output_packet_hash"]

    # Validate CLI command on existing run
    val_proc = subprocess.run(
        [sys.executable, "-m", "motionlab", "validate", "--run", str(run_dir)],
        capture_output=True, text=True, env=_clean_offline_env(), check=True,
    )
    assert json.loads(val_proc.stdout)["overall_status"] == "PASS"

    # Replay command using saved edit_plan.json
    rep_proc = subprocess.run(
        [
            sys.executable, "-m", "motionlab", "render",
            "--plan", str(run_dir / "edit_plan.json"),
            "--input", str(src),
            "--no-cache",
            "--output", str(runs_dir),
        ],
        capture_output=True, text=True, env=_clean_offline_env(), check=True,
    )
    rep_res = json.loads(rep_proc.stdout)
    assert rep_res["overall_status"] == "PASS"
    assert rep_res["plan_hash"] == res["plan_hash"]


def test_e2e_landscape_screen_tutorial_auto_safe_policy(tmp_path: Path):
    src = generate_screen_tutorial_fixture(tmp_path / "tutorial.mp4", width=1280, height=720, duration_s=6.0)
    runs_dir = tmp_path / "runs"
    proc = subprocess.run(
        [
            sys.executable, "-m", "motionlab", "edit",
            "--input", str(src),
            "--profile", "auto",
            "--seed", "42",
            "--offline",
            "--no-cache",
            "--output", str(runs_dir),
        ],
        capture_output=True, text=True, env=_clean_offline_env(), check=True,
    )
    res = json.loads(proc.stdout)
    assert res["overall_status"] == "PASS"
    assert res["profile"] == "tutorial"
    plan = json.loads((Path(res["run_dir"]) / "edit_plan.json").read_text(encoding="utf-8"))
    # Tutorial policy must preserve 1280x720 source framing and avoid disruptive transform/mask effects
    assert (plan["width"], plan["height"]) == (1280, 720)
    assert all(ev["layer"] not in ("transform", "mask") for ev in plan["events"])


def test_e2e_short_edge_clip(tmp_path: Path):
    src = generate_silent_fixture(tmp_path / "short_1s.mp4", width=320, height=480, fps=30, duration_s=1.0)
    proc = subprocess.run(
        [
            sys.executable, "-m", "motionlab", "edit",
            "--input", str(src),
            "--profile", "neutral",
            "--offline",
            "--no-cache",
            "--output", str(tmp_path / "runs"),
        ],
        capture_output=True, text=True, env=_clean_offline_env(), check=True,
    )
    res = json.loads(proc.stdout)
    assert res["overall_status"] == "PASS"
