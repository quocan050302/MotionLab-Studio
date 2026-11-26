"""Phase G7: Regression determinism & sequential vs bounded-parallel consistency tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from tests.fixtures.generator import generate_demo_9x16_15s


def test_cold_run_determinism_and_worker_parity(tmp_path: Path):
    """Two cold runs with different bounded worker counts MUST yield identical plan_hash & events."""
    src = generate_demo_9x16_15s(tmp_path / "reg_6s.mp4", width=360, height=640, fps=30, duration_s=6.0)
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("ELEVENLABS_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY")
    }

    p1 = subprocess.run(
        [
            sys.executable, "-m", "motionlab", "plan",
            "--input", str(src),
            "--profile", "auto",
            "--seed", "42",
            "--no-cache",
            "--max-workers", "1",
            "--output", str(tmp_path / "r1"),
        ],
        capture_output=True, text=True, env=env, check=True,
    )
    p2 = subprocess.run(
        [
            sys.executable, "-m", "motionlab", "plan",
            "--input", str(src),
            "--profile", "auto",
            "--seed", "42",
            "--no-cache",
            "--max-workers", "2",
            "--output", str(tmp_path / "r2"),
        ],
        capture_output=True, text=True, env=env, check=True,
    )
    r1 = json.loads(p1.stdout)
    r2 = json.loads(p2.stdout)
    assert r1["plan_hash"] == r2["plan_hash"]
    assert r1["profile"] == r2["profile"]
    assert r1["events"] == r2["events"]

    plan1 = json.loads(Path(r1["plan_path"]).read_text(encoding="utf-8"))
    plan2 = json.loads(Path(r2["plan_path"]).read_text(encoding="utf-8"))
    assert plan1["events"] == plan2["events"]
