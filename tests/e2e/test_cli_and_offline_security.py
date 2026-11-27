"""Phase G1 & G5 — Offline Zero-API Security Audit, Full CLI Matrix, Negative Tests, and Determinism."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from motionlab.config import PROJECT_ROOT
from tests.fixtures.procedural import (
    make_landscape_screen_ui,
    make_multishot_demo,
    make_short_clip,
)

FORBIDDEN_NETWORK_IMPORTS = {
    "requests",
    "httpx",
    "aiohttp",
    "urllib",
    "urllib3",
    "http",
    "openai",
    "anthropic",
    "elevenlabs",
    "google.genai",
    "google.generativeai",
    "boto3",
    "helpers.transcribe",
    "helpers.transcribe_batch",
}


def _sanitized_offline_subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    for key in (
        "ELEVENLABS_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "HUGGINGFACE_TOKEN",
        "REPLICATE_API_TOKEN",
    ):
        env.pop(key, None)
    # Point all HTTP/HTTPS proxies to an unreachable local discard port so any accidental network call fails closed
    env["HTTP_PROXY"] = "http://127.0.0.1:9"
    env["HTTPS_PROXY"] = "http://127.0.0.1:9"
    env["ALL_PROXY"] = "http://127.0.0.1:9"
    env["NO_PROXY"] = ""
    env["DO_NOT_TRACK"] = "1"
    env["REMOTION_DISABLE_TELEMETRY"] = "1"
    env["HYPERFRAMES_TELEMETRY"] = "0"
    return env


def _run_cli(args: list[str], env: dict[str, str] | None = None, timeout: int = 240) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "motionlab", *args],
        cwd=str(PROJECT_ROOT),
        env=env or _sanitized_offline_subprocess_env(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


class TestZeroApiOfflineSecurity:
    def test_static_ast_audit_zero_cloud_or_network_imports_in_runtime(self) -> None:
        pkg_dir = PROJECT_ROOT / "motionlab"
        violations: list[str] = []
        for py_file in sorted(pkg_dir.rglob("*.py")):
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        top = alias.name.split(".")[0]
                        if alias.name in FORBIDDEN_NETWORK_IMPORTS or top in FORBIDDEN_NETWORK_IMPORTS:
                            violations.append(f"{py_file.name}: import {alias.name}")
                elif isinstance(node, ast.ImportFrom) and node.module:
                    top = node.module.split(".")[0]
                    if node.module in FORBIDDEN_NETWORK_IMPORTS or top in FORBIDDEN_NETWORK_IMPORTS:
                        violations.append(f"{py_file.name}: from {node.module} import ...")
        assert not violations, f"Forbidden network/API imports found in ML-01 runtime: {violations}"

    def test_cli_rejects_unknown_flags_cloud_flag_and_corrupt_media(self, tmp_path: Path) -> None:
        r_unknown = _run_cli(["edit", "--input", "x.mp4", "--totally-unknown-flag"])
        assert r_unknown.returncode != 0

        r_cloud = _run_cli(["edit", "--input", "x.mp4", "--allow-cloud"])
        assert r_cloud.returncode != 0

        bad = tmp_path / "corrupt.mp4"
        bad.write_bytes(b"corrupted data not video")
        r_bad = _run_cli(["edit", "--input", str(bad), "--output", str(tmp_path / "runs")])
        assert r_bad.returncode != 0

    def test_unicode_and_space_filename_safe_without_shell_injection(self, tmp_path: Path) -> None:
        weird_name = "Video Tiếng Việt & (test; echo pwned).mp4"
        src = make_multishot_demo(tmp_path / weird_name, duration_s=3.0, width=360, height=640, fps=24, with_audio=True)
        r = _run_cli(
            [
                "edit",
                "--input", str(src),
                "--profile", "neutral",
                "--quality", "preview",
                "--disable-remotion",
                "--disable-hyperframes",
                "--offline",
                "--min-free-gb", "1.0",
                "--output", str(tmp_path / "runs"),
            ]
        )
        assert r.returncode == 0, f"CLI failed on Unicode/special filename: {r.stderr}"
        data = json.loads(r.stdout)
        assert data["status"] == "TECHNICAL_PASS"
        assert Path(data["final_path"]).exists()


class TestEndToEndMatrix:
    def test_e2e_15s_9x16_demo_cold_repeatability_and_validate_cli(self, tmp_path: Path) -> None:
        """G5(i)+(ii): 15s 9:16 1080x1920 @30fps demo with real cuts + audio cues, cold repeatability, single CLI command."""
        src = make_multishot_demo(
            tmp_path / "demo_15s_9x16.mp4",
            duration_s=15.0,
            width=1080,
            height=1920,
            fps=30,
            with_audio=True,
        )
        runs = tmp_path / "runs"

        # Cold run 1 (with Remotion enabled & proxy sinkhole blocking internet!)
        r1 = _run_cli(
            [
                "edit",
                "--input", str(src),
                "--profile", "auto",
                "--seed", "42",
                "--quality", "preview",
                "--offline",
                "--no-cache",
                "--disable-hyperframes",
                "--min-free-gb", "1.0",
                "--output", str(runs),
            ]
        )
        assert r1.returncode == 0, f"E2E 15s edit failed:\nSTDOUT:{r1.stdout}\nSTDERR:{r1.stderr}"
        d1 = json.loads(r1.stdout)
        assert d1["status"] == "TECHNICAL_PASS"
        assert d1["aesthetic_status"] == "AESTHETIC_PENDING"
        assert Path(d1["final_path"]).exists()
        assert Path(d1["contact_sheet_path"]).exists()

        # Verify effects span >= 3 families on the 15s multi-shot audio demo
        explain1 = json.loads((Path(d1["run_dir"]) / "plan_explain.json").read_text(encoding="utf-8"))
        assert len(explain1["budgets"]["families_used"]) >= 3, f"Expected >=3 families, got {explain1['budgets']}"

        # Cold run 2: dry-run plan must produce IDENTICAL plan_hash
        r2 = _run_cli(
            [
                "plan",
                "--input", str(src),
                "--profile", "auto",
                "--seed", "42",
                "--quality", "preview",
                "--offline",
                "--no-cache",
                "--disable-hyperframes",
                "--min-free-gb", "1.0",
                "--output", str(runs),
            ]
        )
        assert r2.returncode == 0, r2.stderr
        d2 = json.loads(r2.stdout)
        assert d2["plan_hash"] == d1["plan_hash"], "Cold determinism violated: plan_hash differed!"

        # Validate CLI on existing run
        r_val = _run_cli(["validate", "--run", d1["run_dir"]])
        assert r_val.returncode == 0, r_val.stderr
        vdata = json.loads(r_val.stdout)
        assert vdata["overall_status"] == "TECHNICAL_PASS"

    def test_e2e_silent_source_remains_silent(self, tmp_path: Path) -> None:
        """G5(iii): No-audio source produces valid edit with 0 audio streams and no waveform_accent."""
        src = make_multishot_demo(tmp_path / "silent.mp4", duration_s=4.0, width=360, height=640, fps=24, with_audio=False)
        r = _run_cli(
            [
                "edit",
                "--input", str(src),
                "--profile", "auto",
                "--quality", "preview",
                "--offline",
                "--disable-remotion",
                "--disable-hyperframes",
                "--min-free-gb", "1.0",
                "--output", str(tmp_path / "runs"),
            ]
        )
        assert r.returncode == 0, r.stderr
        d = json.loads(r.stdout)
        assert d["status"] == "TECHNICAL_PASS"
        assert d["audio_mode"] == "none"
        plan = json.loads((Path(d["run_dir"]) / "edit_plan.json").read_text(encoding="utf-8"))
        assert all(ev["effect_id"] not in ("waveform_accent", "pulse_border") for ev in plan["events"])

    def test_e2e_landscape_screen_recording_tutorial_safe_policy(self, tmp_path: Path) -> None:
        """G5(iv): Landscape screen-recording auto-selects tutorial safe policy and preserves aspect & readability."""
        src = make_landscape_screen_ui(tmp_path / "screen.mp4", duration_s=4.0, width=960, height=540, fps=24)
        r = _run_cli(
            [
                "edit",
                "--input", str(src),
                "--profile", "auto",
                "--quality", "preview",
                "--offline",
                "--disable-remotion",
                "--disable-hyperframes",
                "--min-free-gb", "1.0",
                "--output", str(tmp_path / "runs"),
            ]
        )
        assert r.returncode == 0, r.stderr
        d = json.loads(r.stdout)
        assert d["status"] == "TECHNICAL_PASS"
        plan = json.loads((Path(d["run_dir"]) / "edit_plan.json").read_text(encoding="utf-8"))
        assert plan["profile"] == "tutorial"
        assert plan["width"] == 960 and plan["height"] == 540
        assert all(ev["layer"] != "layout" for ev in plan["events"])

    def test_e2e_short_edge_clip_emits_no_safe_effects_cleanly(self, tmp_path: Path) -> None:
        """G5(v): Very short 0.33s clip does not crash and emits honest NO_SAFE_EFFECTS."""
        src = make_short_clip(tmp_path / "edge_short.mp4", frames=8, fps=24)
        r = _run_cli(
            [
                "edit",
                "--input", str(src),
                "--profile", "auto",
                "--quality", "preview",
                "--offline",
                "--disable-remotion",
                "--disable-hyperframes",
                "--min-free-gb", "1.0",
                "--output", str(tmp_path / "runs"),
            ]
        )
        assert r.returncode == 0, r.stderr
        d = json.loads(r.stdout)
        assert d["status"] == "TECHNICAL_PASS"
        plan = json.loads((Path(d["run_dir"]) / "edit_plan.json").read_text(encoding="utf-8"))
        assert plan["status"] == "NO_SAFE_EFFECTS"
        assert len(plan["events"]) == 0
