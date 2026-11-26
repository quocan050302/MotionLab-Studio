"""Phase G1: Zero-API static AST audit, runtime import isolation, and URL rejection tests."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

from motionlab.config import REPO_ROOT
from motionlab.preflight import PreflightError, probe_media
from motionlab.schemas import SchemaValidationError, validate_local_path_string

FORBIDDEN_IMPORTS = {
    "requests",
    "httpx",
    "aiohttp",
    "urllib",
    "urllib3",
    "http",
    "socket",
    "openai",
    "anthropic",
    "elevenlabs",
    "google",
    "boto3",
}

FORBIDDEN_ENDPOINTS = (
    "api.elevenlabs.io",
    "api.openai.com",
    "api.anthropic.com",
    "generativelanguage.googleapis.com",
    "video-use.com",
)


def test_motionlab_ast_has_zero_network_or_cloud_imports():
    pkg_dir = REPO_ROOT / "motionlab"
    py_files = sorted(pkg_dir.rglob("*.py"))
    assert len(py_files) >= 15

    violations: list[str] = []
    for pf in py_files:
        src = pf.read_text(encoding="utf-8")
        for ep in FORBIDDEN_ENDPOINTS:
            if ep in src:
                violations.append(f"{pf.relative_to(REPO_ROOT)} references forbidden endpoint {ep}")
        tree = ast.parse(src, filename=str(pf))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    if top in FORBIDDEN_IMPORTS or alias.name.startswith("helpers.transcribe"):
                        violations.append(f"{pf.relative_to(REPO_ROOT)} imports {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                top = mod.split(".")[0]
                if top in FORBIDDEN_IMPORTS or mod.startswith("helpers.transcribe") or mod == "transcribe":
                    violations.append(f"{pf.relative_to(REPO_ROOT)} imports from {mod}")

    assert not violations, "Zero-API audit violations:\n" + "\n".join(violations)


def test_runtime_import_graph_clean_in_fresh_python_process():
    """Verify importing the full motionlab runtime never loads requests or cloud SDKs."""
    env = {
        k: v for k, v in os.environ.items()
        if k not in ("ELEVENLABS_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY")
    }
    code = (
        "import sys, motionlab.cli, motionlab.pipeline, motionlab.analyzer, motionlab.planner, "
        "motionlab.renderers.ffmpeg, motionlab.validation; "
        "bad = [m for m in ('requests','httpx','openai','anthropic','elevenlabs','transcribe') if m in sys.modules]; "
        "assert not bad, f'Forbidden modules loaded: {bad}'; "
        "print('CLEAN_RUNTIME_IMPORTS')"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True)
    assert "CLEAN_RUNTIME_IMPORTS" in proc.stdout


@pytest.mark.parametrize(
    "bad_url",
    [
        "https://example.com/video.mp4",
        "http://127.0.0.1/clip.mp4",
        "rtsp://camera.local/stream",
        "ftp://files.example.org/a.mp4",
    ],
)
def test_remote_url_inputs_fail_closed(bad_url: str):
    with pytest.raises(SchemaValidationError):
        validate_local_path_string(bad_url)
    with pytest.raises(PreflightError):
        probe_media(bad_url)


def test_cli_rejects_allow_cloud_and_unknown_flags():
    proc = subprocess.run(
        [sys.executable, "-m", "motionlab", "edit", "--input", "x.mp4", "--allow-cloud"],
        capture_output=True, text=True,
    )
    assert proc.returncode != 0
    assert "unrecognized arguments" in proc.stderr.lower()


def test_os_sandbox_offline_edit_succeeds_with_outbound_network_denied(tmp_path: Path):
    """OS-level network block proof on macOS (sandbox-exec denying outbound remote IP traffic)."""
    import json
    import shutil
    from tests.fixtures.generator import generate_silent_fixture

    sb = shutil.which("sandbox-exec")
    if not sb:
        pytest.skip("macOS sandbox-exec not available on this OS")

    src = generate_silent_fixture(tmp_path / "offline_clip.mp4", width=360, height=640, duration_s=2.0)
    runs_dir = tmp_path / "runs"
    profile = "(version 1)(allow default)(deny network-outbound (remote ip))"
    env = {
        k: v for k, v in os.environ.items()
        if k not in ("ELEVENLABS_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY")
    }
    cmd = [
        sb, "-p", profile,
        sys.executable, "-m", "motionlab", "edit",
        "--input", str(src),
        "--profile", "neutral",
        "--offline",
        "--no-cache",
        "--output", str(runs_dir),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=120)
    assert proc.returncode == 0, f"Offline sandboxed edit failed:\nSTDOUT:{proc.stdout}\nSTDERR:{proc.stderr}"
    payload = json.loads(proc.stdout)
    assert payload["overall_status"] == "PASS"
    assert Path(payload["final_path"]).is_file()
