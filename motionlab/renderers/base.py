"""Phase E1 — Abstract Renderer Contract and Engine Capability Probe."""

from __future__ import annotations

import os
import shutil
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from motionlab.config import PROJECT_ROOT, env_flag
from motionlab.preflight import find_binary
from motionlab.schemas import EditPlan, EffectEvent, EffectSpec, MediaInfo


class RendererError(RuntimeError):
    """Raised when a renderer fails to execute or produce a valid artifact."""


class BaseRenderer(ABC):
    name: str = "base"

    @abstractmethod
    def probe_capability(self) -> dict[str, Any]:
        """Inspect local binary/license availability without any network call."""

    @abstractmethod
    def supports(self, effect_spec: EffectSpec, media: MediaInfo, capabilities: dict[str, Any]) -> bool:
        """Return True iff this engine can legally and technically render `effect_spec`."""

    @abstractmethod
    def verify_artifact(self, artifact_path: Path, expected_duration_s: float | None = None) -> dict[str, Any]:
        """Probe rendered artifact via ffprobe and verify non-empty valid video stream."""


def find_local_browser() -> str | None:
    """Locate an already-installed local Chrome/Edge/Chromium executable (never downloads anything)."""
    env_override = os.environ.get("MOTIONLAB_BROWSER_EXECUTABLE")
    if env_override and Path(env_override).is_file():
        return str(Path(env_override).resolve())

    candidates = [
        # Windows
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        # macOS
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        # Linux
        "/usr/bin/google-chrome",
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
    ]
    for cand in candidates:
        if cand and Path(cand).is_file():
            return str(Path(cand).resolve())
    for name in ("google-chrome", "chromium", "chromium-browser", "chrome"):
        found = shutil.which(name)
        if found:
            return found
    return None


def probe_artifact_ffprobe(artifact_path: Path, ffprobe_bin: str | None = None) -> dict[str, Any]:
    ffprobe = ffprobe_bin or find_binary("ffprobe")
    if not ffprobe:
        raise RendererError("ffprobe not found")
    if not artifact_path.exists() or artifact_path.stat().st_size == 0:
        raise RendererError(f"Rendered artifact missing or empty: {artifact_path}")
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(artifact_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RendererError(f"ffprobe failed on rendered artifact '{artifact_path.name}': {proc.stderr.strip()}")
    import json

    return json.loads(proc.stdout)


def probe_all_engine_capabilities(
    allow_remotion: bool = True,
    allow_hyperframes: bool = True,
) -> dict[str, Any]:
    """Probe local FFmpeg, Remotion, HyperFrames, and Manim capabilities strictly offline."""
    from motionlab.renderers.ffmpeg import FFmpegRenderer
    from motionlab.renderers.hyperframes import HyperFramesRenderer
    from motionlab.renderers.manim import ManimRenderer
    from motionlab.renderers.remotion import RemotionRenderer

    remotion_enabled = allow_remotion and env_flag("MOTIONLAB_ALLOW_REMOTION", True)
    hf_enabled = allow_hyperframes and env_flag("MOTIONLAB_ALLOW_HYPERFRAMES", True)

    caps = {
        "ffmpeg": FFmpegRenderer().probe_capability(),
        "remotion": RemotionRenderer(license_allowed=remotion_enabled).probe_capability(),
        "hyperframes": HyperFramesRenderer(enabled=hf_enabled).probe_capability(),
        "manim": ManimRenderer().probe_capability(),
    }
    return caps
