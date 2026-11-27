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


def verify_webm_alpha_pixels(
    webm_path: Path,
    width: int = 720,
    height: int = 1280,
    sample_frame: int = 16,
    min_transparent_frac: float = 0.10,
    ffmpeg_bin: str | None = None,
) -> dict[str, Any]:
    """Decode one frame of a VP9/WebM overlay using explicit libvpx-vp9 decoder to verify real RGBA alpha."""
    import numpy as np

    ffmpeg = ffmpeg_bin or find_binary("ffmpeg") or "ffmpeg"
    cmd = [
        ffmpeg,
        "-v", "error",
        "-c:v", "libvpx-vp9",
        "-i", str(webm_path),
        "-vf", f"select='eq(n\\,{max(0, int(sample_frame))})'",
        "-frames:v", "1",
        "-f", "rawvideo",
        "-pix_fmt", "rgba",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=45, shell=False)
    fbytes = width * height * 4
    if proc.returncode != 0 or len(proc.stdout) < fbytes:
        # Fallback: decode frame 0 if sample_frame exceeded clip length
        cmd0 = [
            ffmpeg, "-v", "error", "-c:v", "libvpx-vp9", "-i", str(webm_path),
            "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgba", "pipe:1",
        ]
        proc = subprocess.run(cmd0, capture_output=True, timeout=45, shell=False)
        if proc.returncode != 0 or len(proc.stdout) < fbytes:
            return {
                "alpha_verified": False,
                "has_visible_alpha": False,
                "transparent_frac": 0.0,
                "opaque_stroke_frac": 0.0,
                "nonzero_alpha_ratio": 0.0,
                "reason": f"vp9_rgba_decode_failed:rc={proc.returncode},bytes={len(proc.stdout)}",
            }
    rgba = np.frombuffer(proc.stdout[:fbytes], dtype=np.uint8).reshape((height, width, 4))
    alpha = rgba[..., 3]
    transparent_frac = float(np.mean(alpha <= 10))
    translucent_frac = float(np.mean(alpha <= 245))
    opaque_frac = float(np.mean(alpha >= 20))
    ok = bool(translucent_frac >= min_transparent_frac and opaque_frac >= 0.0005)
    return {
        "alpha_verified": ok,
        "has_visible_alpha": ok,
        "transparent_frac": round(transparent_frac, 5),
        "translucent_frac": round(translucent_frac, 5),
        "opaque_stroke_frac": round(opaque_frac, 5),
        "nonzero_alpha_ratio": round(opaque_frac, 5),
        "sample_frame": int(sample_frame),
    }
