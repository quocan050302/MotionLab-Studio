"""Phase E1: Renderer contract shared by FFmpeg, HyperFrames, Remotion, and optional Manim."""

from __future__ import annotations

import abc
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from motionlab.schemas import EditPlan, EffectEvent, EffectSpec, MediaInfo


def verify_webm_alpha_pixels(
    webm_path: Path,
    width: int,
    height: int,
    sample_frame: int = 10,
    ffmpeg_bin: str | None = None,
) -> dict[str, Any]:
    """Decode midpoint frame with libvpx-vp9 to RGBA and prove real transparency + non-empty strokes survive decode."""
    ffmpeg = ffmpeg_bin or shutil.which("ffmpeg") or "ffmpeg"
    cmd = [
        ffmpeg, "-v", "error",
        "-c:v", "libvpx-vp9",
        "-i", str(webm_path),
        "-vf", f"select=eq(n\\,{max(0, sample_frame)})",
        "-vframes", "1",
        "-f", "rawvideo",
        "-pix_fmt", "rgba",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=30, shell=False)
    expected_bytes = width * height * 4
    if proc.returncode != 0 or len(proc.stdout) < expected_bytes:
        # Fallback to frame 0 if clip shorter than sample_frame
        cmd[cmd.index("-vf") + 1] = "select=eq(n\\,0)"
        proc = subprocess.run(cmd, capture_output=True, timeout=30, shell=False)
        if proc.returncode != 0 or len(proc.stdout) < expected_bytes:
            raise RuntimeError(
                f"Failed to decode VP9 RGBA alpha frame from {webm_path}: {proc.stderr.decode('utf-8', errors='replace')[:300]}"
            )
    arr = np.frombuffer(proc.stdout[:expected_bytes], dtype=np.uint8).reshape((height, width, 4))
    alpha = arr[..., 3]
    a_min = int(alpha.min())
    a_max = int(alpha.max())
    zero_frac = round(float((alpha == 0).mean()), 4)
    nonzero_frac = round(float((alpha > 10).mean()), 5)
    has_true_alpha = (a_min == 0 and zero_frac >= 0.50 and a_max >= 15 and nonzero_frac > 0.0)
    return {
        "alpha_verified": has_true_alpha,
        "alpha_min": a_min,
        "alpha_max": a_max,
        "transparent_frac": zero_frac,
        "opaque_stroke_frac": nonzero_frac,
    }


class BaseRenderer(abc.ABC):
    name: str = "base"

    @abc.abstractmethod
    def supports(
        self,
        spec: EffectSpec,
        media: MediaInfo,
        capabilities: dict[str, Any],
    ) -> bool:
        """Return True if this renderer supports `spec` under current local capabilities & license."""

    @abc.abstractmethod
    def prepare(self, plan: EditPlan, source: Path, workdir: Path) -> dict[str, Any]:
        """Prepare scoped workdir and verify local tool availability."""

    @abc.abstractmethod
    def verify_artifact(self, artifact_path: Path, expected_duration_s: float) -> dict[str, Any]:
        """Verify rendered media artifact via ffprobe."""

    def cleanup(self, workdir: Path) -> None:
        """Optional renderer-owned scratch cleanup."""
        return None
