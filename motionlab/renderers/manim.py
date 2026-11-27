"""Optional Manim CE renderer adapter (explicitly gated off the ML-01 critical path)."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from motionlab.renderers.base import BaseRenderer, RendererError, probe_artifact_ffprobe
from motionlab.schemas import EffectSpec, MediaInfo


class ManimRenderer(BaseRenderer):
    name = "manim"

    def probe_capability(self) -> dict[str, Any]:
        exe = shutil.which("manim")
        return {
            "available": False,  # Intentionally gated off ML-01 auto critical path
            "installed_on_path": bool(exe),
            "executable": exe,
            "license": "MIT License (Manim Community Edition)",
            "reason": "optional_gated_requires_explicit_user_diagram_template",
        }

    def supports(self, effect_spec: EffectSpec, media: MediaInfo, capabilities: dict[str, Any]) -> bool:
        return False

    def verify_artifact(self, artifact_path: Path, expected_duration_s: float | None = None) -> dict[str, Any]:
        info = probe_artifact_ffprobe(artifact_path)
        if not any(s.get("codec_type") == "video" for s in info.get("streams", [])):
            raise RendererError("Manim artifact missing video stream")
        return info
