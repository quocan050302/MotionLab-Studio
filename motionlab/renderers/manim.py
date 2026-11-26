"""Optional out-of-critical-path Manim renderer adapter (explicit data/template only)."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from motionlab.renderers.base import BaseRenderer
from motionlab.schemas import EditPlan, EffectSpec, MediaInfo


class ManimRenderer(BaseRenderer):
    """Optional formal diagram engine — never auto-triggered on general video in ML-01."""

    name = "manim"

    def supports(
        self,
        spec: EffectSpec,
        media: MediaInfo,
        capabilities: dict[str, Any],
    ) -> bool:
        return False  # Explicitly opt-in only with formal user template/data

    def prepare(self, plan: EditPlan, source: Path, workdir: Path) -> dict[str, Any]:
        manim_bin = shutil.which("manim")
        if not manim_bin:
            raise RuntimeError("Manim is optional and not installed in this ML-01 environment.")
        return {"binary": manim_bin, "workdir": str(workdir)}

    def verify_artifact(self, artifact_path: Path, expected_duration_s: float) -> dict[str, Any]:
        raise RuntimeError("Manim renderer is disabled by default at ML-01.")
