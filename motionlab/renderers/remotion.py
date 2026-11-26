"""Phase E3: Remotion (license-gated) React/TypeScript offline overlay renderer adapter."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from motionlab.config import REPO_ROOT, find_browser_executable, find_local_bin
from motionlab.renderers.base import BaseRenderer, verify_webm_alpha_pixels
from motionlab.resource import run_managed_subprocess
from motionlab.schemas import EditPlan, EffectEvent, EffectSpec, MediaInfo, atomic_write_json

REMOTION_DIR = REPO_ROOT / "renderers" / "remotion"
SUPPORTED_OVERLAY_EFFECTS = {"G01", "G02", "G05", "L08"}


class RemotionRenderer(BaseRenderer):
    """Drives project-local Remotion CLI strictly offline using local browser executable.

    License notice: Remotion is source-available (free <=3-person org/individual; paid >=4-person
    company). Only active when user explicitly opts in (`enable_remotion=True`).
    """

    name = "remotion"

    def __init__(self, project_dir: Path | None = None, ffprobe_bin: str | None = None) -> None:
        self.project_dir = (project_dir or REPO_ROOT / "renderers" / "remotion").resolve()
        self.ffprobe_bin = ffprobe_bin or shutil.which("ffprobe") or "ffprobe"

    def binary_path(self) -> Path | None:
        return find_local_bin(self.project_dir, "remotion")

    def supports(
        self,
        spec: EffectSpec,
        media: MediaInfo,
        capabilities: dict[str, Any],
    ) -> bool:
        return (
            spec.id in SUPPORTED_OVERLAY_EFFECTS
            and "remotion" in spec.supported_engines
            and bool(capabilities.get("remotion", {}).get("available"))
        )

    def prepare(self, plan: EditPlan, source: Path, workdir: Path) -> dict[str, Any]:
        rem_bin = self.binary_path()
        browser = find_browser_executable()
        if not rem_bin:
            raise RuntimeError(f"Remotion local CLI not installed in {self.project_dir}")
        if not browser:
            raise RuntimeError("No local Chrome/Chromium executable found for offline Remotion render.")
        workdir.mkdir(parents=True, exist_ok=True)
        return {"binary": str(rem_bin), "browser": browser, "workdir": str(workdir)}

    def render_overlay_event(
        self,
        event: EffectEvent,
        plan: EditPlan,
        workdir: Path,
        audio_envelope: list[float] | None = None,
        timeout_s: float = 120.0,
    ) -> dict[str, Any]:
        """Render a transparent WebM (VP9 yuva420p) overlay composition via local Remotion CLI."""
        prep = self.prepare(plan, Path("."), workdir)
        rem_bin = prep["binary"]
        browser = prep["browser"]

        slot_dir = (workdir.resolve() / f"remotion_{event.id}").resolve()
        slot_dir.mkdir(parents=True, exist_ok=True)
        dur_frames = max(2, event.end_frame_exclusive - event.start_frame)
        fps_int = max(1, int(round(plan.fps.as_float())))

        props = {
            "effectId": event.effect_id,
            "width": plan.width,
            "height": plan.height,
            "fps": fps_int,
            "durationInFrames": dur_frames,
            "parameters": event.parameters,
            "audioEnvelope": (audio_envelope or [])[event.start_frame : event.end_frame_exclusive],
        }
        props_path = slot_dir / "props.json"
        atomic_write_json(props_path, props)

        out_webm = slot_dir / "overlay.webm"
        rem_tmp = slot_dir / "tmp"
        rem_tmp.mkdir(parents=True, exist_ok=True)

        env = dict(os.environ)
        env["REMOTION_TMP"] = str(rem_tmp)
        env["TMPDIR"] = str(rem_tmp)
        env["TEMP"] = str(rem_tmp)
        env["TMP"] = str(rem_tmp)
        env["DO_NOT_TRACK"] = "1"

        cmd = [
            rem_bin, "render",
            "src/index.ts",
            "MotionLabOverlay",
            str(out_webm),
            f"--props={props_path}",
            f"--browser-executable={browser}",
            "--codec=vp9",
            "--pixel-format=yuva420p",
            "--image-format=png",
            "--muted",
            "--concurrency=2",
            "--disable-telemetry",
        ]
        _, stats = run_managed_subprocess(cmd, timeout_s=timeout_s, cwd=self.project_dir, env=env, check=True)
        probe = self.verify_artifact(out_webm, dur_frames / fps_int)
        vstreams = [s for s in probe.get("streams", []) if s.get("codec_type") == "video"]
        pix_fmt = vstreams[0].get("pix_fmt", "") if vstreams else ""
        mid_f = max(1, dur_frames // 2)
        alpha_check = verify_webm_alpha_pixels(out_webm, plan.width, plan.height, sample_frame=mid_f)
        if not alpha_check["alpha_verified"]:
            raise RuntimeError(
                f"Remotion WebM output {out_webm} failed pixel-level alpha verification: {alpha_check}"
            )

        return {
            "event_id": event.id,
            "engine": self.name,
            "artifact_path": str(out_webm),
            "start_frame": event.start_frame,
            "end_frame_exclusive": event.end_frame_exclusive,
            "elapsed_s": stats["elapsed_s"],
            "pix_fmt": pix_fmt,
            **alpha_check,
        }

    def verify_artifact(self, artifact_path: Path, expected_duration_s: float) -> dict[str, Any]:
        proc = subprocess.run(
            [self.ffprobe_bin, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(artifact_path)],
            capture_output=True, text=True, timeout=15, shell=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"ffprobe failed on Remotion artifact {artifact_path}: {proc.stderr}")
        return json.loads(proc.stdout)
