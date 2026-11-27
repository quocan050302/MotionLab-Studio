"""Phase E3 — Local Offline Remotion Renderer Adapter.

Uses pinned local `renderers/remotion/node_modules/.bin/remotion` and an already-installed
local Chrome/Edge browser (`--browser-executable`), with zero `npx`, zero network downloads,
and explicit Remotion license governance.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from motionlab.config import PROJECT_ROOT, atomic_write_json
from motionlab.renderers.base import BaseRenderer, RendererError, find_local_browser, probe_artifact_ffprobe
from motionlab.resource import ResourceBudget, run_subprocess_guarded
from motionlab.schemas import AnalysisResult, EditPlan, EffectEvent, EffectSpec, MediaInfo

REMOTION_SUPPORTED_EFFECTS = {
    "edge_lines",
    "foreground_frame_accent",
    "ring_accent",
    "directional_shapes",
    "ambient_particles",
    "waveform_accent",
    "pulse_border",
}

API_KEY_ENV_NAMES = (
    "ELEVENLABS_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "REPLICATE_API_TOKEN",
    "HUGGINGFACE_TOKEN",
)


def _sanitized_offline_env() -> dict[str, str]:
    env = dict(os.environ)
    for key in API_KEY_ENV_NAMES:
        env.pop(key, None)
    env["REMOTION_DISABLE_TELEMETRY"] = "1"
    env["DO_NOT_TRACK"] = "1"
    env["CI"] = "1"
    return env


class RemotionRenderer(BaseRenderer):
    name = "remotion"

    def __init__(self, license_allowed: bool = True, project_dir: Path | None = None) -> None:
        self.license_allowed = license_allowed
        self.project_dir = (project_dir or (PROJECT_ROOT / "renderers" / "remotion")).resolve()
        self.entry_point = self.project_dir / "src" / "index.ts"
        bin_name = "remotion.cmd" if os.name == "nt" else "remotion"
        self.remotion_bin = self.project_dir / "node_modules" / ".bin" / bin_name
        self.browser_exe = find_local_browser()

    def probe_capability(self) -> dict[str, Any]:
        license_note = (
            "Remotion License: Free for individuals, non-profits, evaluators, and for-profit teams <= 3 people; "
            "Company License required for for-profit organizations >= 4 people (https://www.remotion.dev/license)."
        )
        if not self.license_allowed:
            return {
                "available": False,
                "reason": "disabled_by_license_or_flag",
                "license": license_note,
            }
        if not self.remotion_bin.exists():
            return {
                "available": False,
                "reason": f"local_remotion_cli_missing:{self.remotion_bin}",
                "license": license_note,
            }
        if not self.entry_point.exists():
            return {
                "available": False,
                "reason": f"remotion_entry_missing:{self.entry_point}",
                "license": license_note,
            }
        if not self.browser_exe:
            return {
                "available": False,
                "reason": "local_chrome_or_edge_not_found_offline",
                "license": license_note,
            }
        pkg_json = self.project_dir / "package.json"
        version = "unknown"
        if pkg_json.exists():
            try:
                version = json.loads(pkg_json.read_text(encoding="utf-8")).get("dependencies", {}).get("remotion", "unknown")
            except Exception:
                pass
        return {
            "available": True,
            "version": version,
            "executable": str(self.remotion_bin),
            "browser_executable": self.browser_exe,
            "supported_effects": sorted(REMOTION_SUPPORTED_EFFECTS),
            "license": license_note,
            "reason": "ok",
        }

    def supports(self, effect_spec: EffectSpec, media: MediaInfo, capabilities: dict[str, Any]) -> bool:
        cap = capabilities.get("remotion") or {}
        return bool(cap.get("available")) and effect_spec.id in REMOTION_SUPPORTED_EFFECTS

    def verify_artifact(self, artifact_path: Path, expected_duration_s: float | None = None) -> dict[str, Any]:
        info = probe_artifact_ffprobe(artifact_path)
        v_streams = [s for s in info.get("streams", []) if s.get("codec_type") == "video"]
        if not v_streams:
            raise RendererError(f"Remotion artifact has no video stream: {artifact_path}")
        pix_fmt = str(v_streams[0].get("pix_fmt") or "")
        tags = {str(k).lower(): str(v) for k, v in (v_streams[0].get("tags") or {}).items()}
        has_alpha = ("a" in pix_fmt.lower()) or (tags.get("alpha_mode") == "1")
        if not has_alpha:
            raise RendererError(
                f"Remotion WebM overlay did not preserve alpha channel (pix_fmt={pix_fmt}, tags={tags})"
            )
        return info

    def render_overlay_event(
        self,
        event: EffectEvent,
        plan: EditPlan,
        analysis: AnalysisResult,
        out_webm: Path,
        work_dir: Path,
        budget: ResourceBudget | None = None,
    ) -> dict[str, Any]:
        """Render a single transparent WebM VP9 overlay clip using pinned local Remotion CLI."""
        cap = self.probe_capability()
        if not cap.get("available"):
            raise RendererError(f"Remotion unavailable: {cap.get('reason')}")

        t0 = time.monotonic()
        work_dir.mkdir(parents=True, exist_ok=True)
        out_webm.parent.mkdir(parents=True, exist_ok=True)

        dur_frames = max(2, event.end_frame_exclusive - event.start_frame)
        fps_int = max(1, int(round(plan.fps_num / plan.fps_den)))

        # Build exact per-frame audio envelope slice for audio-reactive overlays
        audio_lookup = {pt.frame: pt.rms_norm for pt in analysis.audio_envelope}
        env_slice: list[float] = []
        for rel_f in range(dur_frames):
            abs_f = event.start_frame + rel_f
            val = audio_lookup.get(abs_f)
            if val is None:
                for d in range(1, 5):
                    if (abs_f - d) in audio_lookup:
                        val = audio_lookup[abs_f - d]
                        break
                    if (abs_f + d) in audio_lookup:
                        val = audio_lookup[abs_f + d]
                        break
            env_slice.append(round(float(val or 0.0), 4))

        props = {
            "effectId": event.effect_id,
            "width": plan.width,
            "height": plan.height,
            "fps": fps_int,
            "durationInFrames": dur_frames,
            "intensity": float(event.parameters.get("intensity", 0.75)),
            "opacity": float(event.parameters.get("opacity", 0.65)),
            "marginRatio": float(event.parameters.get("margin_ratio", 0.045)),
            "seed": int(plan.seed),
            "regionCx": float(event.region.get("cx", 0.84)),
            "regionCy": float(event.region.get("cy", 0.86)),
            "barCount": int(event.parameters.get("bar_count", 16)),
            "particleCount": int(event.parameters.get("particle_count", 12)),
            "strokePx": int(event.parameters.get("stroke_px", 3)),
            "audioEnvelope": env_slice,
        }
        props_file = work_dir / f"remotion_props_{event.id}.json"
        atomic_write_json(props_file, props)

        cmd = [
            str(self.remotion_bin),
            "render",
            str(self.entry_point),
            "MotionLabOverlay",
            str(out_webm),
            f"--props={props_file}",
            "--codec=vp9",
            "--pixel-format=yuva420p",
            "--image-format=png",
            "--concurrency=1",
            "--log=error",
            f"--browser-executable={self.browser_exe}",
        ]
        res = run_subprocess_guarded(
            cmd,
            budget=budget,
            cwd=self.project_dir,
            env=_sanitized_offline_env(),
            timeout_s=min(240, budget.config.subprocess_timeout_s if budget else 240),
            check=False,
        )
        if res.returncode != 0:
            raise RendererError(
                f"Remotion render failed for {event.id} (exit {res.returncode}): {(res.stderr or res.stdout).strip()[:600]}"
            )
        info = self.verify_artifact(out_webm)
        return {
            "engine": "remotion",
            "artifact_path": str(out_webm),
            "elapsed_s": round(time.monotonic() - t0, 4),
            "ffprobe": info,
        }
