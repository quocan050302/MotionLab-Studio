"""Phase E4: HyperFrames (Apache-2.0) HTML/CSS/GSAP offline overlay renderer adapter."""

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
from motionlab.schemas import EditPlan, EffectEvent, EffectSpec, MediaInfo, atomic_write_text

HYPERFRAMES_DIR = REPO_ROOT / "renderers" / "hyperframes"
SUPPORTED_OVERLAY_EFFECTS = {"G01", "G02", "G06", "L08"}


class HyperFramesRenderer(BaseRenderer):
    """Drives local pinned HyperFrames CLI (zero npx network fetch at runtime)."""

    name = "hyperframes"

    def __init__(self, project_dir: Path | None = None, ffprobe_bin: str | None = None) -> None:
        self.project_dir = (project_dir or HYPERFRAMES_DIR).resolve()
        self.ffprobe_bin = ffprobe_bin or shutil.which("ffprobe") or "ffprobe"

    def binary_path(self) -> Path | None:
        return find_local_bin(self.project_dir, "hyperframes")

    def supports(
        self,
        spec: EffectSpec,
        media: MediaInfo,
        capabilities: dict[str, Any],
    ) -> bool:
        return (
            spec.id in SUPPORTED_OVERLAY_EFFECTS
            and "hyperframes" in spec.supported_engines
            and bool(capabilities.get("hyperframes", {}).get("available"))
        )

    def prepare(self, plan: EditPlan, source: Path, workdir: Path) -> dict[str, Any]:
        hf_bin = self.binary_path()
        browser = find_browser_executable()
        if not hf_bin:
            raise RuntimeError(f"HyperFrames local CLI not installed in {self.project_dir}")
        if not browser:
            raise RuntimeError("No local Chrome/Chromium executable found for HyperFrames offline render.")
        workdir.mkdir(parents=True, exist_ok=True)
        return {"binary": str(hf_bin), "browser": browser, "workdir": str(workdir)}

    def _build_overlay_html(
        self,
        event: EffectEvent,
        width: int,
        height: int,
        fps: float,
        duration_s: float,
        gsap_rel_path: str,
    ) -> str:
        params = event.parameters
        accent = str(params.get("accent_hex", "#F4F6FA"))
        opacity = float(params.get("opacity", 0.58)) * float(params.get("intensity", 0.8))
        margin = float(params.get("margin_frac", 0.04))
        mx = max(8, int(round(width * margin)))
        my = max(8, int(round(height * margin)))
        eid = event.effect_id

        if eid == "G02":
            cx = int(round(width * float(params.get("cx", 0.84))))
            cy = int(round(height * float(params.get("cy", 0.16))))
            rad = max(18, int(round(min(width, height) * float(params.get("radius_frac", 0.06)))))
            svg_body = (
                f'<circle id="ring1" cx="{cx}" cy="{cy}" r="{rad}" fill="none" stroke="{accent}" '
                f'stroke-width="3" stroke-dasharray="{int(2 * 3.1416 * rad)}" stroke-dashoffset="{int(2 * 3.1416 * rad)}"/>'
                f'<circle id="ring2" cx="{cx}" cy="{cy}" r="{int(rad * 0.65)}" fill="none" stroke="{accent}" '
                f'stroke-width="2" opacity="0.65"/>'
            )
            gsap_code = (
                f'tl.to("#ring1", {{strokeDashoffset: 0, opacity: {opacity:.3f}, duration: {duration_s * 0.35:.3f}, ease: "power3.out"}}, 0);'
                f'tl.to("#ring1", {{opacity: 0, duration: {duration_s * 0.25:.3f}, ease: "power2.in"}}, {duration_s * 0.75:.3f});'
                f'tl.fromTo("#ring2", {{scale: 0.6, opacity: 0, transformOrigin: "50% 50%"}}, '
                f'{{scale: 1.05, opacity: {opacity * 0.7:.3f}, duration: {duration_s * 0.4:.3f}, ease: "power2.out"}}, 0.08);'
                f'tl.to("#ring2", {{opacity: 0, duration: {duration_s * 0.25:.3f}}}, {duration_s * 0.75:.3f});'
            )
        elif eid in ("L08", "G06"):
            bw = width - 2 * mx
            bh = height - 2 * my
            svg_body = (
                f'<rect id="frameRect" x="{mx}" y="{my}" width="{bw}" height="{bh}" rx="12" ry="12" '
                f'fill="none" stroke="{accent}" stroke-width="4" opacity="0"/>'
            )
            gsap_code = (
                f'tl.to("#frameRect", {{opacity: {opacity:.3f}, duration: {duration_s * 0.28:.3f}, ease: "power2.out"}}, 0);'
                f'tl.to("#frameRect", {{opacity: {opacity * 0.45:.3f}, duration: {duration_s * 0.34:.3f}, ease: "sine.inOut"}}, {duration_s * 0.30:.3f});'
                f'tl.to("#frameRect", {{opacity: 0, duration: {duration_s * 0.28:.3f}, ease: "power2.in"}}, {duration_s * 0.70:.3f});'
            )
        else:  # G01 edge_lines default
            seg_w = int(round((width - 2 * mx) * 0.28))
            svg_body = (
                f'<line id="ltop" x1="{mx}" y1="{my}" x2="{mx + seg_w}" y2="{my}" stroke="{accent}" '
                f'stroke-width="3" stroke-linecap="round" stroke-dasharray="{seg_w}" stroke-dashoffset="{seg_w}" opacity="{opacity:.3f}"/>'
                f'<line id="lbot" x1="{width - mx - seg_w}" y1="{height - my}" x2="{width - mx}" y2="{height - my}" '
                f'stroke="{accent}" stroke-width="3" stroke-linecap="round" stroke-dasharray="{seg_w}" stroke-dashoffset="{seg_w}" opacity="{opacity:.3f}"/>'
            )
            gsap_code = (
                f'tl.to(["#ltop","#lbot"], {{strokeDashoffset: 0, duration: {duration_s * 0.30:.3f}, ease: "power3.out"}}, 0);'
                f'tl.to(["#ltop","#lbot"], {{opacity: 0, duration: {duration_s * 0.25:.3f}, ease: "power2.in"}}, {duration_s * 0.75:.3f});'
            )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width={width}, initial-scale=1" />
<title>MotionLab HyperFrames Overlay {eid}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{ width: {width}px; height: {height}px; overflow: hidden; background: transparent; }}
  #root {{ width: {width}px; height: {height}px; position: relative; background: transparent; }}
</style>
<script src="{gsap_rel_path}"></script>
</head>
<body>
<div id="root" data-composition-id="overlay_{event.id}" data-width="{width}" data-height="{height}" data-fps="{int(round(fps))}" data-start="0" data-duration="{duration_s:.4f}">
  <svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">
    {svg_body}
  </svg>
</div>
<script>
  const tl = gsap.timeline({{ paused: true }});
  {gsap_code}
  window.__timelines = window.__timelines || {{}};
  window.__timelines["overlay_{event.id}"] = tl;
</script>
</body>
</html>
"""

    def render_overlay_event(
        self,
        event: EffectEvent,
        plan: EditPlan,
        workdir: Path,
        timeout_s: float = 120.0,
    ) -> dict[str, Any]:
        """Render a single transparent WebM (VP9 + alpha) overlay clip using local HyperFrames CLI."""
        prep = self.prepare(plan, Path("."), workdir)
        hf_bin = prep["binary"]
        browser = prep["browser"]

        slot_dir = (workdir.resolve() / f"hf_{event.id}").resolve()
        slot_dir.mkdir(parents=True, exist_ok=True)
        fps_f = plan.fps.as_float()
        dur_s = max(0.3, (event.end_frame_exclusive - event.start_frame) / fps_f)

        # Copy local GSAP script directly into slot so zero external URL or traversal is needed
        gsap_src = self.project_dir / "node_modules" / "gsap" / "dist" / "gsap.min.js"
        if not gsap_src.is_file():
            raise RuntimeError(f"Local GSAP asset missing at {gsap_src}")
        shutil.copy2(gsap_src, slot_dir / "gsap.min.js")

        html = self._build_overlay_html(event, plan.width, plan.height, fps_f, dur_s, "./gsap.min.js")
        atomic_write_text(slot_dir / "index.html", html)

        out_webm = slot_dir / "overlay.webm"
        env = dict(os.environ)
        env["CHROME_PATH"] = browser
        env["PUPPETEER_EXECUTABLE_PATH"] = browser
        env["HYPERFRAMES_DISABLE_TELEMETRY"] = "1"
        env["HYPERFRAMES_EXTRACT_CACHE_DIR"] = "off"
        env["DO_NOT_TRACK"] = "1"

        # Run verified HyperFrames lint & render flags (v0.9.98)
        lint_cmd = [hf_bin, "lint", str(slot_dir)]
        lint_res, _ = run_managed_subprocess(lint_cmd, timeout_s=30.0, cwd=self.project_dir, env=env, check=False)

        render_cmd = [
            hf_bin, "render", str(slot_dir),
            "-o", str(out_webm),
            "--format", "webm",
            "-f", str(int(round(fps_f))),
            "-w", "1",
            "--frames-cache-dir=off",
            "--quiet",
            "--no-wait",
        ]
        _, r_stats = run_managed_subprocess(render_cmd, timeout_s=timeout_s, cwd=self.project_dir, env=env, check=True)
        probe = self.verify_artifact(out_webm, dur_s)

        # Verify actual decoded RGBA alpha pixels survive libvpx-vp9 decode (never assume alpha works!)
        vstreams = [s for s in probe.get("streams", []) if s.get("codec_type") == "video"]
        pix_fmt = vstreams[0].get("pix_fmt", "") if vstreams else ""
        mid_f = max(1, (event.end_frame_exclusive - event.start_frame) // 2)
        alpha_check = verify_webm_alpha_pixels(out_webm, plan.width, plan.height, sample_frame=mid_f)
        if not alpha_check["alpha_verified"]:
            raise RuntimeError(
                f"HyperFrames output {out_webm} failed pixel-level alpha verification: {alpha_check}"
            )

        return {
            "event_id": event.id,
            "engine": self.name,
            "artifact_path": str(out_webm),
            "start_frame": event.start_frame,
            "end_frame_exclusive": event.end_frame_exclusive,
            "lint_rc": lint_res.returncode,
            "elapsed_s": r_stats["elapsed_s"],
            "pix_fmt": pix_fmt,
            **alpha_check,
        }

    def verify_artifact(self, artifact_path: Path, expected_duration_s: float) -> dict[str, Any]:
        proc = subprocess.run(
            [self.ffprobe_bin, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(artifact_path)],
            capture_output=True, text=True, timeout=15, shell=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"ffprobe failed on HyperFrames artifact {artifact_path}: {proc.stderr}")
        return json.loads(proc.stdout)
